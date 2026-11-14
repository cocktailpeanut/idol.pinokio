"""Local canvas UI and a small asynchronous image-to-3D API."""
import os
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import io
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from engine import DETAILS, Engine

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 20 * 1024 * 1024


def friendly_error(exc):
    text = str(exc)
    if "out of memory" in text.lower():
        return {"message": "Memory ran out. Close other GPU apps and try again.", "setup_required": False}
    if "gated" in text.lower() or "401" in text or "403" in text:
        return {"message": "Could not download model weights. Check the repository links and Pinokio terminal, then retry.", "setup_required": True}
    return {"message": text or "Generation failed. Check the Pinokio terminal and try again.", "setup_required": False}


class Jobs:
    """One active request, bounded metadata, and no duplicate model copies."""
    def __init__(self, engine):
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="idol")
        self.lock = threading.Lock()
        self.records = {}
        self.busy = False

    def submit(self, image, options):
        with self.lock:
            if self.busy:
                raise HTTPException(409, "A generation is already running. Wait for it to finish, then try again.")
            self.busy = True
            job_id = uuid.uuid4().hex
            self.records[job_id] = {"id": job_id, "state": "running", "phase": "Preparing the model", "_started": time.monotonic()}
            while len(self.records) > 32:
                del self.records[next(iter(self.records))]
        try:
            self.executor.submit(self._run, job_id, image, options)
        except Exception:
            with self.lock:
                self.busy = False
                del self.records[job_id]
            raise
        return job_id

    def _run(self, job_id, image, options):
        def progress(phase):
            with self.lock:
                self.records[job_id]["phase"] = phase
        try:
            ply, summary = self.engine.generate(image, **options, progress=progress)
            result_id = Path(ply).parent.name
            result = {"url": f"/api/results/{result_id}/model.ply", "download_url": f"/api/results/{result_id}/model.ply?download=true", "summary": summary}
            with self.lock:
                record = self.records[job_id]
                record.update(state="done", phase="Ready", result=result, elapsed_seconds=time.monotonic() - record["_started"])
        except Exception as exc:
            logging.exception("IDOL generation failed")
            with self.lock:
                record = self.records[job_id]
                record.update(state="error", error=friendly_error(exc), elapsed_seconds=time.monotonic() - record["_started"])
        finally:
            image.close()
            with self.lock:
                self.busy = False

    def get(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise HTTPException(404, "This generation is no longer available. Choose an image to start again.")
            record = dict(self.records[job_id])
            started = record.pop("_started")
            if record["state"] == "running":
                record["elapsed_seconds"] = time.monotonic() - started
            return record


def create_app(engine=None, output_root=None):
    jobs = Jobs(engine or Engine())
    outputs = Path(output_root) if output_root else ROOT / "outputs"

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.executor.shutdown(wait=False)

    app = FastAPI(title="idol", version="1.0", lifespan=lifespan)
    app.state.jobs = jobs
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static/index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/generate", status_code=202)
    async def generate(
        image: Annotated[UploadFile, File()],
        model: Annotated[Literal["turbo", "standard"], Form()] = "turbo",
        input_type: Annotated[Literal["real", "synthetic"], Form()] = "real",
        low_memory: Annotated[bool, Form()] = True,
        detail: Annotated[int, Form()] = 65536,
        seed: Annotated[int, Form(ge=0, le=4294967295)] = 42,
        device: Annotated[Literal["auto", "cuda", "mps", "cpu"], Form()] = "auto",
    ):
        if detail not in DETAILS:
            raise HTTPException(422, "Choose a supported splat count.")
        data = await image.read(MAX_UPLOAD + 1)
        await image.close()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Choose an image smaller than 20 MB.")
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ("PNG", "JPEG", "WEBP"):
                    raise HTTPException(422, "Choose a PNG, JPEG, or WebP image.")
                source.load()
                prepared = source.copy()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "This file could not be read as an image. Choose a PNG, JPEG, or WebP.") from exc
        try:
            job_id = jobs.submit(prepared, {"model": model, "input_type": input_type, "low_memory": low_memory, "detail": detail, "seed": seed, "device": device})
        except Exception:
            prepared.close()
            raise
        return {"id": job_id, "status_url": f"/api/jobs/{job_id}"}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return jobs.get(job_id)

    @app.get("/api/results/{result_id}/model.ply")
    def result_file(result_id: str, download: bool = False):
        if len(result_id) != 32 or any(c not in "0123456789abcdef" for c in result_id):
            raise HTTPException(404, "Result not found.")
        path = outputs / result_id / "model.ply"
        if not path.is_file():
            raise HTTPException(404, "Result not found.")
        return FileResponse(path, media_type="application/octet-stream", filename="model.ply" if download else None)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Port 0 lets the OS assign an available port. Uvicorn prints the actual
    # serving URL after startup, which Pinokio captures from the terminal.
    uvicorn.run(app, host="127.0.0.1", port=0, access_log=False)
