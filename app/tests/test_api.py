import io
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app


def upload():
    data = io.BytesIO()
    Image.new("RGB", (48, 48), "green").save(data, format="PNG")
    return {"image": ("input.png", data.getvalue(), "image/png")}


class FakeEngine:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.options = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = None

    def generate(self, image, progress, **options):
        self.options = options
        self.entered.set()
        progress("Testing the API")
        self.release.wait(5)
        if self.failure:
            raise self.failure
        folder = self.folder / uuid.uuid4().hex
        folder.mkdir()
        ply = folder / "model.ply"
        ply.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
        return str(ply), "API test fixture"


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = FakeEngine(self.temp.name)
        self.app = create_app(self.engine, self.temp.name)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.engine.release.set()
        self.app.state.jobs.executor.shutdown(wait=True)
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def finish(self, response):
        self.assertEqual(response.status_code, 202, response.text)
        for _ in range(100):
            job = self.client.get(response.json()["status_url"]).json()
            if job["state"] != "running":
                return job
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_upload_status_preview_download_and_persistence(self):
        job = self.finish(self.client.post("/api/generate", files=upload(), data={"model": "standard"}))
        self.assertEqual(job["state"], "done")
        self.assertEqual(self.engine.options["model"], "standard")
        self.assertTrue(self.engine.options["low_memory"])
        result = job["result"]
        preview = self.client.get(result["url"])
        download = self.client.get(result["download_url"])
        self.assertTrue(preview.content.startswith(b"ply\n"))
        self.assertEqual(preview.content, download.content)
        self.assertIn("attachment", download.headers["content-disposition"])
        # Result files are independent of the in-memory job list (and New image).
        self.app.state.jobs.records.clear()
        self.assertEqual(self.client.get(result["download_url"]).status_code, 200)

    def test_busy_request_is_rejected_instead_of_queued(self):
        self.engine.release.clear()
        first = self.client.post("/api/generate", files=upload())
        self.assertTrue(self.engine.entered.wait(2))
        second = self.client.post("/api/generate", files=upload())
        self.assertEqual(second.status_code, 409)
        self.engine.release.set()
        self.assertEqual(self.finish(first)["state"], "done")

    def test_elapsed_time_reconnects_and_stops_on_success_or_failure(self):
        for failure in (None, RuntimeError("interrupted")):
            with self.subTest(failure=failure), patch("app.logging.exception"):
                self.engine.release.clear()
                self.engine.entered.clear()
                self.engine.failure = failure
                response = self.client.post("/api/generate", files=upload())
                self.assertTrue(self.engine.entered.wait(2))
                job_id = response.json()["id"]
                with self.app.state.jobs.lock:
                    self.app.state.jobs.records[job_id]["_started"] -= 123
                running = self.client.get(response.json()["status_url"]).json()
                self.assertEqual(running["state"], "running")
                self.assertGreaterEqual(running["elapsed_seconds"], 123)
                self.assertLess(running["elapsed_seconds"], 128)
                self.assertNotIn("_started", running)
                self.engine.release.set()
                finished = self.finish(response)
                self.assertEqual(finished["state"], "error" if failure else "done")
                again = self.client.get(response.json()["status_url"]).json()
                self.assertEqual(again["elapsed_seconds"], finished["elapsed_seconds"])

    def test_invalid_upload_and_options_never_start_inference(self):
        self.assertEqual(self.client.post("/api/generate").status_code, 422)
        self.assertEqual(self.client.post("/api/generate", files={"image": ("bad.png", b"bad image", "image/png")}).status_code, 422)
        for data in [{"model": "fake"}, {"detail": 123}, {"seed": -1}, {"seed": "1.5"}, {"device": "fake"}]:
            with self.subTest(data=data):
                self.assertEqual(self.client.post("/api/generate", files=upload(), data=data).status_code, 422)
        self.assertFalse(self.engine.entered.is_set())

    def test_upload_size_is_bounded(self):
        with patch("app.MAX_UPLOAD", 10):
            self.assertEqual(self.client.post("/api/generate", files=upload()).status_code, 413)

    def test_encoder_access_error_is_actionable_and_retryable(self):
        self.engine.failure = RuntimeError("gated repo 403")
        with patch("app.logging.exception"):
            job = self.finish(self.client.post("/api/generate", files=upload()))
        self.assertEqual(job["state"], "error")
        self.assertTrue(job["error"]["setup_required"])
        self.engine.failure = None
        self.assertEqual(self.finish(self.client.post("/api/generate", files=upload()))["state"], "done")

    def test_routes_restrict_files_and_report_unknown_jobs(self):
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)
        for key in ["README.md", "..", "a" * 32]:
            self.assertEqual(self.client.get(f"/api/results/{key}/model.ply").status_code, 404)
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        self.assertIn("Drop an image", self.client.get("/").text)


if __name__ == "__main__":
    unittest.main()
