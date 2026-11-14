"""Local IDOL adapter. The pinned upstream SDK remains unmodified."""
import gc
import importlib.util
import json
import math
import threading
import time
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
TURBO_REPO = "speridlabs/idol-turbo"
TURBO_REVISION = "45cdd41f1f4d6d70496daa7d619f57ad79307a37"
DETAILS = [32768, 65536, 131072, 262144, 524288, 1048576]


def encoder_precision(device):
    # The pretrained DINOv3 encoder loses its image features in FP16 on MPS.
    # Use its supported BF16 path, with FP32 on devices without BF16 support.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    if device.type == "mps" and torch.backends.mps.is_macos_or_newer(14, 0):
        return "bf16"
    return "fp32"


def resolve_device(requested="auto"):
    if requested not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError("Choose Auto, NVIDIA CUDA, Apple Silicon, or CPU.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Choose Auto or CPU, or check your NVIDIA driver.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple GPU acceleration is unavailable. Choose Auto or CPU.")
    return torch.device(requested)


def clear_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def chunked_attention(attention, *args, chunk_size=1024):
    """Limit the MPS attention workspace while keeping every key and value."""
    if args[0].shape[1] <= chunk_size:
        return attention(*args)
    q, k, v = args[0].unbind(dim=2) if len(args) == 1 else args
    return torch.cat([attention(q[:, start:start + chunk_size], k, v)
                      for start in range(0, q.shape[1], chunk_size)], dim=1)


@contextmanager
def sampling_progress(denoiser, steps, passes_per_step, progress):
    """Observe the pinned samplers without changing their inference loop."""
    passes = 0

    def completed(module, inputs, output):
        nonlocal passes
        passes += 1
        # Standard uses separate conditional/unconditional passes (optimize=False).
        if passes % passes_per_step == 0:
            progress(f"Generating 3D · step {passes // passes_per_step} of {steps}")

    handle = denoiser.register_forward_hook(completed)
    try:
        yield
    finally:
        handle.remove()


class Residency:
    """Keep just the active stage on the accelerator in low-memory mode."""
    def __init__(self, device, low_memory, progress=lambda *args: None):
        self.device = device
        self.low_memory = low_memory and device.type != "cpu"
        self.active = None
        self.progress = progress

    def activate(self, module, label):
        if self.active is module:
            return
        self.progress(label)
        if self.low_memory and self.active is not None:
            self.active.to("cpu")
            self.active = None
            clear_cache(self.device)
        module.to(self.device)
        self.active = module

    def offload(self):
        if self.low_memory and self.active is not None:
            self.active.to("cpu")
            self.active = None
            clear_cache(self.device)


class StagedEncoder:
    """Stage the encoders independently; avoid hard-coded BF16 on CPU/MPS/older CUDA."""
    def __init__(self, encoder, residency):
        self.encoder = encoder
        self.residency = residency

    @torch.no_grad()
    def __call__(self, images):
        from tensordict import TensorDict

        r, e = self.residency, self.encoder
        r.activate(e.dinov3, "Reading the image")
        feature1 = e._dinov3_tokens(images)
        r.activate(e.flux2_vae, "Encoding appearance")
        vae = e.flux2_vae
        # Same normalization/layout as the pinned SDK's _flux2_tokens, using the
        # VAE's actual precision instead of unconditionally creating BF16 input.
        latents = vae.encode(images.to(next(vae.parameters()).dtype) * 2 - 1).latent_dist.sample()
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        std = (vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(latents)
        tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
        padding = torch.zeros(batch, 5, tokens.shape[2], device=tokens.device)
        return TensorDict({"feature1": feature1, "feature2": torch.cat([padding, tokens], dim=1)}, batch_size=batch)


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.pipeline = None
        self.residency = None
        self.key = None
        self.turbo = None

    def _unload(self):
        device = self.residency.device if self.residency else torch.device("cpu")
        # Turbo changes the SDK's module-level sampler. Restore it even when a
        # device/memory change creates a new, unpatched Standard pipeline.
        if self.turbo is not None and self.turbo._ORIG.get("sample_flow"):
            import idol.pipeline as pipeline_module
            pipeline_module.sample_flow = self.turbo._ORIG["sample_flow"]
        # Remove hooks before releasing the pipeline, breaking hook closure cycles.
        if self.pipeline is not None:
            for handle in self.pipeline._local_hooks:
                handle.remove()
        self.pipeline = self.residency = self.key = None
        clear_cache(device)

    def _load(self, device, low_memory, progress):
        from idol import IdolPipeline
        from idol.models.layers import RePo3DRotaryEmbedding

        key = (str(device), bool(low_memory))
        if self.pipeline is not None and self.key == key:
            self.residency.progress = progress
            return self.pipeline
        self._unload()
        progress("Loading models · first use downloads the weights")
        # CPU-first loading avoids needing space for every model on the GPU at once.
        pipe = IdolPipeline.from_pretrained("speridlabs/idol", device="cpu", dinov3_dtype=encoder_precision(device), optimize=False,
            image_encoders={
                "dinov3": "cocktailpeanut/dinov3-vith16plus-pretrain-lvd1689m",
                "flux2_vae": "cocktailpeanut/FLUX.2-vae",
            })
        pipe.device = device
        encoder = pipe.image_encoder
        encoder.device = device
        if device.type == "cpu":
            for model in (pipe.denoiser, pipe.decoder, pipe.decoder_gs):
                model.float()
                model.dtype = torch.float32
        else:
            # MPS deformable convolution falls back to CPU, where FP16 is very slow.
            if device.type == "cuda":
                pipe.background_remover.half()
        # FP32 VAE on CPU/MPS and pre-Ampere CUDA; BF16 on supported NVIDIA GPUs.
        encoder.flux2_vae.to(dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32)
        if device.type == "mps":
            for module in pipe.denoiser.modules():
                if isinstance(module, RePo3DRotaryEmbedding):
                    module.real_valued = True  # MPS has no complex-tensor support.
        residency = Residency(device, low_memory, progress)
        modules = [pipe.background_remover, encoder.dinov3, encoder.flux2_vae, pipe.denoiser, pipe.decoder, pipe.decoder_gs]
        if not residency.low_memory:
            for module in modules:
                module.to(device)
        preprocess = pipe.preprocess

        def staged_preprocess(*args, **kwargs):
            residency.activate(pipe.background_remover, "Removing the background")
            return preprocess(*args, **kwargs)

        pipe.preprocess = staged_preprocess
        pipe.image_encoder = StagedEncoder(encoder, residency)
        pipe._local_hooks = []
        for module, label in [(pipe.denoiser, "Generating the 3D shape"), (pipe.decoder, "Placing splats"), (pipe.decoder_gs, "Adding color and detail")]:
            pipe._local_hooks.append(module.register_forward_pre_hook(lambda model, inputs, label=label: residency.activate(model, label)))
        self.pipeline, self.residency, self.key = pipe, residency, key
        return pipe

    def _set_model(self, pipe, model):
        if model == "turbo":
            if self.turbo is None:
                from huggingface_hub import hf_hub_download
                filename = hf_hub_download(TURBO_REPO, "turbo.py", revision=TURBO_REVISION)
                spec = importlib.util.spec_from_file_location("idol_local_turbo", filename)
                self.turbo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.turbo)
            if not getattr(pipe, "_local_turbo_patched", False):
                # Pin the LoRA download as well as the adapter code.
                from huggingface_hub import snapshot_download
                folder = snapshot_download(TURBO_REPO, revision=TURBO_REVISION, allow_patterns=["idol_turbo_lora.safetensors"])
                from unittest.mock import patch
                import huggingface_hub
                original_download = huggingface_hub.hf_hub_download

                def pinned_download(repo, filename, **kwargs):
                    if repo == TURBO_REPO and filename == self.turbo.LORA_FILE:
                        return str(Path(folder) / filename)
                    return original_download(repo, filename, **kwargs)

                # The official patcher installs both phase experts and its sampler.
                # Resolve its one LoRA download from our pinned snapshot.
                with patch("huggingface_hub.hf_hub_download", side_effect=pinned_download):
                    self.turbo.patch_pipeline_for_turbo(pipe, TURBO_REPO)
                pipe._local_turbo_patched = True
            self.turbo.set_turbo_mode(pipe, True)
        elif getattr(pipe, "_local_turbo_patched", False):
            self.turbo.set_turbo_mode(pipe, False)
        # New LoRA parameters start on CPU; move them with the rest of this stage.
        if not self.residency.low_memory:
            pipe.denoiser.to(pipe.device)

    def generate(self, image, model="turbo", input_type="real", low_memory=True, detail=65536, seed=42, device="auto", progress=lambda *args: None):
        if image is None:
            raise ValueError("Upload an image first.")
        if model not in ("turbo", "standard") or input_type not in ("real", "synthetic"):
            raise ValueError("Choose a listed model and image type.")
        if detail not in DETAILS or not math.isfinite(float(seed)) or not 0 <= float(seed) <= 2**32 - 1 or float(seed) != int(seed):
            raise ValueError("Choose a listed splat count and a whole-number seed between 0 and 4294967295.")
        image = ImageOps.exif_transpose(image).convert("RGBA")
        if min(image.size) < 16 or image.width / image.height > 8 or image.height / image.width > 8:
            raise ValueError("Use an image at least 16 pixels wide and tall, with an aspect ratio no greater than 8:1.")
        if image.getextrema()[3][1] == 0:
            raise ValueError("This image is fully transparent. Upload an image with a visible subject.")
        with self.lock:
            started = time.monotonic()
            target = resolve_device(device)
            pipe = self._load(target, low_memory, progress)
            import idol.models.layers as layers
            backend, tf32 = layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32
            attention = layers.scaled_dot_product_attention
            try:
                if target.type == "mps" and low_memory:
                    layers.scaled_dot_product_attention = partial(chunked_attention, attention)
                progress("Preparing " + ("IDOL Turbo" if model == "turbo" else "IDOL"))
                self._set_model(pipe, model)
                steps = 5 if model == "turbo" else 50
                with sampling_progress(pipe.denoiser, steps, 1 if model == "turbo" else 2, progress):
                    output = pipe(image, mode=input_type, seed=int(seed), steps=steps, cfg_strength=0.0 if model == "turbo" else 3.0, num_gaussians=int(detail))
                progress("Saving the 3D file")
                folder = ROOT / "outputs" / uuid.uuid4().hex
                folder.mkdir(parents=True)
                ply = folder / "model.ply"
                output.gaussian.save_ply(str(ply))
                output.preprocessed_image.save(folder / "input.png")
                (folder / "camera.json").write_text(json.dumps(output.camera, indent=2))
                elapsed = time.monotonic() - started
                return str(ply), f"{model.title()} · {detail:,} splats · {elapsed:.1f}s · {target.type.upper()}"
            except Exception:
                # An interrupted download/adapter load must not leave a half-patched
                # pipeline cached for the next request.
                self._unload()
                raise
            finally:
                layers.scaled_dot_product_attention = attention
                layers.ATTN_BACKEND = backend
                torch.backends.cuda.matmul.allow_tf32 = tf32
                if self.residency:
                    self.residency.offload()
