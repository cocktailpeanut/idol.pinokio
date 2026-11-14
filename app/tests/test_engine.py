import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from PIL import Image
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Engine, Residency, StagedEncoder, chunked_attention, resolve_device, sampling_progress
from idol.models.dit import LatentSeqMMFlowModel
from idol.models.decoder import OctreeProbabilityFixedlenDecoder, GaussianFixedlenDecoder
from idol.models.layers import RePo3DRotaryEmbedding


def tiny_denoiser(device="cpu"):
    model = LatentSeqMMFlowModel(q_token_length=8, in_channels=4, cam_channels=5,
        model_channels=32, cond_channels=12, cond2_channels=8, num_blocks=1,
        num_refiner_blocks=2, num_heads=4, num_head_channels=8)
    if device == "cpu":
        model.float()
        model.dtype = torch.float32
    else:
        for module in model.modules():
            if isinstance(module, RePo3DRotaryEmbedding):
                module.real_valued = True
    return model.eval().to(device)


class EngineTests(unittest.TestCase):
    def test_progress_counts_guided_steps_in_the_upstream_sampler(self):
        from idol.sampler import sample_flow

        class Velocity(torch.nn.Module):
            def forward(self, sample, *args):
                return sample * .1

        model = Velocity()
        noise = torch.ones(1, 2)
        expected = sample_flow(model, noise.clone(), None, None, steps=50, verbose=False)
        phases = []
        with sampling_progress(model, 50, 2, phases.append):
            actual = sample_flow(model, noise.clone(), None, None, steps=50, verbose=False)
        self.assertEqual(phases, [f"Generating 3D · step {i} of 50" for i in range(1, 51)])
        torch.testing.assert_close(actual, expected)
        self.assertFalse(model._forward_hooks)

    def test_failed_pass_is_not_counted_and_progress_hook_is_removed(self):
        class Interrupted(torch.nn.Module):
            calls = 0
            def forward(self, value):
                self.calls += 1
                if self.calls == 4:
                    raise RuntimeError("interrupted")
                return value

        model, phases = Interrupted(), []
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with sampling_progress(model, 50, 2, phases.append):
                for _ in range(4):
                    model(torch.ones(1))
        self.assertEqual(phases, ["Generating 3D · step 1 of 50"])
        self.assertFalse(model._forward_hooks)

    def test_chunked_attention_preserves_full_self_and_cross_attention(self):
        self._chunked_attention("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_chunked_attention_matches_original(self):
        self._chunked_attention("mps")

    def _chunked_attention(self, device):
        from idol.models.layers import scaled_dot_product_attention
        dtype = torch.float16 if device == "mps" else torch.float32
        q = torch.randn(1, 137, 4, 16, device=device, dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        packed = torch.stack([q, k, v], dim=2)
        for args in [(packed,), (q, k[:, :79], v[:, :79])]:
            with self.subTest(device=device, packed=len(args) == 1), torch.no_grad():
                expected = scaled_dot_product_attention(*args)
                actual = chunked_attention(scaled_dot_product_attention, *args, chunk_size=32)
                torch.testing.assert_close(actual, expected, atol=2e-3 if device == "mps" else 1e-6,
                                           rtol=2e-3 if device == "mps" else 1e-5)

    def test_failed_mps_generation_restores_attention(self):
        import idol.models.layers as layers
        original = layers.scaled_dot_product_attention
        test = self

        class FailedPipeline:
            _local_hooks = []
            denoiser = torch.nn.Identity()
            def __call__(self, *args, **kwargs):
                test.assertIsNot(layers.scaled_dot_product_attention, original)
                raise RuntimeError("inference failed")

        engine = Engine()
        engine.pipeline = FailedPipeline()
        engine.residency = Residency(torch.device("mps"), True)
        with patch("engine.resolve_device", return_value=torch.device("mps")), patch("engine.clear_cache"), \
             patch.object(engine, "_load", return_value=engine.pipeline), patch.object(engine, "_set_model"):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                engine.generate(Image.new("RGB", (32, 32)))
        self.assertIs(layers.scaled_dot_product_attention, original)
        self.assertFalse(FailedPipeline.denoiser._forward_hooks)

    def test_validation_precedes_loading(self):
        engine = Engine()
        with patch.object(engine, "_load", side_effect=AssertionError("should not load")):
            for image, kwargs in [
                (None, {}), (Image.new("RGB", (32, 32)), {"model": "invalid"}),
                (Image.new("RGBA", (32, 32), (0, 0, 0, 0)), {}),
                (Image.new("RGB", (32, 32)), {"seed": math.nan}),
                (Image.new("RGB", (32, 32)), {"seed": 1.5}),
                (Image.new("RGB", (32, 32)), {"detail": 123}),
                (Image.new("RGB", (3, 32)), {}),
                (Image.new("RGB", (1000, 32)), {}),
            ]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    engine.generate(image, **kwargs)

    def test_explicit_unavailable_device_is_actionable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                resolve_device("cuda")

    def test_offload_only_keeps_one_stage_resident(self):
        events = []
        class Module:
            def __init__(self, name): self.name = name
            def to(self, device): events.append((self.name, str(device))); return self
        a, b = Module("a"), Module("b")
        r = Residency(torch.device("cuda"), True)
        with patch("engine.clear_cache"):
            r.activate(a, "first")
            r.activate(a, "same")
            r.activate(b, "next")
            r.offload()
        self.assertEqual(events, [("a", "cuda"), ("a", "cpu"), ("b", "cuda"), ("b", "cpu")])

    def test_cpu_denoiser_uses_full_precision(self):
        self._denoiser_forward("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_real_rotary_denoiser(self):
        self._denoiser_forward("mps")

    def _denoiser_forward(self, device):
        model = tiny_denoiser(device)
        x = TensorDict({"latent": torch.randn(1, 8, 4, device=device), "camera": torch.randn(1, 1, 5, device=device)}, batch_size=1)
        cond = TensorDict({"feature1": torch.randn(1, 5, 12, device=device), "feature2": torch.randn(1, 5, 8, device=device)}, batch_size=1)
        with torch.no_grad():
            result = model(x, torch.ones(1, device=device), cond)
        self.assertEqual(result["latent"].shape, x["latent"].shape)
        self.assertTrue(result["latent"].isfinite().all())

    def test_cpu_decoder_stages(self):
        self._decoder_forward("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_decoder_stages(self):
        self._decoder_forward("mps")

    def _decoder_forward(self, device):
        octree = OctreeProbabilityFixedlenDecoder(32, 4, 1, 4).eval()
        config = {"num_gaussians": 32, "lr": {k: 1.0 for k in ["_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"]}, "perturbe_size": 0.1, "scaling_bias": 0.01, "opacity_bias": 0.1, "3d_filter_kernel_size": 0.001}
        decoder = GaussianFixedlenDecoder(3, 32, 4, 1, 4, config).eval()
        for model in [octree, decoder]:
            if device == "cpu": model.float(); model.dtype = torch.float32
            model.to(device)
        with torch.no_grad():
            cond = torch.randn(1, 8, 4, device=device)
            anchors = octree.sample(octree, cond, 4, 2)
            gaussian = decoder.to_representation(anchors, decoder(anchors, cond))[0]
            with tempfile.TemporaryDirectory() as tmp:
                ply = Path(tmp) / "model.ply"
                gaussian.save_ply(str(ply))
                self.assertTrue(ply.read_bytes().startswith(b"ply\n"))

    def test_staged_encoder_preserves_sdk_token_layout(self):
        from idol.conditioning import ImageEncoder
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
                self.bn = SimpleNamespace(running_mean=torch.zeros(4), running_var=torch.ones(4))
                self.config = SimpleNamespace(batch_norm_eps=1e-5)
            def encode(self, images):
                self.received_dtype = images.dtype
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.arange(16, dtype=images.dtype).view(1, 1, 4, 4)))
        encoder = object.__new__(ImageEncoder)
        encoder.dinov3 = torch.nn.Linear(1, 1)
        encoder.flux2_vae = VAE()
        encoder._dinov3_tokens = lambda images: torch.ones(1, 9, 4)
        image = torch.rand(1, 3, 8, 8)
        expected = ImageEncoder._flux2_tokens(encoder, image)
        actual = StagedEncoder(encoder, Residency(torch.device("cpu"), True))(image)
        torch.testing.assert_close(expected, actual["feature2"])
        self.assertEqual(actual["feature1"].shape, (1, 9, 4))

    def test_cpu_real_encoder_components(self):
        self._encoder_forward("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_real_encoder_components_with_offload(self):
        self._encoder_forward("mps")

    def _encoder_forward(self, device):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from diffusers.models import AutoencoderKLFlux2
        from idol.conditioning import ImageEncoder
        encoder = object.__new__(ImageEncoder)
        encoder.dinov3 = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, patch_size=4, image_size=32, num_register_tokens=4)).eval()
        if device == "mps": encoder.dinov3.half()
        encoder.flux2_vae = AutoencoderKLFlux2(block_out_channels=(32, 32), down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2, layers_per_block=1, latent_channels=4, sample_size=32).eval().float()
        residency = Residency(torch.device(device), True)
        encoded = StagedEncoder(encoder, residency)(torch.rand(1, 3, 32, 32, device=device))
        self.assertEqual(encoded["feature1"].shape[:2], encoded["feature2"].shape[:2])
        self.assertTrue(encoded["feature1"].isfinite().all())
        self.assertTrue(encoded["feature2"].isfinite().all())
        if device == "mps":
            self.assertEqual(next(encoder.dinov3.parameters()).device.type, "cpu")
        residency.offload()
        self.assertEqual(next(encoder.flux2_vae.parameters()).device.type, "cpu")

    def test_oom_message_explains_recovery(self):
        from app import friendly_error
        error = friendly_error(RuntimeError("out of memory"))
        self.assertIn("Close other GPU apps", error["message"])
        self.assertFalse(error["setup_required"])

    def test_unload_restores_standard_sampler_after_turbo(self):
        import idol.pipeline as ip
        original = ip.sample_flow
        engine = Engine()
        engine.turbo = SimpleNamespace(_ORIG={"sample_flow": original})
        with patch.object(ip, "sample_flow", lambda *a, **kw: None):
            engine._unload()
            self.assertIs(ip.sample_flow, original)

    def test_generation_failure_discards_pipeline_and_restores_backend(self):
        import idol.models.layers as layers
        original = layers.ATTN_BACKEND
        class FailedPipeline:
            _local_hooks = []
            denoiser = torch.nn.Identity()
            def __call__(self, *args, **kwargs):
                layers.ATTN_BACKEND = "changed"
                raise RuntimeError("out of memory")
        engine = Engine()
        engine.pipeline = FailedPipeline()
        engine.residency = Residency(torch.device("cpu"), True)
        with patch.object(engine, "_load", return_value=engine.pipeline), patch.object(engine, "_set_model"):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                engine.generate(Image.new("RGB", (32, 32)), device="cpu")
        self.assertIsNone(engine.pipeline)
        self.assertEqual(layers.ATTN_BACKEND, original)


if __name__ == "__main__":
    unittest.main()
