"""Integration with the actual pinned adapter; set IDOL_TURBO_TEST_SOURCE to its turbo.py."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Engine, Residency, sampling_progress
from test_engine import tiny_denoiser


@unittest.skipUnless(os.environ.get("IDOL_TURBO_TEST_SOURCE"), "Set IDOL_TURBO_TEST_SOURCE to the pinned adapter to run integration tests")
class TurboTests(unittest.TestCase):
    def test_cpu_turbo_standard_and_reload(self):
        self._switching("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_turbo_standard_and_reload(self):
        self._switching("mps")

    def _switching(self, device):
        import idol.pipeline as ip
        original = ip.sample_flow
        spec = importlib.util.spec_from_file_location("test_turbo", os.environ["IDOL_TURBO_TEST_SOURCE"])
        turbo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(turbo)
        model = tiny_denoiser(device)
        reference = tiny_denoiser("cpu")
        turbo._inject(reference, turbo.LORA_RANK, turbo.LORA_ALPHA)
        state = {k: torch.randn_like(v) * .01 for k, v in reference.state_dict().items() if "slow_" in k or "fast_" in k}
        noise = TensorDict({"latent": torch.randn(1, 8, 4, device=device), "camera": torch.randn(1, 1, 5, device=device)}, batch_size=1, device=device)
        cond = TensorDict({"feature1": torch.randn(1, 5, 12, device=device), "feature2": torch.randn(1, 5, 8, device=device)}, batch_size=1, device=device)
        with torch.no_grad():
            expected = original(model, noise.clone(), cond, cond.apply(torch.zeros_like), steps=2, cfg_strength=3., verbose=False)
        engine = Engine()
        engine.turbo = turbo
        pipe = SimpleNamespace(denoiser=model, device=torch.device(device), _local_hooks=[])
        engine.pipeline = pipe
        engine.residency = Residency(torch.device(device), False)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                save_file(state, str(Path(tmp) / turbo.LORA_FILE))
                with patch("huggingface_hub.snapshot_download", return_value=tmp):
                    engine._set_model(pipe, "turbo")
                phases = []
                with torch.no_grad(), sampling_progress(model, 5, 1, phases.append):
                    fast = ip.sample_flow(model, noise.clone(), cond, cond.apply(torch.zeros_like), steps=5, verbose=False)
                self.assertEqual(phases, [f"Generating 3D · step {i} of 5" for i in range(1, 6)])
                self.assertFalse(model._forward_hooks)
                self.assertTrue(fast["latent"].isfinite().all())
                self.assertTrue(all(m.active == "fast" for m in model.modules() if isinstance(m, turbo.LoRALinear)))
                engine._set_model(pipe, "standard")
                self.assertIs(ip.sample_flow, original)
                with torch.no_grad():
                    standard = ip.sample_flow(model, noise.clone(), cond, cond.apply(torch.zeros_like), steps=2, cfg_strength=3., verbose=False)
                torch.testing.assert_close(standard["latent"], expected["latent"])
                engine._set_model(pipe, "turbo")
                engine._unload()
                self.assertIs(ip.sample_flow, original)
        finally:
            ip.sample_flow = original


if __name__ == "__main__":
    unittest.main()
