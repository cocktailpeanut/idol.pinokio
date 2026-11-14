"""Opt-in pretrained parity check; uses installed models and a local input image."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Engine


@unittest.skipUnless(os.environ.get("IDOL_ENCODER_PARITY_IMAGE") and torch.backends.mps.is_available(),
                     "Set IDOL_ENCODER_PARITY_IMAGE and run with MPS access for pretrained parity")
class PretrainedEncoderParityTests(unittest.TestCase):
    @torch.inference_mode()
    def test_actual_mps_encoder_preserves_cpu_image_features(self):
        from transformers import AutoModel

        with Image.open(os.environ["IDOL_ENCODER_PARITY_IMAGE"]) as image:
            pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float()[None] / 255
        mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        reference_model = AutoModel.from_pretrained(
            "cocktailpeanut/dinov3-vith16plus-pretrain-lvd1689m", dtype=torch.float32, local_files_only=True
        ).eval().requires_grad_(False)
        expected = reference_model(pixel_values=(pixels - mean) / std).last_hidden_state.float()
        expected = F.layer_norm(expected, expected.shape[-1:]).double()
        del reference_model

        engine = Engine()
        try:
            pipe = engine._load(torch.device("mps"), True, lambda *args: None)
            encoder = pipe.image_encoder.encoder
            engine.residency.activate(encoder.dinov3, "Checking image encoder")
            actual = encoder._dinov3_tokens(pixels.to("mps")).cpu().double()
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(actual.isfinite().all())
            relative_error = float((actual - expected).norm() / expected.norm())
            self.assertLess(relative_error, .05, f"DINOv3 relative feature error: {relative_error:.4f}")
        finally:
            engine._unload()


if __name__ == "__main__":
    unittest.main()
