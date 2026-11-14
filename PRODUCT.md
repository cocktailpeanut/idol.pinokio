# idol

<!-- impeccable:product-schema 1 -->

## Platform
web

## Product Purpose
Turn an uploaded image into a local, downloadable 3D Gaussian splat using either speridlabs/idol-turbo or speridlabs/idol.

## Users and Workflow
The user wants to process both people/characters and objects. The entire page is the dropzone. Upload an image, select the model, generate, rotate the result, and download it. New image returns to the empty canvas without deleting saved results; another image can be dropped onto a result.

## Capabilities and Constraints
- Confirmed: a super minimal light interface, low memory use where possible, and a one-click Pinokio launcher for as many machines as dependencies allow across Windows, macOS, and Linux.
- IDOL outputs Gaussian-splat PLY, not a printable mesh. Complex scenes and people are not guaranteed to reconstruct accurately; upstream training emphasizes single objects.
- Local inference uses public DINOv3 and FLUX.2 VAE mirrors under `cocktailpeanut` on Hugging Face. Downloads happen on first generation without a login or access request.
- Upstream SDK requires Python >=3.10 and PyTorch >=2.5. Native Intel macOS cannot currently satisfy the PyTorch requirement.
- Confirmed interface: no navbar, sidebar, or exposed settings. Empty → image and model/Generate controls → full-page result with Download and New image.
- Stack: plain HTML/CSS/JavaScript, a small FastAPI backend, and locally bundled GaussianSplats3D/Three.js. Gradio is removed.

## Product Principles
- Keep the normal workflow to upload, model, generate, and download.
- Load only one shared base pipeline and swap the Turbo adapter without a second model copy.
- Prefer memory savings over compilation and peak speed by default.
- Report actual setup and inference failures; never show a fabricated result.
