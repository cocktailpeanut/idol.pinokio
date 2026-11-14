# idol

A small image-to-3D app for [IDOL](https://huggingface.co/speridlabs/idol) and [IDOL Turbo](https://huggingface.co/speridlabs/idol-turbo), with a one-click Pinokio launcher. The page is the canvas: no navbar or permanent panels.

## Use

1. Open this project in Pinokio and click **Install**. Dependencies and the local browser viewer are installed automatically.
2. Click **Start**, then **Open Web UI**. Drop an image anywhere on the page, or click the empty canvas to choose one.
3. Select **Turbo** or **IDOL**, then **Generate**. The first request downloads several GB of model weights; subsequent requests reuse the Hugging Face cache.
4. Drag the generated 3D preview to rotate it and scroll or pinch to zoom. **Download** saves the Gaussian-splat PLY. **New image** returns to the empty canvas. You can also drop another image onto a result.

The shared pipeline uses public encoder mirrors: [DINOv3 ViT-H+](https://huggingface.co/cocktailpeanut/dinov3-vith16plus-pretrain-lvd1689m) and [FLUX.2 VAE](https://huggingface.co/cocktailpeanut/FLUX.2-vae). No Hugging Face login or access request is required for downloads.

New image clears the browser preview and releases its graphics resources; it does not delete saved results. Escape also returns to the dropzone. Ctrl/Cmd+Enter generates a loaded image. A running request continues through a browser refresh; the same tab reconnects to its job while the server remains running. New uploads are accepted after the current generation finishes.

During generation, the canvas shows the current stage, completed sampling steps (5 for Turbo, 50 for IDOL), and total elapsed time. Placement, color, and file saving follow sampling. Refreshing the page restores progress and elapsed time.

The server listens only on `127.0.0.1`, at a port assigned by the OS. Pinokio captures the actual URL. The viewer's JavaScript is served locally; no CDN is needed. Stop the app with Pinokio's built-in Stop control.

## Models, memory, and output

**Turbo** uses the distilled slow/fast phase adapters with 5 guidance-free steps. **IDOL** uses the original sampler with 50 guided steps. Both share one base pipeline; switching does not load a second base model.

The UI defaults to low-memory mode, 65,536 splats, photo preprocessing, seed 42, and automatic device selection. Models load on CPU first, then one inference stage at a time moves to the GPU. Inactive weights return to RAM, including after a request. On MPS, low-memory mode also splits IDOL attention into blocks of 1,024 queries, keeping all keys and values while limiting its workspace. Sampling uses PyTorch SDPA without compilation or batched guidance. The API exposes the remaining settings.

The [upstream baseline](https://huggingface.co/speridlabs/idol) reports 7.2 GB peak VRAM. No measured minimum is available for this app's offload mode. It does not promise a particular small GPU will fit. The API can select CPU or 32,768 splats when necessary. CPU needs ample RAM and can be much slower.

| System | Inference path |
| --- | --- |
| Windows / Linux with a supported NVIDIA GPU | CUDA 12.6 PyTorch wheel; a compatible NVIDIA driver is required |
| Apple Silicon macOS | MPS, with supported operations falling back to CPU |
| Windows / Linux without supported GPU acceleration | CPU |
| Native Intel macOS | Unsupported: the SDK requires PyTorch >=2.5, newer than the available native Intel macOS wheels |
| AMD / Intel graphics | CPU by default; this launcher does not install ROCm or DirectML |

DINOv3 uses the SDK's BF16 precision on supported CUDA GPUs and macOS 14+ MPS, with FP32 elsewhere. FP16 is unsuitable for this pretrained encoder. Core models use FP32 on CPU. The VAE and background remover stay in FP32 on MPS; this also avoids slow FP16 CPU fallback in background removal. MPS uses the SDK's real-valued rotary path to avoid complex tensors.

Use a visible single subject. Objects, people, and characters can be uploaded, but upstream training focuses on single objects; people, busy scenes, occlusions, and unseen sides may be inaccurate. PNG, JPEG, and WebP are accepted, up to 20 MB. Avoid extremely narrow images. A transparent PNG can skip background removal.

**The output is a Gaussian-splat `.ply`, not a triangle mesh for 3D printing.** The browser preview requires WebGL2. Each run saves `model.ply`, the prepared `input.png`, and `camera.json` in `app/outputs/<run-id>/`. They remain local until you remove them.

## Launcher actions

- **Install / Reinstall:** installs isolated Python dependencies and locally bundled viewer assets, then checks imports/assets before marking the app installed. It removes the previous Gradio packages when migrating an older installation. Model weights download at generation time.
- **Start:** serves the canvas and API. Model loading is deferred until generation.
- **Update:** pulls this launcher/app repository with `git pull --ff-only`, then installs its declared dependencies. A configured Git upstream is required. SDK revisions change only when declared by the app.
- **Reset dependencies:** removes `app/env`, `app/node_modules`, generated viewer bundles, and the installation marker. App source, results, and the shared model cache remain. Stop the app first, then reset and install again.
- **Hugging Face login (optional):** opens Pinokio's managed login flow. Public model downloads work without it. The app does not store a token in its source tree.

SDK revision: `8db700349a5570aac390e4780fa32e3eed6b587a`. Turbo code and LoRA revision: `45cdd41f1f4d6d70496daa7d619f57ad79307a37`. Base weights follow the SDK's normal Hugging Face cache/download behavior. Upstream source files are not modified. GaussianSplats3D 0.4.7 and Three.js 0.160.1 are pinned in `app/package-lock.json`; their licenses are copied alongside the browser bundles during installation.

## REST API

Use the URL shown by Pinokio as `BASE`. Interactive API documentation is at `BASE/docs`. Requests use the same local model engine as the UI. Only one generation runs at a time; another request receives HTTP 409.

`POST /api/generate` accepts multipart form data:

| Field | Default | Values |
| --- | --- | --- |
| `image` | required | PNG, JPEG, or WebP file |
| `model` | `turbo` | `turbo`, `standard` |
| `input_type` | `real` | `real`, `synthetic` |
| `low_memory` | `true` | Boolean |
| `detail` | `65536` | 32768, 65536, 131072, 262144, 524288, 1048576 |
| `seed` | `42` | Integer, 0–4294967295 |
| `device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |

The response is HTTP 202 with `{ "id": "...", "status_url": "/api/jobs/..." }`. Poll that URL. `state` is `running`, `done`, or `error`; `phase` describes actual progress, including completed sampling steps. `elapsed_seconds` reports total time since submission and stops at completion or failure. A completed job includes `result.url`, `result.download_url`, and `result.summary`. An error includes `error.message` and `error.setup_required`.

The server keeps the most recent 32 job records in memory. Result files are independent of those records: `GET /api/results/RESULT_ID/model.ply` remains available after job eviction or server restart if the file exists. Add `?download=true` for an attachment response. `GET /api/health` checks the web server without loading models.

### Python

Using `requests`, replace `PORT` with the port Pinokio displays:

```python
import time
import requests

base = "http://127.0.0.1:PORT"
with open("photo.png", "rb") as image:
    response = requests.post(base + "/api/generate", files={"image": image},
                             data={"model": "turbo", "low_memory": "true"})
response.raise_for_status()
status_url = base + response.json()["status_url"]
while True:
    response = requests.get(status_url)
    response.raise_for_status()
    job = response.json()
    if job["state"] != "running":
        break
    time.sleep(1)
if job["state"] == "error":
    raise RuntimeError(job["error"]["message"])
result = requests.get(base + job["result"]["download_url"])
result.raise_for_status()
with open("model.ply", "wb") as output:
    output.write(result.content)
```

### JavaScript

From a page on the same server, with an image file picker:

```javascript
const form = new FormData();
form.append("image", document.querySelector('input[type="file"]').files[0]);
form.append("model", "turbo");
let response = await fetch("/api/generate", { method: "POST", body: form });
if (!response.ok) throw new Error((await response.json()).detail);
const { status_url } = await response.json();
let job;
do {
  await new Promise(resolve => setTimeout(resolve, 1000));
  response = await fetch(status_url);
  if (!response.ok) throw new Error("Could not read generation status");
  job = await response.json();
} while (job.state === "running");
if (job.state === "error") throw new Error(job.error.message);
console.log(job.result.download_url);
```

### Curl

Replace `PORT`, `JOB_ID`, and `RESULT_ID` with the server's values:

```sh
curl -sS -F 'image=@photo.png' -F 'model=turbo' http://127.0.0.1:PORT/api/generate
curl -sS http://127.0.0.1:PORT/api/jobs/JOB_ID
curl -sS 'http://127.0.0.1:PORT/api/results/RESULT_ID/model.ply?download=true' -o model.ply
```

## Development and verification

After installation, run the Python tests from the project root:

```sh
app/env/bin/python -m unittest discover -s app/tests -v
```

On Windows use `app\env\Scripts\python.exe`. The Turbo adapter integration tests additionally accept `IDOL_TURBO_TEST_SOURCE`, pointing to `turbo.py` from the pinned revision above. Tests use small randomly initialized components and synthetic API files; they do not claim pretrained model quality. Full image-to-3D testing downloads the pretrained weights from public repositories.

The opt-in pretrained encoder regression compares the app's actual MPS image features with a full-precision CPU reference. It requires cached model weights and a local preprocessed input image:

```sh
HF_HUB_OFFLINE=1 IDOL_ENCODER_PARITY_IMAGE=/path/to/preprocessed.png app/env/bin/python -m unittest discover -s app/tests -v
```

The corrected Turbo path was also checked visually on a portrait against direct upstream SDK inference, using identical preprocessed pixels, seed 42, 5 steps, and 65,536 splats. The face and cape reconstructed in both paths. File-integrity checks alone do not validate reconstruction quality. Memory requirements for this corrected precision path have not been profiled.

Progress reporting was verified with the upstream 50-step guided sampler and pinned Turbo adapter, including interrupted sampling and hook cleanup. A full Turbo run in Brave reported all five steps, restored elapsed time after refresh, and opened the result. The progress layout was checked at desktop and phone widths. Full generation has been exercised on Apple Silicon; Windows/Linux hardware verification remains outstanding.

The browser source is in `app/static/`. There is no frontend framework or build step. `npm ci --omit=dev` installs two viewer packages, and `python prepare_assets.py` copies their browser bundles into `app/static/vendor/`.

Launcher logs are in `logs/api/<script>/latest`. Generation errors appear in the Start terminal. If a download is denied, check the linked public repository and any saved Hugging Face credentials.

The SDK and Turbo models are Apache-2.0; GaussianSplats3D and Three.js are MIT. Encoder terms also apply. A license for the new wrapper source has not been selected; choose one before publishing.
