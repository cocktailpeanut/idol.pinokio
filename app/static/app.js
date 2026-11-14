const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(["canvas", "file", "dropzone", "image", "viewer", "activity", "phase", "elapsed", "error", "error-message", "setup", "dock", "new-image", "models", "generate", "download", "orbit-hint", "drop-overlay"].map(id => [id, $(id)]));
const memoryKey = "idol-local-job";
let state = "empty";
let file = null;
let model = "turbo";
let imageURL = null;
let viewer = null;
let renderer = null;
let resizeObserver = null;
let revision = 0;
let dragDepth = 0;
let elapsedAnchor = null;
let clock = null;

function renderElapsed() {
  if (elapsedAnchor === null) return;
  const seconds = Math.max(0, Math.floor((performance.now() - elapsedAnchor) / 1000));
  const minutes = Math.floor(seconds / 60);
  ui.elapsed.textContent = (minutes ? `${minutes}m ${seconds % 60}s` : `${seconds}s`) + " elapsed";
}
function updateElapsed(seconds) {
  elapsedAnchor = Number.isFinite(seconds) ? performance.now() - seconds * 1000 : null;
  ui.elapsed.hidden = elapsedAnchor === null;
  renderElapsed();
}

function remember(job) {
  try { job ? sessionStorage.setItem(memoryKey, job) : sessionStorage.removeItem(memoryKey); } catch {}
}
function savedJob() {
  try { return sessionStorage.getItem(memoryKey); } catch { return null; }
}
function setState(next) {
  state = next;
  ui.canvas.dataset.state = next;
  ui.dropzone.hidden = next !== "empty";
  ui.image.hidden = !file || !["image", "generating", "loading"].includes(next);
  ui.viewer.hidden = !["loading", "result"].includes(next);
  ui.activity.hidden = !["generating", "loading"].includes(next);
  ui.dock.hidden = !["image", "result"].includes(next);
  ui.models.hidden = next !== "image";
  ui.generate.hidden = next !== "image";
  ui.download.hidden = next !== "result";
  ui["orbit-hint"].hidden = next !== "result" || !viewer;
  ui.canvas.setAttribute("aria-busy", String(["generating", "loading"].includes(next)));
  clearInterval(clock);
  clock = next === "generating" ? setInterval(renderElapsed, 1000) : null;
}
function clearError() {
  ui.error.hidden = true;
  ui.setup.hidden = true;
}
function showError(message, setup = false) {
  ui["error-message"].textContent = message;
  ui.setup.hidden = !setup;
  ui.error.hidden = false;
}
function busy() { return state === "generating" || state === "loading"; }

async function disposeViewer() {
  const previous = viewer;
  const previousRenderer = renderer;
  const previousNodes = [...ui.viewer.childNodes];
  resizeObserver?.disconnect();
  resizeObserver = null;
  viewer = null;
  renderer = null;
  if (previous) {
    try { await previous.dispose(); } catch (error) { console.warn("Viewer cleanup:", error); }
  }
  if (previousRenderer) {
    previousRenderer.dispose();
    previousRenderer.forceContextLoss();
  }
  previousNodes.forEach(node => { if (node.parentNode === ui.viewer) node.remove(); });
}
async function newImage() {
  if (busy()) return;
  const token = ++revision;
  const oldURL = imageURL;
  imageURL = null;
  file = null;
  remember(null);
  clearError();
  setState("empty");
  ui.file.value = "";
  ui.image.removeAttribute("src");
  ui.download.removeAttribute("href");
  if (oldURL) URL.revokeObjectURL(oldURL);
  await disposeViewer();
  if (token === revision) ui.dropzone.focus({ preventScroll: true });
}

async function chooseFile(candidate) {
  if (busy() || !candidate) return;
  if (candidate.size > 20 * 1024 * 1024) {
    showError("Choose an image smaller than 20 MB.");
    return;
  }
  if (!["image/png", "image/jpeg", "image/webp"].includes(candidate.type) && !/\.(png|jpe?g|webp)$/i.test(candidate.name)) {
    showError("Choose a PNG, JPEG, or WebP image.");
    return;
  }
  const token = ++revision;
  const nextURL = URL.createObjectURL(candidate);
  const probe = new Image();
  probe.src = nextURL;
  try { await probe.decode(); }
  catch { URL.revokeObjectURL(nextURL); showError("This image could not be read. Choose another image."); return; }
  if (token !== revision) { URL.revokeObjectURL(nextURL); return; }
  await disposeViewer();
  if (token !== revision) { URL.revokeObjectURL(nextURL); return; }
  if (imageURL) URL.revokeObjectURL(imageURL);
  file = candidate;
  imageURL = nextURL;
  ui.image.src = imageURL;
  remember(null);
  clearError();
  setState("image");
  ui.generate.focus({ preventScroll: true });
}

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "The request could not be completed. Try again.";
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return data;
}
async function openResult(result, token) {
  ui.download.href = result.download_url;
  ui.download.title = result.summary;
  ui.phase.textContent = "Opening 3D preview";
  setState("loading");
  try {
    const [splats, THREE] = await Promise.all([
      import("/static/vendor/gaussian-splats-3d.module.js"), import("three")
    ]);
    if (token !== revision) return;
    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(1);
    renderer.setSize(ui.viewer.clientWidth, ui.viewer.clientHeight);
    ui.viewer.appendChild(renderer.domElement);
    viewer = new splats.Viewer({
      rootElement: ui.viewer,
      renderer,
      cameraUp: [0, -1, 0],
      initialCameraPosition: [1.4, 0.9, 2.0],
      initialCameraLookAt: [0, 0, 0],
      sharedMemoryForWorkers: false,
      gpuAcceleratedSort: false,
      halfPrecisionCovariancesOnGPU: true,
      renderMode: splats.RenderMode.OnChange,
      sceneRevealMode: splats.SceneRevealMode.Instant,
      showLoadingUI: false
    });
    // Own the renderer: the viewer otherwise removes its root from document.body
    // on disposal, while our canvas lives inside <main> and must survive reset.
    const currentViewer = viewer;
    const currentRenderer = renderer;
    resizeObserver = new ResizeObserver(() => {
      if (viewer !== currentViewer) return;
      const width = ui.viewer.clientWidth, height = ui.viewer.clientHeight;
      if (!width || !height) return;
      currentRenderer.setSize(width, height);
      currentViewer.camera.aspect = width / height;
      currentViewer.camera.updateProjectionMatrix();
      currentViewer.forceRenderNextFrame();
    });
    resizeObserver.observe(ui.viewer);
    await viewer.addSplatScene(result.url, { showLoadingUI: false, progressiveLoad: false, splatAlphaRemovalThreshold: 1 });
    if (token !== revision) return;
    viewer.start();
    setState("result");
    ui.viewer.focus({ preventScroll: true });
  } catch (error) {
    console.error("3D preview:", error);
    await disposeViewer();
    setState("result");
    showError("Your 3D file is ready, but the preview could not open. Download it to view in a Gaussian-splat viewer.");
  }
}

async function pollJob(jobID, token) {
  let connectionFailures = 0;
  while (token === revision) {
    let job;
    try {
      job = await request("/api/jobs/" + encodeURIComponent(jobID));
      connectionFailures = 0;
    } catch (error) {
      if (error.status === 404) {
        remember(null);
        setState(file ? "image" : "empty");
        showError(error.message);
        return;
      }
      if (++connectionFailures < 4) { await new Promise(resolve => setTimeout(resolve, 1200)); continue; }
      setState(file ? "image" : "empty");
      showError("Connection lost. Reload the page to reconnect to the generation.");
      return;
    }
    if (token !== revision) return;
    updateElapsed(job.elapsed_seconds);
    if (job.state === "done") { await openResult(job.result, token); return; }
    if (job.state === "error") {
      remember(null);
      setState(file ? "image" : "empty");
      showError(job.error.message, job.error.setup_required);
      return;
    }
    ui.phase.textContent = job.phase;
    await new Promise(resolve => setTimeout(resolve, 600));
  }
}
async function generate() {
  if (!file || busy()) return;
  const token = ++revision;
  clearError();
  ui.phase.textContent = "Preparing the model";
  updateElapsed(0);
  setState("generating");
  const form = new FormData();
  form.append("image", file);
  form.append("model", model);
  try {
    const job = await request("/api/generate", { method: "POST", body: form });
    remember(job.id);
    await pollJob(job.id, token);
  } catch (error) {
    if (token !== revision) return;
    setState("image");
    showError(error.message || "Could not connect to IDOL. Check that the app is running.");
  }
}

ui.dropzone.addEventListener("click", () => { ui.file.value = ""; ui.file.click(); });
ui.file.addEventListener("change", () => chooseFile(ui.file.files[0]));
ui.generate.addEventListener("click", generate);
ui["new-image"].addEventListener("click", newImage);
ui.models.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
  model = button.dataset.model;
  ui.models.querySelectorAll("button").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
}));
window.addEventListener("keydown", event => {
  if (event.key === "Escape" && !busy() && state !== "empty") newImage();
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && state === "image") generate();
});
window.addEventListener("dragenter", event => {
  if (!event.dataTransfer.types.includes("Files")) return;
  event.preventDefault();
  dragDepth++;
  if (!busy()) ui["drop-overlay"].hidden = false;
});
window.addEventListener("dragover", event => { if (event.dataTransfer.types.includes("Files")) event.preventDefault(); });
window.addEventListener("dragleave", event => {
  event.preventDefault();
  if (--dragDepth <= 0) { dragDepth = 0; ui["drop-overlay"].hidden = true; }
});
window.addEventListener("drop", event => {
  event.preventDefault();
  dragDepth = 0;
  ui["drop-overlay"].hidden = true;
  if (busy()) return;
  if (event.dataTransfer.files.length !== 1) { showError("Drop one image at a time."); return; }
  chooseFile(event.dataTransfer.files[0]);
});

setState("empty");
const resumedJob = savedJob();
if (resumedJob) {
  ui.phase.textContent = "Reconnecting to your generation";
  setState("generating");
  pollJob(resumedJob, ++revision);
}
