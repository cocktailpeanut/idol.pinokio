module.exports = {
  requires: { bundle: "ai" },
  run: [{
    when: "{{platform === 'darwin' && arch !== 'arm64'}}",
    method: "log",
    params: { text: "IDOL requires PyTorch >=2.5, which is unavailable for native Intel macOS. Use Apple Silicon macOS, Windows, or Linux." },
    next: null
  }, {
    method: "fs.rm",
    params: { path: "app/.installed" }
  }, {
    method: "script.start",
    params: { uri: "torch.js", params: { path: "app", venv: "env" } }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      venv: "env",
      venv_python: "3.11",
      message: [
        "npm ci --omit=dev",
        "python prepare_assets.py",
        "uv pip install -r requirements.txt",
        "uv pip uninstall gradio gradio-client",
        "python verify_install.py"
      ]
    }
  }]
}
