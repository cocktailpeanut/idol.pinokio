// Platform dispatch follows system/examples/torch.js; IDOL requires torch >=2.5.
module.exports = {
  run: [{
    when: "{{platform === 'darwin' && arch !== 'arm64'}}",
    method: "log",
    params: { text: "Native Intel macOS cannot install IDOL's required PyTorch >=2.5. Use an Apple Silicon Mac, Windows, or Linux. No dependencies were installed." },
    next: null
  }, {
    when: "{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux')}}",
    method: "shell.run",
    params: {
      path: "{{args.path}}",
      venv: "{{args.venv}}",
      venv_python: "3.11",
      message: "uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126"
    },
    next: null
  }, {
    method: "shell.run",
    params: {
      path: "{{args.path}}",
      venv: "{{args.venv}}",
      venv_python: "3.11",
      message: "uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu"
    }
  }]
}
