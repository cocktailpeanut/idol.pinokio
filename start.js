module.exports = {
  daemon: true,
  run: [{
    method: "shell.run",
    params: {
      path: "app",
      venv: "env",
      env: { ATTN_BACKEND: "sdpa", PYTORCH_ENABLE_MPS_FALLBACK: "1" },
      message: "python app.py",
      on: [{
        event: "/(http:\\/\\/[0-9.:]+)/",
        done: true
      }]
    }
  }, {
    when: "{{!!(input.event && input.event[1])}}",
    method: "local.set",
    params: { url: "{{input.event[1]}}" }
  }]
}
