module.exports = {
  version: "8.0.0",
  title: "idol",
  description: "One image to a 3D Gaussian splat. IDOL + Turbo, with low-memory mode.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    for (const [script, text] of [["install.js", "Installing"], ["update.js", "Updating"], ["reset.js", "Resetting"]]) {
      if (info.running(script)) return [{ default: true, icon: "fa-solid fa-terminal", text, href: script }]
    }
    if (!info.exists("app/.installed")) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }]
    }
    if (info.running("start.js")) {
      const local = info.local("start.js")
      return [
        ...(local && local.url ? [{ default: true, icon: "fa-solid fa-cube", text: "Open Web UI", href: local.url }] : []),
        { default: !(local && local.url), icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" },
        { icon: "fa-solid fa-key", text: "Hugging Face login (optional)", href: "login.js" }
      ]
    }
    return [
      { default: true, icon: "fa-solid fa-play", text: "Start", href: "start.js" },
      { icon: "fa-solid fa-key", text: "Hugging Face login (optional)", href: "login.js" },
      { icon: "fa-solid fa-arrows-rotate", text: "Update", href: "update.js" },
      { icon: "fa-solid fa-plug", text: "Reinstall", href: "install.js" },
      { icon: "fa-solid fa-trash-can", text: "Reset dependencies", href: "reset.js" }
    ]
  }
}
