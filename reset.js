module.exports = {
  run: [{
    method: "fs.rm",
    params: { path: "app/.installed" }
  }, {
    method: "fs.rm",
    params: { path: "app/env" }
  }, {
    method: "fs.rm",
    params: { path: "app/node_modules" }
  }, {
    method: "fs.rm",
    params: { path: "app/static/vendor" }
  }]
}
