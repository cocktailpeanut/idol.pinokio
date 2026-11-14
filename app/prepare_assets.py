"""Copy only the browser bundles and their licenses into the public asset folder."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
vendor = root / "static/vendor"
vendor.mkdir(parents=True, exist_ok=True)
for source, target in [
    ("@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js", "gaussian-splats-3d.module.js"),
    ("three/build/three.module.js", "three.module.js"),
    ("three/LICENSE", "THREE-LICENSE.txt"),
    ("@mkkellogg/gaussian-splats-3d/LICENSE", "GAUSSIAN-SPLATS-3D-LICENSE.txt"),
]:
    shutil.copyfile(root / "node_modules" / source, vendor / target)
print("Local viewer assets prepared.")
