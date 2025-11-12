import json
import shutil
import subprocess
import sys
import os
from pathlib import Path

out = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "ffmpeg_on_path": shutil.which("ffmpeg") is not None,
    "ffmpeg_path": shutil.which("ffmpeg"),
    "tkinter_importable": False,
    "packages": {},
    "torch": None,
    "torch_directml": None,
}

try:
    import importlib
    import pkgutil
    import tkinter as tk
    out["tkinter_importable"] = True
except Exception:
    out["tkinter_importable"] = False

# pip freeze
try:
    import pkg_resources
    dists = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    out["packages"] = dists
except Exception:
    try:
        import subprocess
        p = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        lines = p.stdout.splitlines()
        pkgs = {}
        for l in lines:
            if "==" in l:
                k, v = l.split("==", 1)
                pkgs[k.lower()] = v
        out["packages"] = pkgs
    except Exception:
        out["packages"] = {}

# torch info
try:
    import torch
    out["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
except Exception as e:
    out["torch"] = None

# torch-directml
try:
    import torch_directml
    # try a simple device probe
    try:
        dev = torch_directml.device()
        out["torch_directml"] = {"importable": True, "device": str(dev)}
    except Exception as e:
        out["torch_directml"] = {"importable": True, "device": None, "error": str(e)}
except Exception:
    out["torch_directml"] = None

# write report
p = Path.cwd() / "env_report.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Environment report written to:", p)
print(json.dumps(out, indent=2))
