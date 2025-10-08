import os, sys, json, platform, subprocess
from typing import Dict, Any


def _pip_freeze() -> str:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        )
        return out.strip()
    except Exception as e:
        return f"<pip freeze failed: {e}>"


def write_run_info(out_dir: str, extra: Dict[str, Any] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "pip_freeze": _pip_freeze(),
    }
    if extra:
        info.update(extra)
    path = os.path.join(out_dir, "RUN_INFO.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path
