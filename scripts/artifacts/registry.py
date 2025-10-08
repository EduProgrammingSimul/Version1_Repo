import os, json, time, pathlib
from .hashutil import sha256_file

REG_PATH = "artifacts/registry.json"
TAGS_PATH = "artifacts/tags.json"


def _ensure_dirs():
    os.makedirs("artifacts/pid", exist_ok=True)
    os.makedirs("artifacts/flc", exist_ok=True)
    os.makedirs("artifacts/rl", exist_ok=True)


def load_registry():
    _ensure_dirs()
    if os.path.isfile(REG_PATH):
        with open(REG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_registry(entries):
    os.makedirs(os.path.dirname(REG_PATH), exist_ok=True)
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def add_artifact(
    typ, path, status="validated", metrics=None, tags=None, created_at=None
):
    entries = load_registry()
    if created_at is None:
        created_at = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    sha = sha256_file(path)
    rel = str(pathlib.Path(path).as_posix())
    entries.append(
        {
            "type": typ,
            "path": rel,
            "created_at": created_at,
            "sha256": sha,
            "status": status,
            "metrics": metrics or {},
            "tags": tags or [],
        }
    )
    save_registry(entries)
    return sha


def tag(name, path):
    d = {}
    if os.path.isfile(TAGS_PATH):
        with open(TAGS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
    d[name] = path
    os.makedirs(os.path.dirname(TAGS_PATH), exist_ok=True)
    with open(TAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
