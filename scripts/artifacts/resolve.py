import os, json, pathlib, re, time
from .registry import load_registry
from .hashutil import sha256_file

LOCK_PATH = "artifacts/LATEST.lock"


def _read_policy(policy_path="config/artifact_policy.yaml"):
    try:
        import yaml

        with open(policy_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return y
    except Exception:
        return {"policy": "latest", "require_validated": True, "score_key": "val_score"}


def _filter_validated(entries, require_validated: bool):
    if not require_validated:
        return entries
    return [e for e in entries if e.get("status") == "validated"]


def _pick_latest(entries):
    return (
        sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)[0]
        if entries
        else None
    )


def _pick_best(entries, score_key):
    scored = [
        e
        for e in entries
        if isinstance(e.get("metrics", {}).get(score_key, None), (int, float))
    ]
    return (
        sorted(scored, key=lambda e: e["metrics"][score_key], reverse=True)[0]
        if scored
        else None
    )


def _pick_by_tag(entries, tag_name):
    for e in entries:
        if tag_name in (e.get("tags") or []):
            return e
    return None


def resolve(typ: str, override_policy: str = None):
    pol = _read_policy()
    policy = (
        override_policy
        or os.environ.get("ARTIFACT_POLICY")
        or pol.get("policy", "latest")
    )
    require_validated = bool(pol.get("require_validated", True))
    score_key = pol.get("score_key", "val_score")

    entries = [e for e in load_registry() if e.get("type") == typ]
    entries = _filter_validated(entries, require_validated)

    chosen = None
    if policy.startswith("tag:"):
        chosen = _pick_by_tag(entries, policy.split(":", 1)[1])
    elif policy == "best":
        chosen = _pick_best(entries, score_key)
    else:
        chosen = _pick_latest(entries)

    if chosen is None:
        raise RuntimeError(f"No artifact found for type={typ} with policy={policy}.")

    path = chosen["path"]
    sha = chosen["sha256"]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Artifact listed in registry but missing on disk: {path}"
        )

    sha_now = sha256_file(path)
    if sha_now != sha:
        raise RuntimeError(f"SHA256 mismatch for {path}. Registry:{sha} Now:{sha_now}")

    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    lock = {
        "resolved_at": time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()),
        "policy": policy,
        "selection": chosen,
    }
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
    return chosen
