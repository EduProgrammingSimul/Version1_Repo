# Artifact Registry & Auto-Resolution

This repo uses a lightweight artifact registry to **automatically** fetch:
- Optimized PID/FLC parameter files (YAML)
- Trained RL checkpoints (e.g., SB3 .zip)

## Where artifacts live

```
artifacts/
  pid/
    PID_2025-05-10T130455Z_rmse0.123.yaml
    PID_2025-08-02T091122Z_rmse0.118.yaml
  flc/
    FLC_2025-05-10T130455Z_rmse0.140.yaml
  rl/
    RL_2025-06-01T143000Z_return+348.zip
    RL_2025-08-15T101010Z_return+372.zip
  registry.json   # auto-maintained registry (see below)
  tags.json       # optional tags -> artifact path mapping
  LATEST.lock     # last resolved selection (paths + sha256) for reproducibility
```

## Registry structure (`artifacts/registry.json`)

Each entry is a dict with:
- `type`: "PID" | "FLC" | "RL"
- `path`: relative path to the artifact file
- `created_at`: ISO timestamp
- `sha256`: file hash
- `status`: "validated" | "draft"
- `metrics`: { "val_score": 0.0, ... }  # any metrics you want to store
- `tags`: ["experiment-42", "paper-main"]  # optional

## Policies

- **latest**: newest `created_at` among entries (optionally only validated)
- **best**: highest `metrics[score_key]`
- **tag:<name>**: use the artifact explicitly tagged with `<name>`

You can override the policy via env var:
```
export ARTIFACT_POLICY="best"
# or
export ARTIFACT_POLICY="tag:paper-main"
```

## Reproducibility

Every time the resolver runs, it writes `artifacts/LATEST.lock` with **absolute file paths and sha256**.
This is stored under `results/reports/.../RUN_INFO.json` too by the pipeline for audit.
