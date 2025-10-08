import argparse, os, json
from .registry import add_artifact


def main():
    ap = argparse.ArgumentParser(
        description="Add an artifact to artifacts/registry.json"
    )
    ap.add_argument("--type", required=True, choices=["PID", "FLC", "RL"])
    ap.add_argument(
        "--path",
        required=True,
        help="Path to artifact file (YAML for PID/FLC, zip for RL)",
    )
    ap.add_argument("--status", default="validated", choices=["validated", "draft"])
    ap.add_argument(
        "--metrics", default="{}", help='JSON string, e.g., {"val_score":372.1}'
    )
    ap.add_argument("--tags", default="", help="Comma-separated tags")
    args = ap.parse_args()

    metrics = json.loads(args.metrics) if args.metrics.strip() else {}
    tags = [t for t in args.tags.split(",") if t.strip()] if args.tags else []

    sha = add_artifact(
        args.type, args.path, status=args.status, metrics=metrics, tags=tags
    )
    print(f"Added {args.type} -> {args.path}")
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
