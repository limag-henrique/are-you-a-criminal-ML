"""Register the exact SHA-256 of an InsightFace recognition model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--search-root", default=str(Path.home() / ".insightface"))
    parser.add_argument("--output", default="artifacts/model_hashes.json")
    parser.add_argument("--fairface-manifest", default=None)
    args = parser.parse_args()
    candidates = [Path(args.model)] if args.model else list(Path(args.search_root).rglob("w600k_r50.onnx"))
    if not candidates or not candidates[0].is_file():
        raise FileNotFoundError("w600k_r50.onnx was not found; pass --model explicitly")
    model = candidates[0].resolve()
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    payload = {"model": model.name, "path": str(model), "sha256": digest, "size_bytes": model.stat().st_size}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.fairface_manifest:
        manifest_path = Path(args.fairface_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["embedding_model"] = payload
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
