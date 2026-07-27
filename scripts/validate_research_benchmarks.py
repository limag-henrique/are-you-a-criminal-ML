"""Validate local benchmark paths, deterministic image samples, and BFW pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd


def bfw_image_root(faces_dir: Path) -> Path:
    candidates = [item for item in faces_dir.rglob("asian_females") if item.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one BFW image root, found {len(candidates)}.")
    return candidates[0].parent


def _decode_status(root: Path, relative_path: str) -> tuple[str, str]:
    image = cv2.imread(str(root / relative_path), cv2.IMREAD_GRAYSCALE)
    return ("ok" if image is not None else "unreadable"), relative_path


def _stratified_sample(catalog: pd.DataFrame, limit_per_stratum: int) -> tuple[pd.DataFrame, list[str]]:
    strata = [column for column in ("source_race_label", "source_gender_label", "source_group") if column in catalog.columns]
    if not strata:
        return catalog.head(limit_per_stratum).copy(), strata
    sample = catalog.copy()
    sample["_sample_key"] = sample["relative_path"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    sample = sample.sort_values([*strata, "_sample_key"])
    return sample.groupby(strata, dropna=False, group_keys=False).head(limit_per_stratum), strata


def validate_images(
    catalog: pd.DataFrame,
    root: Path,
    name: str,
    workers: int,
    decode_mode: str,
    sample_per_stratum: int,
) -> dict[str, object]:
    paths = catalog["relative_path"].astype(str).tolist()
    missing = [item for item in paths if not (root / item).is_file()]
    print(f"{name}: verified paths {len(paths)}/{len(catalog)}", flush=True)
    if decode_mode == "none":
        decode_paths: list[str] = []
        strata: list[str] = []
    elif decode_mode == "all":
        decode_paths = paths
        strata = []
    else:
        sampled, strata = _stratified_sample(catalog, sample_per_stratum)
        decode_paths = sampled["relative_path"].astype(str).tolist()
    unreadable: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (status, item) in enumerate(executor.map(lambda value: _decode_status(root, value), decode_paths), start=1):
            if status == "unreadable":
                unreadable.append(item)
            if index % 1000 == 0:
                print(f"{name}: decoded {index}/{len(decode_paths)} images", flush=True)
    return {
        "catalogued": int(len(catalog)),
        "paths_checked": len(paths),
        "decoded_images": len(decode_paths),
        "decode_mode": decode_mode,
        "sample_strata": strata,
        "sample_per_stratum": sample_per_stratum if decode_mode == "stratified" else None,
        "missing": len(missing),
        "unreadable": len(unreadable),
        "missing_examples": missing[:10],
        "unreadable_examples": unreadable[:10],
    }


def validate_bfw_pairs(pairs_path: Path, image_paths: set[str]) -> dict[str, object]:
    total = genuine = impostor = missing_references = invalid_labels = 0
    folds: set[int] = set()
    group_counts: dict[str, int] = {}
    for chunk in pd.read_csv(pairs_path, usecols=["p1", "p2", "label", "fold", "a1", "a2"], chunksize=100_000):
        chunk["p1"] = chunk["p1"].astype(str).str.replace("\\\\", "/", regex=False)
        chunk["p2"] = chunk["p2"].astype(str).str.replace("\\\\", "/", regex=False)
        total += len(chunk)
        genuine += int((chunk["label"] == 1).sum())
        impostor += int((chunk["label"] == 0).sum())
        invalid_labels += int((~chunk["label"].isin([0, 1])).sum())
        missing_references += int((~chunk["p1"].isin(image_paths)).sum() + (~chunk["p2"].isin(image_paths)).sum())
        folds.update(chunk["fold"].astype(int).unique().tolist())
        for group, count in chunk.loc[chunk["a1"] == chunk["a2"], "a1"].value_counts().items():
            group_counts[str(group)] = group_counts.get(str(group), 0) + int(count)
    return {
        "pairs": total,
        "genuine_pairs": genuine,
        "impostor_pairs": impostor,
        "invalid_labels": invalid_labels,
        "missing_pair_image_references": missing_references,
        "folds": sorted(folds),
        "same_group_pair_counts": dict(sorted(group_counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local research benchmarks without exposing images.")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--datasets", default="fairface,bfw", help="Comma-separated: fairface,bfw")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decode-mode", choices=["all", "stratified", "none"], default="stratified")
    parser.add_argument("--sample-per-stratum", type=int, default=100)
    args = parser.parse_args(argv)
    selected = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    if not selected or not selected.issubset({"fairface", "bfw"}):
        parser.error("--datasets accepts only fairface,bfw")
    if args.workers < 1 or args.sample_per_stratum < 1:
        parser.error("--workers and --sample-per-stratum must be at least 1")
    root = Path(args.root)
    manifests = root / "manifests"
    result_path = manifests / "benchmark_validation.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    if "fairface" in selected:
        fairface = pd.read_csv(manifests / "fairface_catalog.csv")
        result["fairface"] = validate_images(
            fairface,
            root / "external" / "fairface" / "extracted",
            "FairFace",
            args.workers,
            args.decode_mode,
            args.sample_per_stratum,
        )
    if "bfw" in selected:
        bfw = pd.read_csv(manifests / "bfw_catalog.csv")
        result["bfw"] = validate_images(
            bfw,
            bfw_image_root(root / "external" / "bfw" / "extracted" / "faces-cropped"),
            "BFW",
            args.workers,
            args.decode_mode,
            args.sample_per_stratum,
        )
        result["bfw_pairs"] = validate_bfw_pairs(manifests / "bfw_pairs.csv", set(bfw["relative_path"].astype(str)))
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures: list[int] = []
    for name in selected:
        failures.extend([result[name]["missing"], result[name]["unreadable"]])
        if name == "bfw":
            failures.extend([result["bfw_pairs"]["invalid_labels"], result["bfw_pairs"]["missing_pair_image_references"]])
    if any(failures):
        raise SystemExit("Benchmark validation failed; see benchmark_validation.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
