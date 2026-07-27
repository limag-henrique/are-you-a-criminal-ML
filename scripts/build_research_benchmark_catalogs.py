"""Create local, pseudonymous catalogs for the downloaded research benchmarks.

The catalog is deliberately written under datasets/ (ignored by Git). It keeps
the original benchmark tags as source metadata and does not infer protected
attributes from an image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _existing(path: Path, relative: str) -> bool:
    return (path / relative).is_file()


def _bfw_image_root(faces_dir: Path) -> Path:
    subgroup_dirs = [item for item in faces_dir.rglob("asian_females") if item.is_dir()]
    if len(subgroup_dirs) != 1:
        raise ValueError(f"Expected one BFW image root, found {len(subgroup_dirs)} candidate roots.")
    return subgroup_dirs[0].parent


def build_fairface(root: Path, output: Path) -> dict[str, object]:
    source = root / "external" / "fairface"
    images = source / "extracted"
    labels = [source / "archives" / "fairface_label_train.csv", source / "archives" / "fairface_label_val.csv"]
    frames = []
    for label_path in labels:
        frame = pd.read_csv(label_path)
        required = {"file", "age", "gender", "race"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Unexpected FairFace labels in {label_path}: missing {sorted(missing)}")
        frame = frame.loc[:, ["file", "age", "gender", "race"]].copy()
        frame["source_split"] = label_path.stem.removeprefix("fairface_label_")
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True)
    catalog["file"] = catalog["file"].astype(str).str.replace("\\\\", "/", regex=False)
    catalog["exists"] = catalog["file"].map(lambda item: _existing(images, item))
    catalog = catalog.loc[catalog["exists"]].copy()
    catalog.insert(0, "dataset", "fairface")
    catalog = catalog.rename(
        columns={
            "file": "relative_path",
            "age": "source_age_label",
            "gender": "source_gender_label",
            "race": "source_race_label",
        }
    )
    catalog["source_label_note"] = "Dataset-provided annotation; not self-identification and not inferred by this project."
    catalog["permitted_analysis"] = "detection_coverage_and_impostor_similarity_only"
    catalog.to_csv(output / "fairface_catalog.csv", index=False)
    manifest = pd.DataFrame(
        {
            "path": catalog["relative_path"],
            "subject_id": "fairface_" + catalog["relative_path"].str.replace("/", "_", regex=False).str.rsplit(".", n=1).str[0],
            "quality": "mid",
            "split": "benchmark_fairface",
            "weight": 1.0,
        }
    )
    manifest.to_csv(output / "fairface_extraction_manifest.csv", index=False)
    return {
        "images_catalogued": int(len(catalog)),
        "source_race_label_counts": {str(key): int(value) for key, value in catalog["source_race_label"].value_counts().sort_index().items()},
        "source_gender_label_counts": {str(key): int(value) for key, value in catalog["source_gender_label"].value_counts().sort_index().items()},
    }


def build_bfw(root: Path, output: Path) -> dict[str, object]:
    source = root / "external" / "bfw" / "extracted"
    pairs_path = source / "bfw-v0.1.5-datatable.csv"
    faces = _bfw_image_root(source / "faces-cropped")
    pairs = pd.read_csv(pairs_path, usecols=["p1", "p2", "label", "id1", "id2", "a1", "a2", "g1", "g2", "e1", "e2", "fold"])
    pairs["p1"] = pairs["p1"].astype(str).str.replace("\\\\", "/", regex=False)
    pairs["p2"] = pairs["p2"].astype(str).str.replace("\\\\", "/", regex=False)
    left = pairs[["p1", "id1", "a1", "g1", "e1"]].rename(
        columns={"p1": "relative_path", "id1": "source_subject_id", "a1": "source_group", "g1": "source_gender_label", "e1": "source_ethnicity_label"}
    )
    right = pairs[["p2", "id2", "a2", "g2", "e2"]].rename(
        columns={"p2": "relative_path", "id2": "source_subject_id", "a2": "source_group", "g2": "source_gender_label", "e2": "source_ethnicity_label"}
    )
    catalog = pd.concat([left, right], ignore_index=True).drop_duplicates("relative_path").copy()
    catalog["exists"] = catalog["relative_path"].map(lambda item: _existing(faces, item))
    missing = int((~catalog["exists"]).sum())
    if missing:
        raise ValueError(f"BFW catalog has {missing} pair-referenced images missing from canonical crop release.")
    catalog.insert(0, "dataset", "bfw")
    catalog["source_label_note"] = "Dataset-defined benchmark tag; not self-identification and not inferred by this project."
    catalog["permitted_analysis"] = "one_to_one_verification_research_only"
    catalog.to_csv(output / "bfw_catalog.csv", index=False)
    manifest = pd.DataFrame(
        {
            "path": catalog["relative_path"],
            "subject_id": "bfw_" + catalog["source_subject_id"].astype(str),
            "quality": "mid",
            "split": "benchmark_bfw",
            "weight": 1.0,
        }
    )
    manifest.to_csv(output / "bfw_extraction_manifest.csv", index=False)
    pairs["pair_group"] = pairs["a1"].where(pairs["a1"] == pairs["a2"], pairs["a1"] + "__" + pairs["a2"])
    pairs.to_csv(output / "bfw_pairs.csv", index=False)
    return {
        "images_catalogued": int(len(catalog)),
        "verification_pairs": int(len(pairs)),
        "genuine_pairs": int((pairs["label"] == 1).sum()),
        "impostor_pairs": int((pairs["label"] == 0).sum()),
        "source_group_counts": {str(key): int(value) for key, value in catalog["source_group"].value_counts().sort_index().items()},
        "same_group_pair_counts": {str(key): int(value) for key, value in pairs.loc[pairs["a1"] == pairs["a2"], "a1"].value_counts().sort_index().items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local catalogs for downloaded research benchmarks.")
    parser.add_argument("--root", default="datasets")
    args = parser.parse_args(argv)
    root = Path(args.root)
    output = root / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"fairface": build_fairface(root, output), "bfw": build_bfw(root, output)}
    (output / "benchmark_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
