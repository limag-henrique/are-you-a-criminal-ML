"""Fetch documented face-analysis benchmarks into a git-ignored directory.

The script purposefully downloads complete upstream releases rather than a
hand-picked set of people. It writes a cryptographic checksum and provenance
record so a later study can recreate the exact local materials.

It must be run only for approved, non-operational research. The downloaded
images are biometric data and must not be committed, redistributed, or used to
identify people outside the source dataset's stated purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


SOURCES: dict[str, dict[str, Any]] = {
    "fairface": {
        "files": {
            "fairface-img-margin025-trainval.zip": "https://drive.usercontent.google.com/download?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86&export=download&confirm=t",
            "fairface_label_train.csv": "https://drive.usercontent.google.com/download?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH&export=download&confirm=t",
            "fairface_label_val.csv": "https://drive.usercontent.google.com/download?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D&export=download&confirm=t",
        },
        "citation": "Karkkainen, K. & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age.",
        "license": "CC BY 4.0, as stated by the upstream FairFace repository.",
        "source_page": "https://github.com/joojs/fairface",
        "use_note": (
            "The source labels are externally assigned benchmark categories, not self-identification. "
            "Use them only as documented benchmark strata and report their limitations."
        ),
    },
    "bfw": {
        "files": {
            "BFW-Release.zip": "https://www.dropbox.com/scl/fi/5gindh41lrw8j7bgyv9mq/BFW-Release.zip?rlkey=k7kf4knhm18qi3be661m8qmo4&dl=1",
        },
        "citation": "Robinson, J. P. et al. (2020). Face Recognition: Too Bias, or Not Too Bias?",
        "license": "Upstream release describes BFW as a research-purpose benchmark; do not redistribute downloaded images.",
        "source_page": "https://github.com/visionjo/facerec-bias-bfw",
        "use_note": (
            "The source provides verification pairs and dataset-defined demographic tags. "
            "Those tags are benchmark annotations, not attributes to infer from a new image."
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "face-profile-ml-research-fetcher/1.0"}
    with requests.get(url, stream=True, timeout=(20, 120), headers=headers) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", "0"))
        received = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    received += len(chunk)
                    if expected and received % (100 * 1024 * 1024) < len(chunk):
                        print(f"  {destination.name}: {received / 1e6:.0f}/{expected / 1e6:.0f} MB", flush=True)
    if expected and received != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete download for {destination.name}: {received} of {expected} bytes")
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        invalid = zip_file.testzip()
        if invalid:
            raise RuntimeError(f"Corrupt ZIP member in {archive.name}: {invalid}")
        for item in zip_file.infolist():
            target = (destination / item.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in {archive.name}: {item.filename}")
        zip_file.extractall(destination)


def collect_files(dataset: str, root: Path) -> dict[str, Any]:
    archive_dir = root / "archives"
    extracted_dir = root / "extracted"
    archive_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for filename, url in SOURCES[dataset]["files"].items():
        destination = archive_dir / filename
        if not destination.exists():
            print(f"Downloading {dataset}/{filename}", flush=True)
            download(url, destination)
        else:
            print(f"Reusing {destination}", flush=True)
        record = {
            "filename": filename,
            "url": url,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "zip_crc_verified_during_extraction": destination.suffix.lower() == ".zip",
        }
        records.append(record)
        if destination.suffix.lower() == ".zip":
            marker = extracted_dir / f".{filename}.complete"
            if not marker.exists():
                print(f"Extracting {dataset}/{filename}", flush=True)
                safe_extract(destination, extracted_dir)
                marker.write_text(record["sha256"] + "\n", encoding="utf-8")
    return {"files": records, "extracted_dir": str(extracted_dir.resolve())}


def extract_bfw_face_crops(root: Path) -> dict[str, Any]:
    """Verify and expand the nested canonical cropped-face release from BFW."""
    extracted = root / "extracted"
    archive = extracted / "bfw-faces-cropped.zip"
    expected_file = extracted / "bfw-faces-cropped.md5"
    if not archive.exists() or not expected_file.exists():
        raise FileNotFoundError("BFW release did not contain the expected cropped-face archive and checksum.")
    expected = expected_file.read_text(encoding="utf-8").split()[0].lower()
    actual = hashlib.md5(archive.read_bytes()).hexdigest()  # noqa: S324 - upstream integrity format is MD5.
    if actual != expected:
        raise RuntimeError(f"BFW cropped-face archive checksum mismatch: expected {expected}, got {actual}")
    faces_dir = extracted / "faces-cropped"
    marker = faces_dir / ".complete"
    if not marker.exists():
        print("Extracting BFW canonical cropped faces", flush=True)
        safe_extract(archive, faces_dir)
        marker.write_text(actual + "\n", encoding="utf-8")
    image_count = sum(1 for item in faces_dir.rglob("*") if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"})
    subgroup_dirs = [item for item in faces_dir.rglob("asian_females") if item.is_dir()]
    if len(subgroup_dirs) != 1:
        raise RuntimeError(f"Expected one BFW image root, found {len(subgroup_dirs)} candidate roots.")
    canonical_root = subgroup_dirs[0].parent
    return {
        "canonical_faces_archive": archive.name,
        "canonical_faces_md5": actual,
        "canonical_faces_dir": str(faces_dir.resolve()),
        "canonical_faces_relative_root": str(canonical_root.relative_to(faces_dir)).replace("\\", "/"),
        "canonical_face_image_count": image_count,
        "canonical_faces_zip_crc_verified_during_extraction": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch complete, documented research benchmark releases.")
    parser.add_argument("--root", default="datasets/external")
    parser.add_argument("--datasets", default="fairface,bfw", help="Comma-separated: fairface,bfw")
    args = parser.parse_args(argv)
    selected = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    unknown = set(selected).difference(SOURCES)
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(sorted(unknown))}")
    root = Path(args.root)
    for dataset in selected:
        dataset_root = root / dataset
        result = collect_files(dataset, dataset_root)
        if dataset == "bfw":
            result.update(extract_bfw_face_crops(dataset_root))
        provenance = {
            "dataset": dataset,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "source_page": SOURCES[dataset]["source_page"],
            "citation": SOURCES[dataset]["citation"],
            "license_or_terms": SOURCES[dataset]["license"],
            "use_note": SOURCES[dataset]["use_note"],
            "do_not_commit_or_redistribute_images": True,
            **result,
        }
        (dataset_root / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
