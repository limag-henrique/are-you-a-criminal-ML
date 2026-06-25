#!/usr/bin/env python3
"""
Standardize curated face images for facial-similarity ML datasets.

This script never modifies source files. It writes standardized crops to a new
output tree, copies rejected originals to rejection folders, and emits one CSV
report row per input image.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


QUALITY_FOLDERS = ("Best quality", "Mid quality", "Low quality")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPORT_COLUMNS = [
    "original_path",
    "output_path",
    "quality_folder",
    "status",
    "reason",
    "num_faces_detected",
    "face_confidence",
    "original_width",
    "original_height",
    "output_width",
    "output_height",
]


@dataclass(frozen=True)
class ImageItem:
    path: Path
    quality_folder: str


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    confidence: float
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    backend: str = "unknown"

    @property
    def area(self) -> int:
        return max(0, self.box[2]) * max(0, self.box[3])


class FaceDetector:
    """MediaPipe-first face detector with an OpenCV Haar fallback."""

    def __init__(self, backend: str, min_confidence: float) -> None:
        self.backend = "opencv_haar"
        self.mp_detector = None
        self.mp_segmenter = None
        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
        ]
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

        backend = backend.lower()
        if backend in {"auto", "mediapipe"}:
            try:
                import mediapipe as mp  # type: ignore

                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=min_confidence,
                )
                self.mp_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                self.backend = "mediapipe"
            except Exception as exc:
                if backend == "mediapipe":
                    raise RuntimeError("MediaPipe was requested but could not be initialized") from exc
                print(f"MediaPipe unavailable; using OpenCV fallback: {exc}", file=sys.stderr)
        elif backend != "opencv":
            raise ValueError(f"Unsupported detector backend: {backend}")

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        detections: list[FaceDetection] = []
        if self.mp_detector is not None:
            detections = self._detect_mediapipe(image_rgb)
        if detections:
            return detections
        return self._detect_opencv(image_rgb)

    def segment_person(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if self.mp_segmenter is None:
            return None
        result = self.mp_segmenter.process(image_rgb)
        if result.segmentation_mask is None:
            return None
        return np.asarray(result.segmentation_mask, dtype=np.float32)

    def close(self) -> None:
        if self.mp_detector is not None:
            self.mp_detector.close()
        if self.mp_segmenter is not None:
            self.mp_segmenter.close()

    def _detect_mediapipe(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        h, w = image_rgb.shape[:2]
        result = self.mp_detector.process(image_rgb)
        output: list[FaceDetection] = []
        if not result.detections:
            return output

        keypoint_names = ("right_eye", "left_eye", "nose_tip", "mouth_center", "right_ear", "left_ear")
        for det in result.detections:
            score = float(det.score[0]) if det.score else 0.0
            rel_box = det.location_data.relative_bounding_box
            x = int(max(0, rel_box.xmin * w))
            y = int(max(0, rel_box.ymin * h))
            bw = int(min(w - x, rel_box.width * w))
            bh = int(min(h - y, rel_box.height * h))

            landmarks: dict[str, tuple[float, float]] = {}
            for name, point in zip(keypoint_names, det.location_data.relative_keypoints):
                landmarks[name] = (float(point.x * w), float(point.y * h))

            output.append(FaceDetection((x, y, bw, bh), score, landmarks, "mediapipe"))
        return non_max_suppression(output)

    def _detect_opencv(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        detections: list[FaceDetection] = []
        for cascade in self.face_cascades:
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(32, 32))
            for x, y, w, h in faces:
                landmarks = self._detect_eyes(gray, (int(x), int(y), int(w), int(h)))
                confidence = 0.78 if landmarks else 0.58
                detections.append(FaceDetection((int(x), int(y), int(w), int(h)), confidence, landmarks, "opencv_haar"))
        return non_max_suppression(detections)

    def _detect_eyes(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
        x, y, w, h = box
        upper = gray[y : y + int(h * 0.62), x : x + w]
        if upper.size == 0 or self.eye_cascade.empty():
            return {}

        eyes = self.eye_cascade.detectMultiScale(upper, scaleFactor=1.08, minNeighbors=4, minSize=(12, 12))
        if len(eyes) < 2:
            return {}

        centers = sorted(
            [(x + ex + ew / 2.0, y + ey + eh / 2.0, ew * eh) for ex, ey, ew, eh in eyes],
            key=lambda item: item[2],
            reverse=True,
        )
        best_pair: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        best_score = -1.0
        for i in range(min(4, len(centers))):
            for j in range(i + 1, min(5, len(centers))):
                p1, p2 = centers[i], centers[j]
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])
                if dx < w * 0.18 or dx > w * 0.75 or dy > h * 0.25:
                    continue
                score = dx - dy * 1.5 + min(p1[2], p2[2]) * 0.001
                if score > best_score:
                    best_score = score
                    best_pair = (p1, p2)

        if best_pair is None:
            return {}
        left, right = sorted(best_pair, key=lambda p: p[0])
        return {"left_eye": (left[0], left[1]), "right_eye": (right[0], right[1])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize face photos into aligned 512x512 white-background JPGs or transparent PNGs."
    )
    parser.add_argument("--input_root", default="data_curated", help="Input root containing quality folders.")
    parser.add_argument("--output_root", default="data_processed/face_standardized", help="Output root directory.")
    parser.add_argument("--size", type=int, default=512, help="Final square output size in pixels.")
    parser.add_argument("--background", choices=("white", "transparent"), default="white", help="Output background.")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for white-background output.")
    parser.add_argument("--margin", type=float, default=0.30, help="Face crop margin around the detected box.")
    parser.add_argument(
        "--sample_per_folder",
        type=int,
        default=0,
        help="Process only N images per quality folder. Use 0 for the whole dataset.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "mediapipe", "opencv"),
        default="auto",
        help="Face detector backend. auto prefers MediaPipe and falls back to OpenCV.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.35,
        help="Minimum confidence used by MediaPipe face detection.",
    )
    parser.add_argument(
        "--segmentation_threshold",
        type=float,
        default=0.25,
        help="Foreground threshold for MediaPipe segmentation.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def locate_images(input_root: Path, quality_folders: Iterable[str], sample_per_folder: int = 0) -> list[ImageItem]:
    items: list[ImageItem] = []
    for quality in quality_folders:
        folder = input_root / quality
        if not folder.exists():
            print(f"Warning: quality folder not found: {folder}", file=sys.stderr)
            continue
        paths = sorted(
            (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: str(p).casefold(),
        )
        if sample_per_folder > 0:
            paths = paths[:sample_per_folder]
        items.extend(ImageItem(path=p, quality_folder=quality) for p in paths)
    return items


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.asarray(img)


def detect_faces(detector: FaceDetector, image_rgb: np.ndarray) -> list[FaceDetection]:
    return sorted(detector.detect(image_rgb), key=lambda face: (face.area, face.confidence), reverse=True)


def choose_main_face(faces: list[FaceDetection]) -> FaceDetection:
    return faces[0]


def align_face(image_rgb: np.ndarray, face: FaceDetection) -> tuple[np.ndarray, FaceDetection]:
    angle = roll_from_landmarks(face.landmarks)
    if angle is None or abs(angle) < 0.5:
        return image_rgb, face

    h, w = image_rgb.shape[:2]
    x, y, fw, fh = face.box
    center = (x + fw / 2.0, y + fh / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        image_rgb,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    corners = [(x, y), (x + fw, y), (x + fw, y + fh), (x, y + fh)]
    transformed = np.asarray([transform_point(mat, point) for point in corners], dtype=np.float32)
    x1, y1 = np.floor(transformed.min(axis=0)).astype(int)
    x2, y2 = np.ceil(transformed.max(axis=0)).astype(int)
    new_box = (
        max(0, int(x1)),
        max(0, int(y1)),
        min(w - max(0, int(x1)), max(1, int(x2 - x1))),
        min(h - max(0, int(y1)), max(1, int(y2 - y1))),
    )
    new_landmarks = {name: transform_point(mat, point) for name, point in face.landmarks.items()}
    return aligned, FaceDetection(new_box, face.confidence, new_landmarks, face.backend)


def crop_face_square(image_rgb: np.ndarray, face: FaceDetection, margin: float) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x, y, fw, fh = face.box
    cx, cy = crop_center(face)
    side = int(round(max(fw, fh) * (1.0 + 2.0 * margin)))
    side = max(side, 32)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image_rgb = cv2.copyMakeBorder(
            image_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("empty_face_crop")
    return crop


def remove_or_neutralize_background(
    crop_rgb: np.ndarray,
    detector: FaceDetector,
    background: str,
    segmentation_threshold: float,
) -> np.ndarray:
    mask = detector.segment_person(crop_rgb)
    if mask is None or mask.shape[:2] != crop_rgb.shape[:2] or float(np.max(mask)) < 0.05:
        mask = elliptical_face_mask(crop_rgb.shape[0], crop_rgb.shape[1])
    else:
        mask = (mask >= segmentation_threshold).astype(np.float32)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
        mask = np.clip(mask, 0.0, 1.0)

    if background == "transparent":
        alpha = (mask * 255).round().astype(np.uint8)
        return np.dstack([crop_rgb, alpha])

    alpha = mask[:, :, None].astype(np.float32)
    white = np.full_like(crop_rgb, 255, dtype=np.uint8)
    output = crop_rgb.astype(np.float32) * alpha + white.astype(np.float32) * (1.0 - alpha)
    output = np.clip(output, 0, 255).round().astype(np.uint8)
    output[mask <= 0.01] = (255, 255, 255)
    return output


def save_image(image: np.ndarray, output_path: Path, background: str, jpeg_quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if background == "transparent":
        Image.fromarray(image).save(output_path, format="PNG", optimize=True)
    else:
        rgb = image[:, :, :3] if image.ndim == 3 else image
        Image.fromarray(rgb).save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)


def register_report(rows: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for column in REPORT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    remaining = [column for column in df.columns if column not in REPORT_COLUMNS]
    df = df[REPORT_COLUMNS + remaining]
    df.to_csv(report_path, index=False, encoding="utf-8")


def process_image(
    item: ImageItem,
    input_root: Path,
    output_root: Path,
    detector: FaceDetector,
    size: int,
    background: str,
    jpeg_quality: int,
    margin: float,
    segmentation_threshold: float,
) -> dict[str, object]:
    base_row = {
        "original_path": str(item.path),
        "output_path": "",
        "quality_folder": item.quality_folder,
        "status": "",
        "reason": "",
        "num_faces_detected": 0,
        "face_confidence": "",
        "original_width": "",
        "original_height": "",
        "output_width": "",
        "output_height": "",
        "background": background,
        "detector_backend": "",
        "source_extension": item.path.suffix.lower(),
    }

    try:
        image_rgb = load_image(item.path)
    except Exception as exc:
        rejected_path = copy_rejected_original(item, input_root, output_root, "corrupted")
        return {
            **base_row,
            "output_path": str(rejected_path),
            "status": "corrupted",
            "reason": f"unreadable_or_corrupted:{exc.__class__.__name__}",
        }

    original_height, original_width = image_rgb.shape[:2]
    base_row["original_width"] = original_width
    base_row["original_height"] = original_height

    faces = detect_faces(detector, image_rgb)
    if not faces:
        rejected_path = copy_rejected_original(item, input_root, output_root, "no_face_detected")
        return {
            **base_row,
            "output_path": str(rejected_path),
            "status": "rejected",
            "reason": "no_face_detected",
        }

    main_face = choose_main_face(faces)
    aligned_rgb, aligned_face = align_face(image_rgb, main_face)
    crop_rgb = crop_face_square(aligned_rgb, aligned_face, margin=margin)
    crop_rgb = cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA)
    standardized = remove_or_neutralize_background(
        crop_rgb,
        detector=detector,
        background=background,
        segmentation_threshold=segmentation_threshold,
    )

    output_path = standardized_output_path(item, input_root, output_root, background)
    save_image(standardized, output_path, background=background, jpeg_quality=jpeg_quality)

    return {
        **base_row,
        "output_path": str(output_path),
        "status": "success",
        "reason": "ok",
        "num_faces_detected": len(faces),
        "face_confidence": f"{main_face.confidence:.4f}",
        "output_width": size,
        "output_height": size,
        "detector_backend": main_face.backend,
        "multiple_faces": len(faces) > 1,
        "face_box": ",".join(str(v) for v in main_face.box),
    }


def safe_output_name(path: Path, input_root: Path, extension: str) -> str:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem).strip("._-")
    stem = stem[:90] or "image"
    return f"{stem}_{digest}{extension}"


def standardized_output_path(item: ImageItem, input_root: Path, output_root: Path, background: str) -> Path:
    mode_dir = "white_bg" if background == "white" else "transparent_bg"
    extension = ".jpg" if background == "white" else ".png"
    return output_root / mode_dir / item.quality_folder / safe_output_name(item.path, input_root, extension)


def copy_rejected_original(item: ImageItem, input_root: Path, output_root: Path, reason: str) -> Path:
    suffix = item.path.suffix.lower() if item.path.suffix.lower() in IMAGE_SUFFIXES else item.path.suffix
    target = output_root / "rejected" / reason / item.quality_folder / safe_output_name(item.path, input_root, suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.path, target)
    return target


def non_max_suppression(detections: list[FaceDetection], iou_threshold: float = 0.35) -> list[FaceDetection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda det: (det.confidence, det.area), reverse=True)
    kept: list[FaceDetection] = []
    for detection in detections:
        if all(iou(detection.box, existing.box) < iou_threshold for existing in kept):
            kept.append(detection)
    return kept


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def roll_from_landmarks(landmarks: dict[str, tuple[float, float]]) -> float | None:
    left = landmarks.get("left_eye")
    right = landmarks.get("right_eye")
    if not left or not right:
        return None
    image_left_eye, image_right_eye = sorted((left, right), key=lambda point: point[0])
    dx = image_right_eye[0] - image_left_eye[0]
    dy = image_right_eye[1] - image_left_eye[1]
    if abs(dx) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def transform_point(mat: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (
        float(mat[0, 0] * x + mat[0, 1] * y + mat[0, 2]),
        float(mat[1, 0] * x + mat[1, 1] * y + mat[1, 2]),
    )


def crop_center(face: FaceDetection) -> tuple[float, float]:
    x, y, w, h = face.box
    box_center = (x + w / 2.0, y + h / 2.0)
    useful_landmarks = [
        face.landmarks[name]
        for name in ("left_eye", "right_eye", "nose_tip", "mouth_center")
        if name in face.landmarks
    ]
    if not useful_landmarks:
        return box_center[0], box_center[1] + h * 0.04

    lx = float(np.mean([point[0] for point in useful_landmarks]))
    ly = float(np.mean([point[1] for point in useful_landmarks]))
    cx = box_center[0] * 0.65 + lx * 0.35
    cy = box_center[1] * 0.65 + ly * 0.35 + h * 0.04
    return cx, cy


def elliptical_face_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, int(height * 0.52))
    axes = (int(width * 0.43), int(height * 0.50))
    cv2.ellipse(mask, center, axes, 0, 0, 360, color=1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) * 0.015)
    mask = np.clip(mask, 0.0, 1.0)
    mask[mask < 0.02] = 0.0
    return mask


def print_summary(rows: list[dict[str, object]], total_found: int, elapsed_seconds: float) -> None:
    success = sum(row.get("status") == "success" for row in rows)
    no_face = sum(row.get("reason") == "no_face_detected" for row in rows)
    corrupted = sum(row.get("status") == "corrupted" for row in rows)
    multiple = sum(bool(row.get("multiple_faces")) for row in rows)
    unexpected = sum(row.get("status") == "error" for row in rows)

    print()
    print(f"Total de imagens encontradas: {total_found}")
    print(f"Total processado com sucesso: {success}")
    print(f"Total rejeitado por ausencia de face: {no_face}")
    print(f"Total corrompido: {corrupted}")
    print(f"Total com multiplas faces: {multiple}")
    print(f"Total com erro inesperado: {unexpected}")
    print(f"Tempo total de execucao: {elapsed_seconds:.2f}s")


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("--jpeg_quality must be between 1 and 100")

    input_root = resolve_repo_path(args.input_root)
    output_root = resolve_repo_path(args.output_root)
    report_path = output_root / "processing_report.csv"

    items = locate_images(input_root, QUALITY_FOLDERS, sample_per_folder=args.sample_per_folder)
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Background mode: {args.background}")
    print(f"Total de imagens encontradas: {len(items)}")

    rows: list[dict[str, object]] = []
    detector = FaceDetector(args.detector, min_confidence=args.min_detection_confidence)
    start = time.perf_counter()

    try:
        for item in tqdm(items, desc="Standardizing faces", unit="img"):
            try:
                row = process_image(
                    item=item,
                    input_root=input_root,
                    output_root=output_root,
                    detector=detector,
                    size=args.size,
                    background=args.background,
                    jpeg_quality=args.jpeg_quality,
                    margin=args.margin,
                    segmentation_threshold=args.segmentation_threshold,
                )
            except Exception as exc:
                row = {
                    "original_path": str(item.path),
                    "output_path": "",
                    "quality_folder": item.quality_folder,
                    "status": "error",
                    "reason": f"unexpected_error:{exc.__class__.__name__}:{exc}",
                    "num_faces_detected": "",
                    "face_confidence": "",
                    "original_width": "",
                    "original_height": "",
                    "output_width": "",
                    "output_height": "",
                    "traceback": traceback.format_exc(limit=5),
                }
            rows.append(row)
    finally:
        detector.close()

    elapsed = time.perf_counter() - start
    register_report(rows, report_path)
    print(f"Relatorio CSV: {report_path}")
    print_summary(rows, total_found=len(items), elapsed_seconds=elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
