#!/usr/bin/env python3
"""
Deterministic local image curation pipeline for face-training datasets.

The script never deletes or overwrites source images. It writes approved aligned
face crops, copies objective rejects/quarantine originals when configured, and
emits CSV/JSON reports with one final decision per input image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import DBSCAN
from tqdm import tqdm


DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": "data",
    "output_dir": "data_curated",
    "approved_dir": "approved",
    "rejected_dir": "rejected",
    "quarantine_dir": "quarantine",
    "candidates_dir": "_candidates",
    "reports_dir": "reports",
    "copy_rejected_originals": True,
    "copy_quarantine_originals": True,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
    "output_size": 256,
    "output_format": "jpg",
    "jpeg_quality": 94,
    "crop_margin": 0.35,
    "allow_multiple_faces": False,
    "max_images_per_person": 2,
    "detector_backend": "auto",
    "min_detection_confidence_approve": 0.75,
    "min_detection_confidence_reject": 0.35,
    "min_image_side_approve": 160,
    "min_image_side_reject": 80,
    "min_face_side_approve": 72,
    "min_face_side_reject": 36,
    "min_primary_face_area_ratio": 0.045,
    "reject_face_area_ratio": 0.012,
    "secondary_face_area_ratio": 0.35,
    "max_face_center_offset": 0.38,
    "grayscale_channel_delta_reject": 4.0,
    "grayscale_saturation_reject": 18.0,
    "blur_laplacian_approve": 95.0,
    "blur_laplacian_reject": 28.0,
    "face_blur_laplacian_approve": 80.0,
    "face_blur_laplacian_reject": 24.0,
    "borderline_blur_margin": 18.0,
    "brightness_min_approve": 55.0,
    "brightness_max_approve": 205.0,
    "brightness_min_reject": 25.0,
    "brightness_max_reject": 235.0,
    "contrast_min_approve": 28.0,
    "contrast_min_reject": 12.0,
    "extreme_dark_pixel_ratio_reject": 0.58,
    "extreme_bright_pixel_ratio_reject": 0.58,
    "max_roll_degrees_approve": 18.0,
    "max_roll_degrees_reject": 35.0,
    "quality_score_approve": 72.0,
    "quality_score_reject": 38.0,
    "quality_score_borderline_margin": 7.0,
    "pixelation_blockiness_quarantine": 16.0,
    "pixelation_blockiness_reject": 28.0,
    "collage_grid_line_ratio_quarantine": 0.08,
    "collage_grid_line_ratio_reject": 0.16,
    "non_photo_edge_density_quarantine": 0.24,
    "non_photo_low_color_bins": 64,
    "require_eye_landmarks_for_approval": True,
    "embedding_backend": "insightface",
    "require_embeddings_for_person_limit": False,
    "embedding_model_name": "buffalo_l",
    "embedding_det_size": 640,
    "embedding_ctx_id": -1,
    "embedding_same_person_distance": 0.42,
    "embedding_ambiguous_distance": 0.55,
    "fallback_group_by_filename_identity": True,
    "random_seed": 42,
}


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    confidence: float
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    backend: str = "unknown"

    @property
    def area(self) -> int:
        return max(0, self.box[2]) * max(0, self.box[3])


@dataclass
class Candidate:
    row_index: int
    source_path: Path
    candidate_path: Path
    quality_score: float
    identity_key: str
    embedding: np.ndarray | None = None
    embedding_status: str = "not_attempted"


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        with path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "preprocess.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )


def list_image_files(input_dir: Path, allowed_extensions: Iterable[str]) -> list[Path]:
    allowed = {ext.lower() for ext in allowed_extensions}
    return sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed),
        key=lambda p: str(p).casefold(),
    )


def read_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.asarray(img)


def safe_stem(path: Path, input_dir: Path, max_len: int = 72) -> str:
    rel = path.relative_to(input_dir) if path.is_relative_to(input_dir) else path.name
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem).strip("._-").lower()
    stem = stem[:max_len] or "image"
    return f"{stem}_{digest}"


def inferred_identity_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"_[0-9a-f]{8,}$", "", stem)
    stem = re.sub(r"_[0-9]{3,}_profile(_[0-9a-f]+)?$", "", stem)
    stem = re.sub(r"__(alias|photo|profile|thumb|thumbnail).*$", "", stem)
    stem = re.sub(r"__[a-z_]+$", "", stem)
    stem = re.sub(r"_[0-9]{4}-[0-9]+(_[0-9]+)?$", "", stem)
    stem = re.sub(r"_[0-9]+$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]


def copy_original(path: Path, decision_dir: Path, input_dir: Path) -> Path:
    rel_parent = path.relative_to(input_dir).parent if path.is_relative_to(input_dir) else Path()
    target_dir = decision_dir / rel_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if target.exists():
        target = target_dir / f"{path.stem}_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:8]}{path.suffix}"
    shutil.copy2(path, target)
    return target


def write_rgb_jpeg(image_rgb: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path, format="JPEG", quality=quality, optimize=True)


def image_metrics(image_rgb: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> dict[str, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    channel_delta = float(
        np.mean(np.abs(image_rgb[:, :, 0].astype(np.int16) - image_rgb[:, :, 1].astype(np.int16)))
        + np.mean(np.abs(image_rgb[:, :, 1].astype(np.int16) - image_rgb[:, :, 2].astype(np.int16)))
    ) / 2.0
    metrics = {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "blur_laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "channel_delta": channel_delta,
        "dark_pixel_ratio": float(np.mean(gray < 25)),
        "bright_pixel_ratio": float(np.mean(gray > 245)),
        "edge_density": edge_density(gray),
        "color_bins": color_bins(image_rgb),
        "grid_line_ratio": grid_line_ratio(gray),
        "blockiness": blockiness_score(gray),
    }
    if face_box is not None:
        x, y, w, h = face_box
        roi = gray[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]
        metrics["face_blur_laplacian"] = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
    else:
        metrics["face_blur_laplacian"] = 0.0
    return metrics


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 180)
    return float(np.mean(edges > 0))


def color_bins(image_rgb: np.ndarray) -> int:
    small = cv2.resize(image_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    quantized = (small // 32).reshape(-1, 3)
    return int(np.unique(quantized, axis=0).shape[0])


def grid_line_ratio(gray: np.ndarray) -> float:
    if gray.shape[0] < 80 or gray.shape[1] < 80:
        return 0.0
    row_std = np.std(gray, axis=1)
    col_std = np.std(gray, axis=0)
    row_mean = np.mean(gray, axis=1)
    col_mean = np.mean(gray, axis=0)
    flat_rows = np.mean((row_std < 5) & ((row_mean < 35) | (row_mean > 220)))
    flat_cols = np.mean((col_std < 5) & ((col_mean < 35) | (col_mean > 220)))
    return float(max(flat_rows, flat_cols))


def blockiness_score(gray: np.ndarray) -> float:
    if min(gray.shape) < 32:
        return 0.0
    gray_f = gray.astype(np.float32)
    vdiff = np.abs(np.diff(gray_f, axis=1))
    hdiff = np.abs(np.diff(gray_f, axis=0))
    v_boundary = vdiff[:, 7::8]
    h_boundary = hdiff[7::8, :]
    v_inner = np.delete(vdiff, np.s_[7::8], axis=1) if vdiff.shape[1] > 8 else vdiff
    h_inner = np.delete(hdiff, np.s_[7::8], axis=0) if hdiff.shape[0] > 8 else hdiff
    boundary = float(np.mean(v_boundary) + np.mean(h_boundary)) / 2.0
    inner = float(np.mean(v_inner) + np.mean(h_inner)) / 2.0
    return max(0.0, boundary - inner)


def quality_score(metrics: dict[str, float], face_area_ratio: float, roll_degrees: float | None) -> float:
    score = 100.0
    score -= max(0.0, 95.0 - metrics["face_blur_laplacian"]) * 0.18
    score -= max(0.0, 90.0 - metrics["blur_laplacian"]) * 0.10
    if metrics["brightness"] < 70:
        score -= (70.0 - metrics["brightness"]) * 0.35
    if metrics["brightness"] > 190:
        score -= (metrics["brightness"] - 190.0) * 0.30
    score -= max(0.0, 32.0 - metrics["contrast"]) * 0.65
    score -= max(0.0, 0.05 - face_area_ratio) * 550.0
    score -= max(0.0, metrics["blockiness"] - 8.0) * 1.2
    if roll_degrees is not None:
        score -= max(0.0, abs(roll_degrees) - 12.0) * 0.8
    return float(max(0.0, min(100.0, score)))


class FaceDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend = "opencv_haar"
        self.mp_detector = None
        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
        ]
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

        backend = str(config["detector_backend"]).lower()
        if backend in {"auto", "mediapipe"}:
            try:
                import mediapipe as mp  # type: ignore

                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=float(config["min_detection_confidence_reject"]),
                )
                self.backend = "mediapipe"
            except Exception as exc:
                if backend == "mediapipe":
                    raise RuntimeError("MediaPipe detector requested but unavailable") from exc
                logging.warning("MediaPipe unavailable; falling back to OpenCV Haar detector: %s", exc)

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        if self.mp_detector is not None:
            detections = self._detect_mediapipe(image_rgb)
            if detections:
                return detections
        return self._detect_opencv(image_rgb)

    def close(self) -> None:
        if self.mp_detector is not None:
            self.mp_detector.close()

    def _detect_mediapipe(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        h, w = image_rgb.shape[:2]
        result = self.mp_detector.process(image_rgb)
        output: list[FaceDetection] = []
        if not result.detections:
            return output
        for det in result.detections:
            score = float(det.score[0]) if det.score else 0.0
            box = det.location_data.relative_bounding_box
            x = int(max(0, box.xmin * w))
            y = int(max(0, box.ymin * h))
            bw = int(min(w - x, box.width * w))
            bh = int(min(h - y, box.height * h))
            landmarks: dict[str, tuple[float, float]] = {}
            keypoints = det.location_data.relative_keypoints
            if len(keypoints) >= 2:
                landmarks["right_eye"] = (keypoints[0].x * w, keypoints[0].y * h)
                landmarks["left_eye"] = (keypoints[1].x * w, keypoints[1].y * h)
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
                confidence = 0.82 if landmarks else 0.58
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
        best_pair = None
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
        if not best_pair:
            return {}
        left, right = sorted(best_pair, key=lambda p: p[0])
        return {"left_eye": (left[0], left[1]), "right_eye": (right[0], right[1])}


def non_max_suppression(detections: list[FaceDetection], iou_threshold: float = 0.35) -> list[FaceDetection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: (d.confidence, d.area), reverse=True)
    kept: list[FaceDetection] = []
    for det in detections:
        if all(iou(det.box, old.box) < iou_threshold for old in kept):
            kept.append(det)
    return kept


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def roll_from_landmarks(landmarks: dict[str, tuple[float, float]]) -> float | None:
    left = landmarks.get("left_eye")
    right = landmarks.get("right_eye")
    if not left or not right:
        return None
    return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def aligned_square_crop(
    image_rgb: np.ndarray,
    face: FaceDetection,
    margin: float,
    output_size: int,
) -> tuple[np.ndarray | None, str | None, float | None]:
    roll = roll_from_landmarks(face.landmarks)
    h, w = image_rgb.shape[:2]
    x, y, fw, fh = face.box
    cx, cy = x + fw / 2.0, y + fh / 2.0

    if roll is not None:
        rot_mat = cv2.getRotationMatrix2D((cx, cy), roll, 1.0)
        rotated = cv2.warpAffine(image_rgb, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cx, cy = transform_point(rot_mat, (cx, cy))
    else:
        rotated = image_rgb

    side = int(round(max(fw, fh) * (1.0 + margin * 2.0)))
    side = max(side, 16)
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        rotated = cv2.copyMakeBorder(
            rotated,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None, "aligned_crop_failed", roll
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized, None, roll


def transform_point(mat: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    px = mat[0, 0] * x + mat[0, 1] * y + mat[0, 2]
    py = mat[1, 0] * x + mat[1, 1] * y + mat[1, 2]
    return float(px), float(py)


class InsightFaceEmbedder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.available = False
        self.error = ""
        self.app = None
        backend = str(config.get("embedding_backend", "insightface")).lower()
        if backend in {"none", "disabled", "off"}:
            self.error = "disabled_by_config"
            return
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            self.app = FaceAnalysis(name=str(config["embedding_model_name"]))
            det_size = int(config["embedding_det_size"])
            self.app.prepare(ctx_id=int(config["embedding_ctx_id"]), det_size=(det_size, det_size))
            self.available = True
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {exc}"
            logging.warning("InsightFace embeddings unavailable: %s", self.error)

    def embed(self, image_path: Path) -> tuple[np.ndarray | None, str]:
        if not self.available or self.app is None:
            return None, self.error or "embedding_backend_unavailable"
        bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, "candidate_unreadable_for_embedding"
        faces = self.app.get(bgr)
        if len(faces) != 1:
            return None, f"embedding_face_count_{len(faces)}"
        embedding = np.asarray(faces[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm <= 0:
            return None, "embedding_invalid"
        return embedding / norm, "ok"


def reject_row(
    row: dict[str, Any],
    reason: str,
    rejected_dir: Path,
    input_dir: Path,
    copy_originals: bool,
) -> dict[str, Any]:
    row["decision"] = "rejected"
    row["reason"] = reason
    if copy_originals:
        row["decision_path"] = str(copy_original(Path(row["source_path"]), rejected_dir, input_dir))
    return row


def quarantine_row(
    row: dict[str, Any],
    reason: str,
    quarantine_dir: Path,
    input_dir: Path,
    copy_originals: bool,
) -> dict[str, Any]:
    row["decision"] = "quarantine"
    row["reason"] = reason
    if copy_originals:
        row["decision_path"] = str(copy_original(Path(row["source_path"]), quarantine_dir, input_dir))
    return row


def process_one(
    path: Path,
    index: int,
    detector: FaceDetector,
    config: dict[str, Any],
    paths: dict[str, Path],
    input_dir: Path,
) -> tuple[dict[str, Any], Candidate | None]:
    row: dict[str, Any] = {
        "index": index,
        "source_path": str(path),
        "decision": "",
        "reason": "",
        "decision_path": "",
        "candidate_path": "",
        "cluster_id": "",
        "cluster_method": "",
        "quality_score": "",
        "face_count": 0,
        "detector_backend": "",
        "detection_confidence": "",
        "face_area_ratio": "",
        "roll_degrees": "",
        "embedding_status": "",
        "metrics_json": "",
    }

    try:
        image_rgb = read_rgb_image(path)
    except Exception as exc:
        return reject_row(
            row,
            f"unreadable_or_corrupted:{exc.__class__.__name__}",
            paths["rejected"],
            input_dir,
            bool(config["copy_rejected_originals"]),
        ), None

    h, w = image_rgb.shape[:2]
    if min(h, w) < int(config["min_image_side_reject"]):
        row["metrics_json"] = json.dumps({"width": w, "height": h}, sort_keys=True)
        return reject_row(row, "resolution_far_below_minimum", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    metrics = image_metrics(image_rgb)
    row["metrics_json"] = json.dumps({"width": w, "height": h, **metrics}, sort_keys=True)

    if (
        metrics["channel_delta"] <= float(config["grayscale_channel_delta_reject"])
        and metrics["saturation"] <= float(config["grayscale_saturation_reject"])
    ):
        return reject_row(row, "clear_grayscale_or_black_and_white", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if metrics["grid_line_ratio"] >= float(config["collage_grid_line_ratio_reject"]):
        return reject_row(row, "clear_collage_or_composite_grid", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    faces = sorted(detector.detect(image_rgb), key=lambda f: (f.area, f.confidence), reverse=True)
    row["face_count"] = len(faces)
    if not faces:
        return reject_row(row, "no_detectable_human_face", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    primary = faces[0]
    row["detector_backend"] = primary.backend
    row["detection_confidence"] = f"{primary.confidence:.4f}"
    face_area_ratio = primary.area / float(w * h)
    row["face_area_ratio"] = f"{face_area_ratio:.6f}"

    metrics = image_metrics(image_rgb, primary.box)
    row["metrics_json"] = json.dumps({"width": w, "height": h, **metrics}, sort_keys=True)

    if primary.confidence < float(config["min_detection_confidence_reject"]):
        return reject_row(row, "face_detection_confidence_below_reject_threshold", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if primary.box[2] < int(config["min_face_side_reject"]) or primary.box[3] < int(config["min_face_side_reject"]):
        return reject_row(row, "face_far_too_small", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if face_area_ratio < float(config["reject_face_area_ratio"]):
        return reject_row(row, "person_not_primary_subject_face_too_small", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if metrics["face_blur_laplacian"] < float(config["face_blur_laplacian_reject"]):
        return reject_row(row, "face_region_extremely_blurry", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if metrics["brightness"] < float(config["brightness_min_reject"]) or metrics["brightness"] > float(config["brightness_max_reject"]):
        return reject_row(row, "extreme_under_or_over_exposure", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if metrics["contrast"] < float(config["contrast_min_reject"]):
        return reject_row(row, "extremely_low_contrast", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if (
        metrics["dark_pixel_ratio"] >= float(config["extreme_dark_pixel_ratio_reject"])
        or metrics["bright_pixel_ratio"] >= float(config["extreme_bright_pixel_ratio_reject"])
    ):
        return reject_row(row, "extreme_dark_or_bright_pixel_ratio", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if metrics["blockiness"] >= float(config["pixelation_blockiness_reject"]):
        return reject_row(row, "severe_pixelation_or_compression_artifacts", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    if (
        metrics["saturation"] < 22
        and metrics["edge_density"] >= float(config["non_photo_edge_density_quarantine"])
        and metrics["color_bins"] <= int(config["non_photo_low_color_bins"])
    ):
        return quarantine_row(row, "possible_drawing_avatar_or_non_photo", paths["quarantine"], input_dir, bool(config["copy_quarantine_originals"])), None

    significant_faces = [f for f in faces[1:] if f.area >= primary.area * float(config["secondary_face_area_ratio"])]
    if significant_faces and not bool(config["allow_multiple_faces"]):
        return quarantine_row(row, "multiple_faces_primary_subject_unclear", paths["quarantine"], input_dir, bool(config["copy_quarantine_originals"])), None

    x, y, fw, fh = primary.box
    center_offset = math.hypot((x + fw / 2.0) / w - 0.5, (y + fh / 2.0) / h - 0.5)
    if center_offset > float(config["max_face_center_offset"]):
        return quarantine_row(row, "face_off_center_or_person_not_primary_subject", paths["quarantine"], input_dir, bool(config["copy_quarantine_originals"])), None

    roll = roll_from_landmarks(primary.landmarks)
    row["roll_degrees"] = "" if roll is None else f"{roll:.4f}"
    if roll is not None and abs(roll) > float(config["max_roll_degrees_reject"]):
        return reject_row(row, "extreme_face_roll_pose", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None

    uncertainties: list[str] = []
    if primary.confidence < float(config["min_detection_confidence_approve"]):
        uncertainties.append("face_detection_confidence_borderline")
    if min(h, w) < int(config["min_image_side_approve"]):
        uncertainties.append("image_resolution_borderline")
    if min(primary.box[2], primary.box[3]) < int(config["min_face_side_approve"]):
        uncertainties.append("face_size_borderline")
    if face_area_ratio < float(config["min_primary_face_area_ratio"]):
        uncertainties.append("face_area_borderline")
    if metrics["blur_laplacian"] < float(config["blur_laplacian_approve"]):
        uncertainties.append("image_blur_borderline")
    if metrics["face_blur_laplacian"] < float(config["face_blur_laplacian_approve"]):
        uncertainties.append("face_blur_borderline")
    if abs(metrics["face_blur_laplacian"] - float(config["face_blur_laplacian_approve"])) <= float(config["borderline_blur_margin"]):
        uncertainties.append("face_blur_near_threshold")
    if metrics["brightness"] < float(config["brightness_min_approve"]) or metrics["brightness"] > float(config["brightness_max_approve"]):
        uncertainties.append("brightness_borderline")
    if metrics["contrast"] < float(config["contrast_min_approve"]):
        uncertainties.append("contrast_borderline")
    if metrics["grid_line_ratio"] >= float(config["collage_grid_line_ratio_quarantine"]):
        uncertainties.append("possible_collage_or_composite")
    if metrics["blockiness"] >= float(config["pixelation_blockiness_quarantine"]):
        uncertainties.append("pixelation_borderline")
    if roll is None and bool(config["require_eye_landmarks_for_approval"]):
        uncertainties.append("eye_landmarks_missing")
    if roll is not None and abs(roll) > float(config["max_roll_degrees_approve"]):
        uncertainties.append("pose_roll_borderline")

    score = quality_score(metrics, face_area_ratio, roll)
    row["quality_score"] = f"{score:.4f}"
    if score < float(config["quality_score_reject"]):
        return reject_row(row, "quality_score_below_reject_threshold", paths["rejected"], input_dir, bool(config["copy_rejected_originals"])), None
    if score < float(config["quality_score_approve"]) + float(config["quality_score_borderline_margin"]):
        uncertainties.append("quality_score_borderline")

    crop, crop_error, roll = aligned_square_crop(image_rgb, primary, float(config["crop_margin"]), int(config["output_size"]))
    if crop_error or crop is None:
        return quarantine_row(row, crop_error or "aligned_crop_failed", paths["quarantine"], input_dir, bool(config["copy_quarantine_originals"])), None

    if uncertainties:
        return quarantine_row(
            row,
            "uncertain:" + ",".join(sorted(set(uncertainties))),
            paths["quarantine"],
            input_dir,
            bool(config["copy_quarantine_originals"]),
        ), None

    candidate_name = f"{index:06d}_{safe_stem(path, input_dir)}.jpg"
    candidate_path = paths["candidates"] / candidate_name
    write_rgb_jpeg(crop, candidate_path, int(config["jpeg_quality"]))
    row["decision"] = "candidate"
    row["reason"] = "passed_local_quality_checks_pending_duplicate_review"
    row["candidate_path"] = str(candidate_path)
    row["decision_path"] = str(candidate_path)
    candidate = Candidate(index, path, candidate_path, score, inferred_identity_key(path))
    return row, candidate


def finalize_candidates(
    rows: list[dict[str, Any]],
    candidates: list[Candidate],
    config: dict[str, Any],
    paths: dict[str, Path],
    input_dir: Path,
) -> None:
    if not candidates:
        return

    embedder = InsightFaceEmbedder(config)
    if embedder.available:
        for cand in tqdm(candidates, desc="Embedding candidates"):
            embedding, status = embedder.embed(cand.candidate_path)
            cand.embedding = embedding
            cand.embedding_status = status
            rows[cand.row_index]["embedding_status"] = status
        cluster_by_embeddings(rows, candidates, config, paths, input_dir)
        return

    for cand in candidates:
        cand.embedding_status = embedder.error or "embedding_backend_unavailable"
        rows[cand.row_index]["embedding_status"] = cand.embedding_status

    if bool(config["require_embeddings_for_person_limit"]):
        for cand in candidates:
            row = rows[cand.row_index]
            row["decision"] = "quarantine"
            row["reason"] = "embedding_required_but_unavailable"
            row["cluster_method"] = "none"
            row["decision_path"] = str(copy_original(cand.source_path, paths["quarantine"], input_dir))
        return

    if bool(config["fallback_group_by_filename_identity"]):
        cluster_by_identity_key(rows, candidates, config, paths, input_dir)
    else:
        for cand in sorted(candidates, key=lambda c: (str(c.source_path).casefold(), c.quality_score)):
            approve_candidate(rows[cand.row_index], cand, "unclustered_embedding_unavailable", "none", paths, config)


def cluster_by_embeddings(
    rows: list[dict[str, Any]],
    candidates: list[Candidate],
    config: dict[str, Any],
    paths: dict[str, Path],
    input_dir: Path,
) -> None:
    valid = [cand for cand in candidates if cand.embedding is not None]
    invalid = [cand for cand in candidates if cand.embedding is None]
    for cand in invalid:
        row = rows[cand.row_index]
        row["decision"] = "quarantine"
        row["reason"] = f"embedding_uncertain:{cand.embedding_status}"
        row["cluster_method"] = "insightface"
        row["decision_path"] = str(copy_original(cand.source_path, paths["quarantine"], input_dir))

    if not valid:
        return

    matrix = np.vstack([cand.embedding for cand in valid])
    labels = DBSCAN(
        eps=float(config["embedding_same_person_distance"]),
        min_samples=1,
        metric="cosine",
        n_jobs=1,
    ).fit_predict(matrix)
    clusters: dict[int, list[Candidate]] = {}
    for cand, label in zip(valid, labels):
        clusters.setdefault(int(label), []).append(cand)

    ambiguous_labels = ambiguous_embedding_clusters(clusters, float(config["embedding_ambiguous_distance"]))
    for label, members in sorted(clusters.items()):
        cluster_id = f"emb_{label:05d}"
        if label in ambiguous_labels:
            for cand in members:
                row = rows[cand.row_index]
                row["decision"] = "quarantine"
                row["reason"] = "same_person_cluster_ambiguous"
                row["cluster_id"] = cluster_id
                row["cluster_method"] = "insightface"
                row["decision_path"] = str(copy_original(cand.source_path, paths["quarantine"], input_dir))
            continue
        finalize_cluster(rows, members, cluster_id, "insightface", config, paths, input_dir)


def ambiguous_embedding_clusters(clusters: dict[int, list[Candidate]], threshold: float) -> set[int]:
    if len(clusters) < 2:
        return set()
    centroids: dict[int, np.ndarray] = {}
    for label, members in clusters.items():
        mat = np.vstack([cand.embedding for cand in members if cand.embedding is not None])
        centroid = np.mean(mat, axis=0)
        centroids[label] = centroid / max(np.linalg.norm(centroid), 1e-8)
    labels = sorted(centroids)
    ambiguous: set[int] = set()
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            dist = 1.0 - float(np.dot(centroids[left], centroids[right]))
            if dist < threshold:
                ambiguous.update({left, right})
    return ambiguous


def cluster_by_identity_key(
    rows: list[dict[str, Any]],
    candidates: list[Candidate],
    config: dict[str, Any],
    paths: dict[str, Path],
    input_dir: Path,
) -> None:
    clusters: dict[str, list[Candidate]] = {}
    for cand in candidates:
        clusters.setdefault(cand.identity_key, []).append(cand)
    for key, members in sorted(clusters.items()):
        cluster_id = "src_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        finalize_cluster(rows, members, cluster_id, "filename_identity_fallback", config, paths, input_dir)


def finalize_cluster(
    rows: list[dict[str, Any]],
    members: list[Candidate],
    cluster_id: str,
    method: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    input_dir: Path,
) -> None:
    ranked = sorted(members, key=lambda c: (-c.quality_score, str(c.source_path).casefold()))
    keep = int(config["max_images_per_person"])
    for pos, cand in enumerate(ranked):
        row = rows[cand.row_index]
        row["cluster_id"] = cluster_id
        row["cluster_method"] = method
        if pos < keep:
            approve_candidate(row, cand, cluster_id, method, paths, config)
        else:
            row["decision"] = "rejected"
            row["reason"] = "duplicate_same_person_lower_quality"
            row["decision_path"] = str(copy_original(cand.source_path, paths["duplicates"], input_dir))


def approve_candidate(
    row: dict[str, Any],
    cand: Candidate,
    cluster_id: str,
    method: str,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> None:
    approved_name = f"{cluster_id}_{cand.candidate_path.name}"
    approved_path = paths["approved"] / approved_name
    approved_path.parent.mkdir(parents=True, exist_ok=True)
    if approved_path.exists():
        approved_path = paths["approved"] / f"{approved_path.stem}_{cand.row_index:06d}{approved_path.suffix}"
    shutil.copy2(cand.candidate_path, approved_path)
    row["decision"] = "approved"
    row["reason"] = "passed_local_quality_checks_and_duplicate_limit"
    row["decision_path"] = str(approved_path)
    row["cluster_id"] = cluster_id
    row["cluster_method"] = method
    row["standardized_size"] = f"{config['output_size']}x{config['output_size']}"
    row["standardized_format"] = "RGB_JPEG"


def write_reports(rows: list[dict[str, Any]], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "preprocess_report.json"
    csv_path = reports_dir / "preprocess_report.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def prepare_paths(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    paths = {
        "output": output_dir,
        "approved": output_dir / str(config["approved_dir"]),
        "rejected": output_dir / str(config["rejected_dir"]),
        "duplicates": output_dir / str(config["rejected_dir"]) / "duplicates",
        "quarantine": output_dir / str(config["quarantine_dir"]),
        "candidates": output_dir / str(config["candidates_dir"]),
        "reports": output_dir / str(config["reports_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision", "unknown"))
        summary[decision] = summary.get(decision, 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate, align, and deduplicate local face images.")
    parser.add_argument("--config", type=Path, default=Path("preprocess_config.json"), help="JSON config path.")
    parser.add_argument("--input-dir", type=Path, help="Override configured input directory.")
    parser.add_argument("--output-dir", type=Path, help="Override configured output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N sorted images for testing.")
    parser.add_argument("--keep-candidates", action="store_true", help="Keep temporary candidate crops after finalization.")
    args = parser.parse_args()

    config = load_config(args.config if args.config.exists() else None)
    input_dir = (args.input_dir or Path(config["input_dir"])).resolve()
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    setup_logging(output_dir)
    paths = prepare_paths(config, output_dir)

    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 2

    np.random.seed(int(config["random_seed"]))
    files = list_image_files(input_dir, config["allowed_extensions"])
    if args.limit:
        files = files[: args.limit]
    logging.info("Found %d images under %s", len(files), input_dir)
    if not files:
        return 0

    detector = FaceDetector(config)
    rows: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    try:
        for index, path in enumerate(tqdm(files, desc="Local quality pass")):
            row, candidate = process_one(path, index, detector, config, paths, input_dir)
            row["index"] = len(rows)
            if candidate is not None:
                candidate.row_index = len(rows)
                candidates.append(candidate)
            rows.append(row)
        finalize_candidates(rows, candidates, config, paths, input_dir)
    finally:
        detector.close()

    json_path, csv_path = write_reports(rows, paths["reports"])
    summary = summarize(rows)
    logging.info("Decision summary: %s", summary)
    logging.info("JSON report: %s", json_path)
    logging.info("CSV report: %s", csv_path)

    if not args.keep_candidates:
        shutil.rmtree(paths["candidates"], ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
