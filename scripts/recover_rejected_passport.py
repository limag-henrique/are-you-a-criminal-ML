#!/usr/bin/env python3
"""
Recover rejected low-quality face images as passport-style white-background JPGs.

This is intentionally more permissive than the main standardization pipeline:
when a face detector fails, it still tries to center the visible subject and
produce a usable white-background output instead of rejecting the image again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    confidence: float
    landmarks: dict[str, tuple[float, float]]
    backend: str

    @property
    def area(self) -> int:
        return max(0, self.box[2]) * max(0, self.box[3])


class TolerantFaceDetector:
    def __init__(self) -> None:
        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
        ]
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        scale = max(1.0, 420.0 / max(1, min(h, w)))
        if scale > 1.0:
            gray_work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            gray_work = gray

        detections: list[FaceDetection] = []
        for cascade in self.face_cascades:
            if cascade.empty():
                continue
            for neighbors in (3, 4, 5):
                faces = cascade.detectMultiScale(
                    gray_work,
                    scaleFactor=1.05,
                    minNeighbors=neighbors,
                    minSize=(24, 24),
                )
                for x, y, fw, fh in faces:
                    box = (
                        int(round(x / scale)),
                        int(round(y / scale)),
                        int(round(fw / scale)),
                        int(round(fh / scale)),
                    )
                    box = clamp_box(box, w, h)
                    landmarks = self._detect_eyes(gray, box)
                    confidence = 0.78 if landmarks else max(0.48, 0.68 - neighbors * 0.03)
                    detections.append(FaceDetection(box, confidence, landmarks, "opencv_haar_tolerant"))
        return non_max_suppression(detections)

    def _detect_eyes(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
        x, y, w, h = box
        upper = gray[y : y + int(h * 0.64), x : x + w]
        if upper.size == 0 or self.eye_cascade.empty():
            return {}
        eyes = self.eye_cascade.detectMultiScale(upper, scaleFactor=1.06, minNeighbors=3, minSize=(8, 8))
        if len(eyes) < 2:
            return {}
        centers = sorted(
            [(x + ex + ew / 2.0, y + ey + eh / 2.0, ew * eh) for ex, ey, ew, eh in eyes],
            key=lambda item: item[2],
            reverse=True,
        )
        best_pair = None
        best_score = -1.0
        for i in range(min(5, len(centers))):
            for j in range(i + 1, min(6, len(centers))):
                p1, p2 = centers[i], centers[j]
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])
                if dx < w * 0.16 or dx > w * 0.82 or dy > h * 0.28:
                    continue
                score = dx - dy * 1.4 + min(p1[2], p2[2]) * 0.001
                if score > best_score:
                    best_score = score
                    best_pair = (p1, p2)
        if best_pair is None:
            return {}
        left, right = sorted(best_pair, key=lambda p: p[0])
        return {"left_eye": (left[0], left[1]), "right_eye": (right[0], right[1])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover rejected low-quality images as passport-style JPGs.")
    parser.add_argument(
        "--input_dir",
        default="data_processed/rejected/Low quality",
        help="Directory containing rejected images.",
    )
    parser.add_argument(
        "--output_dir",
        default="data_processed/passport_recovered/white_bg/Low quality",
        help="Directory for recovered white-background JPGs.",
    )
    parser.add_argument(
        "--report_path",
        default="data_processed/passport_recovered/recovery_report.csv",
        help="CSV report path.",
    )
    parser.add_argument("--size", type=int, default=512, help="Output size in pixels.")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="Output JPEG quality.")
    parser.add_argument(
        "--target_face_ratio",
        type=float,
        default=0.68,
        help="Approximate target face height ratio in the final square image.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).casefold(),
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.asarray(img)


def normalize_passport_crop(
    image_rgb: np.ndarray,
    detector: TolerantFaceDetector,
    target_face_ratio: float,
) -> tuple[np.ndarray, dict[str, object]]:
    faces = sorted(detector.detect(image_rgb), key=lambda f: (f.area, f.confidence), reverse=True)
    h, w = image_rgb.shape[:2]
    method = "face"
    face_confidence = ""
    num_faces = len(faces)

    if faces:
        face = faces[0]
        image_rgb, face = align_by_eyes(image_rgb, face)
        x, y, fw, fh = face.box
        cx = x + fw / 2.0
        if face.landmarks:
            eye_y = float(np.mean([p[1] for p in face.landmarks.values()]))
            desired_eye_y = 0.40
            side = max(fw * 1.62, fh / target_face_ratio)
            cy = eye_y + (0.5 - desired_eye_y) * side
        else:
            side = max(fw * 1.68, fh / target_face_ratio)
            cy = y + fh * 0.52
        face_confidence = f"{face.confidence:.4f}"
        guide_box = face.box
    else:
        method = "subject_fallback"
        guide_box = subject_bbox(image_rgb)
        x, y, fw, fh = guide_box
        cx = x + fw / 2.0
        cy = y + fh * 0.48
        side = max(fw * 1.18, fh * 1.08, min(w, h) * 0.85)

    crop = square_crop_with_padding(image_rgb, cx, cy, int(round(side)))
    info = {
        "method": method,
        "num_faces_detected": num_faces,
        "face_confidence": face_confidence,
        "guide_box": ",".join(str(v) for v in guide_box),
    }
    return crop, info


def align_by_eyes(image_rgb: np.ndarray, face: FaceDetection) -> tuple[np.ndarray, FaceDetection]:
    left = face.landmarks.get("left_eye")
    right = face.landmarks.get("right_eye")
    if not left or not right:
        return image_rgb, face
    left, right = sorted((left, right), key=lambda p: p[0])
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    if abs(angle) < 0.5:
        return image_rgb, face

    h, w = image_rgb.shape[:2]
    x, y, fw, fh = face.box
    center = (x + fw / 2.0, y + fh / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image_rgb, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    corners = np.asarray([transform_point(mat, p) for p in [(x, y), (x + fw, y), (x + fw, y + fh), (x, y + fh)]])
    x1, y1 = np.floor(corners.min(axis=0)).astype(int)
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
    new_box = clamp_box((x1, y1, x2 - x1, y2 - y1), w, h)
    new_landmarks = {name: transform_point(mat, point) for name, point in face.landmarks.items()}
    return aligned, FaceDetection(new_box, face.confidence, new_landmarks, face.backend)


def remove_background_to_white(crop_rgb: np.ndarray, guide_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    original_rgb = crop_rgb
    original_h, original_w = original_rgb.shape[:2]
    original_guide_box = guide_box
    max_seg_side = 256
    scale = min(1.0, max_seg_side / float(max(original_h, original_w)))
    if scale < 1.0:
        crop_rgb = cv2.resize(
            original_rgb,
            (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if guide_box is not None:
            gx, gy, gw, gh = guide_box
            guide_box = (
                int(round(gx * scale)),
                int(round(gy * scale)),
                int(round(gw * scale)),
                int(round(gh * scale)),
            )

    h, w = crop_rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(2, int(min(w, h) * 0.035))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    if guide_box is None:
        rect = (border, border, max(1, w - 2 * border), max(1, h - 2 * border))
        cv2.ellipse(
            mask,
            (w // 2, int(h * 0.52)),
            (max(3, int(w * 0.32)), max(3, int(h * 0.42))),
            0,
            0,
            360,
            cv2.GC_FGD,
            -1,
        )
    else:
        x, y, bw, bh = guide_box
        pad_x = int(bw * 0.52)
        pad_top = int(bh * 0.55)
        pad_bottom = int(bh * 1.25)
        x1 = max(border, x - pad_x)
        y1 = max(border, y - pad_top)
        x2 = min(w - border, x + bw + pad_x)
        y2 = min(h - border, y + bh + pad_bottom)
        rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        cv2.ellipse(
            mask,
            (int(x + bw / 2), int(y + bh * 0.56)),
            (max(3, int(bw * 0.67)), max(3, int(bh * 0.88))),
            0,
            0,
            360,
            cv2.GC_FGD,
            -1,
        )

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(crop_rgb, mask, rect, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return composite_with_white(original_rgb, passport_limit_mask(original_h, original_w, original_guide_box))

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    alpha = keep_center_component(alpha)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    limit = passport_limit_mask(h, w, guide_box)
    alpha = np.minimum(alpha, limit)

    if alpha.mean() < 0.08 or alpha.mean() > 0.86:
        alpha = limit
    if scale < 1.0:
        alpha = cv2.resize(alpha, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0.0, 1.0)
    return composite_with_white(original_rgb, alpha)


def composite_with_white(image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    white = np.full_like(image_rgb, 255)
    out = image_rgb.astype(np.float32) * alpha[:, :, None] + white.astype(np.float32) * (1.0 - alpha[:, :, None])
    out = np.clip(out, 0, 255).round().astype(np.uint8)
    out[alpha <= 0.01] = (255, 255, 255)
    return out


def keep_center_component(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha > 0.5).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return alpha
    h, w = alpha.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    best_label = 1
    best_score = -1.0
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        distance = np.linalg.norm(centroids[label] - center)
        score = float(area) - float(distance) * 5.0
        if score > best_score:
            best_score = score
            best_label = label
    return (labels == best_label).astype(np.float32)


def ellipse_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask,
        (width // 2, int(height * 0.51)),
        (int(width * 0.43), int(height * 0.50)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0, sigmaY=3.0)


def passport_limit_mask(height: int, width: int, guide_box: tuple[int, int, int, int] | None) -> np.ndarray:
    if guide_box is None:
        return ellipse_mask(height, width)

    x, y, w, h = clamp_box(guide_box, width, height)
    mask = np.zeros((height, width), dtype=np.float32)
    center = (int(x + w / 2), int(y + h * 0.62))
    axes = (max(8, int(w * 0.92)), max(8, int(h * 1.22)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    shoulder_center = (int(x + w / 2), min(height - 1, int(y + h * 1.65)))
    shoulder_axes = (max(10, int(w * 1.25)), max(8, int(h * 0.52)))
    cv2.ellipse(mask, shoulder_center, shoulder_axes, 0, 180, 360, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.5, sigmaY=2.5)
    return np.clip(mask, 0.0, 1.0)


def subject_bbox(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    edges = cv2.Canny(l_channel, 40, 110)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    non_bg = (gray < 242).astype(np.uint8) * 255
    combined = cv2.bitwise_or(edges, non_bg)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (int(w * 0.15), int(h * 0.08), int(w * 0.70), int(h * 0.82))
    contour = max(contours, key=cv2.contourArea)
    return clamp_box(cv2.boundingRect(contour), w, h)


def square_crop_with_padding(image_rgb: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    side = max(24, side)
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
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
    return image_rgb[y1:y2, x1:x2]


def safe_name(path: Path, input_dir: Path) -> str:
    try:
        rel = path.relative_to(input_dir)
    except ValueError:
        rel = Path(path.name)
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem).strip("._-")[:100] or "image"
    return f"{stem}_{digest}.jpg"


def save_jpeg(image_rgb: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path, format="JPEG", quality=quality, optimize=True)


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def transform_point(mat: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (
        float(mat[0, 0] * x + mat[0, 1] * y + mat[0, 2]),
        float(mat[1, 0] * x + mat[1, 1] * y + mat[1, 2]),
    )


def non_max_suppression(detections: list[FaceDetection], threshold: float = 0.35) -> list[FaceDetection]:
    kept: list[FaceDetection] = []
    for detection in sorted(detections, key=lambda f: (f.confidence, f.area), reverse=True):
        if all(iou(detection.box, previous.box) < threshold for previous in kept):
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


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    report_path = resolve_path(args.report_path)
    images = list_images(input_dir)
    detector = TolerantFaceDetector()
    rows: list[dict[str, object]] = []
    start = time.perf_counter()

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Total de imagens encontradas: {len(images)}")

    for path in tqdm(images, desc="Recovering passport photos", unit="img"):
        row: dict[str, object] = {
            "original_path": str(path),
            "output_path": "",
            "status": "",
            "reason": "",
            "original_width": "",
            "original_height": "",
            "output_width": args.size,
            "output_height": args.size,
            "method": "",
            "num_faces_detected": "",
            "face_confidence": "",
            "guide_box": "",
        }
        try:
            image = load_rgb(path)
            row["original_height"], row["original_width"] = image.shape[:2]
            crop, info = normalize_passport_crop(image, detector, args.target_face_ratio)
            resized = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4)
            resized_faces = sorted(detector.detect(resized), key=lambda f: (f.area, f.confidence), reverse=True)
            segmentation_guide = resized_faces[0].box if resized_faces else None
            white = remove_background_to_white(resized, segmentation_guide)
            output_path = output_dir / safe_name(path, input_dir)
            save_jpeg(white, output_path, args.jpeg_quality)
            row.update(info)
            row["segmentation_guide_box"] = (
                ",".join(str(v) for v in segmentation_guide) if segmentation_guide else ""
            )
            row["output_path"] = str(output_path)
            row["status"] = "success"
            row["reason"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["reason"] = f"{exc.__class__.__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=5)
        rows.append(row)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - start
    success = sum(row["status"] == "success" for row in rows)
    fallback = sum(row.get("method") == "subject_fallback" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    print()
    print(f"Total processado com sucesso: {success}")
    print(f"Total com fallback sem deteccao de face: {fallback}")
    print(f"Total com erro: {errors}")
    print(f"Relatorio CSV: {report_path}")
    print(f"Tempo total de execucao: {elapsed:.2f}s")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
