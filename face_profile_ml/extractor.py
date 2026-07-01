from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps

from .utils import l2_normalize


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: np.ndarray
    bbox: tuple[float, float, float, float]
    det_score: float
    aligned_bgr: np.ndarray | None = None
    face_count: int = 1
    image_shape: tuple[int, int] | None = None

    @property
    def face_area_ratio(self) -> float | None:
        if self.image_shape is None:
            return None
        height, width = self.image_shape
        image_area = max(1.0, float(height * width))
        return _bbox_area(np.asarray(self.bbox, dtype=np.float32)) / image_area


class ArcFaceEmbedder:
    """InsightFace ArcFace extractor.

    The pretrained neural model is used only for inference. No fine-tuning or
    gradient updates are performed anywhere in this project.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
            from insightface.utils import face_align  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "InsightFace is required for detection/alignment/ArcFace embeddings. "
                "Install with: pip install insightface onnxruntime"
            ) from exc

        self._face_align = face_align
        kwargs: dict[str, Any] = {
            "name": model_name,
            "allowed_modules": ["detection", "recognition"],
            "providers": select_available_providers(providers),
        }
        self.app = FaceAnalysis(**kwargs)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def extract_path(self, image_path: str | Path, return_aligned: bool = False) -> FaceEmbedding:
        image = read_bgr_image(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.extract_bgr(image, return_aligned=return_aligned)

    def extract_bgr(self, image_bgr: np.ndarray, return_aligned: bool = False) -> FaceEmbedding:
        faces = self.app.get(image_bgr)
        if not faces:
            raise ValueError("No face detected")
        face = max(faces, key=lambda item: _bbox_area(item.bbox))
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise ValueError("Face detected but embedding was not produced")

        aligned = None
        if return_aligned:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                aligned = self._face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)

        bbox = tuple(float(v) for v in face.bbox)
        score = float(getattr(face, "det_score", 0.0))
        return FaceEmbedding(
            l2_normalize(np.asarray(embedding, dtype=np.float32)),
            bbox,
            score,
            aligned,
            len(faces),
            image_bgr.shape[:2],
        )


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_available_providers(requested: list[str] | None = None) -> list[str]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return requested or ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    preferred = requested or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    selected = [provider for provider in preferred if provider in available]
    if selected:
        return selected
    if "CPUExecutionProvider" in available:
        if requested:
            warnings.warn(
                "None of the requested ONNX Runtime providers are available; using CPUExecutionProvider.",
                RuntimeWarning,
                stacklevel=2,
            )
        return ["CPUExecutionProvider"]
    return list(available)


def read_bgr_image(image_path: str | Path) -> np.ndarray | None:
    """Read an image from disk while preserving Windows Unicode paths and EXIF orientation."""
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
