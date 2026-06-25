#!/usr/bin/env python3
"""Serve a local browser app for realtime face-profile similarity."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd


HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Similaridade com Galeria</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap");

    :root {
      color-scheme: dark;
      font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101316;
      color: #f6f7f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.18) 0%, rgba(255, 255, 255, 0.02) 38%, rgba(31, 34, 33, 0.24) 100%),
        linear-gradient(160deg, #d5dde1 0%, #8f9b9d 42%, #343b3c 100%);
    }
    main {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .glass {
      width: min(940px, 100%);
      display: grid;
      grid-template-columns: minmax(280px, 1.1fr) minmax(240px, 0.9fr);
      gap: 20px;
      padding: 20px;
      border: 1px solid rgba(255, 255, 255, 0.26);
      border-radius: 28px;
      background: rgba(255, 255, 255, 0.16);
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.42);
      backdrop-filter: blur(28px) saturate(145%);
      -webkit-backdrop-filter: blur(28px) saturate(145%);
    }
    .stage {
      position: relative;
      overflow: hidden;
      min-height: 420px;
      border-radius: 22px;
      background: #fff;
      border: 1px solid rgba(255, 255, 255, 0.16);
    }
    video, canvas {
      width: 100%;
      height: 100%;
      min-height: 420px;
      display: block;
      object-fit: cover;
    }
    video {
      position: absolute;
      width: 1px;
      height: 1px;
      min-height: 1px;
      opacity: 0;
      pointer-events: none;
    }
    .preview {
      background: #fff;
    }
    .empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 28px;
      text-align: center;
      color: rgba(34, 42, 46, 0.56);
      font-size: 17px;
      line-height: 1.45;
      pointer-events: none;
    }
    .panel {
      display: flex;
      min-width: 0;
      flex-direction: column;
      justify-content: space-between;
      gap: 24px;
      padding: 8px 6px 8px 2px;
    }
    h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 58px);
      line-height: 0.95;
      letter-spacing: 0;
      font-weight: 700;
    }
    .sub {
      margin: 14px 0 0;
      color: rgba(255, 255, 255, 0.74);
      font-size: 17px;
      line-height: 1.4;
    }
    .score {
      display: grid;
      gap: 10px;
    }
    .number {
      font-size: clamp(64px, 12vw, 112px);
      line-height: 0.9;
      letter-spacing: 0;
      font-weight: 750;
      font-variant-numeric: tabular-nums;
    }
    .label {
      min-height: 24px;
      color: rgba(255, 255, 255, 0.74);
      font-size: 15px;
    }
    .bar {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.22);
    }
    .fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #72d6a3, #e7d274, #ef8c78);
      transition: width 260ms ease;
    }
    .actions {
      display: grid;
      gap: 10px;
    }
    .actions-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button,
    .upload-control {
      width: 100%;
      min-height: 52px;
      border: 1px solid rgba(255, 255, 255, 0.34);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.22);
      color: #fff;
      font: inherit;
      font-size: 16px;
      font-weight: 650;
      cursor: pointer;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.28), 0 12px 28px rgba(0, 0, 0, 0.22);
      transition: transform 160ms ease, background 160ms ease;
      display: grid;
      position: relative;
      place-items: center;
      padding: 0 18px;
      text-align: center;
    }
    button:hover, .upload-control:hover { background: rgba(255, 255, 255, 0.28); }
    button:active, .upload-control:active { transform: scale(0.985); }
    button:disabled {
      cursor: default;
      opacity: 0.56;
      transform: none;
    }
    .upload-control input {
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }
    .capture { display: none; }
    @media (max-width: 760px) {
      body { overflow: auto; }
      main { align-items: start; }
      .glass {
        grid-template-columns: 1fr;
        border-radius: 24px;
      }
      .stage, video { min-height: 360px; }
      .panel { padding: 0 2px 4px; }
      .actions-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="glass" aria-label="Similaridade facial">
      <div class="stage">
        <video id="video" playsinline muted></video>
        <canvas id="preview" class="preview"></canvas>
        <div id="empty" class="empty">Ligue a camera para medir a similaridade.</div>
      </div>
      <div class="panel">
        <div>
          <h1>Similaridade</h1>
          <p class="sub">Comparacao local com a galeria de procurados.</p>
        </div>
        <div class="score" aria-live="polite">
          <div id="number" class="number">--%</div>
          <div class="bar"><div id="fill" class="fill"></div></div>
          <div id="label" class="label">Aguardando camera.</div>
        </div>
        <div class="actions">
          <div class="actions-row">
            <button id="start">Ligar camera</button>
            <button id="stop" disabled>Desligar camera</button>
          </div>
          <label class="upload-control" for="upload">
            Enviar foto
            <input id="upload" type="file" accept="image/*">
          </label>
        </div>
      </div>
    </section>
  </main>
  <canvas id="canvas" class="capture"></canvas>
  <script>
    const video = document.getElementById("video");
    const preview = document.getElementById("preview");
    const canvas = document.getElementById("canvas");
    const start = document.getElementById("start");
    const stop = document.getElementById("stop");
    const upload = document.getElementById("upload");
    const number = document.getElementById("number");
    const fill = document.getElementById("fill");
    const label = document.getElementById("label");
    const empty = document.getElementById("empty");

    let running = false;
    let busy = false;
    let intervalId = null;
    let previewFrameId = null;
    let lastFace = null;

    function setScore(value, text) {
      if (Number.isFinite(value)) {
        const pct = Math.max(0, Math.min(100, value));
        number.textContent = `${pct.toFixed(1)}%`;
        fill.style.width = `${pct}%`;
      }
      label.textContent = text;
    }

    function resetScore(text) {
      number.textContent = "--%";
      fill.style.width = "0%";
      label.textContent = text;
    }

    function syncCanvasSize(target, width, height) {
      const nextWidth = Math.max(1, Math.round(width));
      const nextHeight = Math.max(1, Math.round(height));
      if (target.width !== nextWidth || target.height !== nextHeight) {
        target.width = nextWidth;
        target.height = nextHeight;
      }
    }

    function updateFace(data, sourceWidth, sourceHeight) {
      if (!data.face_box || data.face_box.length !== 4) return;
      lastFace = {
        box: data.face_box.map(Number),
        width: Number(data.frame_width || sourceWidth),
        height: Number(data.frame_height || sourceHeight),
        seenAt: performance.now()
      };
    }

    function drawWhitePreview() {
      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      syncCanvasSize(preview, width, height);
      const ctx = preview.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, preview.width, preview.height);
    }

    function drawFilteredPreview() {
      if (!running || video.readyState < 2) {
        drawWhitePreview();
        if (running) previewFrameId = requestAnimationFrame(drawFilteredPreview);
        return;
      }

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      syncCanvasSize(preview, width, height);
      const ctx = preview.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, preview.width, preview.height);

      const now = performance.now();
      const faceFresh = lastFace && now - lastFace.seenAt < 1500;
      if (faceFresh) {
        const sourceScaleX = width / Math.max(1, lastFace.width);
        const sourceScaleY = height / Math.max(1, lastFace.height);
        const [x1, y1, x2, y2] = lastFace.box;
        const faceWidth = Math.max(1, (x2 - x1) * sourceScaleX);
        const faceHeight = Math.max(1, (y2 - y1) * sourceScaleY);
        const cropX = Math.max(0, x1 * sourceScaleX);
        const cropY = Math.max(0, y1 * sourceScaleY);
        const cropRight = Math.min(width, x2 * sourceScaleX);
        const cropBottom = Math.min(height, y2 * sourceScaleY + faceHeight * 0.12);
        const cropWidth = Math.max(1, cropRight - cropX);
        const cropHeight = Math.max(1, cropBottom - cropY);
        const drawScale = Math.min(preview.width * 0.82 / cropWidth, preview.height * 0.9 / cropHeight);
        const drawWidth = cropWidth * drawScale;
        const drawHeight = cropHeight * drawScale;
        const drawX = (preview.width - drawWidth) / 2;
        const drawY = (preview.height - drawHeight) / 2;

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.save();
        ctx.translate(preview.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, cropX, cropY, cropWidth, cropHeight, preview.width - drawX - drawWidth, drawY, drawWidth, drawHeight);
        ctx.restore();
      }

      if (running) previewFrameId = requestAnimationFrame(drawFilteredPreview);
    }

    async function scoreBlob(blob, mirrorPreview) {
      if (busy) return;
      busy = true;
      try {
        const res = await fetch("/api/score", {
          method: "POST",
          headers: {"Content-Type": blob.type || "image/jpeg"},
          body: blob
        });
        const data = await res.json();
        if (data.ok) {
          updateFace(data, canvas.width, canvas.height);
          if (!mirrorPreview) drawServerPreview(data.preview_jpeg, false);
          setScore(data.similarity_percent, data.accepted ? "Semelhanca visual forte na galeria." : "Sem correspondencia forte na galeria.");
        } else {
          drawWhitePreview();
          setScore(NaN, data.error || "Nenhum rosto detectado.");
        }
      } catch (err) {
        setScore(NaN, "Nao foi possivel calcular agora.");
      } finally {
        busy = false;
      }
    }

    async function scoreFrame() {
      if (!running || busy || video.readyState < 2) return;
      try {
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        const maxSide = 480;
        const scale = Math.min(1, maxSide / Math.max(width, height));
        canvas.width = Math.round(width * scale);
        canvas.height = Math.round(height * scale);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.72));
        await scoreBlob(blob, true);
      } catch (err) {
        setScore(NaN, "Nao foi possivel calcular agora.");
      }
    }

    start.addEventListener("click", async () => {
      start.disabled = true;
      label.textContent = "Abrindo camera...";
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: {ideal: 640}, height: {ideal: 480} },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
        empty.style.display = "none";
        running = true;
        stop.disabled = false;
        start.textContent = "Camera ligada";
        drawWhitePreview();
        setScore(NaN, "Procurando rosto.");
        if (intervalId) clearInterval(intervalId);
        if (previewFrameId) cancelAnimationFrame(previewFrameId);
        drawFilteredPreview();
        scoreFrame();
        intervalId = setInterval(scoreFrame, 650);
      } catch (err) {
        start.disabled = false;
        label.textContent = "Permissao de camera negada.";
      }
    });

    stop.addEventListener("click", () => {
      stopCamera();
    });

    upload.addEventListener("change", async () => {
      const file = upload.files && upload.files[0];
      if (!file) return;
      label.textContent = "Analisando foto...";
      empty.style.display = "none";
      await scoreBlob(file, false);
      upload.value = "";
    });

    function stopCamera() {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
      if (previewFrameId) {
        cancelAnimationFrame(previewFrameId);
        previewFrameId = null;
      }
      running = false;
      busy = false;
      lastFace = null;
      const stream = video.srcObject;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      video.srcObject = null;
      start.disabled = false;
      stop.disabled = true;
      start.textContent = "Ligar camera";
      empty.style.display = "grid";
      drawWhitePreview();
      resetScore("Camera desligada.");
    }

    drawWhitePreview();

    function drawServerPreview(previewJpeg, mirrorPreview) {
      if (!previewJpeg) {
        drawWhitePreview();
        return;
      }
      const img = new Image();
      img.onload = () => {
        const targetWidth = canvas.width || video.videoWidth || img.naturalWidth || img.width;
        const targetHeight = canvas.height || video.videoHeight || img.naturalHeight || img.height;
        syncCanvasSize(preview, targetWidth, targetHeight);
        const ctx = preview.getContext("2d");
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, preview.width, preview.height);
        const imageWidth = img.naturalWidth || img.width;
        const imageHeight = img.naturalHeight || img.height;
        const scale = Math.min(preview.width / imageWidth, preview.height / imageHeight) * 0.98;
        const drawWidth = imageWidth * scale;
        const drawHeight = imageHeight * scale;
        const drawX = (preview.width - drawWidth) / 2;
        const drawY = (preview.height - drawHeight) / 2;
        if (mirrorPreview) {
          ctx.save();
          ctx.translate(preview.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(img, preview.width - drawX - drawWidth, drawY, drawWidth, drawHeight);
          ctx.restore();
        } else {
          ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
        }
      };
      img.src = `data:image/jpeg;base64,${previewJpeg}`;
    }
  </script>
</body>
</html>
"""


class GallerySimilarityScorer:
    def __init__(
        self,
        features_path: Path,
        embeddings_path: Path,
        model_name: str,
        ctx_id: int,
        det_size: int,
        gallery_splits: tuple[str, ...] | None = None,
        calibration_sample: int = 1800,
    ) -> None:
        self.features_path = features_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.gallery_splits = gallery_splits
        self._embedder = None
        self._lock = threading.Lock()

        table = pd.read_csv(features_path)
        embeddings = np.load(embeddings_path)
        valid = table["embedding_index"].astype(int) >= 0
        if gallery_splits:
            valid &= table["split"].astype(str).str.lower().isin(gallery_splits)
        gallery = table.loc[valid].copy()
        if gallery.empty:
            raise ValueError("Nenhum embedding valido encontrado para a galeria.")

        indices = gallery["embedding_index"].astype(int).to_numpy()
        gallery_embeddings = np.asarray(embeddings[indices], dtype=np.float32)
        gallery_embeddings /= np.maximum(np.linalg.norm(gallery_embeddings, axis=1, keepdims=True), 1e-12)

        self.gallery = gallery.reset_index(drop=True)
        self.gallery_embeddings = gallery_embeddings
        self.subject_ids = self.gallery["subject_id"].astype(str).to_numpy()
        self.gallery_count = int(gallery_embeddings.shape[0])
        self.calibration_quantiles = self._calibrate_impostor_distribution(max(64, calibration_sample))
        self.threshold_percent = 85.0

    def score_jpeg(self, payload: bytes) -> dict[str, object]:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"ok": False, "error": "Imagem invalida."}
        try:
            with self._lock:
                result = self._get_embedder().extract_bgr(image)
                score = self.score_embedding(result.embedding)
                filtered_preview = white_face_filter_bgr(image, result.bbox)
            return {
                "ok": True,
                "score_raw": score["best_cosine"],
                "score": score["best_cosine"],
                "similarity_percent": score["similarity_percent"],
                "percentile_rank": score["percentile_rank"],
                "threshold_percent": self.threshold_percent,
                "accepted": bool(score["similarity_percent"] >= self.threshold_percent),
                "det_score": float(result.det_score),
                "source_det_score": float(result.det_score),
                "score_filter_applied": False,
                "face_box": [float(value) for value in result.bbox],
                "frame_width": int(image.shape[1]),
                "frame_height": int(image.shape[0]),
                "preview_jpeg": encode_preview_jpeg(filtered_preview),
                "gallery_count": self.gallery_count,
                "nearest": score["nearest"],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def score_embedding(self, embedding: np.ndarray) -> dict[str, object]:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        similarities = self.gallery_embeddings @ query
        best_index = int(np.argmax(similarities))
        best = float(similarities[best_index])
        row = self.gallery.iloc[best_index]
        percentile_rank = float(np.mean(self.impostor_max_scores <= best) * 100.0)
        return {
            "best_cosine": best,
            "similarity_percent": self._display_percent(best),
            "percentile_rank": percentile_rank,
            "nearest": {
                "path": str(row.get("path", "")),
                "subject_id": str(row.get("subject_id", "")),
                "quality": str(row.get("quality", "")),
                "split": str(row.get("split", "")),
                "cosine": best,
            },
        }

    def _calibrate_impostor_distribution(self, calibration_sample: int) -> dict[str, float]:
        rng = np.random.default_rng(42)
        n = self.gallery_embeddings.shape[0]
        sample_size = min(n, calibration_sample)
        sample_indices = rng.choice(n, size=sample_size, replace=False)
        scores: list[np.ndarray] = []
        for start in range(0, sample_size, 128):
            batch_indices = sample_indices[start : start + 128]
            sims = self.gallery_embeddings[batch_indices] @ self.gallery_embeddings.T
            same_subject = self.subject_ids[batch_indices, None] == self.subject_ids[None, :]
            sims[same_subject] = -np.inf
            scores.append(np.max(sims, axis=1).astype(np.float32))

        self.impostor_max_scores = np.concatenate(scores)
        calibration_scores = self.impostor_max_scores[self.impostor_max_scores < 0.98]
        if calibration_scores.size < max(32, int(self.impostor_max_scores.size * 0.75)):
            calibration_scores = self.impostor_max_scores
        quantiles = np.quantile(calibration_scores, [0.50, 0.95, 0.99, 0.999])
        q50, q95, q99, q999 = [float(value) for value in quantiles]
        if q99 <= q95:
            q99 = q95 + 1e-4
        if q95 <= q50:
            q95 = q50 + 1e-4
        if q999 <= q99:
            q999 = q99 + 1e-4
        return {"q50": q50, "q95": q95, "q99": q99, "q999": q999}

    def _display_percent(self, cosine: float) -> float:
        q = self.calibration_quantiles
        if cosine <= q["q50"]:
            value = 20.0 * max(0.0, cosine / max(q["q50"], 1e-6))
        elif cosine <= q["q95"]:
            value = 20.0 + 35.0 * (cosine - q["q50"]) / (q["q95"] - q["q50"])
        elif cosine <= q["q99"]:
            value = 55.0 + 30.0 * (cosine - q["q95"]) / (q["q99"] - q["q95"])
        else:
            value = 85.0 + 15.0 * (cosine - q["q99"]) / max(q["q999"] - q["q99"], 1e-6)
        return max(0.0, min(100.0, float(value)))

    def _get_embedder(self):
        if self._embedder is None:
            from face_profile_ml.extractor import ArcFaceEmbedder

            self._embedder = ArcFaceEmbedder(
                model_name=self.model_name,
                ctx_id=self.ctx_id,
                det_size=self.det_size,
            )
        return self._embedder


class SimilarityScorer:
    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        ctx_id: int,
        det_size: int,
        score_filter: bool = False,
    ) -> None:
        sys.path.insert(0, str(Path.cwd()))
        from face_profile_ml.calibration import ScoreCalibrator
        from face_profile_ml.profile import FaceProfileModel

        self.model_dir = model_dir
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.score_filter = score_filter
        self.model = FaceProfileModel.load(model_dir)
        self.calibrator = None
        if (model_dir / "calibrator.pkl").exists():
            self.calibrator = ScoreCalibrator.load(model_dir)
        self.threshold = float(self.calibrator.threshold) if self.calibrator else 0.5
        self.display_scale = self._load_display_scale()
        self._embedder = None
        self._lock = threading.Lock()

    def score_jpeg(self, payload: bytes) -> dict[str, object]:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"ok": False, "error": "Imagem invalida."}
        try:
            with self._lock:
                embedder = self._get_embedder()
                source_result = embedder.extract_bgr(image)
                filtered_preview = white_face_filter_bgr(image, source_result.bbox)
                score_result = source_result
                score_filter_applied = False
                if self.score_filter:
                    try:
                        score_result = embedder.extract_bgr(filtered_preview)
                        score_filter_applied = True
                    except Exception:
                        score_result = source_result
                raw = float(self.model.score(score_result.embedding)["score_raw"].iloc[0])
                calibrated_score = (
                    float(self.calibrator.predict_proba(np.asarray([raw], dtype=np.float32))[0])
                    if self.calibrator
                    else raw
                )
                similarity_percent = self._similarity_percent(raw, calibrated_score)
                accepted = self._accepted(calibrated_score, similarity_percent)
            return {
                "ok": True,
                "score_raw": raw,
                "score": calibrated_score,
                "calibrated_similarity_percent": max(0.0, min(100.0, calibrated_score * 100.0)),
                "similarity_percent": similarity_percent,
                "threshold": self.threshold,
                "threshold_percent": self._threshold_percent(),
                "accepted": accepted,
                "det_score": float(score_result.det_score),
                "source_det_score": float(source_result.det_score),
                "score_filter_applied": score_filter_applied,
                "face_box": [float(value) for value in source_result.bbox],
                "frame_width": int(image.shape[1]),
                "frame_height": int(image.shape[0]),
                "preview_jpeg": encode_preview_jpeg(filtered_preview),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _get_embedder(self):
        if self._embedder is None:
            from face_profile_ml.extractor import ArcFaceEmbedder

            self._embedder = ArcFaceEmbedder(
                model_name=self.model_name,
                ctx_id=self.ctx_id,
                det_size=self.det_size,
            )
        return self._embedder

    def _load_display_scale(self) -> tuple[float, float] | None:
        calibration_path = self.model_dir / "calibration_scores.csv"
        if not calibration_path.exists():
            return None

        positives: list[float] = []
        negatives: list[float] = []
        try:
            with calibration_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_text = row.get("score_raw", "")
                    label_text = row.get("label", "")
                    if not raw_text:
                        continue
                    raw = float(raw_text)
                    if label_text == "1":
                        positives.append(raw)
                    elif label_text == "0":
                        negatives.append(raw)
        except Exception:
            return None

        if not positives or not negatives:
            return None
        low = float(np.quantile(np.asarray(negatives, dtype=np.float32), 0.99))
        high = float(np.quantile(np.asarray(positives, dtype=np.float32), 0.50))
        if high <= low:
            return None
        return low, high

    def _similarity_percent(self, raw_score: float, calibrated_score: float) -> float:
        if self.display_scale is None:
            return max(0.0, min(100.0, calibrated_score * 100.0))
        low, high = self.display_scale
        value = (raw_score - low) / (high - low)
        return max(0.0, min(100.0, value * 100.0))

    def _threshold_percent(self) -> float:
        if self.display_scale is None:
            return max(0.0, min(100.0, self.threshold * 100.0))
        return 85.0

    def _accepted(self, calibrated_score: float, similarity_percent: float) -> bool:
        if self.display_scale is None:
            return bool(calibrated_score >= self.threshold)
        return bool(similarity_percent >= self._threshold_percent())


def person_white_filter_bgr(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    mediapipe_mask = _mediapipe_person_mask(image_bgr)
    if mediapipe_mask is not None:
        return compose_white_background(image_bgr, mediapipe_mask)

    grabcut_mask = _grabcut_person_mask(image_bgr, bbox)
    if grabcut_mask is not None:
        return compose_white_background(image_bgr, grabcut_mask)

    return white_face_filter_bgr(image_bgr, bbox)


def _mediapipe_person_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
            result = segmenter.process(image_rgb)
    except Exception:
        return None

    mask = getattr(result, "segmentation_mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape[:2] != image_bgr.shape[:2] or float(mask.max()) < 0.05:
        return None
    mask = (mask >= 0.22).astype(np.float32)
    return smooth_binary_mask(mask)


def _grabcut_person_mask(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    if height < 32 or width < 32:
        return None

    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    rect_x1 = int(round(center_x - face_width * 1.35))
    rect_y1 = int(round(center_y - face_height * 0.95))
    rect_x2 = int(round(center_x + face_width * 1.35))
    rect_y2 = int(round(center_y + face_height * 1.95))
    rect_x1 = max(1, min(width - 3, rect_x1))
    rect_y1 = max(1, min(height - 3, rect_y1))
    rect_x2 = max(rect_x1 + 2, min(width - 2, rect_x2))
    rect_y2 = max(rect_y1 + 2, min(height - 2, rect_y2))

    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    mask[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.GC_PR_BGD

    fg_center = (int(round(center_x)), int(round(center_y + face_height * 0.42)))
    fg_axes = (int(round(face_width * 0.88)), int(round(face_height * 1.55)))
    cv2.ellipse(mask, fg_center, fg_axes, 0, 0, 360, cv2.GC_PR_FGD, thickness=-1)

    definite_face_x1 = max(0, int(round(x1 + face_width * 0.10)))
    definite_face_y1 = max(0, int(round(y1 + face_height * 0.08)))
    definite_face_x2 = min(width, int(round(x2 - face_width * 0.10)))
    definite_face_y2 = min(height, int(round(y2 + face_height * 0.40)))
    if definite_face_x2 > definite_face_x1 and definite_face_y2 > definite_face_y1:
        mask[definite_face_y1:definite_face_y2, definite_face_x1:definite_face_x2] = cv2.GC_FGD

    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image_bgr, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    if float(binary.mean()) < 0.01:
        return None
    return keep_component_near_face(smooth_binary_mask(binary), bbox)


def smooth_binary_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.1).astype(np.float32)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.GaussianBlur(binary, (0, 0), sigmaX=3.0, sigmaY=3.0)
    return np.clip(binary, 0.0, 1.0)


def keep_component_near_face(mask: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    binary = (mask > 0.15).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    height, width = mask.shape[:2]
    face_roi = np.zeros_like(binary)
    face_roi[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = 1

    keep = np.zeros_like(binary)
    best_label = 0
    best_score = -1
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        overlap = int(((labels == label_id) & (face_roi == 1)).sum())
        score = overlap * 100000 + area
        if score > best_score:
            best_score = score
            best_label = label_id
    if best_label > 0:
        keep[labels == best_label] = 1
        keep = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
        return np.clip(keep, 0.0, 1.0)
    return mask


def compose_white_background(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[:, :, None]
    white = np.full_like(image_bgr, 255, dtype=np.uint8)
    filtered = image_bgr.astype(np.float32) * alpha + white.astype(np.float32) * (1.0 - alpha)
    return np.clip(filtered, 0, 255).round().astype(np.uint8)


def encode_preview_jpeg(image_bgr: np.ndarray, max_side: int = 720) -> str:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width, 1))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def white_face_filter_bgr(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Return a rectangular face crop on white, without oval masking."""
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return image_bgr

    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(np.floor(center_x - face_width * 0.72))
    crop_y1 = int(np.floor(center_y - face_height * 0.72))
    crop_x2 = int(np.ceil(center_x + face_width * 0.72))
    crop_y2 = int(np.ceil(center_y + face_height * 0.86))
    crop_width = max(32, crop_x2 - crop_x1)
    crop_height = max(32, crop_y2 - crop_y1)

    crop = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
    src_x1 = max(0, int(np.floor(x1)))
    src_y1 = max(0, int(np.floor(y1)))
    src_x2 = min(width, int(np.ceil(x2)))
    src_y2 = min(height, int(np.ceil(y2 + face_height * 0.12)))
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return image_bgr

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    crop[dst_y1 : dst_y1 + (src_y2 - src_y1), dst_x1 : dst_x1 + (src_x2 - src_x1)] = image_bgr[
        src_y1:src_y2, src_x1:src_x2
    ]
    return crop


class AppHandler(BaseHTTPRequestHandler):
    scorer: object

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._send_json(
                {
                    "ok": True,
                    "mode": self.scorer.__class__.__name__,
                    "model_dir": str(getattr(self.scorer, "model_dir", "")),
                    "features_path": str(getattr(self.scorer, "features_path", "")),
                    "embeddings_path": str(getattr(self.scorer, "embeddings_path", "")),
                    "threshold": getattr(self.scorer, "threshold", None),
                    "threshold_percent": getattr(self.scorer, "threshold_percent", None),
                    "model_name": getattr(self.scorer, "model_name", ""),
                    "det_size": getattr(self.scorer, "det_size", None),
                    "score_filter": getattr(self.scorer, "score_filter", False),
                    "gallery_count": getattr(self.scorer, "gallery_count", None),
                    "calibration_quantiles": getattr(self.scorer, "calibration_quantiles", None),
                }
            )
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/score":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json({"ok": False, "error": "Frame vazio."}, status=400)
            return
        payload = self.rfile.read(length)
        self._send_json(self.scorer.score_jpeg(payload))

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local face similarity browser app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--features", default="artifacts/embedding_manifest.csv")
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument(
        "--gallery-splits",
        default="all",
        help="Comma-separated splits used as the public gallery, or 'all' for every valid embedding.",
    )
    parser.add_argument(
        "--legacy-profile-mode",
        action="store_true",
        help="Use the old aggregate profile model instead of gallery-nearest similarity.",
    )
    parser.add_argument("--model-name", default="buffalo_s")
    parser.add_argument("--ctx-id", type=int, default=-1)
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument(
        "--enable-score-filter",
        dest="score_filter",
        action="store_true",
        help="Score the white-background face crop instead of the original detected face.",
    )
    parser.add_argument(
        "--disable-score-filter",
        dest="score_filter",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(score_filter=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.legacy_profile_mode:
        model_dir = Path(args.model_dir)
        if not (model_dir / "profile_model.pkl").exists():
            raise FileNotFoundError(f"Modelo nao encontrado em {model_dir}")
        AppHandler.scorer = SimilarityScorer(
            model_dir,
            args.model_name,
            args.ctx_id,
            args.det_size,
            score_filter=args.score_filter,
        )
    else:
        gallery_splits = None
        if str(args.gallery_splits).strip().lower() != "all":
            gallery_splits = tuple(item.strip().lower() for item in args.gallery_splits.split(",") if item.strip())
        AppHandler.scorer = GallerySimilarityScorer(
            Path(args.features),
            Path(args.embeddings),
            args.model_name,
            args.ctx_id,
            args.det_size,
            gallery_splits=gallery_splits,
        )
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"App local em http://{args.host}:{args.port}")
    print("Abra no navegador e permita o uso da camera.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
