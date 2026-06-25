#!/usr/bin/env python3
"""Serve a local browser app for realtime face-profile similarity."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Similaridade Facial</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
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
      background: rgba(11, 15, 18, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.16);
    }
    video {
      width: 100%;
      height: 100%;
      min-height: 420px;
      display: block;
      object-fit: cover;
      transform: scaleX(-1);
    }
    .empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 28px;
      text-align: center;
      color: rgba(255, 255, 255, 0.72);
      font-size: 17px;
      line-height: 1.45;
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
    button {
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
    }
    button:hover { background: rgba(255, 255, 255, 0.28); }
    button:active { transform: scale(0.985); }
    button:disabled {
      cursor: default;
      opacity: 0.56;
      transform: none;
    }
    canvas { display: none; }
    @media (max-width: 760px) {
      body { overflow: auto; }
      main { align-items: start; }
      .glass {
        grid-template-columns: 1fr;
        border-radius: 24px;
      }
      .stage, video { min-height: 360px; }
      .panel { padding: 0 2px 4px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="glass" aria-label="Similaridade facial">
      <div class="stage">
        <video id="video" playsinline muted></video>
        <div id="empty" class="empty">Ligue a camera para medir a similaridade.</div>
      </div>
      <div class="panel">
        <div>
          <h1>Similaridade</h1>
          <p class="sub">Comparacao local com o perfil treinado.</p>
        </div>
        <div class="score" aria-live="polite">
          <div id="number" class="number">--%</div>
          <div class="bar"><div id="fill" class="fill"></div></div>
          <div id="label" class="label">Aguardando camera.</div>
        </div>
        <button id="start">Ligar camera</button>
      </div>
    </section>
  </main>
  <canvas id="canvas"></canvas>
  <script>
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const start = document.getElementById("start");
    const number = document.getElementById("number");
    const fill = document.getElementById("fill");
    const label = document.getElementById("label");
    const empty = document.getElementById("empty");

    let running = false;
    let busy = false;

    function setScore(value, text) {
      if (Number.isFinite(value)) {
        const pct = Math.max(0, Math.min(100, value));
        number.textContent = `${pct.toFixed(1)}%`;
        fill.style.width = `${pct}%`;
      }
      label.textContent = text;
    }

    async function scoreFrame() {
      if (!running || busy || video.readyState < 2) return;
      busy = true;
      try {
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        const maxSide = 720;
        const scale = Math.min(1, maxSide / Math.max(width, height));
        canvas.width = Math.round(width * scale);
        canvas.height = Math.round(height * scale);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.84));
        const res = await fetch("/api/score", {
          method: "POST",
          headers: {"Content-Type": "image/jpeg"},
          body: blob
        });
        const data = await res.json();
        if (data.ok) {
          setScore(data.similarity_percent, data.accepted ? "Acima do limiar calibrado." : "Abaixo do limiar calibrado.");
        } else {
          setScore(NaN, data.error || "Nenhum rosto detectado.");
        }
      } catch (err) {
        setScore(NaN, "Nao foi possivel calcular agora.");
      } finally {
        busy = false;
      }
    }

    start.addEventListener("click", async () => {
      start.disabled = true;
      label.textContent = "Abrindo camera...";
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: {ideal: 1280}, height: {ideal: 720} },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
        empty.style.display = "none";
        running = true;
        start.textContent = "Camera ligada";
        setScore(NaN, "Procurando rosto.");
        setInterval(scoreFrame, 900);
      } catch (err) {
        start.disabled = false;
        label.textContent = "Permissao de camera negada.";
      }
    });
  </script>
</body>
</html>
"""


class SimilarityScorer:
    def __init__(self, model_dir: Path, model_name: str, ctx_id: int, det_size: int) -> None:
        sys.path.insert(0, str(Path.cwd()))
        from face_profile_ml.calibration import ScoreCalibrator
        from face_profile_ml.profile import FaceProfileModel

        self.model_dir = model_dir
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.model = FaceProfileModel.load(model_dir)
        self.calibrator = None
        if (model_dir / "calibrator.pkl").exists():
            self.calibrator = ScoreCalibrator.load(model_dir)
        self.threshold = float(self.calibrator.threshold) if self.calibrator else 0.5
        self._embedder = None
        self._lock = threading.Lock()

    def score_jpeg(self, payload: bytes) -> dict[str, object]:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"ok": False, "error": "Imagem invalida."}
        try:
            with self._lock:
                result = self._get_embedder().extract_bgr(image)
                raw = float(self.model.score(result.embedding)["score_raw"].iloc[0])
                if self.calibrator:
                    score = float(self.calibrator.predict_proba(np.asarray([raw], dtype=np.float32))[0])
                else:
                    score = raw
            return {
                "ok": True,
                "score_raw": raw,
                "score": score,
                "similarity_percent": max(0.0, min(100.0, score * 100.0)),
                "threshold": self.threshold,
                "accepted": bool(score >= self.threshold),
                "det_score": float(result.det_score),
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


class AppHandler(BaseHTTPRequestHandler):
    scorer: SimilarityScorer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._send_json(
                {
                    "ok": True,
                    "model_dir": str(self.scorer.model_dir),
                    "threshold": self.scorer.threshold,
                    "model_name": self.scorer.model_name,
                    "det_size": self.scorer.det_size,
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
    parser.add_argument("--model-name", default="buffalo_s")
    parser.add_argument("--ctx-id", type=int, default=-1)
    parser.add_argument("--det-size", type=int, default=320)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not (model_dir / "profile_model.pkl").exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {model_dir}")
    AppHandler.scorer = SimilarityScorer(model_dir, args.model_name, args.ctx_id, args.det_size)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"App local em http://{args.host}:{args.port}")
    print("Abra no navegador e permita o uso da camera.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
