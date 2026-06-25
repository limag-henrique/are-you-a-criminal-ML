#!/usr/bin/env python3
"""Build a visual audit report for face-profile manifest splits."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


DEFAULT_SPLITS = ["profile", "calib_pos", "test_pos", "calib_neg", "test_neg"]
POSITIVE_SPLITS = {"profile", "calib_pos", "test_pos"}
NEGATIVE_SPLITS = {"calib_neg", "test_neg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML/CSV thumbnails for manifest split audit.")
    parser.add_argument("--features", default="artifacts/embedding_manifest.csv")
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--out-dir", default="artifacts/audit")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--samples-per-split", type=int, default=48)
    parser.add_argument("--hard-examples-per-class", type=int, default=24)
    parser.add_argument("--thumb-size", type=int, default=144)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir)
    thumbs_dir = out_dir / "thumbnails"
    if thumbs_dir.exists():
        shutil.rmtree(thumbs_dir)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    features = features[features["embedding_index"].astype(int) >= 0].copy()
    features["label"] = np.where(features["split"].isin(POSITIVE_SPLITS), 1, 0)

    scored = add_scores(features, args.embeddings, args.model_dir)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    audit_rows = choose_audit_rows(scored, splits, args.samples_per_split, args.hard_examples_per_class, args.random_state)
    audit_rows = write_thumbnails(audit_rows, root_dir, out_dir, thumbs_dir, args.thumb_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = out_dir / "audit_samples.csv"
    audit_rows.to_csv(samples_csv, index=False)

    summary = build_summary(scored, splits)
    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    html_path = out_dir / "index.html"
    html_path.write_text(render_html(summary, audit_rows, samples_csv, summary_path), encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Wrote {samples_csv}")
    print(f"Wrote {summary_path}")
    return 0


def add_scores(features: pd.DataFrame, embeddings_path: str | Path, model_dir: str | Path) -> pd.DataFrame:
    model_path = Path(model_dir) / "profile_model.pkl"
    if not model_path.exists():
        features["score_raw"] = np.nan
        features["score_calibrated"] = np.nan
        return features

    sys.path.insert(0, str(Path.cwd()))
    from face_profile_ml.calibration import ScoreCalibrator
    from face_profile_ml.profile import FaceProfileModel

    embeddings = np.load(embeddings_path)
    model = FaceProfileModel.load(model_dir)
    idx = features["embedding_index"].astype(int).to_numpy()
    scores = model.score(embeddings[idx])
    scored = pd.concat([features.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    calibrator_path = Path(model_dir) / "calibrator.pkl"
    if calibrator_path.exists():
        calibrator = ScoreCalibrator.load(model_dir)
        scored["score_calibrated"] = calibrator.predict_proba(scored["score_raw"].to_numpy())
        scored["decision_threshold"] = float(calibrator.threshold)
    else:
        scored["score_calibrated"] = scored["score_raw"]
        scored["decision_threshold"] = 0.5
    return scored


def choose_audit_rows(
    scored: pd.DataFrame,
    splits: list[str],
    samples_per_split: int,
    hard_examples_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    pieces: list[pd.DataFrame] = []
    for split in splits:
        part = scored[scored["split"] == split].copy()
        if part.empty:
            continue
        n = min(samples_per_split, len(part))
        sample_idx = rng.choice(part.index.to_numpy(), size=n, replace=False)
        sample = part.loc[sample_idx].copy()
        sample["sample_reason"] = "random"
        pieces.append(sample)

    if "score_calibrated" in scored.columns and scored["score_calibrated"].notna().any():
        positives = scored[scored["split"].isin(POSITIVE_SPLITS)].sort_values("score_calibrated", ascending=True)
        negatives = scored[scored["split"].isin(NEGATIVE_SPLITS)].sort_values("score_calibrated", ascending=False)
        hard_pos = positives.head(hard_examples_per_class).copy()
        hard_neg = negatives.head(hard_examples_per_class).copy()
        hard_pos["sample_reason"] = "lowest_positive_scores"
        hard_neg["sample_reason"] = "highest_negative_scores"
        pieces.extend([hard_pos, hard_neg])

    if not pieces:
        return scored.head(0).copy()

    selected = pd.concat(pieces, ignore_index=True)
    selected = selected.drop_duplicates(subset=["row_id", "sample_reason"]).copy()
    selected = selected.sort_values(["split", "sample_reason", "score_calibrated", "row_id"], ascending=[True, True, False, True])
    return selected.reset_index(drop=True)


def write_thumbnails(
    rows: pd.DataFrame,
    root_dir: Path,
    out_dir: Path,
    thumbs_dir: Path,
    thumb_size: int,
) -> pd.DataFrame:
    rows = rows.copy()
    thumb_paths: list[str] = []
    for _, row in rows.iterrows():
        source = choose_image_path(row, root_dir)
        filename = f"{row['split']}_{int(row['row_id']):06d}.jpg"
        target = thumbs_dir / filename
        try:
            image = Image.open(source).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb_size, thumb_size), (246, 247, 249))
            x = (thumb_size - image.width) // 2
            y = (thumb_size - image.height) // 2
            canvas.paste(image, (x, y))
            canvas.save(target, quality=88)
            thumb_paths.append(str(target.relative_to(out_dir)).replace("\\", "/"))
        except Exception:
            thumb_paths.append("")
    rows["thumbnail"] = thumb_paths
    return rows


def choose_image_path(row: pd.Series, root_dir: Path) -> Path:
    aligned = str(row.get("aligned_path", "") or "")
    if aligned:
        path = Path(aligned)
        if not path.is_absolute():
            path = root_dir / path
        if path.exists():
            return path

    path = Path(str(row["path"]))
    if not path.is_absolute():
        path = root_dir / path
    return path


def build_summary(scored: pd.DataFrame, splits: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows_valid": int(len(scored)),
        "split_counts": {str(k): int(v) for k, v in scored["split"].value_counts().sort_index().items()},
        "quality_by_split": pd.crosstab(scored["split"], scored["quality"]).to_dict(),
    }
    if "score_calibrated" in scored.columns and scored["score_calibrated"].notna().any():
        split_stats = {}
        for split in splits:
            values = scored.loc[scored["split"] == split, "score_calibrated"].dropna()
            if values.empty:
                continue
            split_stats[split] = {
                "n": int(len(values)),
                "min": float(values.min()),
                "p10": float(values.quantile(0.10)),
                "median": float(values.median()),
                "p90": float(values.quantile(0.90)),
                "max": float(values.max()),
            }
        summary["score_calibrated_by_split"] = split_stats
        if "decision_threshold" in scored.columns:
            summary["decision_threshold"] = float(scored["decision_threshold"].dropna().iloc[0])
    return summary


def render_html(summary: dict[str, object], rows: pd.DataFrame, samples_csv: Path, summary_path: Path) -> str:
    cards = []
    for _, row in rows.iterrows():
        score = row.get("score_calibrated", np.nan)
        score_text = "" if pd.isna(score) else f"{float(score):.3f}"
        raw_score = row.get("score_raw", np.nan)
        raw_text = "" if pd.isna(raw_score) else f"{float(raw_score):.3f}"
        image = html.escape(str(row.get("thumbnail", "")))
        subject = html.escape(str(row.get("subject_id", "")))
        split = html.escape(str(row.get("split", "")))
        quality = html.escape(str(row.get("quality", "")))
        reason = html.escape(str(row.get("sample_reason", "")))
        path = html.escape(str(row.get("path", "")))
        img_tag = f'<img src="{image}" alt="{subject}">' if image else '<div class="missing">sem thumbnail</div>'
        cards.append(
            f"""
            <article class="card {split}">
              {img_tag}
              <div class="meta">
                <strong>{split}</strong><span>{quality}</span><span>{reason}</span>
              </div>
              <div class="score">cal {score_text}<span>raw {raw_text}</span></div>
              <p title="{subject}">{subject}</p>
              <small title="{path}">{path}</small>
            </article>
            """
        )

    summary_json = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Profile Audit</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f7f8;
      color: #172026;
    }}
    body {{ margin: 0; }}
    header {{
      background: #172026;
      color: #fff;
      padding: 24px 32px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    a {{ color: #0b66c3; }}
    header a {{ color: #b8e0ff; }}
    main {{ padding: 24px 32px 40px; }}
    .summary {{
      display: grid;
      grid-template-columns: minmax(260px, 420px) 1fr;
      gap: 24px;
      align-items: start;
      margin-bottom: 24px;
    }}
    pre {{
      margin: 0;
      overflow: auto;
      padding: 16px;
      background: #fff;
      border: 1px solid #d7dde2;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.45;
    }}
    .note {{
      background: #fff;
      border: 1px solid #d7dde2;
      border-left: 4px solid #0b66c3;
      border-radius: 6px;
      padding: 16px;
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(176px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d7dde2;
      border-radius: 6px;
      overflow: hidden;
      min-width: 0;
    }}
    .card img, .missing {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #eef1f4;
      display: block;
    }}
    .missing {{
      display: grid;
      place-items: center;
      color: #64717c;
      font-size: 13px;
    }}
    .meta, .score {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px 0;
      font-size: 12px;
      color: #52616c;
      min-width: 0;
    }}
    .meta strong {{
      color: #172026;
      font-size: 13px;
    }}
    .score {{
      justify-content: space-between;
      color: #172026;
      font-weight: 700;
    }}
    .score span {{
      color: #6b7780;
      font-weight: 400;
    }}
    .card p {{
      margin: 8px 10px 4px;
      font-size: 12px;
      line-height: 1.3;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .card small {{
      display: block;
      margin: 0 10px 10px;
      font-size: 11px;
      line-height: 1.3;
      color: #6b7780;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .summary {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Face Profile Audit</h1>
    <div><a href="{html.escape(samples_csv.name)}">CSV de amostras</a> | <a href="{html.escape(summary_path.name)}">Resumo JSON</a></div>
  </header>
  <main>
    <section class="summary">
      <div class="note">
        Este relatorio audita os splits atuais com faces alinhadas, amostras aleatorias e exemplos dificeis.
        Os rotulos positivos/negativos continuam semanticos por cluster de embedding, nao por anotacao humana.
      </div>
      <pre>{summary_json}</pre>
    </section>
    <section class="grid">
      {"".join(cards)}
    </section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
