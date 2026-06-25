from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .calibration import ScoreCalibrator
from .extractor import ArcFaceEmbedder
from .manifest import read_manifest, split_mask
from .metrics import binary_metrics, metrics_by_quality
from .profile import FaceProfileModel, ScoreWeights
from .utils import ensure_dir, parse_csv_list, write_json


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face profile modeling with pretrained embeddings.")
    sub = parser.add_subparsers(required=True)

    extract = sub.add_parser("extract", help="Detect, align and extract ArcFace embeddings.")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--root-dir", default=None)
    extract.add_argument("--out-dir", default="artifacts")
    extract.add_argument("--model-name", default="buffalo_l")
    extract.add_argument("--ctx-id", type=int, default=-1)
    extract.add_argument("--det-size", type=int, default=640)
    extract.add_argument("--save-aligned", action="store_true")
    extract.set_defaults(func=cmd_extract)

    fit = sub.add_parser("fit", help="Fit a profile model from extracted embeddings.")
    add_feature_args(fit)
    fit.add_argument("--out-dir", default="artifacts/model")
    fit.add_argument("--profile-splits", default="profile,enroll")
    fit.add_argument("--top-k", type=int, default=5)
    fit.add_argument("--mahalanobis-regularization", type=float, default=0.05)
    fit.add_argument("--use-ocsvm", action="store_true")
    fit.add_argument("--ocsvm-nu", type=float, default=0.05)
    fit.set_defaults(func=cmd_fit)

    calibrate = sub.add_parser("calibrate", help="Calibrate score_raw using positive and negative splits.")
    add_feature_args(calibrate)
    calibrate.add_argument("--model-dir", required=True)
    calibrate.add_argument("--positive-splits", default="calib_pos")
    calibrate.add_argument("--negative-splits", default="calib_neg")
    calibrate.set_defaults(func=cmd_calibrate)

    evaluate = sub.add_parser("evaluate", help="Evaluate ROC, AUC, EER, FMR and FNMR.")
    add_feature_args(evaluate)
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--positive-splits", default="test_pos")
    evaluate.add_argument("--negative-splits", default="test_neg")
    evaluate.add_argument("--out-dir", default="artifacts/eval")
    evaluate.set_defaults(func=cmd_evaluate)

    demo = sub.add_parser("demo", help="Run realtime OpenCV demo with median score over multiple frames.")
    demo.add_argument("--model-dir", required=True)
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--frame-window", type=int, default=9)
    demo.add_argument("--model-name", default="buffalo_l")
    demo.add_argument("--ctx-id", type=int, default=-1)
    demo.add_argument("--det-size", type=int, default=640)
    demo.set_defaults(func=cmd_demo)
    return parser


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", required=True, help="Path to embedding_manifest.csv.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy.")


def cmd_extract(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out_dir)
    aligned_dir = ensure_dir(out_dir / "aligned") if args.save_aligned else None
    manifest = read_manifest(args.manifest, root_dir=args.root_dir)
    embedder = ArcFaceEmbedder(model_name=args.model_name, ctx_id=args.ctx_id, det_size=args.det_size)

    embeddings: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract"):
        record = row.to_dict()
        record["embedding_index"] = -1
        record["embedding_status"] = "not_attempted"
        record["det_score"] = np.nan
        record["bbox_x1"] = np.nan
        record["bbox_y1"] = np.nan
        record["bbox_x2"] = np.nan
        record["bbox_y2"] = np.nan
        record["aligned_path"] = ""
        try:
            if not bool(row["exists"]):
                raise FileNotFoundError(row["resolved_path"])
            result = embedder.extract_path(row["resolved_path"], return_aligned=args.save_aligned)
            record["embedding_index"] = len(embeddings)
            record["embedding_status"] = "ok"
            record["det_score"] = result.det_score
            record["bbox_x1"], record["bbox_y1"], record["bbox_x2"], record["bbox_y2"] = result.bbox
            if aligned_dir is not None and result.aligned_bgr is not None:
                aligned_path = aligned_dir / f"{int(row['row_id']):06d}_{Path(row['resolved_path']).stem}.jpg"
                cv2.imwrite(str(aligned_path), result.aligned_bgr)
                record["aligned_path"] = str(aligned_path)
            embeddings.append(result.embedding)
        except Exception as exc:
            record["embedding_status"] = f"error: {exc}"
        rows.append(record)

    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, 0), dtype=np.float32)
    np.save(out_dir / "embeddings.npy", matrix)
    pd.DataFrame(rows).to_csv(out_dir / "embedding_manifest.csv", index=False)
    write_json(out_dir / "extract_metadata.json", {"num_rows": len(rows), "num_embeddings": int(matrix.shape[0])})
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    table, embeddings = load_features(args.features, args.embeddings)
    profile_splits = parse_csv_list(args.profile_splits)
    mask = valid_embedding_mask(table) & split_mask(table, profile_splits)
    if not mask.any():
        raise ValueError(f"No valid embeddings found for profile splits: {profile_splits}")

    idx = table.loc[mask, "embedding_index"].astype(int).to_numpy()
    weights = table.loc[mask, "weight"].astype(float).to_numpy()
    model = FaceProfileModel(
        top_k=args.top_k,
        mahalanobis_regularization=args.mahalanobis_regularization,
        score_weights=ScoreWeights(),
        use_ocsvm=args.use_ocsvm,
        ocsvm_nu=args.ocsvm_nu,
    ).fit(embeddings[idx], weights)
    model.save(args.out_dir)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    table, embeddings = load_features(args.features, args.embeddings)
    model = FaceProfileModel.load(args.model_dir)
    scored = score_feature_table(model, table, embeddings)
    positive = parse_csv_list(args.positive_splits)
    negative = parse_csv_list(args.negative_splits)
    mask_pos = scored["split"].isin(positive)
    mask_neg = scored["split"].isin(negative)
    calibration = scored.loc[mask_pos | mask_neg].copy()
    calibration["label"] = np.where(calibration["split"].isin(positive), 1, 0)
    if calibration.empty:
        raise ValueError("No calibration rows found.")

    calibrator = ScoreCalibrator().fit(calibration["score_raw"].to_numpy(), calibration["label"].to_numpy())
    calibration["score_calibrated"] = calibrator.predict_proba(calibration["score_raw"].to_numpy())
    metric = binary_metrics(calibration["label"].to_numpy(), calibration["score_calibrated"].to_numpy())
    if metric.get("status") == "ok":
        calibrator.threshold = float(metric["eer_threshold"])
    calibrator.save(args.model_dir)
    calibration.to_csv(Path(args.model_dir) / "calibration_scores.csv", index=False)
    write_json(Path(args.model_dir) / "calibration_metrics.json", metric)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out_dir)
    table, embeddings = load_features(args.features, args.embeddings)
    model = FaceProfileModel.load(args.model_dir)
    scored = score_feature_table(model, table, embeddings)
    positive = parse_csv_list(args.positive_splits)
    negative = parse_csv_list(args.negative_splits)
    mask = scored["split"].isin(positive + negative)
    eval_frame = scored.loc[mask].copy()
    eval_frame["label"] = np.where(eval_frame["split"].isin(positive), 1, 0)

    score_column = "score_raw"
    calibrator_path = Path(args.model_dir) / "calibrator.pkl"
    if calibrator_path.exists():
        calibrator = ScoreCalibrator.load(args.model_dir)
        eval_frame["score_calibrated"] = calibrator.predict_proba(eval_frame["score_raw"].to_numpy())
        score_column = "score_calibrated"

    if eval_frame.empty:
        raise ValueError("No evaluation rows found.")
    eval_frame.to_csv(out_dir / "eval_scores.csv", index=False)
    write_json(out_dir / "metrics.json", metrics_by_quality(eval_frame.rename(columns={score_column: "score"}), "score"))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from collections import deque

    model = FaceProfileModel.load(args.model_dir)
    calibrator = None
    if (Path(args.model_dir) / "calibrator.pkl").exists():
        calibrator = ScoreCalibrator.load(args.model_dir)
    embedder = ArcFaceEmbedder(model_name=args.model_name, ctx_id=args.ctx_id, det_size=args.det_size)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    scores: deque[float] = deque(maxlen=max(1, args.frame_window))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        label = "no face"
        color = (40, 40, 255)
        try:
            result = embedder.extract_bgr(frame)
            raw = float(model.score(result.embedding)["score_raw"].iloc[0])
            score = float(calibrator.predict_proba(np.asarray([raw]))[0]) if calibrator else raw
            scores.append(score)
            median_score = float(np.median(scores))
            threshold = calibrator.threshold if calibrator else 0.5
            accepted = median_score >= threshold
            label = f"score={median_score:.3f}"
            color = (40, 180, 40) if accepted else (40, 160, 255)
            x1, y1, x2, y2 = [int(v) for v in result.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        except Exception:
            scores.clear()

        cv2.putText(frame, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imshow("face-profile demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def load_features(features_path: str | Path, embeddings_path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    table = pd.read_csv(features_path)
    embeddings = np.load(embeddings_path)
    if "embedding_index" not in table.columns:
        raise ValueError("features CSV must contain embedding_index.")
    return table, embeddings


def valid_embedding_mask(table: pd.DataFrame) -> pd.Series:
    return table["embedding_index"].astype(int) >= 0


def score_feature_table(model: FaceProfileModel, table: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    valid = table.loc[valid_embedding_mask(table)].copy()
    idx = valid["embedding_index"].astype(int).to_numpy()
    scores = model.score(embeddings[idx])
    return pd.concat([valid.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    raise SystemExit(main())

