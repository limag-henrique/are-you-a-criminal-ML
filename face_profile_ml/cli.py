from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .calibration import ScoreCalibrator
from .bootstrap_ci import bootstrap_metric
from .cross_validation import run_grouped_cluster_cv
from .extractor import ArcFaceEmbedder
from .manifest import read_manifest, split_mask
from .metrics import binary_metrics, metrics_by_quality
from .fairness import audit_group_metrics
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
    extract.add_argument("--det-size", type=int, default=320)
    extract.add_argument("--min-det-score", type=float, default=0.50)
    extract.add_argument("--min-face-area-ratio", type=float, default=0.01)
    extract.add_argument("--allow-multiple-faces", action="store_true")
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

    cv_fit = sub.add_parser("cv-fit", help="Run grouped OOF evaluation of an endogenous cluster target.")
    add_feature_args(cv_fit)
    cv_fit.add_argument("--out-dir", default="artifacts/cv")
    cv_fit.add_argument("--n-splits", type=int, default=5)
    cv_fit.add_argument("--k", type=int, default=64)
    cv_fit.add_argument("--seed", type=int, default=42)
    cv_fit.add_argument(
        "--target-rule",
        choices=["largest", "compact", "separated", "random", "central", "outlier"],
        default="largest",
    )
    cv_fit.add_argument("--bootstrap-rounds", type=int, default=2000)
    cv_fit.set_defaults(func=cmd_cv_fit)

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

    fairness = sub.add_parser("audit-fairness", help="Audit verification performance by documented cohorts.")
    fairness.add_argument("--scores", required=True, help="CSV with label, score and documented cohort columns.")
    fairness.add_argument("--group-columns", required=True, help="Comma-separated documented cohort columns.")
    fairness.add_argument("--score-column", default="score")
    fairness.add_argument("--min-group-n", type=int, default=30)
    fairness.add_argument("--bootstrap-rounds", type=int, default=1000)
    fairness.add_argument("--seed", type=int, default=42)
    fairness.add_argument("--threshold", type=float, default=None, help="Frozen decision threshold from calibration.")
    fairness.add_argument("--no-intersections", action="store_true")
    fairness.add_argument("--out-dir", default="artifacts/fairness")
    fairness.set_defaults(func=cmd_audit_fairness)

    demo = sub.add_parser("demo", help="Run realtime OpenCV demo with median score over multiple frames.")
    demo.add_argument("--model-dir", required=True)
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--frame-window", type=int, default=9)
    demo.add_argument("--model-name", default="buffalo_l")
    demo.add_argument("--ctx-id", type=int, default=-1)
    demo.add_argument("--det-size", type=int, default=320)
    demo.set_defaults(func=cmd_demo)
    return parser


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", required=True, help="Path to embedding_manifest.csv.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy.")


def cmd_extract(args: argparse.Namespace) -> int:
    warnings.filterwarnings("ignore", message="`estimate` is deprecated.*", category=FutureWarning)
    out_dir = ensure_dir(args.out_dir)
    aligned_dir = ensure_dir(out_dir / "aligned") if args.save_aligned else None
    manifest = read_manifest(args.manifest, root_dir=args.root_dir)
    embedder = ArcFaceEmbedder(model_name=args.model_name, ctx_id=args.ctx_id, det_size=args.det_size)

    embeddings: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract", disable=not sys.stderr.isatty()):
        record = row.to_dict()
        record["embedding_index"] = -1
        record["embedding_status"] = "not_attempted"
        record["det_score"] = np.nan
        record["face_count"] = 0
        record["face_area_ratio"] = np.nan
        record["image_width"] = np.nan
        record["image_height"] = np.nan
        record["bbox_x1"] = np.nan
        record["bbox_y1"] = np.nan
        record["bbox_x2"] = np.nan
        record["bbox_y2"] = np.nan
        record["aligned_path"] = ""
        try:
            if not bool(row["exists"]):
                raise FileNotFoundError(row["resolved_path"])
            result = embedder.extract_path(row["resolved_path"], return_aligned=args.save_aligned)
            if result.det_score < float(args.min_det_score):
                raise ValueError(f"Face detection score below threshold: {result.det_score:.4f}")
            face_area_ratio = result.face_area_ratio
            if face_area_ratio is not None and face_area_ratio < float(args.min_face_area_ratio):
                raise ValueError(f"Face area ratio below threshold: {face_area_ratio:.6f}")
            if result.face_count > 1 and not bool(args.allow_multiple_faces):
                raise ValueError(f"Multiple faces detected: {result.face_count}")
            record["embedding_index"] = len(embeddings)
            record["embedding_status"] = "ok"
            record["det_score"] = result.det_score
            record["face_count"] = result.face_count
            record["face_area_ratio"] = np.nan if face_area_ratio is None else face_area_ratio
            if result.image_shape is not None:
                record["image_height"], record["image_width"] = result.image_shape
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
    write_json(
        out_dir / "extract_metadata.json",
        {
            "num_rows": len(rows),
            "num_embeddings": int(matrix.shape[0]),
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0,
            "model_name": args.model_name,
            "ctx_id": args.ctx_id,
            "det_size": args.det_size,
            "min_det_score": args.min_det_score,
            "min_face_area_ratio": args.min_face_area_ratio,
            "allow_multiple_faces": bool(args.allow_multiple_faces),
            "save_aligned": bool(args.save_aligned),
        },
    )
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


def cmd_cv_fit(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out_dir)
    table, embeddings = load_features(args.features, args.embeddings)
    valid = table.loc[valid_embedding_mask(table)].copy()
    indices = valid["embedding_index"].astype(int).to_numpy()
    if "sample_id" not in valid:
        valid["sample_id"] = valid.index.astype(str)
    if "group_id" not in valid:
        valid["group_id"] = valid["sample_id"]
    oof, metrics = run_grouped_cluster_cv(
        valid[["sample_id", "group_id"]].reset_index(drop=True),
        embeddings[indices],
        n_splits=args.n_splits,
        k=args.k,
        seed=args.seed,
        target_rule=args.target_rule,
    )
    output_path = out_dir / "oof_predictions.csv"
    oof.to_csv(output_path, index=False)
    bootstrap = {
        name: bootstrap_metric(
            oof["y_true"].to_numpy(),
            oof["prob_calibrated"].to_numpy(),
            metric=name,
            n_bootstrap=args.bootstrap_rounds,
            seed=args.seed,
        ).as_dict()
        for name in ["auc", "pr_auc", "brier", "balanced_accuracy"]
    }
    write_json(out_dir / "bootstrap_metrics.json", {"metrics": bootstrap})
    write_json(
        out_dir / "run_manifest.json",
        {
            "oof_predictions": str(output_path),
            "oof_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "metrics": metrics,
            "parameters": {
                "n_splits": args.n_splits,
                "k": args.k,
                "seed": args.seed,
                "target_rule": args.target_rule,
                "bootstrap_rounds": args.bootstrap_rounds,
            },
        },
    )
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


def cmd_audit_fairness(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out_dir)
    scores = pd.read_csv(args.scores)
    summary, rows = audit_group_metrics(
        scores,
        score_column=args.score_column,
        group_columns=parse_csv_list(args.group_columns),
        min_group_n=args.min_group_n,
        bootstrap_rounds=args.bootstrap_rounds,
        seed=args.seed,
        include_intersections=not args.no_intersections,
        threshold=args.threshold,
    )
    rows.to_csv(out_dir / "group_metrics.csv", index=False)
    write_json(out_dir / "fairness_summary.json", summary)
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
