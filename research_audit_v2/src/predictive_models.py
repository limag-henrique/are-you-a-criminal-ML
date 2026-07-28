"""Technical-predictability analysis with grouped folds and no social claims."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import write_csv


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prediction = score >= .5
    return {"roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan, "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan, "balanced_accuracy": balanced_accuracy_score(y, prediction), "precision": precision_score(y, prediction, zero_division=0), "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0), "brier": brier_score_loss(y, score)}


def predictive_models(records: pd.DataFrame, membership: np.ndarray, config: dict, tables: Path, reports: Path) -> None:
    technical = [column for column in ["det_score", "face_count", "face_area_ratio", "image_width", "image_height"] if column in records]
    categorical = [column for column in ["quality", "source"] if column in records]
    if membership.sum() == 0 or membership.sum() == len(membership) or not technical and not categorical:
        write_csv(pd.DataFrame(columns=["strategy", "fold", "roc_auc", "pr_auc"]), tables / "predictive_model_metrics.csv")
        reports.joinpath("predictive_models_report.md").write_text("# Predictive models\n\nNot estimable: target prevalence or technical predictors did not support a binary model.\n", encoding="utf-8")
        return
    transforms = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), technical), ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    model = Pipeline([("features", transforms), ("model", LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=config["random_seed"]))])
    y = np.asarray(membership, dtype=int)
    rows = []
    for name, splitter, groups in [("stratified", StratifiedKFold(5, shuffle=True, random_state=config["random_seed"]), None), ("grouped", GroupKFold(min(5, records["group_id"].nunique())), records["group_id"])]:
        for fold, (train, test) in enumerate(splitter.split(records, y, groups)):
            model.fit(records.iloc[train], y[train]); row = {"strategy": name, "fold": fold, **_metrics(y[test], model.predict_proba(records.iloc[test])[:, 1])}; rows.append(row)
    write_csv(pd.DataFrame(rows), tables / "predictive_model_metrics.csv")
    reports.joinpath("predictive_models_report.md").write_text("# Technical predictability\n\nThis analysis measures conditional association between a reconstructed synthetic target and technical/provenance variables. It does not measure recognition, identity, or social validity.\n", encoding="utf-8")
