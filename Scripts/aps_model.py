"""Berth-time model on the APS Gold table.

Two validation protocols are reported, because they answer different questions:

grouped   vessel-grouped 80/20 hold-out, averaged over 5 seeds. Measures
          generalisation to ships not seen in training.
temporal  train on everything before a cut-off, test on what follows. This is
          the deployment question — the model will always predict the future
          from the past — and it is the protocol the manuscript should lead
          with, because a random split silently rewards memorising a berth's
          recent behaviour.

Run:
    uv run python Scripts/aps_model.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "lakehouse_aps" / "gold_berth_time.parquet"
TARGET = "time_at_berth_h"
SEEDS = [42, 7, 13, 30, 2024]

CATEGORICAL = ["berth_location", "vessel_type", "navigation_type", "propulsion", "anchorage"]

NUMERIC = [
    "deadweight_t",
    "gross_tonnage",
    "net_tonnage",
    "beam_m",
    "loa_m",
    "summer_draft_m",
    "max_speed_kn",
    "teu_capacity",
    "vessel_age_years",
    "berth_hour",
    "berth_dayofweek",
    "berth_month",
    "berth_is_weekend",
    "visit_index",
    "wait_from_port_entry_h",
    "anchorage_hours",
    "was_anchored",
    "prev_time_at_berth_h",
    "prev_call_count",
]

MANIFEST = ["manifested_tonnes", "cargo_intensity", "has_manifest"]

FEATURE_SETS = {
    "particulars only": [c for c in NUMERIC if c not in {"prev_time_at_berth_h", "prev_call_count"}]
    + CATEGORICAL,
    "+ vessel history": NUMERIC + CATEGORICAL,
    "+ manifested cargo": NUMERIC + CATEGORICAL + MANIFEST,
}


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [c for c in columns if c not in CATEGORICAL]
    categorical = [c for c in columns if c in CATEGORICAL]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median", add_indicator=True))]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def models(seed: int) -> dict:
    return {
        "Baseline (median stay)": DummyRegressor(strategy="median"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=seed, max_iter=400, learning_rate=0.06
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
    }


def score(y_true_h: np.ndarray, y_pred_h: np.ndarray) -> dict:
    err = y_true_h - y_pred_h
    return {
        "r2_hours": r2_score(y_true_h, y_pred_h),
        "rmse_h": float(np.sqrt(mean_squared_error(y_true_h, y_pred_h))),
        "mae_h": float(mean_absolute_error(y_true_h, y_pred_h)),
        "median_ae_h": float(np.median(np.abs(err))),
        "within_6h": float(np.mean(np.abs(err) <= 6)),
        "within_12h": float(np.mean(np.abs(err) <= 12)),
    }


def fit_predict(model, X_tr, y_tr_log, X_te, columns) -> np.ndarray:
    pipe = Pipeline([("prep", build_preprocessor(columns)), ("model", model)])
    pipe.fit(X_tr, y_tr_log)
    return np.exp(pipe.predict(X_te))


def evaluate(df: pd.DataFrame, columns: list[str], name: str, protocol: str) -> list[dict]:
    X, y_h, groups = df[columns], df[TARGET], df["imo"]
    y_log = np.log(y_h)
    rows = []

    for model_name in models(42):
        runs = []
        if protocol == "temporal":
            cut = pd.Timestamp("2024-01-01")
            tr = (df.berthed_at < cut).to_numpy()
            te = ~tr
            for seed in SEEDS[:3]:
                pred = fit_predict(
                    models(seed)[model_name], X[tr], y_log[tr], X[te], columns
                )
                runs.append(score(y_h[te].to_numpy(), pred))
        else:
            for seed in SEEDS:
                tr, te = next(
                    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                        X, y_log, groups
                    )
                )
                pred = fit_predict(
                    models(seed)[model_name], X.iloc[tr], y_log.iloc[tr], X.iloc[te], columns
                )
                runs.append(score(y_h.iloc[te].to_numpy(), pred))

        frame = pd.DataFrame(runs)
        rows.append(
            {
                "feature_set": name,
                "protocol": protocol,
                "model": model_name,
                "n_train": int(tr.sum()) if tr.dtype == bool else len(tr),
                "n_test": int(te.sum()) if te.dtype == bool else len(te),
                **{k: float(frame[k].mean()) for k in frame.columns},
                "r2_sd": float(frame["r2_hours"].std()),
            }
        )
    return rows


def main(outdir: Path = ROOT / "reports") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(GOLD)
    df = df.dropna(subset=[TARGET])

    rows: list[dict] = []
    for name, columns in FEATURE_SETS.items():
        for protocol in ("grouped", "temporal"):
            rows.extend(evaluate(df, columns, name, protocol))

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "aps_model_results.csv", index=False)

    for protocol in ("temporal", "grouped"):
        view = result[result.protocol == protocol]
        print(f"\n=== {protocol.upper()} validation ===")
        print(
            view[
                [
                    "feature_set",
                    "model",
                    "n_train",
                    "n_test",
                    "r2_hours",
                    "r2_sd",
                    "rmse_h",
                    "mae_h",
                    "median_ae_h",
                    "within_6h",
                    "within_12h",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:0.3f}")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    main(parser.parse_args().out)
