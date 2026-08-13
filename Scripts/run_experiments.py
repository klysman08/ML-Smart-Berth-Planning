"""Reproduce the thesis baseline and run the ablations requested by the
MDPI Logistics reviewers.

Every number reported in the revision material must be produced by this script.
Two conventions matter and are easy to get wrong:

1. The model is fitted on ``log(Time At Berth)``. An R2 computed in log space
   is NOT comparable to an RMSE in hours. This script always reports both, and
   labels them.
2. Random Forest results move by several R2 points depending on the seed at
   this sample size, so every configuration is repeated over several seeds and
   reported as mean +/- sd.

Run:
    uv run python Scripts/run_experiments.py --out reports/
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, KFold, GroupKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from build_modelling_dataset import KEY_COLUMNS, build

warnings.filterwarnings("ignore")

SEEDS = [42, 7, 13, 30, 2024]
TARGET = "Time At Berth"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
@dataclass
class Scores:
    r2_log: float
    rmse_log: float
    r2_hours: float
    rmse_hours: float
    mae_hours: float
    median_ae_hours: float


def score(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Scores:
    """Score in log space (what the model optimises) and in hours (what a port
    operator actually cares about)."""
    y_true_h = np.exp(y_true_log)
    y_pred_h = np.exp(y_pred_log)
    err = y_true_h - y_pred_h
    return Scores(
        r2_log=r2_score(y_true_log, y_pred_log),
        rmse_log=float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        r2_hours=r2_score(y_true_h, y_pred_h),
        rmse_hours=float(np.sqrt(np.mean(err**2))),
        mae_hours=float(np.mean(np.abs(err))),
        median_ae_hours=float(np.median(np.abs(err))),
    )


# --------------------------------------------------------------------------
# feature preparation
# --------------------------------------------------------------------------
def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the remaining object/string columns.

    NOTE: the original notebooks use ``dtype == type(object)``. That test is
    True on pandas 2 but False on pandas 3 (string dtype), where the loop
    silently does nothing and the model then fails to fit. The explicit
    ``select_dtypes`` below is version-proof.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string", "category"]).columns:
        out[col] = LabelEncoder().fit_transform(out[col].astype(str))
    return out


def prepare(
    df: pd.DataFrame,
    *,
    use_time_at_port: bool = True,
    drop_features: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    groups = df["Imo"].astype(int)
    work = df.drop(columns=[c for c in KEY_COLUMNS if c in df.columns])
    if not use_time_at_port:
        work = work.drop(columns=["Time At Port"], errors="ignore")
    work = work.drop(columns=[c for c in drop_features if c in work.columns])

    y = np.log(work[TARGET])
    X = encode(work.drop(columns=[TARGET]))
    if "Time At Port" in X.columns:
        X["Time At Port"] = np.log(X["Time At Port"].clip(lower=1e-3))
    return X, y, groups


def strip_outliers(X: pd.DataFrame, y: pd.Series, groups: pd.Series, z: float = 3.0):
    keep = (np.abs((y - y.mean()) / y.std()) <= z).to_numpy()
    return X[keep], y[keep], groups[keep]


# --------------------------------------------------------------------------
# model zoo
# --------------------------------------------------------------------------
def model_zoo(seed: int) -> dict:
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "KNN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "MLP (50,50)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=seed),
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
        "Random Forest (thesis: 24 trees)": RandomForestRegressor(
            n_estimators=24, random_state=seed, oob_score=True, bootstrap=True
        ),
        "Random Forest (500 trees)": RandomForestRegressor(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "Extra Trees (500 trees)": ExtraTreesRegressor(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
    }


# --------------------------------------------------------------------------
# evaluation protocols
# --------------------------------------------------------------------------
def evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    model_name: str,
    grouped: bool,
    seeds: list[int] = SEEDS,
) -> dict:
    """Fit/score a model over several seeds.

    ``grouped=True`` keeps every call of the same vessel on one side of the
    split, so a vessel seen in training can never be scored in the test set.
    That is the leakage the manuscript's de-duplication step was trying to
    prevent, handled without discarding data.
    """
    rows = []
    for seed in seeds:
        model = model_zoo(seed)[model_name]
        if grouped:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(splitter.split(X, y, groups))
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=0.2, random_state=seed
            )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        rows.append(asdict(score(y.iloc[test_idx].to_numpy(), model.predict(X.iloc[test_idx]))))

    frame = pd.DataFrame(rows)
    return {
        "model": model_name,
        "n": len(X),
        "n_vessels": int(groups.nunique()),
        "split": "grouped by vessel" if grouped else "random",
        **{f"{k}_mean": float(frame[k].mean()) for k in frame.columns},
        **{f"{k}_sd": float(frame[k].std()) for k in frame.columns},
    }


def cv_r2(X, y, groups, *, model_name: str, grouped: bool, seed: int = 42) -> dict:
    model = model_zoo(seed)[model_name]
    if grouped:
        scores = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
    else:
        scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=seed))
    return {"cv_mean": float(scores.mean()), "cv_min": float(scores.min()), "cv_max": float(scores.max())}


# --------------------------------------------------------------------------
# experiment definitions
# --------------------------------------------------------------------------
DATA_CONFIGS = {
    "A. Thesis baseline (as published)": dict(
        drop_service_craft=False, dedup="none", parse_numeric=False
    ),
    "B. + size fields parsed as numbers": dict(
        drop_service_craft=False, dedup="none", parse_numeric=True
    ),
    "C. + service craft removed": dict(
        drop_service_craft=True, dedup="none", parse_numeric=True
    ),
    "D. + one call per vessel per day": dict(
        drop_service_craft=True, dedup="vessel_day", parse_numeric=True
    ),
    "E. + one call per vessel (manuscript-style)": dict(
        drop_service_craft=True, dedup="vessel", parse_numeric=True
    ),
}


def run_all(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    flow_md: list[str] = []
    ablation_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    leakage_rows: list[dict] = []

    for label, cfg in DATA_CONFIGS.items():
        df, log = build(**cfg)
        flow_md.append(f"### {label}\n\n{log.to_markdown()}\n")

        X, y, g = prepare(df)
        X, y, g = strip_outliers(X, y, g)

        for grouped in (False, True):
            for model_name in ("Random Forest (thesis: 24 trees)", "Extra Trees (500 trees)"):
                row = evaluate(X, y, g, model_name=model_name, grouped=grouped)
                row["config"] = label
                ablation_rows.append(row)

        # leakage probe: does Time At Port carry the answer?
        for use_tap in (True, False):
            Xl, yl, gl = prepare(df, use_time_at_port=use_tap)
            Xl, yl, gl = strip_outliers(Xl, yl, gl)
            row = evaluate(Xl, yl, gl, model_name="Random Forest (500 trees)", grouped=True)
            row["config"] = label
            row["time_at_port_used"] = use_tap
            leakage_rows.append(row)

    # full model benchmark on the recommended configuration
    df, _ = build(**DATA_CONFIGS["C. + service craft removed"])
    X, y, g = prepare(df)
    X, y, g = strip_outliers(X, y, g)
    for model_name in model_zoo(42):
        row = evaluate(X, y, g, model_name=model_name, grouped=True)
        row.update(cv_r2(X, y, g, model_name=model_name, grouped=True))
        benchmark_rows.append(row)

    pd.DataFrame(ablation_rows).to_csv(outdir / "ablation.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(outdir / "benchmark.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(outdir / "leakage_probe.csv", index=False)
    (outdir / "filtering_flow.md").write_text("\n".join(flow_md), encoding="utf-8")

    _print_summary(ablation_rows, benchmark_rows, leakage_rows)


def _fmt(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    frame = pd.DataFrame(rows)
    out = pd.DataFrame({label: frame[key] for key, label in cols})
    return out.to_string(index=False, float_format=lambda v: f"{v:0.3f}")


def _print_summary(ablation, benchmark, leakage) -> None:
    print("\n=== ABLATION: data-construction decisions ===")
    print(
        _fmt(
            ablation,
            [
                ("config", "configuration"),
                ("model", "model"),
                ("split", "split"),
                ("n", "n"),
                ("n_vessels", "vessels"),
                ("r2_log_mean", "R2(log)"),
                ("r2_log_sd", "+-"),
                ("rmse_hours_mean", "RMSE(h)"),
                ("mae_hours_mean", "MAE(h)"),
            ],
        )
    )
    print("\n=== LEAKAGE PROBE: with vs without 'Time At Port' ===")
    print(
        _fmt(
            leakage,
            [
                ("config", "configuration"),
                ("time_at_port_used", "uses TimeAtPort"),
                ("n", "n"),
                ("r2_log_mean", "R2(log)"),
                ("rmse_hours_mean", "RMSE(h)"),
            ],
        )
    )
    print("\n=== BENCHMARK (config C, grouped split, 5 seeds) ===")
    print(
        _fmt(
            benchmark,
            [
                ("model", "model"),
                ("r2_log_mean", "R2(log)"),
                ("r2_log_sd", "+-"),
                ("cv_mean", "CV R2"),
                ("rmse_hours_mean", "RMSE(h)"),
                ("mae_hours_mean", "MAE(h)"),
                ("median_ae_hours_mean", "MedAE(h)"),
            ],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run_all(args.out)
