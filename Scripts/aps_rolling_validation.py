"""Rolling-origin temporal validation, plus a drift diagnosis for the manifest.

A single train/test cut-off can flatter or punish a model by accident. This
script repeats the temporal evaluation at several origins, each time training
on everything before the cut-off and testing on the following six months, so
the manuscript can report stability rather than one lucky split.

It also diagnoses why the manifested-cargo feature set helps under
vessel-grouped validation but hurts under temporal validation: the coverage and
scale of the manifest field are compared across periods.

Run:
    uv run python Scripts/aps_rolling_validation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from aps_model import FEATURE_SETS, GOLD, TARGET, build_preprocessor, score

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = ["2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01", "2024-04-01"]
HORIZON_MONTHS = 6


def rolling(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for name, columns in FEATURE_SETS.items():
        for cutoff in CUTOFFS:
            start = pd.Timestamp(cutoff)
            end = start + pd.DateOffset(months=HORIZON_MONTHS)
            train = df[df.berthed_at < start]
            test = df[(df.berthed_at >= start) & (df.berthed_at < end)]
            if len(train) < 500 or len(test) < 100:
                continue

            pipe = Pipeline(
                [
                    ("prep", build_preprocessor(columns)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1
                        ),
                    ),
                ]
            )
            pipe.fit(train[columns], np.log(train[TARGET]))
            predicted = np.exp(pipe.predict(test[columns]))

            baseline = np.full(len(test), train[TARGET].median())
            rows.append(
                {
                    "feature_set": name,
                    "cutoff": cutoff,
                    "n_train": len(train),
                    "n_test": len(test),
                    **score(test[TARGET].to_numpy(), predicted),
                    "baseline_mae_h": float(np.mean(np.abs(test[TARGET].to_numpy() - baseline))),
                }
            )
    return pd.DataFrame(rows)


def manifest_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = df.berthed_at.dt.to_period("Y").astype(str)
    grouped = df.groupby("period").agg(
        n=("call_id", "size"),
        manifest_coverage=("has_manifest", "mean"),
        median_tonnes=("manifested_tonnes", "median"),
        mean_tonnes=("manifested_tonnes", "mean"),
        median_stay_h=(TARGET, "median"),
    )
    corr = (
        df.dropna(subset=["manifested_tonnes"])
        .groupby("period")
        .apply(
            lambda g: np.corrcoef(np.log(g[TARGET]), np.log1p(g["manifested_tonnes"]))[0, 1],
            include_groups=False,
        )
        .rename("corr_log_stay_vs_log_tonnes")
    )
    return grouped.join(corr)


def main(outdir: Path = ROOT / "reports") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(GOLD).dropna(subset=[TARGET])

    results = rolling(df)
    results.to_csv(outdir / "aps_rolling_validation.csv", index=False)
    print("Rolling-origin validation (Random Forest, train = all prior, test = next 6 months)\n")
    print(
        results[
            [
                "feature_set",
                "cutoff",
                "n_train",
                "n_test",
                "r2_hours",
                "rmse_h",
                "mae_h",
                "baseline_mae_h",
                "within_6h",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:0.3f}")
    )

    summary = results.groupby("feature_set")[["r2_hours", "mae_h", "within_6h"]].agg(["mean", "std"])
    print("\nStability across origins")
    print(summary.to_string(float_format=lambda v: f"{v:0.3f}"))

    drift = manifest_drift(df)
    drift.to_csv(outdir / "aps_manifest_drift.csv")
    print("\nManifest coverage and signal by year")
    print(drift.to_string(float_format=lambda v: f"{v:0.3f}"))


if __name__ == "__main__":
    main()
