"""Answer the reviewers' operational question: is an error of ~10 h useful?

An R2 in isolation says nothing to a berth planner. What matters is whether the
model beats the heuristics a planner already has for free -- "assume the median
stay", "assume the median stay for this vessel type", "assume the median stay at
this terminal". This script compares the model against those baselines on the
same vessel-grouped hold-out folds, and reports the error distribution in hours
alongside the typical stay, so the manuscript can state the comparison honestly.

Run:
    uv run python Scripts/operational_value.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from build_modelling_dataset import build
from run_experiments import SEEDS, prepare, strip_outliers

TARGET = "Time At Berth"


def _heuristic(train: pd.DataFrame, test: pd.DataFrame, by: str | None) -> np.ndarray:
    if by is None:
        return np.full(len(test), train[TARGET].median())
    lookup = train.groupby(by)[TARGET].median()
    fallback = train[TARGET].median()
    return test[by].map(lookup).fillna(fallback).to_numpy()


def main(outdir: Path = Path("reports")) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Honest configuration: commercial calls only, numeric size fields,
    # `Time At Port` excluded because it is measured concurrently with the
    # target and is unknown at prediction time.
    df, _ = build(drop_service_craft=True, dedup="vessel_day", parse_numeric=True)
    X, y, groups = prepare(df, use_time_at_port=False)
    X, y, groups = strip_outliers(X, y, groups)
    raw = df.loc[X.index]

    rows = []
    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(splitter.split(X, y, groups))
        train_raw, test_raw = raw.iloc[tr], raw.iloc[te]
        actual = test_raw[TARGET].to_numpy()

        predictions = {
            "Global median": _heuristic(train_raw, test_raw, None),
            "Median by vessel type": _heuristic(train_raw, test_raw, "Vessel Type - Generic"),
            "Median by terminal": _heuristic(train_raw, test_raw, "Terminal Name"),
        }
        model = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
        model.fit(X.iloc[tr], y.iloc[tr])
        predictions["Random Forest"] = np.exp(model.predict(X.iloc[te]))

        for name, pred in predictions.items():
            err = actual - pred
            rows.append(
                {
                    "seed": seed,
                    "estimator": name,
                    "rmse_h": float(np.sqrt(np.mean(err**2))),
                    "mae_h": float(np.mean(np.abs(err))),
                    "median_ae_h": float(np.median(np.abs(err))),
                    "within_6h": float(np.mean(np.abs(err) <= 6)),
                    "within_12h": float(np.mean(np.abs(err) <= 12)),
                }
            )

    frame = pd.DataFrame(rows)
    summary = frame.groupby("estimator").mean(numeric_only=True).drop(columns="seed")
    summary = summary.sort_values("mae_h")
    summary.to_csv(outdir / "operational_value.csv")

    stay = raw[TARGET]
    print("Typical stay in the modelling sample (hours)")
    print(
        f"  median {stay.median():.1f} | mean {stay.mean():.1f} | "
        f"IQR {stay.quantile(0.25):.1f}-{stay.quantile(0.75):.1f} | sd {stay.std():.1f}\n"
    )
    print("Vessel-grouped hold-out, mean over 5 seeds:")
    print(summary.to_string(float_format=lambda v: f"{v:0.3f}"))


if __name__ == "__main__":
    main()
