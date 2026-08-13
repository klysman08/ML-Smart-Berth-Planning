"""Diagnostics for the APS berth-time model.

Two things the manuscript needs and does not currently have:

1. A metric-space consistency check. R2 and RMSE are only mutually consistent
   if RMSE ~= sd(y) * sqrt(1 - R2) on the *same* scale. The submitted paper
   pairs R2 = 0.311 with RMSE = 10.03 h, which implies sd(y) ~= 12.1 h. This
   script prints the sd of the target under several outlier policies so the
   pairing can be confirmed or corrected rather than argued about.

2. SHAP attribution for the model that is actually reported.

Run:
    uv run python Scripts/aps_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from aps_model import FEATURE_SETS, GOLD, TARGET, build_preprocessor, score

ROOT = Path(__file__).resolve().parents[1]


def variance_check(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    log_y = np.log(y)
    policies = {
        "all visits (<= 336 h)": y,
        "<= 168 h": y[y <= 168],
        "z-score 3 on hours": y[np.abs((y - y.mean()) / y.std()) <= 3],
        "z-score 3 on log hours": y[np.abs((log_y - log_y.mean()) / log_y.std()) <= 3],
    }
    rows = []
    for name, sample in policies.items():
        sd = sample.std()
        rows.append(
            {
                "outlier policy": name,
                "n": len(sample),
                "mean_h": sample.mean(),
                "median_h": sample.median(),
                "sd_h": sd,
                "rmse_at_r2_0.311": sd * np.sqrt(1 - 0.311),
            }
        )
    return pd.DataFrame(rows)


def main(outdir: Path = ROOT / "reports", seed: int = 42) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(GOLD).dropna(subset=[TARGET])

    check = variance_check(df)
    check.to_csv(outdir / "aps_metric_consistency.csv", index=False)
    print("Metric-space consistency: what RMSE an R2 of 0.311 implies\n")
    print(check.to_string(index=False, float_format=lambda v: f"{v:0.2f}"))

    columns = FEATURE_SETS["+ vessel history"]
    cut = pd.Timestamp("2024-01-01")
    train, test = df[df.berthed_at < cut], df[df.berthed_at >= cut]

    pipe = Pipeline(
        [
            ("prep", build_preprocessor(columns)),
            ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                            random_state=seed, n_jobs=-1)),
        ]
    )
    pipe.fit(train[columns], np.log(train[TARGET]))
    predicted = np.exp(pipe.predict(test[columns]))
    metrics = score(test[TARGET].to_numpy(), predicted)
    print("\nTemporal hold-out (train <2024, test 2024), Random Forest + vessel history")
    for k, v in metrics.items():
        print(f"  {k:14s} {v:0.3f}")

    matrix = pipe.named_steps["prep"].transform(test[columns])
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    sample = shap.utils.sample(matrix, min(600, len(matrix)), random_state=seed)
    values = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(sample)

    importance = (
        pd.Series(np.abs(values).mean(axis=0), index=names)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
    )
    importance.to_csv(outdir / "aps_shap_importance.csv")
    print("\nTop predictors (mean |SHAP|, log-hours scale)")
    print(importance.head(12).to_string(float_format=lambda v: f"{v:0.4f}"))

    shap.summary_plot(values, sample, feature_names=names, show=False, plot_size=(10, 7))
    plt.tight_layout()
    plt.savefig(outdir / "aps_shap_summary.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
