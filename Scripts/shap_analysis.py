"""SHAP interpretation of the honest (leakage-free) model.

The thesis-era SHAP figure is dominated by `Time At Port`, which is measured
concurrently with the target. This script re-runs the interpretation without it
so the reported drivers are ones a planner actually knows before the vessel
berths.

Run:
    uv run python Scripts/shap_analysis.py
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
from sklearn.model_selection import GroupShuffleSplit

from build_modelling_dataset import build
from run_experiments import prepare, strip_outliers


def main(outdir: Path = Path("reports"), seed: int = 42) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df, _ = build(drop_service_craft=True, dedup="vessel_day", parse_numeric=True)
    X, y, groups = prepare(df, use_time_at_port=False)
    X, y, groups = strip_outliers(X, y, groups)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    model = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    model.fit(X.iloc[tr], y.iloc[tr])

    values = shap.TreeExplainer(model).shap_values(X.iloc[te])
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)

    impurity = pd.Series(model.feature_importances_, index=X.columns)
    table = pd.DataFrame(
        {"mean_abs_shap": mean_abs, "impurity_importance": impurity.reindex(mean_abs.index)}
    )
    table.to_csv(outdir / "feature_importance_no_leakage.csv")

    shap.summary_plot(values, X.iloc[te], show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_no_leakage.png", dpi=200)
    plt.close()

    print(table.to_string(float_format=lambda v: f"{v:0.4f}"))


if __name__ == "__main__":
    main()
