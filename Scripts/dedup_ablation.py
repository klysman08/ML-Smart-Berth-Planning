"""Isolate the two decisions that reviewers ask about:

  1. de-duplicating repeated vessels inside the modelling window, and
  2. keeping `Time At Port` as a predictor.

Everything else is held constant (commercial calls only, size fields parsed as
numbers, z-score outlier removal, Random Forest, 5 seeds). Both a random 80/20
split and a vessel-grouped 80/20 split are reported, because a random split
lets the *same vessel* appear in train and test and that is the optimism the
de-duplication step was meant to remove.

Run:
    uv run python Scripts/dedup_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_modelling_dataset import build
from run_experiments import evaluate, prepare, strip_outliers

MODEL = "Random Forest (500 trees)"
DEDUP_LABELS = {
    "none": "all calls kept",
    "vessel_day": "1 call / vessel / day",
    "vessel": "1 call / vessel",
}


def main(outdir: Path = Path("reports")) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dedup, dedup_label in DEDUP_LABELS.items():
        df, _ = build(drop_service_craft=True, dedup=dedup, parse_numeric=True)
        for use_tap in (True, False):
            X, y, g = prepare(df, use_time_at_port=use_tap)
            X, y, g = strip_outliers(X, y, g)
            for grouped in (False, True):
                row = evaluate(X, y, g, model_name=MODEL, grouped=grouped)
                row["dedup"] = dedup_label
                row["time_at_port"] = "kept" if use_tap else "removed"
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "dedup_ablation.csv", index=False)

    view = pd.DataFrame(
        {
            "de-duplication": frame["dedup"],
            "Time At Port": frame["time_at_port"],
            "split": frame["split"],
            "n": frame["n"],
            "vessels": frame["n_vessels"],
            "R2 (log)": frame["r2_log_mean"],
            "+-": frame["r2_log_sd"],
            "RMSE (h)": frame["rmse_hours_mean"],
            "MedAE (h)": frame["median_ae_hours_mean"],
        }
    )
    print(view.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))


if __name__ == "__main__":
    main()
