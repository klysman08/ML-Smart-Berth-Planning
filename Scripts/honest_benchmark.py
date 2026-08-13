"""Model benchmark on the leakage-free configuration.

`run_experiments.py` benchmarks on the configuration as the thesis defined it,
which keeps `Time At Port` and therefore flatters every model equally. This
script repeats the comparison with that feature removed, which is the table the
manuscript should report: commercial calls only, numeric size fields, one call
per vessel per day, vessel-grouped 80/20 hold-out, 5 seeds.

Run:
    uv run python Scripts/honest_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_modelling_dataset import build
from run_experiments import cv_r2, evaluate, model_zoo, prepare, strip_outliers


def main(outdir: Path = Path("reports")) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df, _ = build(drop_service_craft=True, dedup="vessel_day", parse_numeric=True)
    X, y, g = prepare(df, use_time_at_port=False)
    X, y, g = strip_outliers(X, y, g)

    rows = []
    for name in model_zoo(42):
        row = evaluate(X, y, g, model_name=name, grouped=True)
        row.update(cv_r2(X, y, g, model_name=name, grouped=True))
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("r2_log_mean", ascending=False)
    frame.to_csv(outdir / "honest_benchmark.csv", index=False)

    view = pd.DataFrame(
        {
            "model": frame["model"],
            "R2 (log)": frame["r2_log_mean"],
            "+-": frame["r2_log_sd"],
            "CV R2": frame["cv_mean"],
            "R2 (hours)": frame["r2_hours_mean"],
            "RMSE (h)": frame["rmse_hours_mean"],
            "MAE (h)": frame["mae_hours_mean"],
        }
    )
    print(f"n = {len(X)} calls, {g.nunique()} vessels, Time At Port excluded\n")
    print(view.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))


if __name__ == "__main__":
    main()
