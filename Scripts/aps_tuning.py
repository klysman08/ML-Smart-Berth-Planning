"""Optuna hyperparameter search under temporal validation, and the final
numeric benchmark that replaces the manuscript's qualitative Table 6.

Design choices that matter for defensibility:

* Tuning is scored on a validation period that sits *before* the test period,
  never on the test period itself. The order is train -> validate -> test in
  time, so no future information reaches model selection.
* Every model family gets the same trial budget, so the comparison between
  Random Forest and Extra Trees is decided by performance rather than by how
  much search each one received. Two reviewers asked why Random Forest was
  chosen; this is the evidence that settles it.
* Both tuned and untuned results are reported, because the manuscript's
  original benchmark was untuned and the comparison should be like-for-like.

Run:
    uv run python Scripts/aps_tuning.py --trials 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline

from aps_model import FEATURE_SETS, GOLD, TARGET, build_preprocessor, score

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SET = "+ vessel history"
TRAIN_END = pd.Timestamp("2023-07-01")
VALID_END = pd.Timestamp("2024-01-01")


def _search_space(trial: optuna.Trial, family: str) -> dict:
    if family in {"Random Forest", "Extra Trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 6, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
    if family == "HistGradientBoosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 150, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
    }


def _build(family: str, params: dict, seed: int):
    common = dict(random_state=seed)
    if family == "Random Forest":
        return RandomForestRegressor(**params, **common, n_jobs=-1)
    if family == "Extra Trees":
        return ExtraTreesRegressor(**params, **common, n_jobs=-1)
    if family == "HistGradientBoosting":
        return HistGradientBoostingRegressor(**params, **common)
    return GradientBoostingRegressor(**params, **common)


DEFAULTS = {
    "Random Forest": dict(n_estimators=500, min_samples_leaf=2),
    "Extra Trees": dict(n_estimators=500, min_samples_leaf=2),
    "HistGradientBoosting": dict(max_iter=400, learning_rate=0.06),
    "Gradient Boosting": dict(),
}


def tune(family: str, splits: dict, columns: list[str], trials: int, seed: int = 42):
    train, valid = splits["train"], splits["valid"]

    def objective(trial: optuna.Trial) -> float:
        model = _build(family, _search_space(trial, family), seed)
        pipe = Pipeline([("prep", build_preprocessor(columns)), ("model", model)])
        pipe.fit(train[columns], np.log(train[TARGET]))
        predicted = np.exp(pipe.predict(valid[columns]))
        # Optimise MAE in hours: it is the quantity a berth planner feels, and
        # it is robust to the long right tail of berth stays.
        return float(np.mean(np.abs(valid[TARGET].to_numpy() - predicted)))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main(trials: int, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(GOLD).dropna(subset=[TARGET]).sort_values("berthed_at")
    columns = FEATURE_SETS[FEATURE_SET]

    splits = {
        "train": df[df.berthed_at < TRAIN_END],
        "valid": df[(df.berthed_at >= TRAIN_END) & (df.berthed_at < VALID_END)],
        "test": df[df.berthed_at >= VALID_END],
    }
    print(
        f"train {len(splits['train'])} (<{TRAIN_END:%Y-%m}) | "
        f"valid {len(splits['valid'])} | test {len(splits['test'])} (>={VALID_END:%Y-%m})\n"
    )

    # Final models are refit on train+valid so the test period is predicted
    # from every observation that precedes it.
    fit_pool = pd.concat([splits["train"], splits["valid"]])
    rows, best_params = [], {}

    for family in DEFAULTS:
        for mode in ("untuned", "tuned"):
            if mode == "untuned":
                params = DEFAULTS[family]
            else:
                params, valid_mae = tune(family, splits, columns, trials)
                best_params[family] = {"params": params, "validation_mae_h": valid_mae}
                print(f"{family:22s} best validation MAE {valid_mae:0.2f} h  {params}")

            runs = []
            for seed in (42, 7, 13):
                pipe = Pipeline(
                    [
                        ("prep", build_preprocessor(columns)),
                        ("model", _build(family, params, seed)),
                    ]
                )
                pipe.fit(fit_pool[columns], np.log(fit_pool[TARGET]))
                predicted = np.exp(pipe.predict(splits["test"][columns]))
                runs.append(score(splits["test"][TARGET].to_numpy(), predicted))

            frame = pd.DataFrame(runs)
            rows.append(
                {
                    "model": family,
                    "mode": mode,
                    **{k: float(frame[k].mean()) for k in frame.columns},
                    "r2_sd": float(frame["r2_hours"].std()),
                }
            )

    result = pd.DataFrame(rows).sort_values(["mode", "r2_hours"], ascending=[True, False])
    result.to_csv(outdir / "aps_tuned_benchmark.csv", index=False)
    (outdir / "aps_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print("\nFinal benchmark on the held-out test period (mean of 3 seeds)")
    print(
        result[
            ["model", "mode", "r2_hours", "r2_sd", "rmse_h", "mae_h", "median_ae_h",
             "within_6h", "within_12h"]
        ].to_string(index=False, float_format=lambda v: f"{v:0.3f}")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    main(args.trials, args.out)
