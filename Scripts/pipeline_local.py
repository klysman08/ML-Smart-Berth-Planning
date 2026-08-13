"""Local medallion pipeline (Bronze -> Silver -> Gold) over the port-call export.

Replaces the lost Databricks/Delta implementation with a single-node pipeline
built on Parquet, and — unlike the original — instruments itself. Every layer
records wall-clock time, row and byte volume, and the data-quality actions it
took. Those measurements are the quantitative architecture evidence Reviewer 2
asked for and that the manuscript currently asserts without numbers.

Layers
------
Bronze  raw ingestion, no semantic change: types parsed, column names
        standardised, source file and ingestion timestamp recorded.
Silver  quality assessment and cleansing: service craft separated from
        commercial calls, impossible values quarantined, missingness profiled.
Gold    analytics-ready vessel-call table: one row per call, target computed,
        features engineered and tiered by when they become knowable.

Run:
    uv run python Scripts/pipeline_local.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"
LAKE = ROOT / "lakehouse"

SERVICE_CRAFT = {
    "Tug",
    "Pilot Vessel",
    "Fire Fighting Vessel",
    "SAR",
    "Dredger",
    "Pleasure Craft",
    "Law Enforce",
    "Inland, Pleasure Craft, >20 metres",
}

TIMESTAMP_COLUMNS = [
    "Dock Timestamp",
    "Undock Timestamp",
    "Current Port Ata",
    "Current Port Atd",
    "Voyage Origin Port Atd",
]

# Physical plausibility bounds. Values outside these are quarantined rather than
# dropped silently, so the Silver layer can report what it rejected.
BOUNDS = {
    "Time At Berth": (0.0, 24 * 30),
    "Time At Port": (0.0, 24 * 60),
    "Draught At Arrival": (0.5, 30.0),
    "Draught At Departure": (0.5, 30.0),
    "Voyage Speed Average": (0.0, 40.0),
    "Voyage Speed Max": (0.0, 50.0),
    "Voyage Distance Travelled": (0.0, 30000.0),
}


@dataclass
class LayerMetrics:
    layer: str
    rows_in: int
    rows_out: int
    bytes_out: int
    seconds: float
    actions: dict = field(default_factory=dict)


class PipelineRun:
    def __init__(self, root: Path = LAKE):
        self.root = root
        self.metrics: list[LayerMetrics] = []
        for layer in ("bronze", "silver", "gold"):
            (self.root / layer).mkdir(parents=True, exist_ok=True)

    def _write(self, df: pd.DataFrame, layer: str, name: str) -> int:
        path = self.root / layer / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        return path.stat().st_size

    def record(self, layer, rows_in, rows_out, nbytes, seconds, **actions) -> None:
        self.metrics.append(LayerMetrics(layer, rows_in, rows_out, nbytes, seconds, actions))

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "layer": m.layer,
                    "rows_in": m.rows_in,
                    "rows_out": m.rows_out,
                    "bytes_out": m.bytes_out,
                    "mb_out": round(m.bytes_out / 1e6, 3),
                    "seconds": round(m.seconds, 3),
                    "rows_per_second": round(m.rows_in / m.seconds) if m.seconds else None,
                    **m.actions,
                }
                for m in self.metrics
            ]
        )


# --------------------------------------------------------------------------
# Bronze
# --------------------------------------------------------------------------
def bronze(run: PipelineRun, source: Path | None = None) -> pd.DataFrame:
    source = source or DATASETS / "SINES.csv"
    started = time.perf_counter()

    # utf-8-sig strips the BOM that otherwise corrupts the first column name.
    df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    rows_in = len(df)

    df.columns = [c.strip() for c in df.columns]
    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["_source_file"] = source.name
    df["_ingested_at"] = pd.Timestamp.utcnow().tz_localize(None)

    nbytes = run._write(df, "bronze", "vessel_calls")
    run.record(
        "bronze",
        rows_in,
        len(df),
        nbytes,
        time.perf_counter() - started,
        columns=df.shape[1],
        timestamps_parsed=len([c for c in TIMESTAMP_COLUMNS if c in df.columns]),
    )
    return df


# --------------------------------------------------------------------------
# Silver
# --------------------------------------------------------------------------
def silver(run: PipelineRun, raw: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
    rows_in = len(raw)
    df = raw.copy()

    service = df["Vessel Type - Generic"].isin(SERVICE_CRAFT)
    n_service = int(service.sum())
    df = df[~service]

    quarantined = 0
    for col, (low, high) in BOUNDS.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ~df[col].between(low, high)
        quarantined += int(bad.sum())
        df.loc[bad, col] = np.nan

    # A berth interval must sit inside its port call; records violating this are
    # inconsistent at source and cannot be used to define the target.
    inconsistent = (
        df["Dock Timestamp"].notna()
        & df["Current Port Ata"].notna()
        & (df["Dock Timestamp"] < df["Current Port Ata"])
    )
    n_inconsistent = int(inconsistent.sum())
    df = df[~inconsistent]

    before = len(df)
    df = df.drop_duplicates(subset=["Imo", "Dock Timestamp"], keep="first")
    n_dupes = before - len(df)

    missing = df.isna().mean().sort_values(ascending=False)
    (run.root / "silver" / "missingness.json").write_text(
        json.dumps({k: round(float(v), 4) for k, v in missing.items()}, indent=2),
        encoding="utf-8",
    )

    nbytes = run._write(df, "silver", "commercial_calls")
    run.record(
        "silver",
        rows_in,
        len(df),
        nbytes,
        time.perf_counter() - started,
        service_craft_removed=n_service,
        values_quarantined=quarantined,
        inconsistent_intervals_removed=n_inconsistent,
        exact_duplicates_removed=n_dupes,
    )
    return df


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------
def gold(run: PipelineRun, curated: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
    rows_in = len(curated)
    df = curated.copy()

    df = df.dropna(subset=["Time At Berth", "Dock Timestamp", "Current Port Ata"])
    df = df[df["Time At Berth"] > 0]

    ata = df["Current Port Ata"]
    df["arrival_hour"] = ata.dt.hour
    df["arrival_dayofweek"] = ata.dt.dayofweek
    df["arrival_month"] = ata.dt.month
    df["arrival_is_weekend"] = (ata.dt.dayofweek >= 5).astype(int)

    # Hours spent at anchor / in the port area before the vessel got a berth.
    # Known the moment berthing starts, so legitimate for a berth-time model.
    df["wait_before_berth_h"] = (
        df["Dock Timestamp"] - df["Current Port Ata"]
    ).dt.total_seconds() / 3600
    df["wait_before_berth_h"] = df["wait_before_berth_h"].clip(lower=0)

    # Port-internal quantities: the cargo actually exchanged is only observable
    # once the vessel departs. Kept in the Gold table but tagged, so the
    # modelling layer can quantify their value without using them by accident.
    df["cargo_draught_delta"] = df["Draught At Departure"] - df["Draught At Arrival"]
    df["cargo_draught_delta_abs"] = df["cargo_draught_delta"].abs()

    df = df.sort_values("Dock Timestamp")
    before = len(df)
    df = df.assign(_d=df["Dock Timestamp"].dt.date)
    df = df.drop_duplicates(subset=["Imo", "_d"], keep="first").drop(columns="_d")
    n_same_day = before - len(df)

    nbytes = run._write(df, "gold", "berth_time")
    run.record(
        "gold",
        rows_in,
        len(df),
        nbytes,
        time.perf_counter() - started,
        engineered_features=7,
        same_day_repeat_calls_removed=n_same_day,
        unique_vessels=int(df["Imo"].nunique()),
    )
    return df.reset_index(drop=True)


def run_pipeline(outdir: Path = ROOT / "reports") -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    run = PipelineRun()

    raw = bronze(run)
    curated = silver(run, raw)
    analytical = gold(run, curated)

    metrics = run.metrics_frame()
    metrics.to_csv(outdir / "pipeline_metrics.csv", index=False)

    raw_bytes = (DATASETS / "SINES.csv").stat().st_size
    gold_bytes = int(metrics.loc[metrics.layer == "gold", "bytes_out"].iloc[0])
    print(metrics.to_string(index=False))
    print(
        f"\nsource CSV {raw_bytes/1e6:.2f} MB -> Gold parquet {gold_bytes/1e6:.2f} MB "
        f"({raw_bytes/gold_bytes:.1f}x smaller)"
    )
    print(f"end-to-end: {metrics['seconds'].sum():.2f} s")
    print(f"Gold table: {len(analytical)} calls, {analytical['Imo'].nunique()} vessels")
    return analytical, metrics


if __name__ == "__main__":
    run_pipeline()
