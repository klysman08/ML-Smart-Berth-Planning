"""Quantitative validation of the lakehouse blueprint.

Section 4.5 of the manuscript asserts that the architecture met its objectives
without measuring anything. This script produces the evidence: per-layer
runtime and volume, storage compaction, query latency against the Gold table
versus the raw sources, scalability across data volumes, and the data-quality
actions the pipeline took.

The point is not that a laptop rivals a cluster. It is that the same medallion
blueprint is portable: the design can be realised on a managed platform
(Delta Lake, Unity Catalog, Spark) or on open-source components on a single
machine (Parquet, DuckDB, scikit-learn, MLflow), and the governance properties
survive the move. That portability is the architectural contribution.

Run:
    uv run python Scripts/architecture_benchmark.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from aps_pipeline import DEFAULT_SOURCE, LAKE, bronze, gold, silver

ROOT = Path(__file__).resolve().parents[1]

# How each element of the Databricks implementation maps onto the local
# open-source realisation of the same blueprint.
BLUEPRINT = [
    ("Storage format", "Delta Lake", "Apache Parquet", "Columnar, typed, portable"),
    ("Processing", "Apache Spark", "pandas / DuckDB", "Single-node is sufficient at this volume"),
    ("Catalogue", "Unity Catalog", "Directory layout + schema manifest", "Lineage kept, governance simplified"),
    ("Layers", "Bronze / Silver / Gold", "Bronze / Silver / Gold", "Identical medallion semantics"),
    ("Experiment tracking", "Managed MLflow", "Local MLflow", "Same API, local backing store"),
    ("Orchestration", "Databricks Workflows", "Python entry points", "Same DAG, different scheduler"),
    ("Quality controls", "Delta constraints", "Bounds + quarantine in Silver", "Explicit and auditable"),
    ("Time travel", "Delta versioning", "Immutable dated snapshots", "Reproducibility retained"),
]


def scalability(source: Path, fractions=(0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Run the whole pipeline on progressively larger slices of the source."""
    tables = bronze(source)
    points = tables["points"].sort_values("event_time")
    rows = []

    for fraction in fractions:
        # Slice by call, not by row, so berth events stay paired.
        calls = points.call_id.dropna().unique()
        keep = set(calls[: int(len(calls) * fraction)])
        subset = dict(tables)
        subset["points"] = points[points.call_id.isin(keep)]

        started = time.perf_counter()
        visits, _ = silver(subset, max_hours=336.0)
        analytical, _ = gold(visits, subset, drop_service=True)
        elapsed = time.perf_counter() - started

        rows.append(
            {
                "fraction": fraction,
                "reporting_points": len(subset["points"]),
                "berth_visits": len(analytical),
                "seconds": elapsed,
                "points_per_second": len(subset["points"]) / elapsed,
            }
        )
    return pd.DataFrame(rows)


def query_latency(repeats: int = 5) -> pd.DataFrame:
    """Time the analytical question the Gold table exists to answer, against
    the curated table and against the raw sources."""
    rows = []

    gold_path = LAKE / "gold_berth_time.parquet"
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        df = pd.read_parquet(gold_path)
        df.groupby("berth_location")["time_at_berth_h"].agg(["count", "median"])
        times.append(time.perf_counter() - started)
    rows.append({"source": "Gold table (Parquet)", "seconds": float(np.mean(times))})

    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        tables = bronze(DEFAULT_SOURCE)
        visits, _ = silver(tables, max_hours=336.0)
        analytical, _ = gold(visits, tables, drop_service=True)
        analytical.groupby("berth_location")["time_at_berth_h"].agg(["count", "median"])
        times.append(time.perf_counter() - started)
    rows.append({"source": "Raw CSV sources (full recompute)", "seconds": float(np.mean(times))})

    frame = pd.DataFrame(rows)
    frame["speedup"] = frame.seconds.max() / frame.seconds
    return frame


def storage() -> pd.DataFrame:
    rows = []
    for path in sorted(DEFAULT_SOURCE.glob("*.csv")):
        rows.append({"artefact": f"source/{path.name}", "layer": "source", "mb": path.stat().st_size / 1e6})
    for path in sorted(LAKE.glob("*.parquet")):
        layer = path.stem.split("_")[0]
        rows.append({"artefact": path.name, "layer": layer, "mb": path.stat().st_size / 1e6})
    return pd.DataFrame(rows)


def main(outdir: Path = ROOT / "reports") -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    blueprint = pd.DataFrame(
        BLUEPRINT, columns=["Concern", "Managed platform", "Open-source local", "Note"]
    )
    blueprint.to_csv(outdir / "architecture_blueprint.csv", index=False)
    print("Blueprint portability\n")
    print(blueprint.to_string(index=False))

    scale = scalability(DEFAULT_SOURCE)
    scale.to_csv(outdir / "architecture_scalability.csv", index=False)
    print("\n\nScalability\n")
    print(scale.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
    growth = np.polyfit(np.log(scale.reporting_points), np.log(scale.seconds), 1)[0]
    print(f"\nEmpirical scaling exponent: runtime ~ n^{growth:0.2f} (1.0 = linear)")

    latency = query_latency()
    latency.to_csv(outdir / "architecture_query_latency.csv", index=False)
    print("\n\nAnalytical query latency\n")
    print(latency.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    space = storage()
    space.to_csv(outdir / "architecture_storage.csv", index=False)
    by_layer = space.groupby("layer").mb.sum()
    print("\n\nStorage by layer (MB)\n")
    print(by_layer.to_string(float_format=lambda v: f"{v:0.3f}"))
    print(f"\nSource -> Gold compaction: {by_layer.get('source', 0) / by_layer.get('gold', 1):0.1f}x")


if __name__ == "__main__":
    main()
