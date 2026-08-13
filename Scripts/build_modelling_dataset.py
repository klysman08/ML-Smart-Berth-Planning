"""Rebuild the modelling table from the raw port-call export, with an auditable
record of every filtering step.

This replaces the ad-hoc chain `previ.ipynb` -> `marge_datasets.py` with a single
deterministic function that (a) produces the same columns as the shipped
`Datasets/dataset_modelagem.csv` and (b) additionally keeps the join keys
(`Imo`, `Dock Timestamp`) needed for de-duplication, temporal windowing and
group-aware validation.

Run:
    uv run python Scripts/build_modelling_dataset.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"

# Port service craft: these are harbour tugs, pilot boats and fire-fighting
# vessels operated by the port itself. They berth for minutes, not hours, and
# they are not "vessel calls" in the berth-planning sense.
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

MODEL_COLUMNS = [
    "Berth Name",
    "Terminal Name",
    "Time At Berth",
    "Time At Port",
    "Vessel Type - Generic",
    "Commercial Market",
    "Voyage Distance Travelled",
    "Voyage Speed Average",
    "Year of build",
    "Voyage Origin Port",
    "Flag",
    "Gross tonnage",
    "Deadweight",
    "Length",
    "Breadth",
]

KEY_COLUMNS = ["Imo", "Dock Timestamp"]

# Columns delivered by the Baltic Shipping scrape as strings with a unit
# suffix ("54774 tons", "294 m"). The original notebooks pass these through
# LabelEncoder, which turns a continuous quantity into an arbitrary category
# ordered lexicographically. `parse_numeric=True` fixes that.
UNIT_COLUMNS = ["Gross tonnage", "Deadweight", "Length", "Breadth"]


@dataclass
class BuildLog:
    """Row counts after each filtering step, for the PRISMA-style flowchart."""

    steps: list[tuple[str, int, int]] = field(default_factory=list)

    def record(self, label: str, df: pd.DataFrame) -> pd.DataFrame:
        previous = self.steps[-1][1] if self.steps else len(df)
        self.steps.append((label, len(df), len(df) - previous))
        return df

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "records", "delta"])

    def to_markdown(self) -> str:
        lines = ["| Step | Records retained | Removed |", "| --- | ---: | ---: |"]
        for label, n, delta in self.steps:
            lines.append(f"| {label} | {n:,} | {0 if delta == 0 else delta:,} |")
        return "\n".join(lines)


def _parse_unit_column(series: pd.Series) -> pd.Series:
    """'54774 tons' -> 54774.0 ; '294 m' -> 294.0 ; '-' -> NaN."""
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw_calls(path: Path | None = None) -> pd.DataFrame:
    path = path or DATASETS / "SINES.csv"
    # utf-8-sig strips the BOM that otherwise corrupts the first column name.
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in ["Dock Timestamp", "Undock Timestamp", "Current Port Ata", "Current Port Atd"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_particulars(path: Path | None = None) -> pd.DataFrame:
    path = path or DATASETS / "df_imos_caracteristicas.csv"
    df = pd.read_csv(path)
    df["IMO number"] = pd.to_numeric(df["IMO number"], errors="coerce")
    return df.dropna(subset=["IMO number"]).drop_duplicates(subset=["IMO number"])


def build(
    *,
    window: tuple[str, str] | None = None,
    drop_service_craft: bool = False,
    dedup: str = "none",
    parse_numeric: bool = False,
    max_time_at_berth: float | None = None,
    log: BuildLog | None = None,
) -> tuple[pd.DataFrame, BuildLog]:
    """Build the modelling table.

    Parameters
    ----------
    window
        ``(start, end)`` half-open date bounds applied to ``Dock Timestamp``,
        e.g. ``("2023-01-01", "2024-01-01")``. ``None`` keeps the full export.
    drop_service_craft
        Remove tugs, pilot boats and fire-fighting vessels.
    dedup
        ``"none"``, ``"vessel_day"`` (one call per IMO per calendar day, the
        rule described in the manuscript) or ``"vessel"`` (one call per IMO
        across the whole window).
    parse_numeric
        Convert the unit-suffixed size columns to floats instead of leaving
        them as strings destined for LabelEncoder.
    max_time_at_berth
        Drop stays longer than this many hours (the manuscript uses 168 h for
        the descriptive distribution).
    """
    log = log or BuildLog()

    calls = load_raw_calls()
    log.record("Raw port-call export", calls)

    if window is not None:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        calls = calls[calls["Dock Timestamp"].between(start, end, inclusive="left")]
        log.record(f"Within window {window[0]} to {window[1]}", calls)

    calls = calls.dropna(subset=["Time At Port", "Time At Berth", "Voyage Speed Average"])
    log.record("Complete berth/port/voyage timings", calls)

    calls = calls[calls["Imo"] != 0]
    log.record("Valid IMO identifier", calls)

    if drop_service_craft:
        calls = calls[~calls["Vessel Type - Generic"].isin(SERVICE_CRAFT)]
        log.record("Commercial calls only (service craft removed)", calls)

    particulars = load_particulars()
    merged = calls.merge(particulars, left_on="Imo", right_on="IMO number", how="inner")
    log.record("Matched to vessel particulars (IMO join)", merged)

    keep = KEY_COLUMNS + MODEL_COLUMNS
    df = merged[[c for c in keep if c in merged.columns]].copy()

    if parse_numeric:
        for col in UNIT_COLUMNS:
            df[col] = _parse_unit_column(df[col])
        df["Year of build"] = pd.to_numeric(df["Year of build"], errors="coerce")

    df = df.dropna(subset=MODEL_COLUMNS)
    log.record("Complete cases across modelling features", df)

    df = df[df["Time At Berth"] > 0]
    log.record("Positive time at berth (log-transformable)", df)

    if max_time_at_berth is not None:
        df = df[df["Time At Berth"] <= max_time_at_berth]
        log.record(f"Time at berth <= {max_time_at_berth:g} h", df)

    if dedup == "vessel_day":
        df = df.sort_values("Dock Timestamp").assign(
            _call_date=lambda d: d["Dock Timestamp"].dt.date
        )
        df = df.drop_duplicates(subset=["Imo", "_call_date"], keep="first").drop(
            columns="_call_date"
        )
        log.record("One call per vessel per day", df)
    elif dedup == "vessel":
        df = df.sort_values("Dock Timestamp").drop_duplicates(subset=["Imo"], keep="first")
        log.record("One call per vessel in window", df)
    elif dedup != "none":
        raise ValueError(f"unknown dedup strategy: {dedup!r}")

    return df.reset_index(drop=True), log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--drop-service-craft", action="store_true")
    parser.add_argument("--dedup", default="none", choices=["none", "vessel_day", "vessel"])
    parser.add_argument("--parse-numeric", action="store_true")
    parser.add_argument("--max-time-at-berth", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df, log = build(
        window=tuple(args.window) if args.window else None,
        drop_service_craft=args.drop_service_craft,
        dedup=args.dedup,
        parse_numeric=args.parse_numeric,
        max_time_at_berth=args.max_time_at_berth,
    )

    print(log.to_markdown())
    print(f"\nFinal shape: {df.shape}")
    print(json.dumps({"unique_vessels": int(df['Imo'].nunique())}, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
