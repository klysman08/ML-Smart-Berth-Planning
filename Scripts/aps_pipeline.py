"""Medallion pipeline over the APS (Janela Unica Logistica) source extracts.

This is the manuscript's actual data lineage, rebuilt to run locally on a single
machine after the Databricks workspace was lost. The source CSVs are the same
ones the `nexus-databricks` notebooks ingested from the SharePoint volume:

    FN_APS_JUL.csv                ship particulars, one row per IMO
    PR_JUL.csv                    reporting points (berth, unberth, anchor, pilot)
    ROP_JUL.csv                   equipment/operations report (sparse)
    Ton_movimentadas_navio_jul.csv  manifested tonnage per call

Unlike the lost implementation this one derives time at berth the way the
manuscript describes it — from the berthing and unberthing events — instead of
from the equipment-usage timestamps used in `06/07_ML_Experiment`.

Bronze  parse and standardise the four sources, no semantic change.
Silver  pair Atracar/Largar events into berth visits, quality-filter, profile.
Gold    join particulars and manifested tonnage, engineer arrival-time features.

Run:
    uv run python Scripts/aps_pipeline.py --source <dir-with-the-four-csvs>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "Datasets" / "aps"
LAKE = ROOT / "lakehouse_aps"

BERTH_EVENT, UNBERTH_EVENT = "Atracar", "Largar"
ANCHOR_EVENT, WEIGH_EVENT = "Fundear", "Suspender"
PORT_IN_EVENT, PORT_OUT_EVENT = "PLF - Entrada", "PLF - Saída"

# Port service craft, by APS vessel-type nomenclature.
SERVICE_TYPES = {
    "Rebocador (sem reboque)",
    "Rebocador",
    "Rebocador / Empurrador",
    "Embarcação de Pilotagem",
    "Embarcação de Serviço",
    "Draga",
    "Embarcação de Recreio",
}

PARTICULARS = {
    "Porte Bruto": "deadweight_t",
    "Arqueação Bruta (GT)": "gross_tonnage",
    "Arqueação Líquida (NT) (t)": "net_tonnage",
    "Boca de Sinal (m)": "beam_m",
    "CFF (LOA) (m)": "loa_m",
    "Calado Máximo (Summer Draft) (m)": "summer_draft_m",
    "Velocidade máxima (nós)": "max_speed_kn",
    "Capacidade de Teus": "teu_capacity",
    "Tipo de Navio": "vessel_type",
    "Tipo de Navegação": "navigation_type",
    "Energia de Propulsão": "propulsion",
    "Data de Construção": "build_date",
}


def _read(path: Path) -> pd.DataFrame:
    """APS exports are semicolon-separated Latin-1 with stray '?' in headers."""
    df = pd.read_csv(path, sep=";", encoding="ISO-8859-1", low_memory=False)
    df.columns = [c.replace("?", "").strip() for c in df.columns]
    return df


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d/%m/%Y %H:%M", errors="coerce")


# --------------------------------------------------------------------------
# Bronze
# --------------------------------------------------------------------------
def bronze(source: Path) -> dict[str, pd.DataFrame]:
    points = _read(source / "PR_JUL.csv").rename(
        columns={
            "Escala navio": "call_id",
            "Tipo de Ponto de Relato": "event_type",
            "Local do Ponto de Relato": "location",
            "Hora do Ponto de Relato": "event_time",
            "Número_IMO_Navio": "imo",
        }
    )
    points["event_time"] = _ts(points["event_time"])
    points["imo"] = pd.to_numeric(points["imo"], errors="coerce")

    ships = _read(source / "FN_APS_JUL.csv")
    ships["imo"] = pd.to_numeric(ships["Número IMO"], errors="coerce")
    ships = ships.dropna(subset=["imo"]).drop_duplicates("imo")
    ships = ships[["imo", *PARTICULARS]].rename(columns=PARTICULARS)
    for col in [
        "deadweight_t",
        "gross_tonnage",
        "net_tonnage",
        "beam_m",
        "loa_m",
        "summer_draft_m",
        "max_speed_kn",
        "teu_capacity",
    ]:
        ships[col] = pd.to_numeric(
            ships[col].astype("string").str.replace(",", ".", regex=False), errors="coerce"
        )
    ships["build_year"] = _ts(ships["build_date"]).dt.year
    ships = ships.drop(columns="build_date")

    tonnage = _read(source / "Ton_movimentadas_navio_jul.csv").rename(
        columns={"Número de Escala": "call_id", "Toneladas Manifestadas": "manifested_kg"}
    )
    tonnage["manifested_kg"] = pd.to_numeric(tonnage["manifested_kg"], errors="coerce")
    tonnage = tonnage.groupby("call_id", as_index=False)["manifested_kg"].sum()

    return {"points": points, "ships": ships, "tonnage": tonnage}


# --------------------------------------------------------------------------
# Silver — pair berth events into visits
# --------------------------------------------------------------------------
def _pair_visits(points: pd.DataFrame) -> pd.DataFrame:
    """Match each Atracar to the next Largar within the same call.

    A call can contain several berth visits (shifting between berths), so a
    stack-free forward scan per call is the correct pairing, not a groupby-min.
    """
    events = points[points.event_type.isin([BERTH_EVENT, UNBERTH_EVENT])]
    events = events.sort_values(["call_id", "event_time"])

    visits = []
    for call_id, group in events.groupby("call_id", sort=False):
        open_berth = None
        index = 0
        for row in group.itertuples():
            if row.event_type == BERTH_EVENT:
                open_berth = row
            elif open_berth is not None:
                index += 1
                visits.append(
                    {
                        "call_id": call_id,
                        "imo": open_berth.imo,
                        "berth_location": open_berth.location,
                        "berthed_at": open_berth.event_time,
                        "unberthed_at": row.event_time,
                        "visit_index": index,
                    }
                )
                open_berth = None
    df = pd.DataFrame(visits)
    df["time_at_berth_h"] = (
        df["unberthed_at"] - df["berthed_at"]
    ).dt.total_seconds() / 3600
    return df


def _call_context(points: pd.DataFrame) -> pd.DataFrame:
    """Per-call timestamps that are known before the vessel takes the berth."""

    def first_time(event: str) -> pd.Series:
        sub = points[points.event_type == event]
        return sub.groupby("call_id")["event_time"].min()

    context = pd.DataFrame(
        {
            "port_entry_at": first_time(PORT_IN_EVENT),
            "anchored_at": first_time(ANCHOR_EVENT),
            "weighed_anchor_at": first_time(WEIGH_EVENT),
        }
    )
    anchorage = points[points.event_type == ANCHOR_EVENT].groupby("call_id")["location"].first()
    context["anchorage"] = anchorage
    return context.reset_index()


def silver(tables: dict[str, pd.DataFrame], *, max_hours: float) -> tuple[pd.DataFrame, dict]:
    points = tables["points"].dropna(subset=["event_time", "call_id"])
    visits = _pair_visits(points)
    stats = {"paired_visits": len(visits)}

    visits = visits[visits.time_at_berth_h > 0]
    stats["after_positive_duration"] = len(visits)

    visits = visits[visits.time_at_berth_h <= max_hours]
    stats["after_max_stay_filter"] = len(visits)

    visits = visits.merge(_call_context(points), on="call_id", how="left")
    return visits, stats


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------
def gold(visits: pd.DataFrame, tables: dict[str, pd.DataFrame], *, drop_service: bool) -> tuple[pd.DataFrame, dict]:
    df = visits.merge(tables["ships"], on="imo", how="left")
    stats = {"matched_particulars": int(df.vessel_type.notna().sum())}

    if drop_service:
        df = df[~df.vessel_type.isin(SERVICE_TYPES)]
        stats["after_service_craft_removed"] = len(df)

    df = df.merge(tables["tonnage"], on="call_id", how="left")
    stats["with_manifested_tonnage"] = int(df.manifested_kg.notna().sum())

    # Manifested cargo is declared before the call, so it is legitimately known
    # at berthing time. Stored in tonnes; a null means "not declared", which is
    # itself informative and is flagged rather than imputed away.
    df["manifested_tonnes"] = df["manifested_kg"] / 1000.0
    df["has_manifest"] = df["manifested_kg"].notna().astype(int)

    berthed = df["berthed_at"]
    df["berth_hour"] = berthed.dt.hour
    df["berth_dayofweek"] = berthed.dt.dayofweek
    df["berth_month"] = berthed.dt.month
    df["berth_year"] = berthed.dt.year
    df["berth_is_weekend"] = (berthed.dt.dayofweek >= 5).astype(int)

    # Waiting already elapsed when the vessel takes the berth: both are known.
    df["wait_from_port_entry_h"] = (
        (berthed - df["port_entry_at"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    df["anchorage_hours"] = (
        (df["weighed_anchor_at"] - df["anchored_at"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    df["was_anchored"] = df["anchored_at"].notna().astype(int)

    df["vessel_age_years"] = df["berth_year"] - df["build_year"]
    df["cargo_intensity"] = df["manifested_tonnes"] / df["deadweight_t"].replace(0, np.nan)

    # Lagged vessel history: the previous stay of the same ship, strictly
    # earlier in time. Legitimate, and the closest honest analogue of the
    # manuscript's "prior time at port".
    df = df.sort_values("berthed_at")
    df["prev_time_at_berth_h"] = df.groupby("imo")["time_at_berth_h"].shift(1)
    df["prev_call_count"] = df.groupby("imo").cumcount()

    stats["final_rows"] = len(df)
    stats["unique_vessels"] = int(df.imo.nunique())
    stats["unique_calls"] = int(df.call_id.nunique())
    return df.reset_index(drop=True), stats


def run(source: Path, *, max_hours: float = 336.0, drop_service: bool = True) -> pd.DataFrame:
    LAKE.mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.perf_counter()
    tables = bronze(source)
    timings["bronze_s"] = time.perf_counter() - t0
    for name, frame in tables.items():
        frame.to_parquet(LAKE / f"bronze_{name}.parquet", index=False)

    t0 = time.perf_counter()
    visits, silver_stats = silver(tables, max_hours=max_hours)
    timings["silver_s"] = time.perf_counter() - t0
    visits.to_parquet(LAKE / "silver_berth_visits.parquet", index=False)

    t0 = time.perf_counter()
    analytical, gold_stats = gold(visits, tables, drop_service=drop_service)
    timings["gold_s"] = time.perf_counter() - t0
    analytical.to_parquet(LAKE / "gold_berth_time.parquet", index=False)

    print("Source tables")
    for name, frame in tables.items():
        print(f"  {name:10s} {frame.shape}")
    print("\nSilver")
    for k, v in silver_stats.items():
        print(f"  {k:32s} {v:,}")
    print("\nGold")
    for k, v in gold_stats.items():
        print(f"  {k:32s} {v:,}")
    print("\nTimings (s)")
    for k, v in timings.items():
        print(f"  {k:12s} {v:0.2f}")
    stay = analytical.time_at_berth_h
    print(
        f"\nTime at berth: median {stay.median():.1f} h | mean {stay.mean():.1f} h | "
        f"IQR {stay.quantile(.25):.1f}-{stay.quantile(.75):.1f} h"
    )
    print(f"Period: {analytical.berthed_at.min():%Y-%m-%d} to {analytical.berthed_at.max():%Y-%m-%d}")
    return analytical


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--max-hours", type=float, default=336.0)
    parser.add_argument("--keep-service-craft", action="store_true")
    args = parser.parse_args()
    run(args.source, max_hours=args.max_hours, drop_service=not args.keep_service_craft)
