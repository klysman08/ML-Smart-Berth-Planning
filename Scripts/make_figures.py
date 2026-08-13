"""Publication figures for the revised manuscript.

Produces four PNGs at 300 dpi, sized for a single MDPI column-width block:

  fig_flowchart.png     record counts through the medallion pipeline
  fig_distribution.png  distribution of vessel time at berth
  fig_validation.png    rolling-origin validation across five cut-offs
  fig_shap.png          mean absolute SHAP attribution, leakage-free model

Palette: #4271c4 / #d4761a / #8a5aa8 — validated for colour-vision deficiency
separation, chroma and contrast against a light surface.

Run:
    uv run python Scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

BLUE, ORANGE, PURPLE = "#4271c4", "#d4761a", "#8a5aa8"
INK, MUTED, GRID = "#1b1b1b", "#5c5c5c", "#dcdcdc"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def flowchart(out: Path) -> None:
    steps = [
        ("Reporting points ingested (Bronze)", "36,773"),
        ("Berthing/unberthing events paired (Silver)", "8,205"),
        ("Positive berth duration", "8,205"),
        ("Duration ≤ 336 h", "8,180"),
        ("Service craft removed", "8,148"),
        ("Gold analytical table\n8,148 visits · 6,550 calls · 1,923 vessels", "8,148"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.7)
    ax.axis("off")

    for index, (label, count) in enumerate(steps):
        y = (len(steps) - index - 1) * 1.7 + 0.35
        last = index == len(steps) - 1
        box = FancyBboxPatch(
            (0.6, y), 8.8, 1.05,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=BLUE if last else MUTED,
            facecolor="#eaf0fb" if last else "#ffffff",
        )
        ax.add_patch(box)
        ax.text(1.0, y + 0.52, label, va="center", ha="left", fontsize=8.6,
                color=INK, linespacing=1.4)
        ax.text(9.1, y + 0.52, count, va="center", ha="right", fontsize=9.4,
                color=BLUE if last else INK, fontweight="bold")
        if not last:
            ax.add_patch(
                FancyArrowPatch((5.0, y), (5.0, y - 0.62),
                                arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.1, color=MUTED)
            )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def distribution(gold: pd.DataFrame, out: Path) -> None:
    stay = gold["time_at_berth_h"]
    shown = stay[stay <= 168]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.08},
    )

    top.boxplot(shown, vert=False, widths=0.5, showfliers=False,
                medianprops=dict(color=ORANGE, linewidth=2),
                boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED),
                capprops=dict(color=MUTED))
    top.scatter(shown.mean(), 1, marker="^", s=42, color=PURPLE, zorder=3)
    top.set_yticks([])
    top.grid(False)
    for side in ("left", "bottom"):
        top.spines[side].set_visible(False)

    bottom.hist(shown, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    bottom.set_xlabel("Time at berth (hours)")
    bottom.set_ylabel("Berth visits")
    bottom.set_xlim(0, 168)

    for value, label, colour in [
        (shown.median(), f"median {shown.median():.1f} h", ORANGE),
        (shown.mean(), f"mean {shown.mean():.1f} h", PURPLE),
    ]:
        bottom.axvline(value, color=colour, linewidth=1.6, linestyle="--")
        bottom.text(value + 2.5, bottom.get_ylim()[1] * 0.88, label,
                    color=colour, fontsize=8.4)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validation(rolling: pd.DataFrame, out: Path) -> None:
    order = ["particulars only", "+ vessel history", "+ manifested cargo"]
    colours = {"particulars only": PURPLE, "+ vessel history": BLUE,
               "+ manifested cargo": ORANGE}
    labels = {"particulars only": "Vessel particulars only",
              "+ vessel history": "+ vessel history",
              "+ manifested cargo": "+ manifested cargo"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    cutoffs = sorted(rolling.cutoff.unique())
    ticks = [c[:7] for c in cutoffs]

    for name in order:
        sub = rolling[rolling.feature_set == name].sort_values("cutoff")
        left.plot(range(len(sub)), sub.r2_hours, marker="o", markersize=6,
                  linewidth=2, color=colours[name], label=labels[name])
        right.plot(range(len(sub)), sub.mae_h, marker="o", markersize=6,
                   linewidth=2, color=colours[name], label=labels[name])

    baseline = rolling.groupby("cutoff").baseline_mae_h.first().mean()
    right.axhline(baseline, color=MUTED, linestyle="--", linewidth=1.4)
    right.text(0.05, baseline - 0.45, f"median-stay baseline ({baseline:.1f} h)",
               fontsize=8, color=MUTED)

    for axis, ylabel in ((left, "R² (hours)"), (right, "Mean absolute error (h)")):
        axis.set_xticks(range(len(ticks)))
        axis.set_xticklabels(ticks, rotation=30, ha="right", fontsize=8)
        axis.set_xlabel("Training cut-off")
        axis.set_ylabel(ylabel)
    left.set_ylim(0, 0.36)
    right.set_ylim(8, 14.4)
    left.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


PRETTY = {
    "num__prev_time_at_berth_h": "Previous stay of the same vessel",
    "cat__berth_location": "Berth",
    "num__visit_index": "Berth visit index within the call",
    "num__max_speed_kn": "Maximum speed",
    "num__deadweight_t": "Deadweight",
    "num__beam_m": "Beam",
    "num__gross_tonnage": "Gross tonnage",
    "cat__vessel_type": "Vessel type",
    "num__prev_call_count": "Prior calls by the vessel",
    "num__loa_m": "Length overall",
    "num__summer_draft_m": "Summer draft",
    "num__wait_from_port_entry_h": "Wait from port entry to berthing",
}


def shap_figure(importance: pd.DataFrame, out: Path, top: int = 12) -> None:
    frame = importance.head(top).iloc[::-1]
    names = [PRETTY.get(n, n.split("__")[-1].replace("_", " ")) for n in frame.iloc[:, 0]]
    values = frame.iloc[:, 1].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.barh(names, values, color=BLUE, height=0.66)
    bars[-1].set_color(ORANGE)
    ax.set_xlabel("Mean |SHAP value| (log-hours)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, values.max() * 1.16)
    for name, value in zip(names, values):
        ax.text(value + values.max() * 0.015, name, f"{value:.3f}",
                va="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    gold = pd.read_parquet(ROOT / "lakehouse_aps" / "gold_berth_time.parquet")
    rolling = pd.read_csv(REPORTS / "aps_rolling_validation.csv")
    importance = pd.read_csv(REPORTS / "aps_shap_importance.csv")

    flowchart(FIGURES / "fig_flowchart.png")
    distribution(gold, FIGURES / "fig_distribution.png")
    validation(rolling, FIGURES / "fig_validation.png")
    shap_figure(importance, FIGURES / "fig_shap.png")
    print("written:", *[p.name for p in sorted(FIGURES.glob("*.png"))])


if __name__ == "__main__":
    main()
