"""Feature policy for the berth-time model.

Features are grouped by *when the information becomes available*, which is the
only defensible way to decide what a planning model may use. A feature that is
predictive but only observable after the vessel leaves is not a feature, it is
the answer.

ARRIVAL_KNOWN   available at, or before, the moment the vessel takes the berth.
                This is the operational feature set: a planner standing on the
                quay at berthing time knows all of it.
PORT_INTERNAL   observable only at departure in this dataset, but held by the
                port in advance as booked cargo volume. Used exclusively to
                quantify what publishing that data would be worth. Never part
                of a reported operational model.
EXCLUDED        defines or contains the target.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Tier 1 — knowable at berthing time
# --------------------------------------------------------------------------
ARRIVAL_CATEGORICAL = [
    "Berth Name",
    "Terminal Name",
    "Vessel Type - Generic",
    "Commercial Market",
    "Commercial Size Class",
    "Load Condition At Arrival",
    "Port Operation",
    "Voyage Origin Port",
    "Origin Port Country",
    "Destination Port",
]

ARRIVAL_NUMERIC = [
    "Capacity - Dwt",
    "Capacity - Gt",
    "Capacity - Teu",
    "Capacity - Liquid Gas",
    "Built",
    "Draught At Arrival",
    "Voyage Distance Travelled",
    "Voyage Speed Average",
    "Voyage Speed Max",
    "Voyage Idle Time",
    "Voyage Time Underway",
    "wait_before_berth_h",
    "arrival_hour",
    "arrival_dayofweek",
    "arrival_month",
    "arrival_is_weekend",
]

ARRIVAL_KNOWN = ARRIVAL_CATEGORICAL + ARRIVAL_NUMERIC

# --------------------------------------------------------------------------
# Tier 2 — held by the port, absent from the open feed
# --------------------------------------------------------------------------
PORT_INTERNAL_CATEGORICAL = ["Load Condition At Departure"]
PORT_INTERNAL_NUMERIC = ["cargo_draught_delta_abs"]
PORT_INTERNAL = PORT_INTERNAL_CATEGORICAL + PORT_INTERNAL_NUMERIC

# --------------------------------------------------------------------------
# Never usable
# --------------------------------------------------------------------------
EXCLUDED = [
    "Time At Port",            # contains the target (berth interval sits inside it)
    "Current Port Atd",        # departure defines the target
    "Undock Timestamp",        # departure defines the target
    "Draught At Departure",    # raw departure state; only the delta is meaningful
    "cargo_draught_delta",     # signed version of the tier-2 feature
    "Time At Berth",           # the target itself
]

# Replicates the thesis feature set, for a like-for-like comparison.
THESIS_SUBSET = [
    "Berth Name",
    "Terminal Name",
    "Vessel Type - Generic",
    "Commercial Market",
    "Voyage Distance Travelled",
    "Voyage Speed Average",
    "Voyage Origin Port",
    "Built",
    "Capacity - Gt",
    "Capacity - Dwt",
]

FEATURE_SETS = {
    "thesis-equivalent": THESIS_SUBSET,
    "arrival-known": ARRIVAL_KNOWN,
    "arrival-known + port-internal": ARRIVAL_KNOWN + PORT_INTERNAL,
}

CATEGORICAL = set(ARRIVAL_CATEGORICAL + PORT_INTERNAL_CATEGORICAL)
