"""EOQ and reorder-point utilities for the project's simulated inventory.

All cost, lead-time and safety-stock inputs are scenario assumptions unless
the user supplies external business data. Demand is expressed in units/month.
"""
from __future__ import annotations
import math


def calculate_eoq(annual_demand: float, ordering_cost: float,
                  annual_holding_cost_per_unit: float) -> int:
    """Return classical EOQ rounded to at least one whole unit."""
    d = max(float(annual_demand), 0.0)
    s = max(float(ordering_cost), 0.0)
    h = max(float(annual_holding_cost_per_unit), 0.0)
    if d == 0:
        return 0
    if s == 0:
        return 1
    if h == 0:
        raise ValueError("Annual holding cost H must be greater than zero for EOQ.")
    return max(1, int(round(math.sqrt((2.0 * d * s) / h))))


def calculate_rop(average_demand_per_period: float, lead_time_periods: float,
                  safety_stock: float = 0.0) -> int:
    """Return reorder point; demand and lead time must use matching periods."""
    return max(0, int(math.ceil(
        max(float(average_demand_per_period), 0.0)
        * max(float(lead_time_periods), 0.0)
        + max(float(safety_stock), 0.0)
    )))


def inventory_position(on_hand: float, on_order: float, backorders: float = 0.0) -> float:
    """Inventory position = on-hand + on-order - backorders."""
    return max(0.0, float(on_hand) + float(on_order) - float(backorders))
