"""Day-by-day inventory simulation for the ordering policies.

"par" policy: every Monday, order up to a par level.
    - Current par = the habit (2 weeks of average usage for every item)
    - Smart par   = a statistical par for each item (same Monday routine)
"rop" policy: every business day, if inventory position is at or below
    the reorder point, order a set quantity (EOQ).

Inventory position = on hand + on order. Using it (not just on hand) prevents
double-ordering while a delivery is still on the truck.
"""
import numpy as np
import pandas as pd


def simulate(usage: np.ndarray, dates: pd.DatetimeIndex, policy: str, *,
             start_on_hand: float, lead_time_mean: float, lead_time_sd: float,
             par: float = 0.0, rop: float = 0.0, order_qty: float = 0.0,
             rng: np.random.Generator) -> dict:
    on_hand = float(start_on_hand)
    pipeline = []                      # list of (arrival_day_index, qty)
    short_units = 0.0
    stockout_days = 0
    orders = 0
    on_hand_history = []
    for t, (demand, day) in enumerate(zip(usage, dates)):
        # 1) Receive deliveries due today (arrive at start of day).
        arrived = sum(q for d, q in pipeline if d == t)
        pipeline = [(d, q) for d, q in pipeline if d != t]
        on_hand += arrived

        # 2) Fill today's demand.
        served = min(on_hand, demand)
        if demand > served:
            short_units += demand - served
            stockout_days += 1
        on_hand -= served

        # 3) End of day: decide whether to order (business days only).
        position = on_hand + sum(q for _, q in pipeline)
        qty = 0.0
        if day.weekday() < 5:
            if policy == "par" and day.weekday() == 0:          # Mondays
                qty = max(0.0, par - position)
            elif policy == "rop" and position <= rop:
                qty = order_qty
        if qty > 0:
            lt = max(1, int(round(rng.normal(lead_time_mean, lead_time_sd))))
            pipeline.append((t + lt, np.ceil(qty)))
            orders += 1

        on_hand_history.append(on_hand)
        total_demand = float(np.sum(usage))
    return {
        "orders": orders,
        "avg_on_hand": float(np.mean(on_hand_history)),
        "units_short": short_units,
        "stockout_days": stockout_days,
        "fill_rate": 1 - short_units / total_demand if total_demand else 1.0,
    }