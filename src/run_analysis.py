"""Compare the current par-level habit vs. statistical reorder points.

Parameters are learned on Jan-Jun (train) and tested on Jul-Dec (test),
so the results show how the policy performs on months it has never seen.

Run:  python src/run_analysis.py
Out:  reports/results_by_item.csv, reports/service_level_sweep.csv, reports/figures/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from inventory import eoq, order_up_to_level, reorder_point, safety_stock
from simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, FIGS = ROOT / "data", ROOT / "reports", ROOT / "reports" / "figures"

# ---- Assumptions (document every one in the README) ----
SPLIT_DATE = "2025-07-01"
ORDER_COST = 3.00       # $ per extra order line (one daily wholesaler order covers many lines)
HOLDING_RATE = 0.25     # 25% of item value per year (capital, space, expiry, damage)
PAR_MULTIPLIER = 2.0    # current habit: "keep two weeks on the shelf" for every item
SERVICE_LEVEL = 0.95    # target for the reorder-point policy
MIN_WEEKS, MAX_WEEKS = 0.5, 4   # shelf space and expiry limits on order size
SEED = 7

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # current par, smart par, reorder point
POLICIES = ["Current par", "Smart par", "Reorder point"]
INK, MUTED = "#0b0b0b", "#52514e"
def item_parameters(items: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    # Measure variability on WEEKLY totals. Daily numbers include zero-usage weekends,
    # which would make usage look far more erratic than it really is.
    weekly = (train.set_index("date").groupby("item_id")["qty_used"]
                   .resample("W-SUN").sum().reset_index())
    stats = weekly.groupby("item_id")["qty_used"].agg(weekly_mean="mean", weekly_sd="std")
    p = items.set_index("item_id").join(stats)
    p["daily_mean"] = p["weekly_mean"] / 7
    p["daily_sd"] = p["weekly_sd"] / np.sqrt(7)   # assumes days are independent
    p["annual_demand"] = p["daily_mean"] * 365
    p["annual_spend"] = p["annual_demand"] * p["unit_cost"]
    p["par_level"] = np.ceil(p["weekly_mean"] * PAR_MULTIPLIER)
    return p
def rop_settings(row, service_level: float) -> tuple[float, float, float]:
    ss = safety_stock(service_level, row.daily_mean, row.daily_sd, row.lead_time_mean, row.lead_time_sd)
    rop = reorder_point(row.daily_mean, row.lead_time_mean, ss)
    q = eoq(row.annual_demand, ORDER_COST, row.unit_cost, HOLDING_RATE)
    # Real-world limits: no order smaller than half a week or bigger than 4 weeks of usage.
    q = float(np.clip(q, MIN_WEEKS * row.weekly_mean, MAX_WEEKS * row.weekly_mean))
    q = max(1.0, np.ceil(q))
    return ss, rop, q
def run_policies(params: pd.DataFrame, test: pd.DataFrame, service_level: float) -> pd.DataFrame:
    test_days = test["date"].nunique()
    out = []
    for item_id, row in params.iterrows():
        series = test[test["item_id"] == item_id].sort_values("date")
        usage, dates = series["qty_used"].to_numpy(), pd.DatetimeIndex(series["date"])
        ss, rop, q = rop_settings(row, service_level)
        common = dict(lead_time_mean=row.lead_time_mean, lead_time_sd=row.lead_time_sd)
        # Same random seed for all three policies -> same delivery luck, fair comparison.
        smart = float(np.ceil(order_up_to_level(service_level, row.daily_mean, row.daily_sd,
                                                row.lead_time_mean, row.lead_time_sd)))
        par = simulate(usage, dates, "par", start_on_hand=row.par_level, par=row.par_level,
                       rng=np.random.default_rng(SEED), **common)
        spar = simulate(usage, dates, "par", start_on_hand=smart, par=smart,
                        rng=np.random.default_rng(SEED), **common)
        stat = simulate(usage, dates, "rop", start_on_hand=np.ceil(rop + q), rop=rop, order_qty=q,
                        rng=np.random.default_rng(SEED), **common)
        for name, r in (("Current par", par), ("Smart par", spar), ("Reorder point", stat)):
            holding = r["avg_on_hand"] * row.unit_cost * HOLDING_RATE * test_days / 365
            ordering = r["orders"] * ORDER_COST
            out.append({"item_id": item_id, "item_name": row.item_name, "category": row.category,
                        "policy": name, **r,
                        "avg_inventory_value": r["avg_on_hand"] * row.unit_cost,
                        "holding_cost": holding, "ordering_cost": ordering,
                        "total_cost": holding + ordering,
                        "safety_stock": ss if name == "Reorder point" else np.nan,
                        "reorder_point": rop if name == "Reorder point" else np.nan,
                        "order_qty": q if name == "Reorder point" else np.nan,
                        "par_level": {"Current par": row.par_level, "Smart par": smart}.get(name, np.nan)})
    return pd.DataFrame(out)
def overall(results: pd.DataFrame, usage_total: float) -> pd.DataFrame:
    g = results.groupby("policy").agg(total_cost=("total_cost", "sum"),
                                      avg_inventory_value=("avg_inventory_value", "sum"),
                                      orders=("orders", "sum"), units_short=("units_short", "sum"),
                                      stockout_days=("stockout_days", "sum"))
    g["fill_rate"] = 1 - g["units_short"] / usage_total
    # Unit fill rate is dominated by cheap, high-volume items, so also count
    # how many ITEMS ran out at least once - that is what the techs feel.
    g["items_with_stockouts"] = results[results["stockout_days"] > 0].groupby("policy").size()
    g["items_with_stockouts"] = g["items_with_stockouts"].fillna(0).astype(int)
    return g
def main():
    items = pd.read_csv(DATA / "items.csv")
    usage = pd.read_csv(DATA / "daily_usage.csv", parse_dates=["date"])
    train, test = usage[usage["date"] < SPLIT_DATE], usage[usage["date"] >= SPLIT_DATE]

    params = item_parameters(items, train)
    results = run_policies(params, test, SERVICE_LEVEL)
    summary = overall(results, test["qty_used"].sum())

    sweep_rows = []
    for sl in [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]:
        o = overall(run_policies(params, test, sl), test["qty_used"].sum())
        for pol in ("Smart par", "Reorder point"):
            sweep_rows.append({"service_level": sl, "policy": pol, **o.loc[pol].to_dict()})
    sweep = pd.DataFrame(sweep_rows)

    REPORTS.mkdir(exist_ok=True)
    params.to_csv(REPORTS / "item_parameters.csv")
    results.to_csv(REPORTS / "results_by_item.csv", index=False)
    sweep.to_csv(REPORTS / "service_level_sweep.csv", index=False)
    # charts(params, results)      <- remove the # in Session 9
    summary.to_csv(REPORTS / "summary.csv")

    pd.set_option("display.float_format", "{:,.3f}".format)
    print(summary.T)
    print(sweep[["policy", "service_level", "fill_rate", "total_cost", "avg_inventory_value"]])


if __name__ == "__main__":
    main()