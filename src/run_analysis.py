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

def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED); ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e7e6e2", linewidth=0.8); ax.set_axisbelow(True)
def charts(params, results):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    # 1) ABC analysis: a few items drive most of the spend.
    abc = params.sort_values("annual_spend", ascending=False)
    cum = abc["annual_spend"].cumsum() / abc["annual_spend"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(abc)), abc["annual_spend"] / 1000, color=BLUE, width=0.75)
    style(ax, "Annual spend by item (ABC analysis)", "Items, highest spend first", "Annual spend ($ thousands)")
    n_a = int((cum <= 0.80).sum()) + 1
    ax.axvline(n_a - 0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.text(n_a - 0.2, ax.get_ylim()[1] * 0.9, f"A items: top {n_a} = 80% of spend", color=INK, fontsize=9)
    ax.set_xticks([])
    fig.tight_layout(); fig.savefig(FIGS / "01_abc_spend.png", dpi=150); plt.close(fig)

    # 2) Par change by item: the habit over-stocks some items and under-stocks others.
    w = results.pivot_table(index="item_name", columns="policy", values="par_level")
    w["change"] = (w["Smart par"] / w["Current par"] - 1) * 100
    w = w.sort_values("change")
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [BLUE if c < 0 else ORANGE for c in w["change"]]
    ax.barh(w.index, w["change"], color=colors, height=0.7)
    ax.axvline(0, color=MUTED, linewidth=1)
    style(ax, "Smart par vs. current par (2-week habit)", "Change in par level (%)", "")
    ax.grid(axis="x", color="#e7e6e2"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    handles = [Patch(color=BLUE, label="Lower par (over-stocked today)"),
               Patch(color=ORANGE, label="Higher par (under-stocked today)")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "02_par_change_by_item.png", dpi=150); plt.close(fig)

    # 3) Units short by category, grouped bars for all three policies.
    cat = results.pivot_table(index="category", columns="policy", values="units_short", aggfunc="sum")
    cat = cat.sort_values("Current par", ascending=True)
    y = np.arange(len(cat)); h = 0.27
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (pol, col) in enumerate(zip(POLICIES, (BLUE, ORANGE, AQUA))):
        ax.barh(y + (1 - i) * h, cat[pol], height=h, color=col, label=pol, edgecolor="#fcfcfb", linewidth=1)
    ax.set_yticks(y, cat.index)
    style(ax, "Units short by category, Jul–Dec test period", "Units requested but not on the shelf", "")
    ax.grid(axis="x", color="#e7e6e2"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "03_units_short_by_category.png", dpi=150); plt.close(fig)
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
    charts(params, results)    
    summary.to_csv(REPORTS / "summary.csv")

    pd.set_option("display.float_format", "{:,.3f}".format)
    print(summary.T)
    print(sweep[["policy", "service_level", "fill_rate", "total_cost", "avg_inventory_value"]])


if __name__ == "__main__":
    main()