"""What-if: test a par level for ONE item using your own usage numbers.

Run from the main project folder:  python examples/what_if.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import simulate  # noqa: E402

# ---- Change these to your numbers ----
ITEM = "Syringe 60 mL"
START_DATE = "2026-11-02"           # a Monday
DAILY_USAGE = [44, 41, 38, 35, 30, 0, 0,     # week 1, Mon-Sun
               46, 40, 39, 36, 28, 0, 0,     # week 2
               43, 42, 37, 33, 31, 0, 0,     # week 3
               48, 39, 40, 34, 29, 0, 0]     # week 4
LEAD_TIME_DAYS = 1                  # average days for an order to arrive
LEAD_TIME_SD = 0                    # 0 = always on time; try 0.5 for occasional late trucks
PAR_LEVELS_TO_TEST = [150, 200, 250, 300]
# --------------------------------------

usage = np.array(DAILY_USAGE)
dates = pd.date_range(START_DATE, periods=len(usage))
print(f"{ITEM}: {usage.sum()} used over {len(usage)} days\n")
print(f"{'Par':>5} {'Orders':>7} {'Avg on hand':>12} {'Units short':>12} {'Fill rate':>10}")
for par in PAR_LEVELS_TO_TEST:
    r = simulate(usage, dates, "par", start_on_hand=par, par=par,
                 lead_time_mean=LEAD_TIME_DAYS, lead_time_sd=LEAD_TIME_SD,
                 rng=np.random.default_rng(7))
    print(f"{par:>5} {r['orders']:>7} {r['avg_on_hand']:>12.1f} {r['units_short']:>12.0f} {r['fill_rate']:>10.1%}")