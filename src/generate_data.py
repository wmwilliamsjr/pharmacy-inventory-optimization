"""Generate one year of synthetic daily usage for oncology compounding supplies.

All data is fake. Item types, usage levels, costs and lead times are modeled on a
typical infusion-center compounding room, not on any real facility's records.

Run:  python src/generate_data.py
Out:  data/items.csv, data/daily_usage.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42                     # same seed -> same data every run (reproducible)
START, END = "2025-01-01", "2025-12-31"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# name, category, unit cost ($), avg weekday usage, lead time mean (days), lead time sd (days)
ITEMS = [
    ("Sodium Chloride 0.9% 50 mL bag",   "IV fluids",  1.45, 22, 1.5, 0.5),
    ("Sodium Chloride 0.9% 100 mL bag",  "IV fluids",  1.55, 30, 1.5, 0.5),
    ("Sodium Chloride 0.9% 250 mL bag",  "IV fluids",  1.80, 38, 1.5, 0.5),
    ("Sodium Chloride 0.9% 500 mL bag",  "IV fluids",  2.10, 12, 1.5, 0.5),
    ("Dextrose 5% 100 mL bag",           "IV fluids",  1.70,  8, 2.0, 1.0),
    ("Dextrose 5% 250 mL bag",           "IV fluids",  1.95, 10, 2.0, 1.0),
    ("Syringe 1 mL",                     "Syringes",   0.14, 25, 1.0, 0.3),
    ("Syringe 3 mL",                     "Syringes",   0.16, 30, 1.0, 0.3),
    ("Syringe 5 mL",                     "Syringes",   0.18, 35, 1.0, 0.3),
    ("Syringe 10 mL",                    "Syringes",   0.22, 60, 1.0, 0.3),
    ("Syringe 20 mL",                    "Syringes",   0.35, 40, 1.0, 0.3),
    ("Syringe 30 mL",                    "Syringes",   0.48, 18, 1.0, 0.3),
    ("Syringe 60 mL",                    "Syringes",   0.62, 14, 1.0, 0.3),
    ("Needle 18G",                       "Needles",    0.06, 70, 1.0, 0.3),
    ("Needle 20G",                       "Needles",    0.06, 30, 1.0, 0.3),
    ("CSTD vial adaptor 20 mm",          "CSTD",       9.50, 28, 7.0, 2.5),
    ("CSTD vial adaptor 13 mm",          "CSTD",       9.50,  9, 7.0, 2.5),
    ("CSTD syringe adaptor",             "CSTD",       6.75, 45, 7.0, 2.5),
    ("CSTD bag spike adaptor",           "CSTD",      11.25, 32, 7.0, 2.5),
    ("Chemo gloves (pair)",              "PPE",        1.20, 48, 2.0, 0.8),
    ("Chemo gown",                       "PPE",        2.60, 16, 3.0, 1.5),
    ("Hair cover",                       "PPE",        0.09, 16, 3.0, 1.5),
    ("Shoe covers (pair)",               "PPE",        0.25, 16, 3.0, 1.5),
    ("Sterile alcohol wipes",            "Aseptic",    0.03, 120, 1.0, 0.3),
    ("0.2 micron in-line filter",        "Admin sets", 6.10,  6, 5.0, 2.0),
    ("Non-PVC IV admin set",             "Admin sets", 4.25, 18, 3.0, 1.0),
    ("Port caps",                        "Aseptic",    0.11, 55, 1.0, 0.3),
    ("Chemo transport bag",              "Transport",  0.30, 40, 3.0, 1.0),
    ("Chemo waste bag",                  "Waste",      0.45, 10, 3.0, 1.0),
    ("Sharps container (chemo)",         "Waste",      5.20,  2, 5.0, 2.0),
    ("Syringe 50 mL",                    "Syringes",   0.55, 12, 1.0, 0.3),
    ("Dextrose 5% 50 mL bag",            "IV fluids",  1.60,  6, 2.0, 1.0),
    ("Lactated Ringer's 1000 mL bag",    "IV fluids",  2.40,  5, 2.0, 1.0),
    ("Red caps (syringe tip caps)",      "Aseptic",    0.08, 40, 1.0, 0.3),
    ("Amber line covers",                "Light protection", 0.65, 6, 5.0, 2.0),
]

# Infusion centers run Mon-Fri; Mondays are busiest. Index 0 = Monday.
WEEKDAY_FACTOR = {0: 1.15, 1: 1.05, 2: 1.00, 3: 0.95, 4: 0.85, 5: 0.0, 6: 0.0}


def build_items() -> pd.DataFrame:
    items = pd.DataFrame(ITEMS, columns=["item_name", "category", "unit_cost",
                                         "avg_weekday_usage", "lead_time_mean", "lead_time_sd"])
    items.insert(0, "item_id", [f"SUP{n:03d}" for n in range(1, len(items) + 1)])
    return items

def build_usage(items: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    rows = []
    for item in items.itertuples():
        # Each item gets its own "lumpiness": higher dispersion = less predictable usage.
        dispersion = rng.uniform(2, 12)            # smaller = more variable
        for d in dates:
            factor = WEEKDAY_FACTOR[d.weekday()]
            if factor == 0:
                qty = 0
            else:
                mean = item.avg_weekday_usage * factor
                # Negative binomial = Poisson with extra variability (real usage is lumpy).
                p = dispersion / (dispersion + mean)
                qty = int(rng.negative_binomial(dispersion, p))
            rows.append((d.date(), item.item_id, qty))
    usage = pd.DataFrame(rows, columns=["date", "item_id", "qty_used"])

    # Clinic closed on major holidays.
    holidays = pd.to_datetime(["2025-01-01", "2025-05-26", "2025-07-04",
                               "2025-09-01", "2025-11-27", "2025-12-25"]).date
    usage.loc[usage["date"].isin(holidays), "qty_used"] = 0
    return usage

def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    items = build_items()
    usage = build_usage(items, rng)
    items.to_csv(DATA_DIR / "items.csv", index=False)
    usage.to_csv(DATA_DIR / "daily_usage.csv", index=False)
    print(f"Wrote {len(items)} items and {len(usage):,} usage rows to {DATA_DIR}")


if __name__ == "__main__":
    main()