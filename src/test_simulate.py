"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import simulate  # noqa: E402

MON_TO_FRI = pd.date_range("2026-10-12", periods=5)   # Oct 12, 2026 is a Monday


def run_par(usage, par):
    return simulate(np.array(usage), MON_TO_FRI, "par", start_on_hand=par, par=par,
                    lead_time_mean=1, lead_time_sd=0, rng=np.random.default_rng(0))


def test_matches_hand_trace():
    # Session 6 paper trace: par 20, usage 6, 8, 5, 7, 9
    r = run_par([6, 8, 5, 7, 9], par=20)
    assert r["orders"] == 1
    assert r["units_short"] == 9
    assert r["stockout_days"] == 1
    assert r["avg_on_hand"] == pytest.approx(6.6)
    assert r["fill_rate"] == pytest.approx(1 - 9 / 35)


def test_big_enough_par_never_runs_out():
    r = run_par([6, 8, 5, 7, 9], par=100)
    assert r["units_short"] == 0 and r["fill_rate"] == 1.0


def test_inventory_never_negative():
    r = run_par([50, 50, 50, 50, 50], par=10)
    assert r["avg_on_hand"] >= 0