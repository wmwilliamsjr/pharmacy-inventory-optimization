"""Run with:  pytest -q"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inventory import eoq, order_up_to_level, reorder_point, safety_stock, z_score  # noqa: E402


def test_z_score_95_percent():
    assert z_score(0.95) == pytest.approx(1.645, abs=0.001)


def test_z_score_rejects_bad_input():
    with pytest.raises(ValueError):
        z_score(1.5)


def test_safety_stock_usage_variability_only():
    # z=1.645, sd=10/day, L=4 days, no lead-time variability -> 1.645 * 10 * 2 = 32.9
    assert safety_stock(0.95, 50, 10, 4, 0) == pytest.approx(32.9, abs=0.1)


def test_late_deliveries_raise_safety_stock():
    assert safety_stock(0.95, 50, 10, 4, 2) > safety_stock(0.95, 50, 10, 4, 0)


def test_reorder_point():
    assert reorder_point(daily_mean=50, lead_time_mean=4, ss=33) == 233


def test_eoq_textbook_example():
    # D=1000/yr, S=$10/order, unit cost $5, 20% holding -> H=$1 -> sqrt(20000) = 141.4
    assert eoq(1000, 10, 5, 0.20) == pytest.approx(math.sqrt(20000))


def test_eoq_zero_demand():
    assert eoq(0, 10, 5, 0.2) == 0


def test_smart_par_covers_review_plus_lead_time():
    # No variability: par = daily usage * (7 review days + 2 lead days)
    assert order_up_to_level(0.95, 10, 0, 2, 0) == pytest.approx(90)