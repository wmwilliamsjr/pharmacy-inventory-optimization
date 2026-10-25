"""Inventory formulas: safety stock, reorder point, EOQ.

All quantities are in units; all time is in calendar days.
"""
import math

from scipy.stats import norm


def z_score(service_level: float) -> float:
    """Service level (e.g. 0.95) -> z value (1.645).
    Service level = chance of NOT running out while waiting for an order to arrive."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1")
    return float(norm.ppf(service_level))


def safety_stock(service_level: float, daily_mean: float, daily_sd: float,
                 lead_time_mean: float, lead_time_sd: float = 0.0) -> float:
    """Extra units held to cover usage spikes AND late deliveries.

    SS = z * sqrt( L * sd_d^2  +  d^2 * sd_L^2 )
         usage variability     lead-time variability
    """
    z = z_score(service_level)
    variance = lead_time_mean * daily_sd ** 2 + daily_mean ** 2 * lead_time_sd ** 2
    return z * math.sqrt(variance)


def reorder_point(daily_mean: float, lead_time_mean: float, ss: float) -> float:
    """Order when inventory position drops to this level.
    ROP = expected usage during lead time + safety stock."""
    return daily_mean * lead_time_mean + ss


def eoq(annual_demand: float, order_cost: float, unit_cost: float, holding_rate: float) -> float:
    """Economic order quantity: the order size that balances ordering cost vs holding cost.
    EOQ = sqrt( 2 * D * S / H ),  H = unit_cost * holding_rate."""
    if annual_demand <= 0:
        return 0.0
    h = unit_cost * holding_rate
    return math.sqrt(2 * annual_demand * order_cost / h)


def order_up_to_level(service_level: float, daily_mean: float, daily_sd: float,
                      lead_time_mean: float, lead_time_sd: float = 0.0,
                      review_days: float = 7) -> float:
    """'Smart par' for a fixed weekly order day (periodic review).

    The order must cover usage until the NEXT order arrives: review period + lead time.
    Par = d * (R + L) + z * sqrt( (R + L) * sd_d^2 + d^2 * sd_L^2 )
    """
    protection = review_days + lead_time_mean
    z = z_score(service_level)
    sigma = math.sqrt(protection * daily_sd ** 2 + daily_mean ** 2 * lead_time_sd ** 2)
    return daily_mean * protection + z * sigma