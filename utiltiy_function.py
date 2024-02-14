from datetime import datetime
import numpy as np


def get_day_count_fraction(d1: datetime, d2: datetime):
    """
    get time differences in fraction of years between two dates, d2 - d1, using basis of 365 days
    """
    return (d2 - d1).days / 365


def interpolate_log_discount_factor(tenors_to_interpolate: list, base_tenors: np.array, base_rates: np.array):
    """
    interpolate rate for a given tenor based on a range of tenors and rates, with interpolation method to be linear on log discount factors
    """
    log_dfs = -base_rates * base_tenors
    interpolated_log_dfs = np.interp(tenors_to_interpolate, xp=base_tenors, fp=log_dfs)
    interpolated_rates = -interpolated_log_dfs / tenors_to_interpolate
    return interpolated_rates
