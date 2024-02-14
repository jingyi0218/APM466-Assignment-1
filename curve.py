import numpy as np
from datetime import datetime
from scipy.optimize import fsolve
from bond import Bond
from utiltiy_function import get_day_count_fraction, interpolate_log_discount_factor


class Curve:
    __pillar_years = [1, 2, 3, 4, 5]

    def __init__(self, bonds_list: list[Bond], reference_date: datetime):
        self.bonds_list = bonds_list
        self.reference_date = reference_date
        self.yield_curve = self.build_yield_curve()
        self.spot_curve = self.build_spot_curve()
        self.forward_curve = self.build_forward_curve()

    @classmethod
    def get_pillar_years(cls):
        return cls.__pillar_years

    def build_yield_curve(self):
        tenors = np.array([get_day_count_fraction(self.reference_date, b.maturity) for b in self.bonds_list])
        yields = np.array([b.calculate_yield_to_maturity() for b in self.bonds_list])
        return interpolate_log_discount_factor(self.__pillar_years, tenors, yields) * 100

    def build_spot_curve(self):
        spot_rates = np.array([])
        for i in range(len(self.bonds_list)):
            bond = self.bonds_list[i]
            coupon_tenors = np.array([get_day_count_fraction(self.reference_date, c) for c in bond.coupon_schedule])

            def bond_value_func(s):
                return np.sum(bond.coupon / 2 * 100 * np.exp(-spot_rates * coupon_tenors[:-1])) \
                       + 100 * (1 + bond.coupon / 2) * np.exp(-s * coupon_tenors[-1]) - bond.price

            sol = fsolve(bond_value_func, x0=np.array([bond.coupon]))
            spot_rates = np.append(spot_rates, sol[0])

        tenors = np.array([get_day_count_fraction(self.reference_date, b.maturity) for b in self.bonds_list])
        return interpolate_log_discount_factor(self.__pillar_years, tenors, spot_rates) * 100

    def build_forward_curve(self):
        delta_rt = self.spot_curve[1:] * np.array(self.__pillar_years[1:]) - self.spot_curve[0] * self.__pillar_years[0]
        delta_t = np.array(self.__pillar_years[1:]) - self.__pillar_years[0]
        forward_rates = delta_rt / delta_t
        return forward_rates
