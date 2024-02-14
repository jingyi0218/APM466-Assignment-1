import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve
from utiltiy_function import get_day_count_fraction


class Bond:

    def __init__(self, bond_name: str, reference_date: datetime, price: float):
        self.name = bond_name
        self.reference_date = reference_date
        self.price = price
        self.coupon, self.maturity = self.parse_bond_name()
        self.coupon_schedule = self.generate_current_coupon_schedule()

    def parse_bond_name(self):
        _, coupon, month, year = self.name.split(' ')
        coupon_rate = float(coupon) / 100
        maturity_date = datetime.strptime(f"1/{month}/{year}", "%d/%b/%y")
        return coupon_rate, maturity_date

    def generate_current_coupon_schedule(self):
        schedule = []
        i = 0
        while True:
            coupon_date = self.maturity - relativedelta(months=6 * i)
            if coupon_date <= self.reference_date:
                break
            else:
                schedule.append(coupon_date)
                i += 1
        schedule.reverse()

        return schedule

    def calculate_yield_to_maturity(self):
        coupon_tenors = np.array([get_day_count_fraction(self.reference_date, c) for c in self.coupon_schedule])

        def bond_value_func(ytm):
            return np.sum(self.coupon / 2 * 100 * np.exp(-ytm * coupon_tenors)) + 100 * np.exp(-ytm * coupon_tenors[-1]) - self.price

        sol = fsolve(bond_value_func, x0=np.array([self.coupon]))
        return sol[0]
