import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from bond import Bond
from curve import Curve


def main():
    # read data
    df_raw = pd.read_csv("Bond Prices.csv", index_col="Dates")
    df_raw.index = pd.to_datetime(df_raw.index)

    # bonds to use
    pillar_bonds = ["CAN 2.25 Mar 24", "CAN 1.5 Sep 24", "CAN 1.25 Mar 25", "CAN 0.5 Sep 25", "CAN 0.25 Mar 26", "CAN 1.0 Sep 26",
                    "CAN 1.25 Mar 27", "CAN 2.75 Sep 27", "CAN 3.5 Mar 28", "CAN 3.25 Sep 28"]
    df = df_raw[pillar_bonds]

    # Question 4: build curves --------------------------------------------------
    curves_of_all_dates = []
    for this_date in df.index:
        bonds_list = [Bond(bond_name=col, reference_date=this_date, price=df.loc[this_date, col]) for col in df.columns]
        curves = Curve(bonds_list=bonds_list, reference_date=this_date)
        curves_of_all_dates.append(curves)

    # Question 4a: plot yield curves
    for date, curves in zip(df.index, curves_of_all_dates):
        # print(curves.yield_curve)
        plt.plot(curves.get_pillar_years(), curves.yield_curve, label=date.date())
    plt.title("Yield Curves of All Dates")
    plt.xticks(Curve.get_pillar_years())
    plt.xlabel("Maturity (years)")
    plt.ylabel("Yields (%)")
    plt.legend()
    plt.show()

    # Question 4b: plot spot curves
    for date, curves in zip(df.index, curves_of_all_dates):
        # print(curves.spot_curve)
        plt.plot(curves.get_pillar_years(), curves.spot_curve, label=date.date())
    plt.title("Spot Curves of All Dates")
    plt.xticks(Curve.get_pillar_years())
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spot Rates (%)")
    plt.legend()
    plt.show()

    # Question 4c: plot forward curves
    for date, curves in zip(df.index, curves_of_all_dates):
        # print(curves.spot_curve)
        plt.plot(curves.get_pillar_years()[1:], curves.forward_curve, label=date.date())
    plt.title("1-Year Forward Curves  of All Dates")
    plt.xticks(Curve.get_pillar_years()[1:])
    plt.xlabel("Maturity (years)")
    plt.ylabel("Forward Rates (%)")
    plt.legend()
    plt.show()

    # Question 5: Calculate covariance matrix --------------------------------------------------

    # covariance matrix of yields return
    yields = np.array([curve.yield_curve for curve in curves_of_all_dates]).T
    yields_log_return = np.log(yields[:, 1:] / yields[:, :-1])
    yields_cov = np.cov(yields_log_return)
    print("Covariance Matrix of Yields Returns:\n", yields_cov)

    # covariance matrix of forward rates return
    forward_rates = np.array([curve.forward_curve for curve in curves_of_all_dates]).T
    forward_rates_log_return = np.log(forward_rates[:, 1:] / forward_rates[:, :-1])
    forward_rates_cov = np.cov(forward_rates_log_return)
    print("Covariance Matrix of Forward Rates Returns:\n", forward_rates_cov)

    # Question 6: Principal components analysis --------------------------------------------------

    # PCA of yields return covariance matrix
    yields_eigenvalues, yields_eigenvectors = np.linalg.eig(yields_cov)
    print("Eigenvalues of Yields Returns Covariance Matrix:\n", yields_eigenvalues)
    print("Eigenvectors of Yields Returns Covariance Matrix:\n", np.round(yields_eigenvectors, 4))

    # PCA of forward rates return covariance matrix
    forward_rates_eigenvalues, forward_rates_eigenvectors = np.linalg.eig(forward_rates_cov)
    print("Eigenvalues of Forward Rates Returns Covariance Matrix:\n", forward_rates_eigenvalues)
    print("Eigenvectors of Forward Rates Returns Covariance Matrix:\n", np.round(forward_rates_eigenvectors, 4))


if __name__ == "__main__":
    main()
