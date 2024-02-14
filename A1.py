import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
pd.options.mode.chained_assignment = None

import pandas as pd

def load_and_prepare_data(file_path):
    date_columns = [str(day) for day in range(9, 14)] + [str(day) for day in range(16, 21)]  # Exclude missing dates
    new_column_names = {date: f'2023-01-{int(date):02d}' for date in date_columns}
    df = pd.read_csv(file_path, encoding='latin1').drop(['Name'], axis=1).rename(columns=new_column_names)
    df['maturity date'] = pd.to_datetime(df['maturity date'])
    return df


def process_data(df, date_columns):
    for date in date_columns:
        df[f"time to maturity {date}"] = (df['maturity date'] - datetime.fromisoformat(date)).dt.days
        df[f"accrued interest {date}"] = ((182 - df[f"time to maturity {date}"] % 182) * df["coupon"].str.strip('%').astype(float) / 365)
        df[f"dirty price {date}"] = df[date] + df[f"accrued interest {date}"]
        df[f"x {date}"] = df[f"time to maturity {date}"] / 365

        def calculate_yield(row, current_date):
            day = np.asarray([(row[f"time to maturity {current_date}"] % 182) / 182 + n for n in range(int(row[f"time to maturity {current_date}"] / 182) + 1)])
            pay = np.asarray([float(row["coupon"].strip('%')) / 2] * int(row[f"time to maturity {current_date}"] / 182) + [float(row["coupon"].strip('%')) / 2 + 100])
            return fsolve(lambda y: np.dot(pay, (1 + y / 2) ** (-day)) - row[f"dirty price {current_date}"], .05)[0]
        
        df[f"yield {date}"] = df.apply(calculate_yield, axis=1, current_date=date)

    return df


def plot_curves(df, date_columns, proc_func, x_label, y_label, title, legend_prefix='Day'):
    plt.figure(figsize=(10, 6))
    for date in date_columns:
        x, y = proc_func(df, date)
        plt.plot(x, y, label=f"{date}")
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.close()


def rtn_raw(df, date):
    return df[f"x {date}"], df[f"yield {date}"]


def rtn_interpolation(df, date):
    df = copy.deepcopy(df)
    x, y = df[f"x {date}"], df[f"yield {date}"]
    inter = np.linspace(0.5, 5, len(y))
    return np.asarray(inter), np.asarray(np.poly1d(np.polyfit(x, y, 2))(inter))




def rtn_spot_rate(df, date):
    df = copy.deepcopy(df)
    n = len(df)
    spot_rates = np.zeros(n)  # Initialize spot rates with zeros

    for i, bond in df.iterrows():
        # Extract necessary values
        coupon = float(bond["coupon"].strip('%')) / 2
        dirty_price = bond[f"dirty price {date}"]
        time_to_maturity = bond[f"x {date}"]

        # Calculate spot rate for the first bond directly
        if i == 0:
            spot_rates[i] = -np.log(dirty_price / (coupon + 100)) / time_to_maturity
        else:
            # For subsequent bonds, define a function for fsolve to find the spot rate
            def objective(spot):
                cash_flows = np.array([coupon if j < i else coupon + 100 for j in range(i + 1)])
                times = df[f"x {date}"][:i+1]
                pv = sum(cash_flows[j] * np.exp(-spot_rates[j] * times.iloc[j]) for j in range(i)) + cash_flows[-1] * np.exp(-spot * time_to_maturity)
                return pv - dirty_price

            # Solve for the spot rate that zeros the objective function
            spot_rates[i] = fsolve(objective, 0.05)[0]

    # Return time to maturity and calculated spot rates
    return df[f"x {date}"], spot_rates
    

def rtn_fwd_rate(df, date):
    x, y = rtn_spot_rate(df, date)

    # Perform polynomial fitting
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)

    # Interpolate forward rates
    fwd_rates = []
    for i in range(4):
        fwd_rate = (poly(x[i*2+3]) * (i+2) - poly(x[1])) / (i+1)
        fwd_rates.append(fwd_rate)
    
    return ['1yr','2yr','3yr','4yr'], fwd_rates



def cov(df, date_columns):
    log = np.empty([5, 9])
    npl = np.empty([5, 10])
    for i, date in enumerate(date_columns):
        x, y = rtn_interpolation(df, date)
        for j in range(5):
            npl[j, i] = y[j*2+1]
    for i in range(9):
        for j in range(5):
            log[j, i] = np.log(npl[j, i+1] / npl[j, i])
    return np.cov(log), log


def matrix(df, date_columns):
    npl = np.empty([4, 10])
    for i, date in enumerate(date_columns):
        _, y = rtn_fwd_rate(df, date)
        npl[:, i] = y
    return npl


def main():
    file_path = "APM466 A1 data_V1.csv"
    df = load_and_prepare_data(file_path)
    date_columns = [f'2023-01-{day:02d}' for day in range(9, 14)] + [f'2023-01-{day:02d}' for day in range(16, 21)]
    df = process_data(df, date_columns)
    
    plot_curves(df, date_columns, rtn_spot_rate, 'Time to Maturity', 'Spot Rate', '5years Uninterpolated Spot Curve')
    plot_curves(df, date_columns, rtn_fwd_rate, 'Time to Maturity', 'Forward Rate', '1year Forward Rate Curve')
    plot_curves(df, date_columns, rtn_interpolation, 'Time to Maturity', 'Yield', '5years Interpolated Yield Curve')
    plot_curves(df, date_columns, rtn_raw, 'Time to Maturity', 'Yield', '5years Uninterpolated Yield Curve')

    print('covariance matrix')
    print(cov(df, date_columns)[0])

    print("covariance matrix: ", np.cov(matrix(df, date_columns)))
    print('-----------------------------------------------------------------------------------------------')
    w1, v1 = np.linalg.eig(np.cov(cov(df, date_columns)[1]))
    print("eigenvalue of the matrix :", w1)
    print('------------------------------------------------------------------------------------------------')
    print("eigenvector of the matrix is: ", v1)
    print('-------------------------------------------------------------------------------------------------')
    w2, v2 = np.linalg.eig(np.cov(matrix(df, date_columns)))
    print("eigenvalue of the matrix :", w2)
    print('-------------------------------------------------------------------------------------------------')
    print("eigenvector of the matrix is: ", v2)


if __name__ == "__main__":
    main()
