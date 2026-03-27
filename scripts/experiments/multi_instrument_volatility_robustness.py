"""
Multi-Instrument Volatility Robustness Experiment

Coverage-aware vs naive panel construction
across multiple instruments.

Models:
- ARIMA(1,0,1)
- GARCH(1,1)
"""

import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ======================================
# Configuration
# ======================================

DATA_DIR = "data_sample/Unadjusted"
META_PATH = "metadata/company_metadata.csv"

RESULT_PATH = "results/tables/multi_instrument_volatility_results.csv"

GLOBAL_START = "2012-10-01"

ARIMA_ORDER = (1,0,1)

MIN_OBS = 400

os.makedirs("results/tables", exist_ok=True)

# ======================================
# Load metadata
# ======================================

meta = pd.read_csv(META_PATH)

meta["First_Date"] = pd.to_datetime(meta["First_Date"])

# filter relatively new listings
eligible = meta[meta["First_Date"] >= "2017-01-01"]

tickers = eligible["Ticker"].tolist()

print("Eligible instruments:", len(tickers))
print(tickers)

results = []

# ======================================
# Loop instruments
# ======================================

for ticker in tqdm(tickers):

    try:

        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        print(f'Analyzing {ticker}')

        if not os.path.exists(path):
            continue

        data = pd.read_csv(path)

        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")
        data.set_index("Date", inplace=True)

        prices = data["Close"].astype(float).dropna()

        if len(prices) < MIN_OBS:
            continue

        listing_date = prices.index.min()

        # ======================================
        # Coverage-aware pipeline
        # ======================================

        log_A = np.log(prices)

        returns_A = log_A.diff().dropna()

        if len(returns_A) < MIN_OBS:
            continue

        std_A = returns_A.std()

        # ARIMA
        arima_A = ARIMA(returns_A, order=ARIMA_ORDER).fit()

        AIC_A = arima_A.aic

        # GARCH
        garch_A = arch_model(returns_A*100, p=1, q=1).fit(disp="off")

        omega_A = garch_A.params["omega"]
        alpha_A = garch_A.params["alpha[1]"]
        beta_A = garch_A.params["beta[1]"]

        if alpha_A + beta_A >= 0.999:
            continue

        UV_A = omega_A/(1-alpha_A-beta_A)

        # ======================================
        # Naive pipeline
        # ======================================

        full_index = pd.date_range(
            start=GLOBAL_START,
            end=prices.index.max(),
            freq="D"
        )

        df_full = pd.DataFrame(index=full_index)

        df_full = df_full.join(prices.rename("Close"))

        df_full["Close"] = df_full["Close"].bfill()

        df_full = df_full.dropna()

        log_B = np.log(df_full["Close"])

        returns_B = log_B.diff().dropna()

        if len(returns_B) < MIN_OBS:
            continue

        std_B = returns_B.std()

        # ARIMA
        arima_B = ARIMA(returns_B, order=ARIMA_ORDER).fit()

        AIC_B = arima_B.aic

        # GARCH
        garch_B = arch_model(returns_B*100, p=1, q=1).fit(disp="off")

        omega_B = garch_B.params["omega"]
        alpha_B = garch_B.params["alpha[1]"]
        beta_B = garch_B.params["beta[1]"]

        if alpha_B + beta_B >= 0.999:
            continue

        UV_B = omega_B/(1-alpha_B-beta_B)

        # ======================================
        # Distortion metrics
        # ======================================

        distortion_std = (std_A - std_B)/std_A

        distortion_garch = (UV_A - UV_B)/UV_A

        results.append({

            "Ticker": ticker,

            "Listing_Date": listing_date,

            "Std_Aware": std_A,
            "Std_Naive": std_B,
            "Distortion_STD": distortion_std,

            "AIC_ARIMA_Aware": AIC_A,
            "AIC_ARIMA_Naive": AIC_B,

            "UV_GARCH_Aware": UV_A,
            "UV_GARCH_Naive": UV_B,
            "Distortion_GARCH": distortion_garch

        })

    except:
        print(f'Failed Analyzing {ticker}')
        continue

# ======================================
# Save results
# ======================================

results_df = pd.DataFrame(results)

results_df.to_csv(RESULT_PATH, index=False)

print("\nCompleted experiment.")
print("Instruments analyzed:", len(results_df))

# ======================================
# Summary statistics
# ======================================

if len(results_df) > 0:

    print("\nSummary")

    print("Mean STD distortion:",
          results_df["Distortion_STD"].mean())

    print("Median STD distortion:",
          results_df["Distortion_STD"].median())

    print("Mean GARCH distortion:",
          results_df["Distortion_GARCH"].mean())

    print("Median GARCH distortion:",
          results_df["Distortion_GARCH"].median())

    print("Positive distortion %:",
          (results_df["Distortion_STD"]>0).mean())