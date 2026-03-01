"""
Cross-Instrument ARIMA Robustness Study (Returns-Based)

Evaluates ARIMA(1,0,1) performance across
multiple instrument types using log returns.

This experiment demonstrates structural heterogeneity
in forecasting error and volatility across instrument categories.
"""

import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

warnings.filterwarnings("ignore")

# ==================================================
# Configuration
# ==================================================

INSTRUMENTS = {
    "SQURPHARMA": "Equity",
    "BATBC": "Equity",
    "GP": "Equity",
    "TB20Y0744": "TreasuryBill",
    "1JANATAMF": "MutualFund"
}

DATA_DIR = "data_sample/Unadjusted"  # change if needed
RESULT_DIR = "results/tables"

ARIMA_ORDER = (1, 0, 1)
TRAIN_RATIO = 0.8

os.makedirs(RESULT_DIR, exist_ok=True)

results = []

# ==================================================
# Loop Through Instruments
# ==================================================

for ticker, inst_type in INSTRUMENTS.items():

    print(f"\nProcessing: {ticker}")

    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        print("File not found.")
        continue

    data = pd.read_csv(path)

    # Strict date parsing
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data = data.sort_values("Date")
    data.set_index("Date", inplace=True)

    prices = data["Close"].astype(float).dropna()

    if len(prices) < 250:
        print("Insufficient observations.")
        continue

    # ==================================================
    # Compute Log Returns
    # ==================================================
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna()

    split_idx = int(len(returns) * TRAIN_RATIO)

    train = returns.iloc[:split_idx]
    test = returns.iloc[split_idx:]

    # ==================================================
    # Fit ARIMA on Returns
    # ==================================================
    model = ARIMA(train, order=ARIMA_ORDER)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    # ==================================================
    # Evaluation Metrics
    # ==================================================
    rmse = sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    results.append({
        "Ticker": ticker,
        "Instrument_Type": inst_type,
        "Observations": len(returns),
        "Train_Size": len(train),
        "Test_Size": len(test),
        "Return_STD": returns.std(),
        "AIC": model_fit.aic,
        "BIC": model_fit.bic,
        "RMSE": rmse,
        "MAE": mae
    })

# ==================================================
# Save Results
# ==================================================

results_df = pd.DataFrame(results)

output_path = os.path.join(
    RESULT_DIR,
    "cross_instrument_metrics_returns.csv"
)

results_df.to_csv(output_path, index=False)

print("\nCross-instrument returns-based experiment completed.")
print(f"Results saved to: {output_path}")