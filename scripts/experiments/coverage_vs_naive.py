"""
Coverage-Aware vs Naive Modeling Comparison

Demonstrates distortion caused by ignoring listing start dates
using AAMRANET as case study.
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

TICKER = "AAMRANET"
DATA_DIR = "data_sample"
RESULT_DIR = "results/tables"

ARIMA_ORDER = (1, 0, 1)
TRAIN_RATIO = 0.8

os.makedirs(RESULT_DIR, exist_ok=True)

# ==================================================
# Load Data
# ==================================================

path = os.path.join(DATA_DIR, f"{TICKER}.csv")
data = pd.read_csv(path)

data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data = data.sort_values("Date")
data.set_index("Date", inplace=True)

prices_actual = data["Close"].astype(float).dropna()

# ==================================================
# MODEL A — Coverage-Aware
# ==================================================

log_prices_A = np.log(prices_actual)
returns_A = log_prices_A.diff().dropna()

split_A = int(len(returns_A) * TRAIN_RATIO)
train_A = returns_A.iloc[:split_A]
test_A = returns_A.iloc[split_A:]

model_A = ARIMA(train_A, order=ARIMA_ORDER)
fit_A = model_A.fit()

forecast_A = fit_A.forecast(steps=len(test_A))

rmse_A = sqrt(mean_squared_error(test_A, forecast_A))
mae_A = mean_absolute_error(test_A, forecast_A)

# ==================================================
# MODEL B — Naive (Backward Forward-Fill)
# ==================================================

# Create full date index from dataset start
full_dates = pd.date_range(
    start="2012-10-01",
    end=prices_actual.index.max(),
    freq="D"
)

df_full = pd.DataFrame(index=full_dates)
df_full = df_full.join(prices_actual.rename("Close"))

# Forward-fill (backward to pre-listing period)
df_full["Close"] = df_full["Close"].ffill()

# Drop remaining NaNs (if any at beginning)
df_full = df_full.dropna()

log_prices_B = np.log(df_full["Close"])
returns_B = log_prices_B.diff().dropna()

split_B = int(len(returns_B) * TRAIN_RATIO)
train_B = returns_B.iloc[:split_B]
test_B = returns_B.iloc[split_B:]

model_B = ARIMA(train_B, order=ARIMA_ORDER)
fit_B = model_B.fit()

forecast_B = fit_B.forecast(steps=len(test_B))

rmse_B = sqrt(mean_squared_error(test_B, forecast_B))
mae_B = mean_absolute_error(test_B, forecast_B)

# ==================================================
# Save Comparison
# ==================================================

results = pd.DataFrame([
    {
        "Model": "Coverage-Aware",
        "Observations": len(returns_A),
        "Return_STD": returns_A.std(),
        "AIC": fit_A.aic,
        "BIC": fit_A.bic,
        "RMSE": rmse_A,
        "MAE": mae_A
    },
    {
        "Model": "Naive_Filled",
        "Observations": len(returns_B),
        "Return_STD": returns_B.std(),
        "AIC": fit_B.aic,
        "BIC": fit_B.bic,
        "RMSE": rmse_B,
        "MAE": mae_B
    }
])

output_path = os.path.join(
    RESULT_DIR,
    "coverage_vs_naive_comparison.csv"
)

results.to_csv(output_path, index=False)

print("\nCoverage-aware vs naive comparison completed.")
print(f"Results saved to: {output_path}")