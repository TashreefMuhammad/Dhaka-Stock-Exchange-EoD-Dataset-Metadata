"""
A1 — Rolling ARIMA Demonstration

Illustrates usability of the Dhaka Stock Exchange EoD dataset
using a rolling one-step ARIMA(1,1,1) model.

This experiment serves as a methodological illustration
and not as a predictive benchmarking study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# ==================================================
# Suppress Convergence Warnings Only
# ==================================================
warnings.simplefilter("ignore", ConvergenceWarning)


# ==================================================
# Configuration
# ==================================================
TICKER = "SQURPHARMA"
DATA_PATH = f"datasample/{TICKER}.csv"   # adjust if needed
FIGURE_DIR = "figures"

TRAIN_RATIO = 0.8
ARIMA_ORDER = (1, 1, 1)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ==================================================
# Load and Prepare Data
# ==================================================
data = pd.read_csv(DATA_PATH)

# Strict date parsing (YYYY-MM-DD)
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")

data = data.sort_values("Date")
data.set_index("Date", inplace=True)

series = data["Close"].astype(float).dropna()

print(f"Ticker: {TICKER}")
print(f"Total observations: {len(series)}")

if len(series) < 200:
    raise ValueError("Insufficient observations for ARIMA illustration.")


# ==================================================
# Log Transformation (Financial Standard)
# ==================================================
log_series = np.log(series)


# ==================================================
# Train / Test Split
# ==================================================
split_index = int(len(log_series) * TRAIN_RATIO)

train = log_series.iloc[:split_index]
test = log_series.iloc[split_index:]

print(f"Training observations: {len(train)}")
print(f"Testing observations: {len(test)}")


# ==================================================
# Rolling One-Step Forecast
# ==================================================
history = train.tolist()
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=ARIMA_ORDER)
    model_fit = model.fit()

    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)

    history.append(test.iloc[t])

# Convert predictions back to pandas series
forecast_log = pd.Series(predictions, index=test.index)

# Convert back from log space
train_exp = np.exp(train)
test_exp = np.exp(test)
forecast_exp = np.exp(forecast_log)


# ==================================================
# Evaluation Metrics (Reported but Not Emphasized)
# ==================================================
rmse = sqrt(mean_squared_error(test_exp, forecast_exp))
mae = mean_absolute_error(test_exp, forecast_exp)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")


# ==================================================
# Visualization
# ==================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Training data
ax.plot(train_exp.index, train_exp,
        linewidth=1,
        label="Training Data")

# Testing data (thicker, slightly transparent)
ax.plot(test_exp.index, test_exp,
        linewidth=2.2,
        alpha=0.6,
        label="Testing Data")

# Forecast (slightly thinner than test)
ax.plot(forecast_exp.index, forecast_exp,
        linewidth=1.3,
        label="Rolling ARIMA Forecast")

ax.set_title(f"Rolling ARIMA(1,1,1) Forecast Illustration ({TICKER})")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")

ax.legend()
fig.tight_layout()

fig.savefig(f"{FIGURE_DIR}/A1_arima_example.png", dpi=300)
fig.savefig(f"{FIGURE_DIR}/A1_arima_example.pdf")

plt.close(fig)

print("Rolling ARIMA demonstration figure generated successfully.")
