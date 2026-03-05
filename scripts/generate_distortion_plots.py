"""
Generate plots and summary statistics
for volatility distortion experiment
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================
# Paths
# ============================

RESULT_FILE = "results/tables/multi_instrument_volatility_results.csv"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ============================
# Load results
# ============================

df = pd.read_csv(RESULT_FILE)

df["STD_pct"] = df["Distortion_STD"] * 100
df["GARCH_pct"] = df["Distortion_GARCH"] * 100

# ============================
# Summary statistics
# ============================

mean_std = df["STD_pct"].mean()
mean_garch = df["GARCH_pct"].mean()

print("\nSummary statistics")
print("------------------")
print("Instruments analyzed:", len(df))
print("Mean STD distortion:", mean_std)
print("Mean GARCH distortion:", mean_garch)

# ============================
# FIGURE V1
# Distortion Distribution
# ============================

plt.figure(figsize=(8,5))

plt.hist(
    df["STD_pct"],
    bins=10,
    alpha=0.6,
    label="Return Volatility Distortion"
)

plt.hist(
    df["GARCH_pct"],
    bins=10,
    alpha=0.6,
    label="GARCH Variance Distortion"
)

# mean lines

plt.axvline(
    mean_std,
    linestyle="--",
    linewidth=2,
    label=f"STD mean = {mean_std:.1f}%"
)

plt.axvline(
    mean_garch,
    linestyle="--",
    linewidth=2,
    label=f"GARCH mean = {mean_garch:.1f}%"
)

plt.xlabel("Distortion (%)")
plt.ylabel("Number of Instruments")

plt.title("Distribution of Volatility Distortion")

plt.xlim(-5, 45)

plt.legend()

plt.tight_layout()

png_path = os.path.join(FIG_DIR, "V1_distortion_distribution.png")
pdf_path = os.path.join(FIG_DIR, "V1_distortion_distribution.pdf")

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

plt.close()

print("Saved:", png_path)

# ============================
# FIGURE V2
# Distortion Boxplot
# ============================

plt.figure(figsize=(6,5))

data = [
    df["STD_pct"],
    df["GARCH_pct"]
]

plt.boxplot(
    data,
    labels=[
        "Return STD Distortion",
        "GARCH Variance Distortion"
    ],
    showmeans=True
)

plt.ylabel("Distortion (%)")

plt.title("Volatility Distortion Comparison")

plt.tight_layout()

png_path = os.path.join(FIG_DIR, "V2_distortion_boxplot.png")
pdf_path = os.path.join(FIG_DIR, "V2_distortion_boxplot.pdf")

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

plt.close()

print("Saved:", png_path)