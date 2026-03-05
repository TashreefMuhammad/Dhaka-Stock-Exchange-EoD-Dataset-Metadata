"""
Generate figures and summary statistics
for volatility distortion experiment
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# Paths
# =====================================================

RESULT_FILE = "results/tables/multi_instrument_volatility_results.csv"
RESULT_DIR = "results/tables"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# Load data
# =====================================================

df = pd.read_csv(RESULT_FILE)

df["STD_pct"] = df["Distortion_STD"] * 100
df["GARCH_pct"] = df["Distortion_GARCH"] * 100

df["Listing_Date"] = pd.to_datetime(df["Listing_Date"])
df["Listing_Year"] = df["Listing_Date"].dt.year

# Dataset start date (from your dataset)
DATASET_START = pd.Timestamp("2012-10-01")

# Padding length in days
df["Padding_Days"] = (df["Listing_Date"] - DATASET_START).dt.days

# =====================================================
# Summary statistics
# =====================================================

summary = {
    "Instruments_analyzed": len(df),

    "Mean_STD_distortion_pct": df["STD_pct"].mean(),
    "Median_STD_distortion_pct": df["STD_pct"].median(),

    "Mean_GARCH_distortion_pct": df["GARCH_pct"].mean(),
    "Median_GARCH_distortion_pct": df["GARCH_pct"].median(),

    "Pct_STD_distortion_gt_10pct": (df["STD_pct"] > 10).mean(),
    "Pct_STD_distortion_gt_20pct": (df["STD_pct"] > 20).mean(),

    "Pct_GARCH_distortion_gt_10pct": (df["GARCH_pct"] > 10).mean(),
}

summary_df = pd.DataFrame([summary])

summary_path = os.path.join(
    RESULT_DIR,
    "distortion_summary_statistics.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\nSummary statistics")
print("------------------")

for k, v in summary.items():
    print(f"{k}: {v}")

print("\nSaved summary statistics to:", summary_path)

mean_std = df["STD_pct"].mean()
mean_garch = df["GARCH_pct"].mean()

# =====================================================
# FIGURE V1 — Distortion Distribution
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["STD_pct"],
    bins=8,
    color="#1f77b4",
    alpha=0.7,
    label="Return Volatility Distortion"
)

plt.hist(
    df["GARCH_pct"],
    bins=8,
    color="#ff7f0e",
    alpha=0.6,
    label="GARCH Variance Distortion"
)

plt.axvline(
    mean_std,
    color="#1f77b4",
    linestyle="--",
    linewidth=2,
    label=f"STD mean = {mean_std:.1f}%"
)

plt.axvline(
    mean_garch,
    color="#ff7f0e",
    linestyle="--",
    linewidth=2,
    label=f"GARCH mean = {mean_garch:.1f}%"
)

plt.xlabel("Distortion (%)")
plt.ylabel("Number of Instruments")

plt.title("Distribution of Volatility Distortion")

plt.xlim(10,45)

plt.legend()

plt.tight_layout()

plt.savefig(f"{FIG_DIR}/V1_distortion_distribution.png", dpi=300)
plt.savefig(f"{FIG_DIR}/V1_distortion_distribution.pdf")

plt.close()

print("Saved figure: V1_distortion_distribution")

# =====================================================
# FIGURE V2 — Boxplot Comparison
# =====================================================

plt.figure(figsize=(7,5))

box = plt.boxplot(
    [df["STD_pct"], df["GARCH_pct"]],
    patch_artist=True,
    labels=[
        "Return STD Distortion",
        "GARCH Variance Distortion"
    ],
    showmeans=True
)

colors = ["#1f77b4", "#ff7f0e"]

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.ylabel("Distortion (%)")

plt.title("Volatility Distortion Comparison")

plt.tight_layout()

plt.savefig(f"{FIG_DIR}/V2_distortion_boxplot.png", dpi=300)
plt.savefig(f"{FIG_DIR}/V2_distortion_boxplot.pdf")

plt.close()

print("Saved figure: V2_distortion_boxplot")

# =====================================================
# FIGURE V3 — Distortion vs Listing Year
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(
    df["Listing_Year"],
    df["STD_pct"],
    color="#1f77b4",
    label="Return STD Distortion"
)

plt.scatter(
    df["Listing_Year"],
    df["GARCH_pct"],
    color="#ff7f0e",
    label="GARCH Distortion"
)

plt.xlabel("Listing Year")
plt.ylabel("Distortion (%)")

plt.title("Distortion vs Listing Year")

plt.legend(loc="upper left")

plt.tight_layout()

plt.savefig(f"{FIG_DIR}/V3_distortion_vs_listing_year.png", dpi=300)
plt.savefig(f"{FIG_DIR}/V3_distortion_vs_listing_year.pdf")

plt.close()

print("Saved figure: V3_distortion_vs_listing_year")

# =====================================================
# FIGURE V4 — Distortion vs Padding Length
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(
    df["Padding_Days"],
    df["STD_pct"],
    color="#1f77b4",
    label="Return STD Distortion"
)

plt.scatter(
    df["Padding_Days"],
    df["GARCH_pct"],
    color="#ff7f0e",
    label="GARCH Distortion"
)

plt.xlabel("Padding Length (Days)")
plt.ylabel("Distortion (%)")

plt.title("Distortion vs Padding Length")

plt.legend(loc="upper left")

plt.tight_layout()

plt.savefig(f"{FIG_DIR}/V4_distortion_vs_padding_length.png", dpi=300)
plt.savefig(f"{FIG_DIR}/V4_distortion_vs_padding_length.pdf")

plt.close()

print("Saved figure: V4_distortion_vs_padding_length")

print("\nAll figures and summary statistics generated successfully.")