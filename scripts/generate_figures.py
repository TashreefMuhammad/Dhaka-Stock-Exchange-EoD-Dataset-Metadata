"""
Generate Descriptive and Coverage Figures
Dhaka Stock Exchange EoD Dataset

Figures:
D1 - Instrument Composition
D2 - Instrument Lifespan Distribution
C1 - Available Instruments Over Time
C2 - Coverage Ratio Over Time

All figures are saved as:
- PNG (300 DPI)
- PDF (vector format)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


# ==================================================
# Configuration
# ==================================================
FIGURE_DIR = "figures"
COMPANY_METADATA_PATH = "metadata/company_metadata.csv"
COVERAGE_PATH = "metadata/date_coverage_summary.csv"

os.makedirs(FIGURE_DIR, exist_ok=True)


# ==================================================
# Load Data
# ==================================================
company_metadata = pd.read_csv(COMPANY_METADATA_PATH)
coverage = pd.read_csv(COVERAGE_PATH)

coverage["Date"] = pd.to_datetime(coverage["Date"])
coverage = coverage.sort_values("Date")

print(f"Total instruments: {len(company_metadata)}")
print(f"Date range: {coverage['Date'].min()} to {coverage['Date'].max()}")


# ==================================================
# D1 — Instrument Composition
# ==================================================
counts = (
    company_metadata["Instrument_Type"]
    .value_counts()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

counts.plot(kind="bar", ax=ax)

for i, v in enumerate(counts):
    ax.text(i, v + 2, str(v), ha="center")

ax.set_xlabel("Instrument Type")
ax.set_ylabel("Number of Instruments")
ax.set_title("Instrument Composition in DSE EoD Dataset")

fig.tight_layout()

fig.savefig(f"{FIGURE_DIR}/D1_instrument_composition.png", dpi=300)
fig.savefig(f"{FIGURE_DIR}/D1_instrument_composition.pdf")

plt.close(fig)

print("D1 generated.")


# ==================================================
# D2 — Lifespan Distribution
# ==================================================
lifespans = company_metadata["Calendar_Days"]

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(lifespans, bins=50)

ax.set_xlabel("Instrument Lifespan (Calendar Days)")
ax.set_ylabel("Number of Instruments")
ax.set_title("Distribution of Instrument Lifespans")

fig.tight_layout()

fig.savefig(f"{FIGURE_DIR}/D2_lifespan_distribution.png", dpi=300)
fig.savefig(f"{FIGURE_DIR}/D2_lifespan_distribution.pdf")

plt.close(fig)

print("D2 generated.")


# ==================================================
# C1 — Available Instruments Over Time
# ==================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(coverage["Date"], coverage["Available_Any"], linewidth=1)
ax.plot(coverage["Date"], coverage["Available_Both"], linewidth=1)

ax.set_xlabel("Date")
ax.set_ylabel("Number of Available Instruments")
ax.set_title("Available Instruments Over Time")

ax.legend([
    "Available (Adjusted OR Unadjusted)",
    "Available (Both Versions)"
])

fig.tight_layout()

fig.savefig(f"{FIGURE_DIR}/C1_available_instruments_over_time.png", dpi=300)
fig.savefig(f"{FIGURE_DIR}/C1_available_instruments_over_time.pdf")

plt.close(fig)

print("C1 generated.")


# ==================================================
# C2 — Coverage Ratio Over Time
# ==================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(coverage["Date"], coverage["Coverage_Ratio_Full"], linewidth=1)

ax.set_xlabel("Date")
ax.set_ylabel("Coverage Ratio")
ax.set_title("Coverage Ratio Over Time")

fig.tight_layout()

fig.savefig(f"{FIGURE_DIR}/C2_coverage_ratio_over_time.png", dpi=300)
fig.savefig(f"{FIGURE_DIR}/C2_coverage_ratio_over_time.pdf")

plt.close(fig)

print("C2 generated.")


print("All figures successfully generated.")
