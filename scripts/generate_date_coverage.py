import pandas as pd
import os

if __name__ == "__main__":

    # -----------------------------
    # Load availability matrix
    # -----------------------------
    df = pd.read_csv("metadata/availability_matrix.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    tickers = df.columns[1:]  # exclude Date
    total_instruments = len(tickers)

    records = []

    for _, row in df.iterrows():
        values = row[tickers]

        available_any = (values > 0).sum()
        available_unadjusted = values.isin([2, 3]).sum()
        available_adjusted = values.isin([1, 3]).sum()
        available_both = (values == 3).sum()

        coverage_ratio = (
            available_both / total_instruments
            if total_instruments > 0 else 0
        )

        day_name = row["Date"].strftime("%A")
        is_weekend = day_name in ["Friday", "Saturday"]  # DSE weekend

        records.append({
            "Date": row["Date"].date(),
            "DayOfWeek": day_name,
            "IsWeekend": is_weekend,
            "Total_Instruments": total_instruments,
            "Available_Any": available_any,
            "Available_Unadjusted": available_unadjusted,
            "Available_Adjusted": available_adjusted,
            "Available_Both": available_both,
            "Coverage_Ratio": round(coverage_ratio, 4)
        })

    date_coverage = pd.DataFrame(records)

    os.makedirs("metadata", exist_ok=True)
    date_coverage.to_csv(
        "metadata/date_coverage_summary.csv",
        index=False
    )

    print("Per-date coverage metadata generated.")
    print("Number of dates:", len(date_coverage))

#%%
avail = pd.read_csv("metadata/availability_matrix.csv")
avail["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

print(avail.iloc[0]["Date"], avail.iloc[0]["Date"].strftime("%A"))
