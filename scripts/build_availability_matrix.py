import pandas as pd
import os

if __name__ == "__main__":

    # Ensure output directory exists
    os.makedirs("metadata", exist_ok=True)

    # -----------------------------
    # 1. Load data [It is expected that you have the main CSV files stored in the same folder as this code is running]
    # -----------------------------
    combine1 = pd.read_csv("UnAdjusted-AmarStock.csv")
    combine2 = pd.read_csv("Adjusted-AmarStock.csv")

    # -----------------------------
    # 2. Normalize dates
    # -----------------------------
    combine1["Date"] = pd.to_datetime(combine1["Date"]).dt.normalize()
    combine2["Date"] = pd.to_datetime(combine2["Date"]).dt.normalize()

    # -----------------------------
    # 3. Extract trading codes
    # -----------------------------
    tradingCodes1 = combine1["Ticker"].unique()
    tradingCodes2 = combine2["Ticker"].unique()

    all_codes = sorted(set(tradingCodes1) | set(tradingCodes2))

    # -----------------------------
    # 4. Generate full calendar
    # -----------------------------
    start_date = "2012-10-01"
    end_date   = "2026-01-25"

    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # -----------------------------
    # 5. Initialize availability matrix
    # -----------------------------
    all_options = pd.DataFrame(
        0,
        index=all_dates,
        columns=all_codes,
        dtype="uint8"
    )

    all_options.index.name = "Date"

    # -----------------------------
    # 6. Mark adjusted availability (+1)
    # -----------------------------
    adj_pairs = (
        combine2[["Date", "Ticker"]]
        .drop_duplicates()
        .set_index(["Date", "Ticker"])
        .index
    )

    stacked = all_options.stack()
    stacked.loc[adj_pairs] += 1

    # -----------------------------
    # 7. Mark unadjusted availability (+2)
    # -----------------------------
    unadj_pairs = (
        combine1[["Date", "Ticker"]]
        .drop_duplicates()
        .set_index(["Date", "Ticker"])
        .index
    )

    stacked.loc[unadj_pairs] += 2

    # -----------------------------
    # 8. Restore DataFrame shape
    # -----------------------------
    all_options = stacked.unstack().reset_index()

    # -----------------------------
    # 9. Sanity checks
    # -----------------------------
    print("Value counts (must be 0,1,2,3 only):")
    print(all_options.iloc[:, 1:].stack().value_counts())

    print("\nNumber of instruments:")
    print(len(all_codes))

    # -----------------------------
    # 10. Save output
    # -----------------------------
    all_options.to_csv(
        "metadata/availability_matrix.csv",
        index=False
    )
