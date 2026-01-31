import pandas as pd
import os

def infer_instrument_type(ticker: str) -> str:
    t = ticker.upper()

    if t.startswith("00"):
        return "Index"
    if t.startswith("TB"):
        return "TreasuryBill"
    if "SUKUK" in t:
        return "Sukuk"
    if "BOND" in t:
        return "Bond"
    if "MF" in t:
        return "MutualFund"
    return "Equity"


if __name__ == "__main__":

    # -----------------------------
    # Load availability matrix
    # -----------------------------
    df = pd.read_csv("metadata/availability_matrix.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    tickers = df.columns[1:]  # exclude Date

    records = []

    for ticker in tickers:
        series = df[["Date", ticker]].copy()
        present = series[series[ticker] > 0]

        if present.empty:
            continue

        first_date = present["Date"].min()
        last_date  = present["Date"].max()
        calendar_days = (last_date - first_date).days + 1

        days_adjusted   = (present[ticker].isin([1, 3])).sum()
        days_unadjusted = (present[ticker].isin([2, 3])).sum()
        days_both       = (present[ticker] == 3).sum()

        coverage_ratio = days_both / calendar_days if calendar_days > 0 else 0

        records.append({
            "Ticker": ticker,
            "Instrument_Type": infer_instrument_type(ticker),
            "First_Date": first_date.date(),
            "Last_Date": last_date.date(),
            "Calendar_Days": calendar_days,
            "Days_Adjusted": days_adjusted,
            "Days_Unadjusted": days_unadjusted,
            "Days_Both": days_both,
            "Coverage_Ratio": round(coverage_ratio, 4)
        })

    company_meta = pd.DataFrame(records)

    os.makedirs("metadata", exist_ok=True)
    company_meta.to_csv("metadata/company_metadata.csv", index=False)

    print("Company metadata generated.")
    print("Number of instruments:", len(company_meta))
