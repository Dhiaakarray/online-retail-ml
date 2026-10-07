
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/transactions_raw.parquet")
CLEAN_TRANSACTIONS_PATH = Path("data/processed/transactions_clean.parquet")
CLEAN_RETURNS_PATH = Path("data/processed/returns_clean.parquet")
REPORT_PATH = Path("data/processed/etl_report.txt")

NON_PRODUCT_STOCK_CODES = {
    "POST", "DOT", "M", "m", "D", "S",
    "ADJUST", "AMAZONFEE", "B", "CRUK", "GIFT",
}


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    log = []
    n_start = len(df)
    log.append(f"Starting rows: {n_start:,}")

    before = len(df)
    df = df.drop_duplicates()
    log.append(f"Dropped exact duplicate rows: {before - len(df):,} "
               f"(remaining: {len(df):,})")

    is_return = df["Invoice"].str.startswith("C") | (df["Quantity"] < 0)
    returns_df = df[is_return].copy()
    purchases_df = df[~is_return].copy()
    log.append(f"Split off return/cancellation rows: {len(returns_df):,} "
               f"(remaining candidate purchases: {len(purchases_df):,})")

    before = len(returns_df)
    returns_df = returns_df.dropna(subset=["Customer ID"])
    log.append(f"Returns: dropped rows with missing Customer ID: "
               f"{before - len(returns_df):,} (remaining: {len(returns_df):,})")


    before = len(purchases_df)
    purchases_df = purchases_df.dropna(subset=["Customer ID"])
    log.append(f"Purchases: dropped rows with missing Customer ID: "
               f"{before - len(purchases_df):,} (remaining: {len(purchases_df):,})")


    before = len(purchases_df)
    purchases_df = purchases_df[purchases_df["Price"] > 0]
    log.append(f"Purchases: dropped Price <= 0 rows: "
               f"{before - len(purchases_df):,} (remaining: {len(purchases_df):,})")


    before = len(purchases_df)
    purchases_df = purchases_df[purchases_df["Quantity"] > 0]
    log.append(f"Purchases: dropped Quantity <= 0 rows: "
               f"{before - len(purchases_df):,} (remaining: {len(purchases_df):,})")

    before = len(purchases_df)
    purchases_df = purchases_df[~purchases_df["StockCode"].isin(NON_PRODUCT_STOCK_CODES)]
    log.append(f"Purchases: dropped non-product StockCodes (postage, fees, "
               f"discounts, etc.): {before - len(purchases_df):,} "
               f"(remaining: {len(purchases_df):,})")

    purchases_df["LineTotal"] = purchases_df["Quantity"] * purchases_df["Price"]
    returns_df["LineTotal"] = returns_df["Quantity"] * returns_df["Price"]

    log.append(f"\nFinal clean purchases: {len(purchases_df):,} rows "
               f"({len(purchases_df) / n_start * 100:.1f}% of raw data)")
    log.append(f"Final clean returns: {len(returns_df):,} rows "
               f"({len(returns_df) / n_start * 100:.1f}% of raw data)")
    log.append(f"Total rows discarded entirely: "
               f"{n_start - len(purchases_df) - len(returns_df):,}")

    return purchases_df, returns_df, log


def load_raw_data() -> pd.DataFrame:
    """
    Load raw transaction data. If data/raw/transactions_raw.parquet exists,
    read it directly (fast). Otherwise, read from online_retail_II.xlsx,
    combine all sheets, save to data/raw/transactions_raw.parquet, and return it.
    """
    if RAW_PATH.exists():
        print(f"Loading raw data from {RAW_PATH}...")
        return pd.read_parquet(RAW_PATH)

    excel_path = Path("online_retail_II.xlsx")
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Neither raw parquet file ({RAW_PATH}) nor Excel file ({excel_path}) were found!"
        )

    print(f"Raw parquet file not found. Converting {excel_path} sheets to {RAW_PATH}...")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(excel_path)
    df_list = [pd.read_excel(xl, sheet_name=sheet) for sheet in xl.sheet_names]
    df_raw = pd.concat(df_list, ignore_index=True)

    for col in ["Invoice", "StockCode", "Description"]:
        df_raw[col] = df_raw[col].apply(lambda x: str(x) if pd.notna(x) else None)

    df_raw.to_parquet(RAW_PATH, index=False)
    print(f"Saved raw transactions parquet to {RAW_PATH} ({len(df_raw):,} rows).\n")
    return df_raw


def main():
    df = load_raw_data()
    purchases_df, returns_df, log = clean_transactions(df)

    CLEAN_TRANSACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    purchases_df.to_parquet(CLEAN_TRANSACTIONS_PATH, index=False)
    returns_df.to_parquet(CLEAN_RETURNS_PATH, index=False)

    report_text = "\n".join(log)
    REPORT_PATH.write_text(report_text)

    print(report_text)
    print(f"\nSaved clean purchases to {CLEAN_TRANSACTIONS_PATH}")
    print(f"Saved clean returns to {CLEAN_RETURNS_PATH}")
    print(f"Saved ETL report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

