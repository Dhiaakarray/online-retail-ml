

from pathlib import Path
import pandas as pd

RAW_XLSX_PATH = Path("C:/Users/tun/Desktop/project 1/data/raw/online_retail_II.xlsx")
RAW_OUTPUT_PATH = Path("data/raw/transactions_raw.parquet")


def load_raw_data(xlsx_path: Path = RAW_XLSX_PATH) -> pd.DataFrame:
    sheet_names = ["Year 2009-2010", "Year 2010-2011"]

    frames = []
    for sheet in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    combined["Invoice"] = combined["Invoice"].astype(str)
    combined["StockCode"] = combined["StockCode"].astype(str)
    combined["Description"] = combined["Description"].astype(str)

    return combined


def main():
    df = load_raw_data()
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_OUTPUT_PATH, index=False)
    print(f"Loaded {len(df):,} rows from {RAW_XLSX_PATH}")
    print(f"Saved combined raw data to {RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
