

from pathlib import Path
import pandas as pd

TRANSACTIONS_PATH = Path("C:/Users/tun/Desktop/project 1/data/processed/transactions_clean.parquet")
RETURNS_PATH = Path("C:/Users/tun/Desktop/project 1/data/processed/returns_clean.parquet")
OUTPUT_PATH = Path("C:/Users/tun/Desktop/project 1/data/processed/customer_features.parquet")

HOLDOUT_DAYS = 90


def build_features(transactions: pd.DataFrame, returns: pd.DataFrame,
                    holdout_days: int = HOLDOUT_DAYS) -> pd.DataFrame:
    """
    Build one row per customer: behavior features from the observation
    window, plus a churn label derived from the holdout window.
    """
    max_date = transactions["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    obs = transactions[transactions["InvoiceDate"] < cutoff_date].copy()
    holdout = transactions[transactions["InvoiceDate"] >= cutoff_date].copy()
    obs_returns = returns[returns["InvoiceDate"] < cutoff_date].copy()

    print(f"Cutoff date: {cutoff_date}")
    print(f"Observation window: {obs['InvoiceDate'].min()} to {obs['InvoiceDate'].max()} "
          f"({len(obs):,} purchase rows)")
    print(f"Holdout window: {holdout['InvoiceDate'].min()} to {holdout['InvoiceDate'].max()} "
          f"({len(holdout):,} purchase rows)")

    # --- RFM + behavior features, computed ONLY from the observation window ---
    grouped = obs.groupby("Customer ID")

    features = grouped.agg(
        recency_days=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
        frequency=("Invoice", "nunique"),
        monetary_total=("LineTotal", "sum"),
        distinct_products=("StockCode", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        country=("Country", lambda x: x.mode().iloc[0]),
    )

    features["avg_order_value"] = features["monetary_total"] / features["frequency"]
    features["tenure_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    features = features.drop(columns=["first_purchase", "last_purchase"])

    # --- Return behavior, also from the observation window only ---
    return_counts = obs_returns.groupby("Customer ID").size().rename("num_returns")
    features = features.join(return_counts, how="left")
    features["num_returns"] = features["num_returns"].fillna(0)
    features["return_rate"] = features["num_returns"] / features["frequency"]

    # --- Churn label, from the HOLDOUT window only ---
    customers_active_in_holdout = set(holdout["Customer ID"].unique())
    features["churned"] = (~features.index.isin(customers_active_in_holdout)).astype(int)

    features = features.reset_index()

    return features


def main():
    transactions = pd.read_parquet(TRANSACTIONS_PATH)
    returns = pd.read_parquet(RETURNS_PATH)

    features = build_features(transactions, returns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nBuilt features for {len(features):,} customers")
    print(f"Churn rate: {features['churned'].mean() * 100:.1f}%")
    print(f"\nColumn summary:\n{features.describe()}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
