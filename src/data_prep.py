import pandas as pd

from config import (
    DATA_RAW_PATH,
    DATA_PROCESSED_PATH,
    TARGET_COL,
    TIME_COL,
    DROP_COLS,
)


def main():
    # 1) load raw data (raw stays read-only)
    df = pd.read_csv(DATA_RAW_PATH)
    print(f"[load] raw shape = {df.shape}")

    # 2) parse timestamp (validation needs time ordering)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    # 3) basic row validity: must have time + label
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()

    # 4) make label explicit binary int (avoid "0/1 as strings" issues)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 5) quick data quality summary (for validation report)
    pos_rate = df[TARGET_COL].mean()
    print(f"[label] positive rate = {pos_rate:.4f} (N={len(df)})")

    missing_top10 = df.isna().mean().sort_values(ascending=False).head(10)
    print("[missing] top 10 columns by missing rate:")
    print(missing_top10)

    # 6) leakage control: drop identifiers if present
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[leakage] dropped columns: {to_drop}")

    # 7) save processed modeling table
    df.to_parquet(DATA_PROCESSED_PATH, index=False)
    print(f"[save] wrote {DATA_PROCESSED_PATH} with shape={df.shape}")


if __name__ == "__main__":
    main()
