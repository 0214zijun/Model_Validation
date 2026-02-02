"""
AML Model Validation project config.

Put all "project-wide rules" here so the pipeline is reproducible:
- where data lives
- what the label / time column is
- what fields must be excluded to avoid leakage
- how we define the OOT backtest split
"""

# ---- data locations (raw is read-only; processed is reproducible output)
DATA_RAW_PATH = "data/raw/aml_transactions_raw.csv"
DATA_PROCESSED_PATH = "data/processed/modeling_table.parquet"

# ---- schema: columns we rely on across scripts
TARGET_COL = "label_suspicious"   # 1 = suspicious, 0 = normal
TIME_COL = "timestamp"           # used for out-of-time split

# ---- leakage controls: IDs can cause memorization + unrealistic performance
DROP_COLS = ["transaction_id", "customer_id"]

# ---- reproducibility / evaluation setup
RANDOM_STATE = 42

# Out-of-time (OOT) split: everything after this date is treated as "future"
OOT_SPLIT_DATE = "2025-06-30"

