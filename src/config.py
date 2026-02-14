"""
AML Model Validation project config.

Put all "project-wide rules" here so the pipeline is reproducible:
- where data lives
- what the label / time column is
- what fields must be excluded to avoid leakage
- how we define the OOT backtest split
"""

from pathlib import Path

# ============================================================
# Project root (ALWAYS correct regardless of where you run from)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------
# Data locations (raw is read-only; processed is reproducible output)
# ------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure processed dir exists (raw dir should exist because you place files there)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Use Path objects internally (best for cross-platform)
DATA_RAW_PATH = RAW_DIR / "aml_transactions_raw.csv"
DATA_PROCESSED_PATH = PROCESSED_DIR / "modeling_table.parquet"

# If you also want to reference RFtest locally:
RF_TEST_PATH = RAW_DIR / "RFtest.csv"

# ------------------------------------------------------------
# Schema: columns we rely on across scripts
# ------------------------------------------------------------
TARGET_COL = "label_suspicious"   # 1 = suspicious, 0 = normal
TIME_COL = "timestamp"           # used for out-of-time split

# ------------------------------------------------------------
# Leakage controls: IDs can cause memorization + unrealistic performance
# ------------------------------------------------------------
DROP_COLS = ["transaction_id", "customer_id"]

# ------------------------------------------------------------
# Reproducibility / evaluation setup
# ------------------------------------------------------------
RANDOM_STATE = 42

# Out-of-time (OOT) split: everything after this date is treated as "future"
OOT_SPLIT_DATE = "2025-06-30"
