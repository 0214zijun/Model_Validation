import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (
    DATA_PROCESSED_PATH,
    TARGET_COL,
    TIME_COL,
    RANDOM_STATE,
    OOT_SPLIT_DATE,
)


def main():
    # 1) load processed data
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    print(f"[load] processed shape = {df.shape}")

    # 2) time-based split (train vs OOT)
    cutoff = pd.to_datetime(OOT_SPLIT_DATE)
    train_df = df[df[TIME_COL] < cutoff].copy()
    oot_df = df[df[TIME_COL] >= cutoff].copy()

    print(f"[split] train={len(train_df)}, oot={len(oot_df)}")

    # 3) separate X / y
    X_train = train_df.drop(columns=[TARGET_COL, TIME_COL])
    y_train = train_df[TARGET_COL].astype(int)

    X_oot = oot_df.drop(columns=[TARGET_COL, TIME_COL])
    y_oot = oot_df[TARGET_COL].astype(int)

    # 4) baseline: logistic regression
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)

    p_lr_train = lr.predict_proba(X_train)[:, 1]
    p_lr_oot = lr.predict_proba(X_oot)[:, 1]

    print(
        f"[LR] train AUC={roc_auc_score(y_train, p_lr_train):.4f}, "
        f"OOT AUC={roc_auc_score(y_oot, p_lr_oot):.4f}, "
        f"OOT PR-AUC={average_precision_score(y_oot, p_lr_oot):.4f}"
    )

    # 5) challenger: random forest
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)

    p_rf_train = rf.predict_proba(X_train)[:, 1]
    p_rf_oot = rf.predict_proba(X_oot)[:, 1]

    print(
        f"[RF] train AUC={roc_auc_score(y_train, p_rf_train):.4f}, "
        f"OOT AUC={roc_auc_score(y_oot, p_rf_oot):.4f}, "
        f"OOT PR-AUC={average_precision_score(y_oot, p_rf_oot):.4f}"
    )


if __name__ == "__main__":
    main()
