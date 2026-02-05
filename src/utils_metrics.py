import numpy as np
from sklearn.metrics import roc_auc_score


def ks_stat(y_true, y_prob):
    """KS = max separation between positive and negative score CDFs (common in risk)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    thresholds = np.unique(y_prob)
    ks = 0.0
    for t in thresholds:
        tpr = (pos >= t).mean()
        fpr = (neg >= t).mean()
        ks = max(ks, abs(tpr - fpr))
    return float(ks)


def psi(expected, actual, bins=10, eps=1e-6):
    """
    PSI compares score distribution drift (train vs OOT).
    bins are based on expected (train) quantiles.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, qs)
    cuts[0], cuts[-1] = -np.inf, np.inf

    e_cnt = np.histogram(expected, bins=cuts)[0].astype(float)
    a_cnt = np.histogram(actual, bins=cuts)[0].astype(float)

    e_pct = np.clip(e_cnt / e_cnt.sum(), eps, None)
    a_pct = np.clip(a_cnt / a_cnt.sum(), eps, None)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def top_decile_capture(y_true, y_prob, top_pct=0.10):
    """How many true positives are captured in the top X% highest scores (very practical in AML)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_prob)
    k = max(1, int(np.ceil(n * top_pct)))
    idx = np.argsort(-y_prob)[:k]  # top scores
    captured_tp = y_true[idx].sum()
    total_tp = y_true.sum()
    if total_tp == 0:
        return np.nan
    return float(captured_tp / total_tp)
