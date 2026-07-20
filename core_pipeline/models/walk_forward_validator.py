import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# ── CONFIG ──────────────────────────────────────────────────────────────
FEATURES_DIR = Path(r"E:\quant-research-ofi\features")
TICKER       = "SPY"
TRAIN_MONTHS = 3
TEST_MONTHS  = 1
TARGET_COLS = ["fwd_ret_10s", "fwd_ret_30s", "fwd_ret_1m", "fwd_ret_5m", "fwd_ret_10m"]

FEATURE_COLS = [
    "ofi", "ofi_10s", "ofi_30s", "ofi_1m", "ofi_5m", "ofi_10m",
    "queue_imbalance", "trade_imbalance",
    "spread", "spread_change",
    "microprice", "vwap",
    "kyle_lambda", "amihud",
    "realized_vol", "ofi_norm"
]

# ── LOAD DATA ────────────────────────────────────────────────────────────
def load_features(ticker: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{ticker}_features.parquet"
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.sort_index()
    return df

# ── WALK-FORWARD FOLDS (timestamp-exact, no normalize) ───────────────────
def make_folds(df: pd.DataFrame, train_months: int, test_months: int) -> list[dict]:
    """
    Expanding-window walk-forward folds using exact timestamps.
    Train window grows each fold. Test window is always test_months wide.
    Returns list of dicts: {fold, train_start, train_end, test_start, test_end}
    """
    start = df.index.min()
    end   = df.index.max()

    folds     = []
    fold_num  = 0
    train_end = start + pd.DateOffset(months=train_months)

    while train_end + pd.DateOffset(months=test_months) <= end + pd.Timedelta(days=1):
        test_end = train_end + pd.DateOffset(months=test_months)

        # get actual timestamps that fall in train and test windows
        train_mask = (df.index >= start)     & (df.index < train_end)
        test_mask  = (df.index >= train_end) & (df.index < test_end)

        train_actual_end   = df.index[train_mask].max() if train_mask.any() else train_end
        test_actual_start  = df.index[test_mask].min()  if test_mask.any()  else train_end
        test_actual_end    = df.index[test_mask].max()  if test_mask.any()  else test_end

        folds.append({
            "fold"        : fold_num,
            "train_start" : start,
            "train_end"   : train_actual_end,
            "test_start"  : test_actual_start,
            "test_end"    : test_actual_end,
            # keep DateOffset boundaries for slicing
            "_train_end_boundary" : train_end,
            "_test_end_boundary"  : test_end,
        })

        fold_num  += 1
        train_end  = test_end

    return folds

# ── IC PER FOLD ──────────────────────────────────────────────────────────
def compute_fold_ic(df: pd.DataFrame, fold: dict, feature: str, target: str) -> dict:
    """
    For one fold: slice train (for temporal integrity check), slice test,
    apply shift(1) to feature, drop NaNs, compute Spearman IC vs target.
    shift(1) applied HERE only — never saved to parquet.
    """
    train = df[
        (df.index >= fold["train_start"]) &
        (df.index <= fold["train_end"])
    ].copy()

    test = df[
        (df.index >= fold["test_start"]) &
        (df.index <= fold["test_end"])
    ].copy()

    # temporal integrity: confirm no test row leaks into train
    if len(train) > 0 and len(test) > 0:
        assert train.index.max() < test.index.min(), \
            f"Fold {fold['fold']}: train/test overlap detected — lookahead bias!"

    # shift(1) here — lags feature by one bar so no future data used
    test["feature_lagged"] = test[feature].shift(1)
    test = test.dropna(subset=["feature_lagged", target])

    if len(test) < 50:
        return {
            "fold": fold["fold"], "feature": feature, "target": target,
            "ic": np.nan, "pval": np.nan, "n": len(test),
            "train_rows": len(train),
            "train_start": fold["train_start"].date(),
            "train_end"  : fold["train_end"].date(),
            "test_start" : fold["test_start"].date(),
            "test_end"   : fold["test_end"].date(),
        }

    ic, pval = spearmanr(test["feature_lagged"], test[target])
    return {
        "fold"       : fold["fold"],
        "feature"    : feature,
        "target"     : target,
        "ic"         : round(ic,   6),
        "pval"       : round(pval, 6),
        "n"          : len(test),
        "train_rows" : len(train),
        "train_start": fold["train_start"].date(),
        "train_end"  : fold["train_end"].date(),
        "test_start" : fold["test_start"].date(),
        "test_end"   : fold["test_end"].date(),
    }

# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading {TICKER} features...")
    df    = load_features(TICKER)
    folds = make_folds(df, TRAIN_MONTHS, TEST_MONTHS)

    print(f"\nTotal folds: {len(folds)}")
    print(f"{'Fold':<6} {'Train Start':<14} {'Train End':<14} {'Test Start':<14} {'Test End':<14} {'Train Rows':<12}")
    print("-" * 75)
    for f in folds:
        print(f"{f['fold']:<6} {str(f['train_start'].date()):<14} {str(f['train_end'].date()):<14} "
              f"{str(f['test_start'].date()):<14} {str(f['test_end'].date()):<14} {str(f['train_rows'] if 'train_rows' in f else '?'):<12}")

    # ── COMPUTE IC ───────────────────────────────────────────────────────
    print(f"\nComputing IC across {len(folds)} folds x {len(FEATURE_COLS)} features x {len(TARGET_COLS)} targets...")
    results = []
    for fold in folds:
        for feat in FEATURE_COLS:
            for target in TARGET_COLS:
                results.append(compute_fold_ic(df, fold, feat, target))

    results_df = pd.DataFrame(results)

    # ── IC DECAY TABLE (per target horizon) ──────────────────────────────
    print(f"\n── IC Decay by Horizon (mean across all folds, feature=ofi) ──")
    ofi_rows = results_df[results_df["feature"] == "ofi"]
    decay = ofi_rows.groupby("target")["ic"].agg(mean_ic="mean", std_ic="std").reset_index()
    decay["icir"] = (decay["mean_ic"] / decay["std_ic"]).round(4)
    decay["mean_ic"] = decay["mean_ic"].round(6)
    print(decay.to_string(index=False))

    # ── FULL SUMMARY TABLE ───────────────────────────────────────────────
    summary = (
        results_df[results_df["target"] == "fwd_ret_5m"]
        .groupby("feature")["ic"]
        .agg(mean_ic="mean", icir=lambda x: x.mean() / x.std() if x.std() > 0 else np.nan)
        .reset_index()
        .sort_values("icir", ascending=False)
    )
    summary["mean_ic"] = summary["mean_ic"].round(6)
    summary["icir"]    = summary["icir"].round(4)

    print(f"\n── IC Summary (target=fwd_ret_5m, {len(folds)} folds) ──")
    print(f"{'Feature':<20} {'Mean IC':<12} {'ICIR':<8}")
    print("-" * 42)
    for _, row in summary.iterrows():
        print(f"{row['feature']:<20} {row['mean_ic']:<12} {row['icir']:<8}")

    # ── SAVE ─────────────────────────────────────────────────────────────
    out_path = FEATURES_DIR / "walk_forward_ic.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Total rows: {len(results_df)} | Folds: {len(folds)} | Features: {len(FEATURE_COLS)} | Targets: {len(TARGET_COLS)}")

if __name__ == "__main__":
    main()
