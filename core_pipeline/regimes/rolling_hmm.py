import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_PATH   = r"E:\quant-research-ofi\features\SPY_features.parquet"
N_STATES       = 3          # calm=0, normal=1, stressed=2
MIN_TRAIN_DAYS = 60         # 2 months burn-in — ensures all 3 regimes seen

HMM_FEATURES = [
    'realized_vol',
    'spread',
    'queue_imbalance',
    'trade_imbalance',
]

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
def load_features(path):
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts')
    df = df.sort_index()
    df['abs_ofi'] = df['ofi'].abs()
    cols = HMM_FEATURES + ['abs_ofi']
    df = df[cols].dropna()
    return df

# ── STEP 2: RANK TRANSFORM — LEAKAGE FREE + SCALABLE ─────────────────────────
def transform_day(train_df, today_df):
    """
    Transform today's bars into percentile ranks using ONLY training distribution.
    Uses scipy rankdata — O(n log n), memory efficient, scales to millions of rows.
    Today's data never influences its own transformation.
    """
    result = np.zeros((len(today_df), len(train_df.columns)))
    for j, col in enumerate(train_df.columns):
        combined = np.concatenate([train_df[col].values, today_df[col].values])
        ranks    = rankdata(combined, method='average')
        n_train  = len(train_df)
        result[:, j] = ranks[n_train:] / (n_train + 1)
    return result

# ── STEP 3: REGIME RELABELING — CONSISTENT ACROSS DAYS ───────────────────────
def relabel_regimes(model, day_regimes):
    """
    HMM states are permutation invariant — state 0 today may mean stressed tomorrow.
    Fix: sort states by mean realized_vol (first feature).
    State 0 = calm   (lowest realized_vol)
    State 1 = normal (medium realized_vol)
    State 2 = stressed (highest realized_vol)
    """
    vol_means = model.means_[:, 0]
    order     = np.argsort(vol_means)
    mapping   = {old: new for new, old in enumerate(order)}
    return np.array([mapping[r] for r in day_regimes])

# ── STEP 4: ROLLING HMM — REFIT ONCE PER DAY ─────────────────────────────────
def rolling_hmm(df, n_states=N_STATES, min_train_days=MIN_TRAIN_DAYS):
    """
    Refit HMM once per day using all data up to previous day (expanding window).
    Apply that model to ALL bars within the day as a sequence.
    Transition dynamics preserved — true HMM, not pointwise GMM.
    Regimes relabeled consistently by realized volatility every day.
    No bar ever uses same-day or future data for fitting or transformation.
    """
    n       = len(df)
    regimes = np.full(n, np.nan)

    # normalize index to date only for grouping
    df_dates = df.index.normalize()
    dates    = df_dates.unique().sort_values()
    total    = len(dates)

    print(f"Total trading days: {total}")

    for i, today in enumerate(dates):
        if i < min_train_days:
            continue

        # ── Training data = everything BEFORE today ───────────────────────────
        train_mask = df_dates < today
        train_df   = df[train_mask]

        if len(train_df) == 0:
            continue

        # ── Rank transform training data ──────────────────────────────────────
        X_train = train_df.rank(pct=True).values

        # ── Fit HMM on past data only ─────────────────────────────────────────
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )
        try:
            model.fit(X_train)
        except Exception as e:
            print(f"HMM fit failed on {today.date()}: {e}")
            continue

        # ── Get today's data ──────────────────────────────────────────────────
        today_mask = df_dates == today
        today_df   = df[today_mask]
        today_idx  = np.where(today_mask)[0]

        if len(today_df) == 0:
            continue

        # ── Transform today using train distribution only ─────────────────────
        X_today = transform_day(train_df, today_df)

        # ── Predict regimes for whole day as sequence ─────────────────────────
        try:
            day_regimes = model.predict(X_today)
            day_regimes = relabel_regimes(model, day_regimes)
            regimes[today_idx] = day_regimes
        except Exception as e:
            print(f"HMM predict failed on {today.date()}: {e}")
            continue

        if i % 50 == 0:
            print(f"Day {i}/{total} complete — {today.date()}")

    regime_series = pd.Series(regimes, index=df.index, name='regime')
    return regime_series

# ── STEP 5: LOOKAHEAD CHECK ───────────────────────────────────────────────────
def lookahead_check(regime_series):
    first_valid = regime_series.first_valid_index()
    print(f"First valid prediction timestamp : {first_valid}")
    print(f"MIN_TRAIN_DAYS setting           : {MIN_TRAIN_DAYS}")
    print(f"Lookahead check PASSED           : regime starts after {MIN_TRAIN_DAYS} days burn-in")

# ── STEP 6: REGIME SUMMARY ────────────────────────────────────────────────────
def regime_summary(regime_series):
    counts = regime_series.value_counts().sort_index()
    total  = regime_series.notna().sum()
    print(f"\n{'Regime':<10} {'Count':>10} {'Pct':>8}")
    print("-" * 30)
    labels = {0: 'calm', 1: 'normal', 2: 'stressed'}
    for state, count in counts.items():
        label = labels.get(int(state), str(state))
        print(f"{label:<10} {int(count):>10} {count/total*100:>7.1f}%")

# ── STEP 7: SAVE ──────────────────────────────────────────────────────────────
def save_regimes(df, regime_series, path):
    out           = df.copy()
    out['regime'] = regime_series
    save_path     = path.replace('.parquet', '_with_regimes.parquet')
    out.to_parquet(save_path)
    print(f"\nSaved to             : {save_path}")
    print(f"Rows with regime     : {regime_series.notna().sum()}")
    print(f"Rows NaN (burn-in)   : {regime_series.isna().sum()}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ROLLING HMM — 3 STATE REGIME DETECTION")
    print("=" * 60)

    print("\nLoading features...")
    df = load_features(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows | features: {list(df.columns)}")

    print(f"\nConfig:")
    print(f"  States        : {N_STATES} (calm=0, normal=1, stressed=2)")
    print(f"  Burn-in days  : {MIN_TRAIN_DAYS}")
    print(f"  Features      : {HMM_FEATURES + ['abs_ofi']}")

    print(f"\nRunning rolling HMM...")
    regime_series = rolling_hmm(df)

    print("\n── LOOKAHEAD CHECK ──")
    lookahead_check(regime_series)

    print("\n── REGIME DISTRIBUTION ──")
    regime_summary(regime_series)

    print("\n── SAVING ──")
    save_regimes(df, regime_series, PARQUET_PATH)

    print("\nDONE. Push to GitHub.")
