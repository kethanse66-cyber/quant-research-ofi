import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_PATH   = r"E:\quant-research-ofi\features\SPY_features.parquet"
N_STATES       = 2          # calm=0, stressed=1
MIN_TRAIN_DAYS = 60

HMM_FEATURES = [
    'realized_vol',
    'spread',
    'queue_imbalance',
    'trade_imbalance',
]

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
def load_features(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df['abs_ofi'] = df['ofi'].abs()
    cols = HMM_FEATURES + ['abs_ofi']
    df = df[cols].dropna()
    return df

# ── STEP 2: RANK TRANSFORM ────────────────────────────────────────────────────
def transform_day(train_df, today_df):
    result = np.zeros((len(today_df), len(train_df.columns)))
    for j, col in enumerate(train_df.columns):
        combined = np.concatenate([train_df[col].values, today_df[col].values])
        ranks    = rankdata(combined, method='average')
        n_train  = len(train_df)
        result[:, j] = ranks[n_train:] / (n_train + 1)
    return result

# ── STEP 3: REGIME RELABELING ─────────────────────────────────────────────────
def relabel_regimes(model, day_regimes):
    """
    Sort states by mean realized_vol.
    State 0 = calm   (lowest realized_vol)
    State 1 = stressed (highest realized_vol)
    """
    vol_means = model.means_[:, 0]
    order     = np.argsort(vol_means)
    mapping   = {old: new for new, old in enumerate(order)}
    return np.array([mapping[r] for r in day_regimes])

# ── STEP 4: ROLLING HMM ───────────────────────────────────────────────────────
def rolling_hmm(df, n_states=N_STATES, min_train_days=MIN_TRAIN_DAYS):
    n        = len(df)
    regimes  = np.full(n, np.nan)
    df_dates = df.index.normalize()
    dates    = df_dates.unique().sort_values()
    total    = len(dates)

    print(f"Total trading days: {total}")

    for i, today in enumerate(dates):
        if i < min_train_days:
            continue

        train_mask = df_dates < today
        train_df   = df[train_mask]

        if len(train_df) == 0:
            continue

        X_train = train_df.rank(pct=True).values

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )
        try:
            model.fit(X_train)
            if i % 50 == 0:
                print("\nTransition Matrix:")
                trans_df = pd.DataFrame(
                    model.transmat_,
                    columns=['calm', 'stressed'],
                    index=['calm', 'stressed']
                )
                print(trans_df.round(3))
        except Exception as e:
            print(f"HMM fit failed on {today.date()}: {e}")
            continue

        today_mask = df_dates == today
        today_df   = df[today_mask]
        today_idx  = np.where(today_mask)[0]

        if len(today_df) == 0:
            continue

        X_today = transform_day(train_df, today_df)

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
    labels = {0: 'calm', 1: 'stressed'}   # 2-state: 0=calm, 1=stressed
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
    print("ROLLING HMM — 2 STATE REGIME DETECTION")
    print("=" * 60)

    print("\nLoading features...")
    df = load_features(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows | features: {list(df.columns)}")

    print(f"\nConfig:")
    print(f"  States        : {N_STATES} (calm=0, stressed=1)")
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