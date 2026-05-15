import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
REGIMES_PATH  = r"E:\quant-research-ofi\features\SPY_features_with_regimes.parquet"
ORIGINAL_PATH = r"E:\quant-research-ofi\features\SPY_features.parquet"

ALL_16_FEATURES = [
    'ofi', 'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m', 'ofi_10m',
    'ofi_norm', 'queue_imbalance', 'trade_imbalance',
    'spread', 'spread_change', 'microprice', 'vwap',
    'kyle_lambda', 'amihud', 'realized_vol'
]

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
def load_data(regimes_path, original_path):
    df = pd.read_parquet(regimes_path)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts')
    df = df.sort_index()

    original = pd.read_parquet(original_path)
    if 'ts' in original.columns:
        original['ts'] = pd.to_datetime(original['ts'])
        original = original.set_index('ts')
    original = original.sort_index()

    cols_available = [c for c in ALL_16_FEATURES if c in original.columns]
    cols_to_drop   = [c for c in cols_available if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.join(original[cols_available], how='left')
    return df

# ── STEP 2: DYNAMIC REGIME LABELING ──────────────────────────────────────────
def label_regimes(df):
    vol_by_regime = df.groupby('regime')['realized_vol'].mean().sort_values()
    states        = vol_by_regime.index.tolist()
    if len(states) == 3:
        mapping = {states[0]: 'calm', states[1]: 'normal', states[2]: 'stressed'}
    elif len(states) == 2:
        mapping = {states[0]: 'calm', states[1]: 'stressed'}
    else:
        mapping = {s: f'regime_{int(s)}' for s in states}
    df = df.copy()
    df['regime_label'] = df['regime'].map(mapping)
    return df

# ── STEP 3: FULL CHARACTERIZATION TABLE ──────────────────────────────────────
def full_characterization(df):
    features = [f for f in ALL_16_FEATURES if f in df.columns]
    regimes  = ['calm', 'normal', 'stressed']
    rows     = []

    for regime in regimes:
        subset = df[df['regime_label'] == regime]
        for feat in features:
            col = subset[feat].dropna()
            if len(col) < 10:
                continue
            rows.append({
                'regime':   regime,
                'feature':  feat,
                'mean':     round(col.mean(), 6),
                'std':      round(col.std(),  6),
                'skewness': round(col.skew(), 4),
                'n':        len(col)
            })

    return pd.DataFrame(rows)

# ── STEP 4: STATISTICAL TESTS WITH COHEN'S D ─────────────────────────────────
def regime_significance_tests(df):
    """
    Mann-Whitney U test + Cohen's d effect size.
    Mann-Whitney: are calm and stressed distributions different?
    Cohen's d: how large is the difference economically?
    Rule of thumb: d<0.2 small, 0.2-0.5 medium, >0.8 large
    """
    features = [f for f in ALL_16_FEATURES if f in df.columns]
    calm     = df[df['regime_label'] == 'calm']
    stressed = df[df['regime_label'] == 'stressed']

    results = []
    for feat in features:
        x = calm[feat].dropna()
        y = stressed[feat].dropna()
        if len(x) < 30 or len(y) < 30:
            continue

        stat, pval   = stats.mannwhitneyu(x, y, alternative='two-sided')
        pooled_std   = np.sqrt((x.std()**2 + y.std()**2) / 2)
        cohens_d     = round((y.mean() - x.mean()) / (pooled_std + 1e-10), 4)

        results.append({
            'feature':       feat,
            'calm_mean':     round(x.mean(), 6),
            'stressed_mean': round(y.mean(), 6),
            'p_value':       round(pval,      6),
            'significant':   'YES' if pval < 0.05 else 'NO',
            'cohens_d':      cohens_d,
            'effect_size':   'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.2 else 'small'
        })

    return pd.DataFrame(results)

# ── STEP 5: REGIME STABILITY SCORE ───────────────────────────────────────────
def regime_stability_score(df):
    df         = df.copy()
    df['date'] = df.index.normalize()
    total_days = df['date'].nunique()

    print(f"\n{'Regime':<12} {'Days Present':>14} {'Avg Daily Pct':>16}")
    print("-"*45)

    for regime in ['calm', 'normal', 'stressed']:
        daily        = df.groupby('date').apply(
            lambda x: (x['regime_label'] == regime).mean()
        )
        days_present = (daily > 0).sum()
        avg_pct      = daily[daily > 0].mean()
        print(f"{regime:<12} {days_present:>14} {avg_pct*100:>15.1f}%")

    print(f"\nTotal trading days: {total_days}")

# ── STEP 6: PRINT + SAVE ──────────────────────────────────────────────────────
def print_and_save(char_df, sig_df):
    print("\n" + "="*70)
    print("FULL CHARACTERIZATION TABLE — ALL 16 FEATURES PER REGIME")
    print("="*70)
    pivot = char_df.pivot_table(
        index='feature',
        columns='regime',
        values=['mean', 'std', 'skewness']
    )
    print(pivot.to_string())

    print("\n" + "="*85)
    print("MANN-WHITNEY + COHEN'S D — CALM vs STRESSED")
    print("="*85)
    print(f"{'Feature':<20} {'Calm Mean':>12} {'Stressed Mean':>14} {'p-value':>10} {'Sig':>5} {'Cohen d':>10} {'Effect':>8}")
    print("-"*85)
    for _, row in sig_df.iterrows():
        print(f"{row['feature']:<20} {row['calm_mean']:>12} {row['stressed_mean']:>14} "
              f"{row['p_value']:>10} {row['significant']:>5} {row['cohens_d']:>10} {row['effect_size']:>8}")

    os.makedirs('reports', exist_ok=True)
    char_df.to_csv('reports/full_characterization.csv', index=False)
    sig_df.to_csv('reports/regime_significance_tests.csv',  index=False)
    print("\nSaved:")
    print("  reports/full_characterization.csv")
    print("  reports/regime_significance_tests.csv")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("REGIME CHARACTERIZATION — FULL FEATURE STATS + SIGNIFICANCE TESTS")
    print("="*70)

    print("\nLoading data...")
    df = load_data(REGIMES_PATH, ORIGINAL_PATH)

    print("Labeling regimes...")
    df = label_regimes(df)

    print("Computing full characterization table...")
    char_df = full_characterization(df)

    print("Running Mann-Whitney + Cohen's d tests...")
    sig_df = regime_significance_tests(df)

    print("Computing regime stability...")
    regime_stability_score(df)

    print_and_save(char_df, sig_df)

    print("\nDONE. Push to GitHub.")
