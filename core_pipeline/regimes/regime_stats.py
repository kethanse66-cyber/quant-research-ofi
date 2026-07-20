import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
REGIMES_PATH  = r"E:\quant-research-ofi\features\SPY_features_with_regimes.parquet"
ORIGINAL_PATH = r"E:\quant-research-ofi\features\SPY_features.parquet"

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

    cols_to_join = [
        'ofi', 'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m', 'ofi_10m',
        'ofi_norm', 'queue_imbalance', 'trade_imbalance',
        'spread', 'spread_change', 'microprice', 'vwap',
        'kyle_lambda', 'amihud', 'realized_vol'
    ]
    cols_available = [c for c in cols_to_join if c in original.columns]
    
    # drop columns from df that already exist in original to avoid overlap
    cols_to_drop = [c for c in cols_available if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    df = df.join(original[cols_available], how='left')
    return df

# ── STEP 2: DYNAMIC REGIME LABELING ──────────────────────────────────────────
def label_regimes(df):
    vol_by_regime = df.groupby('regime')['realized_vol'].mean().sort_values()
    states = vol_by_regime.index.tolist()
    if len(states) == 3:
        mapping = {states[0]: 'calm', states[1]: 'normal', states[2]: 'stressed'}
    elif len(states) == 2:
        mapping = {states[0]: 'calm', states[1]: 'stressed'}
    else:
        mapping = {s: f'regime_{int(s)}' for s in states}
    df = df.copy()
    df['regime_label'] = df['regime'].map(mapping)
    return df, mapping

# ── STEP 3: TRANSITION MATRIX ─────────────────────────────────────────────────
def compute_transition_matrix(df):
    """
    Transition matrix: probability of moving from regime A to regime B.
    Only look at consecutive rows within same day — no overnight transitions.
    """
    df = df.copy()
    df['date']         = df.index.normalize()
    df['next_regime']  = df['regime_label'].shift(-1)
    df['next_date']    = df['date'].shift(-1)

    # only keep rows where next row is same day
    same_day = df['date'] == df['next_date']
    transitions = df[same_day & df['regime_label'].notna() & df['next_regime'].notna()]

    regimes = sorted(transitions['regime_label'].unique())
    matrix  = pd.DataFrame(0.0, index=regimes, columns=regimes)

    for _, row in transitions.iterrows():
        matrix.loc[row['regime_label'], row['next_regime']] += 1

    # normalize rows to get probabilities
    matrix = matrix.div(matrix.sum(axis=1), axis=0).round(4)
    return matrix

# ── STEP 4: AVERAGE REGIME DURATION ──────────────────────────────────────────
def compute_regime_duration(df):
    """
    Average duration of each regime in minutes.
    A regime run = consecutive bars with same label within same day.
    Each bar = 10 seconds = 1/6 minute.
    """
    df = df.copy()
    df['date'] = df.index.normalize()

    results = {}
    regimes = sorted(df['regime_label'].dropna().unique())

    for regime in regimes:
        run_lengths = []
        for date, day_df in df.groupby('date'):
            labels = day_df['regime_label'].values
            count  = 0
            for label in labels:
                if label == regime:
                    count += 1
                else:
                    if count > 0:
                        run_lengths.append(count)
                    count = 0
            if count > 0:
                run_lengths.append(count)

        if run_lengths:
            avg_bars    = np.mean(run_lengths)
            avg_minutes = avg_bars * 10 / 60  # 10 seconds per bar
            results[regime] = {
                'avg_duration_bars':    round(avg_bars, 1),
                'avg_duration_minutes': round(avg_minutes, 2),
                'n_runs':               len(run_lengths)
            }

    return results

# ── STEP 5: REGIME CHARACTERIZATION TABLE ────────────────────────────────────
def regime_characterization(df):
    """
    Mean and std of key features per regime.
    This goes directly into paper Section 5.
    """
    features = [
        'realized_vol', 'spread', 'queue_imbalance',
        'trade_imbalance', 'ofi', 'ofi_norm'
    ]
    features = [f for f in features if f in df.columns]

    rows = []
    for regime in sorted(df['regime_label'].dropna().unique()):
        subset = df[df['regime_label'] == regime]
        row    = {'regime': regime, 'n_bars': len(subset)}
        for feat in features:
            col = subset[feat].dropna()
            row[f'{feat}_mean'] = round(col.mean(), 6)
            row[f'{feat}_std']  = round(col.std(),  6)
        rows.append(row)

    return pd.DataFrame(rows).set_index('regime')

# ── STEP 6: PRINT RESULTS ─────────────────────────────────────────────────────
def print_results(transition_matrix, duration_results, char_table):
    print("\n" + "="*60)
    print("TRANSITION MATRIX — P(next regime | current regime)")
    print("="*60)
    print(transition_matrix.to_string())
    print("\n→ Diagonal = probability of staying in same regime")
    print("→ High diagonal = regime is persistent")

    print("\n" + "="*60)
    print("AVERAGE REGIME DURATION")
    print("="*60)
    print(f"{'Regime':<12} {'Avg Bars':>10} {'Avg Minutes':>14} {'N Runs':>10}")
    print("-"*50)
    for regime, r in duration_results.items():
        print(f"{regime:<12} {r['avg_duration_bars']:>10} {r['avg_duration_minutes']:>14} {r['n_runs']:>10}")

    print("\n" + "="*60)
    print("REGIME CHARACTERIZATION TABLE")
    print("="*60)
    print(char_table.to_string())

# ── STEP 7: SAVE ──────────────────────────────────────────────────────────────
def save_results(transition_matrix, duration_results, char_table):
    os.makedirs('reports', exist_ok=True)
    transition_matrix.to_csv('reports/transition_matrix.csv')
    pd.DataFrame(duration_results).T.to_csv('reports/regime_duration.csv')
    char_table.to_csv('reports/regime_characterization.csv')
    print("\nSaved:")
    print("  reports/transition_matrix.csv")
    print("  reports/regime_duration.csv")
    print("  reports/regime_characterization.csv")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("REGIME STATS — TRANSITION MATRIX + DURATION + CHARACTERIZATION")
    print("="*60)

    print("\nLoading data...")
    df = load_data(REGIMES_PATH, ORIGINAL_PATH)

    print("Labeling regimes...")
    df, mapping = label_regimes(df)
    print(f"Mapping: {mapping}")

    print("\nComputing transition matrix...")
    transition_matrix = compute_transition_matrix(df)

    print("Computing regime durations...")
    duration_results = compute_regime_duration(df)

    print("Computing characterization table...")
    char_table = regime_characterization(df)

    print_results(transition_matrix, duration_results, char_table)

    print("\nSaving results...")
    save_results(transition_matrix, duration_results, char_table)
    # ── STATIONARY DISTRIBUTION ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("STATIONARY DISTRIBUTION — long run time in each regime")
    print("="*60)
    T = transition_matrix.values
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx        = np.argmin(np.abs(eigenvalues - 1))
    stationary = eigenvectors[:, idx].real
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()
    for i, regime in enumerate(transition_matrix.index):
        print(f"  {regime:<12} : {stationary[i]:.4f}") 
    print("\nDONE. Push to GitHub.")
