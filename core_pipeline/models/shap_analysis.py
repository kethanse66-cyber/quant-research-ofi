# shap_analysis.py
# Phase 5 ML Models — SHAP Feature Importance
# SHAP values on Ridge model (best performing model)
# Waterfall plot for one prediction
# Bar plot of mean absolute SHAP values
# Reference: Lundberg & Lee (2017) | Lopez de Prado (2018) Ch.8

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import warnings
warnings.filterwarnings('ignore')

SELECTED_FEATURES = [
    'ofi_10s', 'ofi_30s', 'ofi_10m',
    'queue_imbalance', 'trade_imbalance',
    'spread', 'spread_change',
    'kyle_lambda', 'amihud',
    'realized_vol', 'ofi_norm',
    'microprice_ret', 'vwap_deviation'
]

TARGET = 'fwd_ret_1m'
SKIP   = 6

# ── LOAD DATA ─────────────────────────────────────────────────────
def load_data(ticker='SPY'):
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts').sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if 'microprice' in df.columns:
        df['microprice_ret'] = np.log(df['microprice'] / df['microprice'].shift(1))
    if 'microprice' in df.columns and 'vwap' in df.columns and 'spread' in df.columns:
        denom = df['spread'].replace(0, np.nan)
        df['vwap_deviation'] = (df['microprice'] - df['vwap']) / denom
    return df

# ── TRAIN FINAL RIDGE MODEL ───────────────────────────────────────

def train_final_ridge(df, train_months=9):
    """
    Train Ridge on first train_months of data.
    Use remaining data as test set for SHAP analysis.
    
    Why 9 months train: maximises training data while keeping
    3 months of unseen test data for SHAP explanation.
    
    Shift features by 1 — no lookahead.
    Scale with StandardScaler fit on train only.
    """
    feature_cols = [f for f in SELECTED_FEATURES if f in df.columns]

    df = df.copy()
    df[feature_cols] = df[feature_cols].shift(1)
    df_clean = df[feature_cols + [TARGET]].dropna()

    # Split by time
    months = df_clean.resample('ME').size().index
    if len(months) < train_months + 1:
        train_months = len(months) - 1

    train_end = months[train_months - 1]
    train_df  = df_clean[df_clean.index <= train_end]
    test_df   = df_clean[df_clean.index > train_end].iloc[::SKIP]

    print(f"  Train: {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df):,} rows)")
    print(f"  Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} ({len(test_df):,} rows)")

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[TARGET].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)

    y_pred = ridge.predict(X_test_sc)
    ic, pval = stats.spearmanr(y_test, y_pred)
    print(f"  Test IC: {ic:.4f} | p-value: {pval:.4f}")

    return ridge, scaler, X_test_sc, y_test, test_df, feature_cols

# ── SHAP ANALYSIS ─────────────────────────────────────────────────

def run_shap_analysis(ridge, scaler, X_test_sc, feature_cols, ticker='SPY'):
    """
    SHAP values for Ridge regression using LinearExplainer.
    
    Why LinearExplainer for Ridge:
    Ridge is a linear model. LinearExplainer is exact and fast for linear models.
    For tree models we would use TreeExplainer.
    For black-box models we would use KernelExplainer (slow).
    
    SHAP value meaning:
    For each prediction, SHAP tells you how much each feature
    pushed the prediction up or down from the baseline.
    
    Baseline = mean prediction across all test observations.
    
    Manual example:
    Baseline = 0.0001 (mean predicted return)
    ofi_10s SHAP = +0.0003 → OFI pushed prediction up by 0.0003
    spread SHAP  = -0.0001 → spread pushed prediction down by 0.0001
    Final prediction = 0.0001 + 0.0003 - 0.0001 = 0.0003
    
    Reference: Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions
    
    Important caveat:
    SHAP importance is split-based for trees, coefficient-based for linear models.
    For Ridge: SHAP values are proportional to scaled coefficients * feature values.
    High SHAP does not mean causal importance — it means predictive contribution.
    """
    print(f"\n  Computing SHAP values using LinearExplainer...")

    # LinearExplainer — exact for linear models
    explainer = shap.LinearExplainer(
        ridge,
        X_test_sc,
        feature_perturbation='interventional'
    )

    shap_values = explainer.shap_values(X_test_sc)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    print(f"  SHAP values computed: {shap_df.shape[0]} observations x {shap_df.shape[1]} features")

    return shap_values, shap_df, explainer

# ── PLOT 1: MEAN ABSOLUTE SHAP BAR CHART ──────────────────────────

def plot_shap_bar(shap_values, feature_cols, ticker='SPY'):
    """
    Bar chart of mean absolute SHAP values.
    Shows which features contribute most on average.
    OFI features should dominate for this project.
    
    Mean |SHAP| = average magnitude of contribution across all predictions.
    High mean |SHAP| = feature consistently moves predictions.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_shap     = mean_abs_shap[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(sorted_features, sorted_shap, color='steelblue', alpha=0.8)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(f'Feature Importance (SHAP) — Ridge on {ticker} fwd_ret_30s', fontsize=13)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, sorted_shap):
        ax.text(val + max(sorted_shap) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.5f}', va='center', fontsize=9)

    plt.tight_layout()
    out = rf"E:\quant-research-ofi\reports\{ticker}_shap_bar.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {ticker}_shap_bar.png")

    # Print table
    print(f"\n  Mean |SHAP| ranking:")
    print(f"  {'Feature':<20} {'Mean |SHAP|':>12} {'Rank':>6}")
    print(f"  {'─'*40}")
    for rank, (feat, val) in enumerate(
            zip(reversed(sorted_features), reversed(sorted_shap)), 1):
        print(f"  {feat:<20} {val:>12.6f} {rank:>6}")

    return dict(zip(feature_cols, mean_abs_shap))

# ── PLOT 2: WATERFALL PLOT FOR ONE PREDICTION ─────────────────────

def plot_shap_waterfall(shap_values, explainer, X_test_sc,
                        feature_cols, test_df, ticker='SPY'):
    """
    Waterfall plot for one specific prediction.
    Shows step-by-step how each feature built up the prediction.
    
    Pick the prediction with highest predicted return — most interesting to explain.
    
    Reading the waterfall:
    Start at baseline (mean prediction).
    Each bar = one feature's contribution.
    Red = pushed prediction up.
    Blue = pushed prediction down.
    Final bar = actual prediction.
    """
    # Pick observation with highest absolute predicted return
    pred_magnitudes = np.abs(shap_values).sum(axis=1)
    best_idx = np.argmax(pred_magnitudes)

    print(f"\n  Waterfall plot for observation {best_idx}")
    print(f"  Timestamp: {test_df.index[best_idx]}")

    shap_explanation = shap.Explanation(
        values=shap_values[best_idx],
        base_values=explainer.expected_value,
        data=X_test_sc[best_idx],
        feature_names=feature_cols
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation, max_display=13, show=False)
    plt.title(f'SHAP Waterfall — {ticker} — {test_df.index[best_idx].date()}',
              fontsize=12)
    plt.tight_layout()
    out = rf"E:\quant-research-ofi\reports\{ticker}_shap_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {ticker}_shap_waterfall.png")

# ── PLOT 3: SHAP BEESWARM ─────────────────────────────────────────

def plot_shap_beeswarm(shap_values, X_test_sc, feature_cols, ticker='SPY'):
    """
    Beeswarm plot — shows distribution of SHAP values per feature.
    
    Each dot = one observation.
    Color = feature value (red=high, blue=low).
    Position on x-axis = SHAP value (impact on prediction).
    
    This shows not just average importance but HOW each feature affects predictions.
    Example: if ofi_10s has all red dots on the right side — high OFI consistently
    pushes predictions up. That confirms the directional relationship.
    """
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=X_test_sc,
        feature_names=feature_cols
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=13, show=False)
    plt.title(f'SHAP Beeswarm — {ticker} — Ridge fwd_ret_30s', fontsize=12)
    plt.tight_layout()
    out = rf"E:\quant-research-ofi\reports\{ticker}_shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {ticker}_shap_beeswarm.png")

# ── MAIN ──────────────────────────────────────────────────────────

def run_shap_analysis_main(ticker='SPY'):
    print(f"\n{'='*65}")
    print(f"SHAP ANALYSIS — {ticker} — Ridge on {TARGET}")
    print(f"{'='*65}")

    # Load and prepare data
    df = load_data(ticker)

    # Train final Ridge model
    print(f"\n{'─'*65}")
    print("Training final Ridge model...")
    print(f"{'─'*65}")
    ridge, scaler, X_test_sc, y_test, test_df, feature_cols = train_final_ridge(df)

    # SHAP analysis
    print(f"\n{'─'*65}")
    print("Computing SHAP values...")
    print(f"{'─'*65}")
    shap_values, shap_df, explainer = run_shap_analysis(
        ridge, scaler, X_test_sc, feature_cols, ticker)

    # Plot 1: Bar chart
    print(f"\n{'─'*65}")
    print("Plot 1 — Mean absolute SHAP bar chart")
    print(f"{'─'*65}")
    importance_dict = plot_shap_bar(shap_values, feature_cols, ticker)

    # Plot 2: Waterfall
    print(f"\n{'─'*65}")
    print("Plot 2 — Waterfall plot for one prediction")
    print(f"{'─'*65}")
    plot_shap_waterfall(shap_values, explainer, X_test_sc,
                        feature_cols, test_df, ticker)

    # Plot 3: Beeswarm
    print(f"\n{'─'*65}")
    print("Plot 3 — Beeswarm plot")
    print(f"{'─'*65}")
    plot_shap_beeswarm(shap_values, X_test_sc, feature_cols, ticker)

    # Save SHAP importance table
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'mean_abs_shap': list(importance_dict.values())
    }).sort_values('mean_abs_shap', ascending=False)

    out = rf"E:\quant-research-ofi\reports\{ticker}_shap_importance.csv"
    importance_df.to_csv(out, index=False)
    print(f"\n  Saved: {ticker}_shap_importance.csv")

    print(f"\n{'='*65}")
    print(f"SHAP ANALYSIS COMPLETE — {ticker}")
    print(f"Top 3 features driving Ridge predictions:")
    for _, row in importance_df.head(3).iterrows():
        print(f"  {row['feature']:<20} mean|SHAP|={row['mean_abs_shap']:.6f}")
    print(f"{'='*65}")

    return importance_df


if __name__ == "__main__":
    # Install shap first if needed:
    # .venv311\Scripts\pip install shap matplotlib
    importance_df = run_shap_analysis_main(ticker='SPY')
