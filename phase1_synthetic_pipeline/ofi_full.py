"""
ofi_full.py
-----------
Order Flow Imbalance (OFI) computed at multiple time horizons (30s, 1min, 5min)
on synthetic tick data, with autocorrelation structure analysis.

Reference: Cont, Kukanov & Stoikov (2014) - The Price Impact of Order Book Events
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────

def generate_synthetic_ticks(n_ticks: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic order book tick data.

    Parameters
    ----------
    n_ticks : int
        Number of tick observations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, bid_price, ask_price, bid_size, ask_size, trade_price, trade_size, trade_side
    """
    rng = np.random.default_rng(seed)

    # Simulate timestamps: market hours 9:30-16:00, random inter-arrival (Poisson)
    start = pd.Timestamp('2024-01-02 09:30:00')
    inter_arrival_ms = rng.exponential(scale=500, size=n_ticks).astype(int)  # avg 0.5s
    timestamps = pd.to_datetime(
        start.value + np.cumsum(inter_arrival_ms) * 1_000_000  # ms -> ns
    )

    # Simulate mid-price as random walk
    mid_price = 500.0 + np.cumsum(rng.normal(0, 0.05, n_ticks))

    # Spread: 1-3 cents, wider during volatile periods
    spread = rng.uniform(0.01, 0.03, n_ticks)

    bid_price = mid_price - spread / 2
    ask_price = mid_price + spread / 2

    # Queue sizes: log-normal, correlated with recent volatility
    bid_size = np.maximum(1, rng.lognormal(mean=6, sigma=1, size=n_ticks)).astype(int)
    ask_size = np.maximum(1, rng.lognormal(mean=6, sigma=1, size=n_ticks)).astype(int)

    # Trade side: slightly buy-biased when price is rising
    price_momentum = np.sign(np.diff(mid_price, prepend=mid_price[0]))
    trade_side_prob = 0.5 + 0.1 * price_momentum
    trade_side = np.where(rng.random(n_ticks) < trade_side_prob, 1, -1)  # 1=buy, -1=sell

    trade_size = np.maximum(1, rng.lognormal(mean=5, sigma=1.2, size=n_ticks)).astype(int)
    trade_price = np.where(trade_side == 1, ask_price, bid_price)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid_price': bid_price,
        'ask_price': ask_price,
        'bid_size': bid_size,
        'ask_size': ask_size,
        'trade_price': trade_price,
        'trade_size': trade_size,
        'trade_side': trade_side,
        'mid_price': mid_price,
    })

    # Keep only market hours (9:30 - 16:00)
    df = df[
        (df['timestamp'].dt.time >= pd.Timestamp('09:30').time()) &
        (df['timestamp'].dt.time <= pd.Timestamp('16:00').time())
    ].reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# 2. CORE OFI COMPUTATION
# ─────────────────────────────────────────────

def compute_tick_level_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tick-level OFI per Cont, Kukanov & Stoikov (2014).

    OFI_t = delta_BidSize_t * I(BidPrice_t >= BidPrice_{t-1})
            - delta_AskSize_t * I(AskPrice_t <= AskPrice_{t-1})

    Where:
        delta_BidSize_t = BidSize_t - BidSize_{t-1}  (if bid price unchanged or improved)
        delta_AskSize_t = AskSize_t - AskSize_{t-1}  (if ask price unchanged or worsened)

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with bid_price, ask_price, bid_size, ask_size.

    Returns
    -------
    pd.DataFrame
        Original df with added column: ofi_tick
    """
    df = df.copy()

    bid_price_change = df['bid_price'] - df['bid_price'].shift(1)
    ask_price_change = df['ask_price'] - df['ask_price'].shift(1)

    bid_size_change = df['bid_size'] - df['bid_size'].shift(1)
    ask_size_change = df['ask_size'] - df['ask_size'].shift(1)

    # Bid contribution: positive if bid improved or held with more size
    bid_contrib = np.where(
        bid_price_change > 0, df['bid_size'],
        np.where(bid_price_change == 0, bid_size_change, -df['bid_size'].shift(1))
    )

    # Ask contribution: negative if ask worsened or held with more size
    ask_contrib = np.where(
        ask_price_change < 0, df['ask_size'],
        np.where(ask_price_change == 0, ask_size_change, -df['ask_size'].shift(1))
    )

    df['ofi_tick'] = bid_contrib - ask_contrib
    df['ofi_tick'] = df['ofi_tick'].fillna(0)

    return df


# ─────────────────────────────────────────────
# 3. RESAMPLE OFI TO MULTIPLE HORIZONS
# ─────────────────────────────────────────────

def compute_ofi_horizons(df: pd.DataFrame) -> dict:
    """
    Aggregate tick-level OFI to 30s, 1min, and 5min bars.
    Also computes forward returns at each horizon for IC analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Tick data with ofi_tick and mid_price, indexed by timestamp.

    Returns
    -------
    dict
        Keys: '30s', '1min', '5min' — each a DataFrame with:
        ofi, ofi_norm, mid_price, forward_return
    """
    df = df.set_index('timestamp')

    horizons = {
        '30s':  '30s',
        '1min': '1min',
        '5min': '5min',
    }

    result = {}

    for label, freq in horizons.items():
        bars = df.resample(freq).agg(
            ofi=('ofi_tick', 'sum'),
            mid_price=('mid_price', 'last'),
            volume=('trade_size', 'sum'),
            n_trades=('trade_side', 'count'),
        ).dropna()

        # Normalise OFI by volume to make it comparable across bars
        bars['ofi_norm'] = bars['ofi'] / bars['volume'].replace(0, np.nan)

        # Forward return: next bar's mid-price change
        bars['forward_return'] = bars['mid_price'].shift(-1) / bars['mid_price'] - 1
        bars['forward_return'] = bars['forward_return'].shift(-1)  # align: use lag=1

        # Lag OFI by 1 to avoid lookahead
        bars['ofi_lag1'] = bars['ofi'].shift(1)
        bars['ofi_norm_lag1'] = bars['ofi_norm'].shift(1)

        bars = bars.dropna()
        result[label] = bars

    return result


# ─────────────────────────────────────────────
# 4. AUTOCORRELATION STRUCTURE
# ─────────────────────────────────────────────

def compute_acf_manual(series: pd.Series, max_lags: int = 20) -> pd.DataFrame:
    """
    Compute autocorrelation function (ACF) manually.

    ACF(k) = Cov(X_t, X_{t-k}) / Var(X_t)

    Parameters
    ----------
    series : pd.Series
        Time series to analyse.
    max_lags : int
        Maximum number of lags.

    Returns
    -------
    pd.DataFrame
        Columns: lag, acf, conf_upper, conf_lower (95% Bartlett bands)
    """
    n = len(series)
    mean = series.mean()
    var = series.var()

    acf_vals = []
    for k in range(0, max_lags + 1):
        if k == 0:
            acf_vals.append(1.0)
        else:
            cov = ((series[k:].values - mean) * (series[:-k].values - mean)).mean()
            acf_vals.append(cov / var if var > 0 else 0)

    # 95% Bartlett confidence bands
    conf = 1.96 / np.sqrt(n)

    return pd.DataFrame({
        'lag': range(0, max_lags + 1),
        'acf': acf_vals,
        'conf_upper': conf,
        'conf_lower': -conf,
    })


def analyse_autocorrelation(ofi_dict: dict, max_lags: int = 20) -> dict:
    """
    Compute ACF for OFI at each horizon.

    Parameters
    ----------
    ofi_dict : dict
        Output of compute_ofi_horizons().
    max_lags : int
        Max lags for ACF.

    Returns
    -------
    dict
        Keys match ofi_dict keys; values are ACF DataFrames.
    """
    acf_results = {}
    for label, bars in ofi_dict.items():
        acf_results[label] = compute_acf_manual(bars['ofi'].dropna(), max_lags=max_lags)
    return acf_results


# ─────────────────────────────────────────────
# 5. INFORMATION COEFFICIENT (IC)
# ─────────────────────────────────────────────

def compute_ic(ofi_dict: dict) -> pd.DataFrame:
    """
    Compute rank IC (Spearman correlation) between lagged OFI and forward return
    at each horizon.

    IC = Spearman(OFI_{t-1}, Return_t)

    Parameters
    ----------
    ofi_dict : dict
        Output of compute_ofi_horizons().

    Returns
    -------
    pd.DataFrame
        Columns: horizon, IC, t_stat, p_value, n_obs
    """
    rows = []
    for label, bars in ofi_dict.items():
        clean = bars[['ofi_lag1', 'forward_return']].dropna()
        if len(clean) < 30:
            continue
        ic, pval = stats.spearmanr(clean['ofi_lag1'], clean['forward_return'])
        n = len(clean)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        rows.append({
            'horizon': label,
            'IC': round(ic, 4),
            't_stat': round(t_stat, 3),
            'p_value': round(pval, 4),
            'n_obs': n,
            'significant': pval < 0.05,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────

def plot_ofi_analysis(ofi_dict: dict, acf_dict: dict, ic_df: pd.DataFrame) -> None:
    """
    Three-panel plot:
    1. OFI time series at each horizon
    2. ACF at each horizon
    3. IC bar chart across horizons

    Parameters
    ----------
    ofi_dict : dict
        Output of compute_ofi_horizons().
    acf_dict : dict
        Output of analyse_autocorrelation().
    ic_df : pd.DataFrame
        Output of compute_ic().
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('OFI Multi-Horizon Analysis', fontsize=14, fontweight='bold', y=1.01)

    colors = {'30s': '#1f77b4', '1min': '#ff7f0e', '5min': '#2ca02c'}

    for col, (label, bars) in enumerate(ofi_dict.items()):
        # --- Row 1: OFI time series ---
        ax = axes[0, col]
        plot_bars = bars['ofi'].iloc[:200]
        ax.plot(plot_bars.values, color=colors[label], linewidth=0.8, alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.set_title(f'OFI — {label}', fontsize=11)
        ax.set_xlabel('Bar index')
        ax.set_ylabel('OFI (raw)')
        ax.tick_params(labelsize=8)

        # --- Row 2: ACF ---
        ax2 = axes[1, col]
        acf = acf_dict[label]
        ax2.bar(acf['lag'], acf['acf'], color=colors[label], alpha=0.7, width=0.6)
        ax2.fill_between(
            acf['lag'], acf['conf_lower'], acf['conf_upper'],
            alpha=0.15, color='gray', label='95% CI'
        )
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_title(f'ACF — {label}', fontsize=11)
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrelation')
        ax2.tick_params(labelsize=8)
        ax2.legend(fontsize=7)

    # --- Row 3: IC summary bar chart (span all 3 cols) ---
    ax3 = axes[2, 0]
    bar_colors = [colors[h] for h in ic_df['horizon']]
    bars_ic = ax3.bar(ic_df['horizon'], ic_df['IC'], color=bar_colors, alpha=0.8, edgecolor='white')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_title('Information Coefficient by Horizon', fontsize=11)
    ax3.set_ylabel('Rank IC (Spearman)')
    ax3.set_xlabel('Horizon')
    for bar, sig in zip(bars_ic, ic_df['significant']):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            '*' if sig else '',
            ha='center', va='bottom', fontsize=12, color='black'
        )
    ax3.tick_params(labelsize=9)

    # IC table in middle panel
    ax4 = axes[2, 1]
    ax4.axis('off')
    table_data = ic_df[['horizon', 'IC', 't_stat', 'p_value', 'n_obs']].values.tolist()
    table = ax4.table(
        cellText=table_data,
        colLabels=['Horizon', 'IC', 't-stat', 'p-value', 'N'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax4.set_title('IC Summary Table', fontsize=11)

    # OFI normalised distribution in right panel
    ax5 = axes[2, 2]
    for label, bars in ofi_dict.items():
        ofi_norm = bars['ofi_norm'].dropna()
        ofi_clipped = ofi_norm.clip(
            ofi_norm.quantile(0.01), ofi_norm.quantile(0.99)
        )
        ax5.hist(ofi_clipped, bins=50, alpha=0.5, label=label, color=colors[label], density=True)
    ax5.set_title('OFI Normalised Distribution', fontsize=11)
    ax5.set_xlabel('OFI / Volume')
    ax5.set_ylabel('Density')
    ax5.legend(fontsize=8)
    ax5.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig('ofi_analysis.png', dpi=150, bbox_inches='tight')
    print("Plot saved: ofi_analysis.png")
    plt.show()


# ─────────────────────────────────────────────
# 7. STATIONARITY TESTS
# ─────────────────────────────────────────────

def test_stationarity(ofi_dict: dict) -> pd.DataFrame:
    """
    Run ADF test on OFI series at each horizon.
    OFI should be stationary (it is a differenced quantity by construction).

    Parameters
    ----------
    ofi_dict : dict
        Output of compute_ofi_horizons().

    Returns
    -------
    pd.DataFrame
        ADF results per horizon.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for label, bars in ofi_dict.items():
        series = bars['ofi'].dropna()
        result = adfuller(series, autolag='AIC')
        rows.append({
            'horizon': label,
            'ADF_stat': round(result[0], 4),
            'p_value': round(result[1], 4),
            'n_lags': result[2],
            'stationary (p<0.05)': result[1] < 0.05,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OFI MULTI-HORIZON ANALYSIS")
    print("Cont, Kukanov & Stoikov (2014)")
    print("=" * 60)

    # Step 1: Generate synthetic tick data
    print("\n[1] Generating synthetic tick data...")
    ticks = generate_synthetic_ticks(n_ticks=50000, seed=42)
    print(f"    Ticks generated: {len(ticks):,}")
    print(f"    Date range: {ticks['timestamp'].min()} → {ticks['timestamp'].max()}")

    # Step 2: Compute tick-level OFI
    print("\n[2] Computing tick-level OFI...")
    ticks = compute_tick_level_ofi(ticks)
    print(f"    OFI mean: {ticks['ofi_tick'].mean():.4f}")
    print(f"    OFI std:  {ticks['ofi_tick'].std():.4f}")

    # Step 3: Resample to multiple horizons
    print("\n[3] Resampling to 30s, 1min, 5min horizons...")
    ofi_dict = compute_ofi_horizons(ticks)
    for label, bars in ofi_dict.items():
        print(f"    {label}: {len(bars):,} bars | OFI mean={bars['ofi'].mean():.2f} | std={bars['ofi'].std():.2f}")

    # Step 4: Stationarity tests
    print("\n[4] ADF Stationarity Tests on OFI:")
    stat_df = test_stationarity(ofi_dict)
    print(stat_df.to_string(index=False))

    # Step 5: Autocorrelation structure
    print("\n[5] Computing ACF (20 lags)...")
    acf_dict = analyse_autocorrelation(ofi_dict, max_lags=20)
    for label, acf in acf_dict.items():
        sig_lags = acf[(acf['acf'].abs() > acf['conf_upper']) & (acf['lag'] > 0)]['lag'].tolist()
        print(f"    {label}: significant lags = {sig_lags}")

    # Step 6: Information Coefficient
    print("\n[6] Information Coefficient (IC) Analysis:")
    ic_df = compute_ic(ofi_dict)
    print(ic_df.to_string(index=False))

    # Step 7: Plot
    print("\n[7] Generating plots...")
    plot_ofi_analysis(ofi_dict, acf_dict, ic_df)

    print("\n" + "=" * 60)
    print("DONE. Next file: feature_normalizer.py (Day 11)")
    print("=" * 60)

    return ofi_dict, acf_dict, ic_df


if __name__ == '__main__':
    ofi_dict, acf_dict, ic_df = main()
