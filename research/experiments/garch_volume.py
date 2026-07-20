# garch_vol.py
# Phase 5 ML Models — GARCH Volatility Feature
# Fit GARCH(1,1) on SPY returns
# Extract conditional volatility forecast
# Compare GARCH vol vs realized vol as regime feature
# Reference: Engle (1982) | Bollerslev (1986)

import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ─────────────────────────────────────────────────────

def load_ticker(ticker):
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

# ── COMPUTE RETURNS ───────────────────────────────────────────────

def compute_returns(df, freq='1min'):
    """
    Compute log returns at specified frequency for GARCH fitting.
    
    Why not 10s returns for GARCH:
    GARCH needs enough data points to estimate volatility clustering.
    At 10s frequency the signal-to-noise ratio is too low.
    1-minute returns give ~97,500 observations — sufficient for GARCH.
    
    Manual example:
    price_t   = 500.00
    price_t+1 = 500.25
    return_t  = log(500.25/500.00) = 0.0005 = 0.05%
    """
    if 'microprice' in df.columns:
        price = df['microprice']
    else:
        raise ValueError("No microprice column")

    price_1m = price.resample(freq).last().dropna()
    returns  = np.log(price_1m / price_1m.shift(1)).dropna()
    returns  = returns * 100  # Scale to percentage — GARCH works better

    print(f"  Returns computed at {freq} frequency")
    print(f"  Observations: {len(returns)}")
    print(f"  Mean: {returns.mean():.4f}% | Std: {returns.std():.4f}%")

    return returns

# ── FIT GARCH(1,1) ────────────────────────────────────────────────

def fit_garch(returns, ticker='SPY'):
    """
    Fit GARCH(1,1) model to return series.
    
    GARCH(1,1) equation:
    sigma²_t = omega + alpha * epsilon²_{t-1} + beta * sigma²_{t-1}
    
    Where:
    omega = long-run variance (baseline volatility)
    alpha = ARCH term — how much yesterday's shock affects today's vol
    beta  = GARCH term — how much yesterday's vol persists today
    
    Stationarity condition: alpha + beta < 1
    If alpha + beta close to 1 — volatility is very persistent (typical in finance)
    
    Manual example:
    omega=0.01, alpha=0.10, beta=0.85
    alpha + beta = 0.95 < 1 — stationary, persistent vol
    Yesterday big shock (epsilon² large) → today higher vol
    Yesterday high vol (sigma² large) → today still high vol
    
    Reference: Bollerslev (1986)
    """
    print(f"\n  Fitting GARCH(1,1) on {ticker} 1-min returns...")

    model = arch_model(
        returns,
        vol='Garch',
        p=1,
        q=1,
        dist='t',
        rescale=False
    )

    result = model.fit(disp='off', show_warning=False)

    omega = result.params['omega']
    alpha = result.params.get('alpha[1]', np.nan)
    beta  = result.params.get('beta[1]', np.nan)
    persistence = alpha + beta

    print(f"  GARCH parameters:")
    print(f"    omega = {omega:.6f}")
    print(f"    alpha = {alpha:.4f}  (ARCH term — shock impact)")
    print(f"    beta  = {beta:.4f}  (GARCH term — vol persistence)")
    print(f"    alpha + beta = {persistence:.4f}  ({'stationary' if persistence < 1 else 'NON-STATIONARY'})")
    print(f"  Log-likelihood: {result.loglikelihood:.2f}")
    print(f"  AIC: {result.aic:.2f}")

    # Extract conditional volatility
    cond_vol = result.conditional_volatility

    return result, cond_vol

# ── ROLLING GARCH FORECAST ────────────────────────────────────────

def rolling_garch_forecast(returns, window_days=60, ticker='SPY'):
    """
    Rolling GARCH volatility forecast — no lookahead bias.
    
    At each point T, fit GARCH only on data up to T-1.
    Forecast volatility for T.
    
    This is the same rolling principle as rolling HMM and rolling hedge ratio.
    Never use future data in any feature computation.
    
    Why rolling GARCH instead of full-sample GARCH:
    Full-sample GARCH uses future returns to estimate parameters.
    Rolling GARCH gives a genuine real-time forecast.
    
    Note: rolling GARCH is slow — we use daily refitting not bar-by-bar.
    Refit once per day, apply forecast to all bars that day.
    This is standard practice at quant firms.
    """
    print(f"\n  Rolling GARCH forecast — daily refit, window={window_days} days")

    # Get unique trading days
    days = returns.index.normalize().unique()
    window_bars = window_days * 390  # 390 min bars per day

    garch_vol_daily = {}

    for i, day in enumerate(days):
        if i < window_days:
            continue

        # Use only data up to end of previous day
        prev_day = days[i - 1]
        train = returns[returns.index.normalize() <= prev_day]

        if len(train) < window_bars:
            continue

        # Use last window_days of data — true rolling window
        train = train.iloc[-window_bars:]

        try:
            model = arch_model(train, vol='Garch', p=1, q=1,
                             dist='t', rescale=False)
            res = model.fit(disp='off', show_warning=False)
            # 1-step ahead forecast
            forecast = res.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0])
            garch_vol_daily[day] = vol_forecast
        except:
            continue

    garch_vol_series = pd.Series(garch_vol_daily)
    print(f"  GARCH forecasts computed: {len(garch_vol_series)} days")

    return garch_vol_series

# ── COMPARE GARCH VS REALIZED VOL ────────────────────────────────

def compare_garch_vs_realized(garch_vol_daily, df_10s, target_col='fwd_ret_30s', skip=3):
    """
    Compare GARCH conditional vol vs realized vol as predictive features.
    
    Which one better predicts forward returns IC?
    
    Method:
    1. Map daily GARCH forecast to all 10s bars in that day
    2. Compute IC of GARCH vol vs forward returns
    3. Compute IC of realized vol vs forward returns
    4. Compare — which is stronger signal?
    
    Note: we are not testing vol as a DIRECTION predictor.
    We are testing if high vol days have different signal characteristics.
    This is a REGIME feature, not a return predictor directly.
    """
    from scipy import stats as scipy_stats

    # Map daily GARCH to 10s bars
    garch_10s = pd.Series(index=df_10s.index, dtype=float)
    for day, vol in garch_vol_daily.items():
        mask = df_10s.index.normalize() == day
        garch_10s[mask] = vol

    # Build comparison DataFrame
    comp = pd.DataFrame({
        'garch_vol':   garch_10s,
        'realized_vol': df_10s['realized_vol'],
        'target':      df_10s[target_col]
    }).dropna()

    # Lag both features — no lookahead
    comp['garch_lagged']    = comp['garch_vol'].shift(1)
    comp['realized_lagged'] = comp['realized_vol'].shift(1)
    comp = comp.dropna().iloc[::skip]

    if len(comp) < 100:
        print(f"  Insufficient data for comparison")
        return

    ic_garch,    _ = scipy_stats.spearmanr(comp['garch_lagged'],    comp['target'])
    ic_realized, _ = scipy_stats.spearmanr(comp['realized_lagged'], comp['target'])

    print(f"\n  Feature comparison vs {target_col}:")
    print(f"  GARCH conditional vol IC    : {ic_garch:.4f}")
    print(f"  Realized vol IC             : {ic_realized:.4f}")
    print(f"  Winner: {'GARCH' if abs(ic_garch) > abs(ic_realized) else 'Realized vol'}")

    return ic_garch, ic_realized

# ── ADD GARCH TO FEATURE LIBRARY ──────────────────────────────────

def add_garch_to_features(garch_vol_daily, ticker='SPY'):
    """
    Add GARCH conditional volatility to the feature parquet.
    
    Map daily GARCH forecast to all 10s bars in that day.
    Save as new column 'garch_vol' in the feature parquet.
    """
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Map daily GARCH to 10s bars
    garch_10s = pd.Series(index=df.index, dtype=float)
    for day, vol in garch_vol_daily.items():
        mask = df.index.normalize() == day
        garch_10s[mask] = vol

    df['garch_vol'] = garch_10s

    # Save back
    df.to_parquet(path)
    print(f"\n  garch_vol added to {ticker}_features.parquet")
    print(f"  Non-null garch_vol rows: {df['garch_vol'].notna().sum():,}")

    return df

# ── MAIN ──────────────────────────────────────────────────────────

def run_garch_analysis(ticker='SPY'):
    print(f"\n{'='*65}")
    print(f"GARCH VOLATILITY ANALYSIS — {ticker}")
    print(f"{'='*65}")

    df = load_ticker(ticker)

    # Step 1: Compute returns
    print(f"\n{'─'*65}")
    print("STEP 1 — Compute 1-minute returns")
    print(f"{'─'*65}")
    returns = compute_returns(df, freq='1min')

    # Step 2: Fit full-sample GARCH for parameter inspection
    print(f"\n{'─'*65}")
    print("STEP 2 — Full-sample GARCH(1,1) parameter inspection")
    print(f"{'─'*65}")
    result, cond_vol_full = fit_garch(returns, ticker)

    # Step 3: Rolling GARCH forecast — no lookahead
    print(f"\n{'─'*65}")
    print("STEP 3 — Rolling GARCH forecast (daily refit, no lookahead)")
    print(f"{'─'*65}")
    garch_vol_daily = rolling_garch_forecast(returns, window_days=60, ticker=ticker)

    if len(garch_vol_daily) == 0:
        print("  Rolling GARCH failed — insufficient data")
        return

    # Step 4: Compare GARCH vs realized vol
    print(f"\n{'─'*65}")
    print("STEP 4 — Compare GARCH vol vs realized vol")
    print(f"{'─'*65}")
    compare_garch_vs_realized(garch_vol_daily, df, 'fwd_ret_30s', skip=3)
    compare_garch_vs_realized(garch_vol_daily, df, 'fwd_ret_1m',  skip=6)

    # Step 5: Add to feature library
    print(f"\n{'─'*65}")
    print("STEP 5 — Add garch_vol to feature parquet")
    print(f"{'─'*65}")
    df_updated = add_garch_to_features(garch_vol_daily, ticker)

    # Save GARCH daily series
    out = rf"E:\quant-research-ofi\reports\{ticker}_garch_vol_daily.csv"
    garch_vol_daily.to_csv(out, header=True)
    print(f"  Saved: {ticker}_garch_vol_daily.csv")

    print(f"\n{'='*65}")
    print(f"GARCH ANALYSIS COMPLETE — {ticker}")
    print(f"If GARCH IC > realized vol IC → replace realized_vol in SELECTED_FEATURES")
    print(f"{'='*65}")

    return garch_vol_daily


if __name__ == "__main__":
    garch_vol_daily = run_garch_analysis(ticker='SPY')
