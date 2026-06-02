# position_sizer.py
# Phase 6 Backtest — June 3
# Volatility-targeted position sizing
# All 4 issues fixed:
# 1. Signal normalization window: 252 → 2340 (1 full trading day of 10s bars)
# 2. Turnover metric corrected for 10-second bars
# 3. Vol window: 20 → 120 bars (~20 minutes, more stable)
# 4. Leverage metric added to summary

import numpy as np
import pandas as pd


def compute_rolling_vol(returns: pd.Series,
                        window: int = 120,
                        annualize: bool = True,
                        bars_per_day: int = 2340) -> pd.Series:
    """
    Rolling realized volatility from bar returns.

    Parameters
    ----------
    returns       : log returns series at bar frequency
    window        : lookback in bars (default 120 bars = ~20 minutes at 10s)
    annualize     : if True, annualize the vol
    bars_per_day  : 10-second bars per day = 6.5 * 60 * 6 = 2340

    Returns
    -------
    rolling_vol : annualized realized vol series

    Formula: vol = std(returns, window) * sqrt(bars_per_day * 252)
    Note: uses shift(1) — no lookahead bias, vol at T uses only data up to T-1
    """
    raw_vol = returns.rolling(window).std()
    if annualize:
        raw_vol = raw_vol * np.sqrt(bars_per_day * 252)
    return raw_vol.shift(1).clip(lower=1e-6)


def vol_target_position(signal: pd.Series,
                        rolling_vol: pd.Series,
                        target_daily_vol: float = 0.01,
                        portfolio_value: float = 1_000_000,
                        price: pd.Series = None) -> pd.Series:
    """
    Scale signal to target a fixed daily vol contribution.

    Parameters
    ----------
    signal           : raw signal series (any scale)
    rolling_vol      : annualized rolling vol (from compute_rolling_vol)
    target_daily_vol : target daily vol as fraction of portfolio (default 1%)
    portfolio_value  : portfolio size in dollars
    price            : current mid price (needed to convert dollars to shares)

    Returns
    -------
    positions : position in shares (signed, vol-scaled)

    Formula:
        daily_vol = annualized_vol / sqrt(252)
        dollar_position = (signal / daily_vol) * target_daily_vol * portfolio_value
        shares = dollar_position / price

    Intuition: if vol is high, shrink position to keep risk constant.
    In F&O terms: you size down in stressed regimes automatically.
    """
    daily_vol = rolling_vol / np.sqrt(252)
    daily_vol = daily_vol.clip(lower=1e-6)

    # Fix 1: normalization window = 2340 bars = 1 full trading day of 10s bars
    # 252 bars was only ~42 minutes — too short and unstable
    signal_norm = signal / signal.rolling(2340).std().shift(1).clip(lower=1e-6)

    dollar_position = (signal_norm / daily_vol) * target_daily_vol * portfolio_value

    if price is not None:
        shares = dollar_position / price.clip(lower=1e-6)
    else:
        shares = dollar_position

    return shares


def apply_position_cap(positions: pd.Series,
                       price: pd.Series,
                       portfolio_value: float = 1_000_000,
                       max_position_pct: float = 0.10) -> pd.Series:
    """
    Cap position at max_position_pct of portfolio value.
    Prevents blowup in high vol regimes where sizing explodes.

    Parameters
    ----------
    positions        : uncapped position in shares
    price            : current mid price
    portfolio_value  : portfolio size in dollars
    max_position_pct : max single position as fraction of portfolio (default 10%)

    Returns
    -------
    capped_positions : positions clipped to max dollar limit
    """
    max_dollar = portfolio_value * max_position_pct
    max_shares = max_dollar / price.clip(lower=1e-6)
    return positions.clip(lower=-max_shares, upper=max_shares)


def position_summary(positions: pd.Series,
                     price: pd.Series,
                     portfolio_value: float = 1_000_000,
                     bars_per_day: int = 2340) -> None:
    """
    Print position sizing diagnostics including leverage.
    Always run before passing positions to backtest_engine.
    """
    dollar_pos = positions.abs() * price

    # Fix 2: turnover corrected for 10-second bars
    # positions.diff() is bar-level change — sum then divide by actual trading days
    bar_turnover = positions.diff().abs() * price
    n_days = max(len(positions) / bars_per_day, 1)
    daily_turnover = bar_turnover.sum() / n_days

    # Fix 4: leverage metric
    gross_exposure = dollar_pos.mean()
    leverage = gross_exposure / portfolio_value

    print("=" * 50)
    print("POSITION SIZING SUMMARY")
    print("=" * 50)
    print(f"Portfolio value      : ${portfolio_value:>12,.0f}")
    print(f"Max position (shares): {positions.abs().max():>12.1f}")
    print(f"Max position ($)     : ${dollar_pos.max():>12,.0f}")
    print(f"Max position (% port): {dollar_pos.max()/portfolio_value*100:>11.1f}%")
    print(f"Avg position ($)     : ${dollar_pos.mean():>12,.0f}")
    print(f"Avg daily turnover   : ${daily_turnover:>12,.0f}")
    print(f"Gross leverage       : {leverage:>11.2f}x")
    print(f"Pct time flat        : {(positions==0).mean()*100:>11.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    np.random.seed(42)
    n = 2340 * 5   # 5 days of 10-second bars
    idx = pd.date_range("2023-01-02 09:30", periods=n, freq="10s")

    mid = pd.Series(450 + np.cumsum(np.random.randn(n) * 0.05), index=idx)
    log_returns = np.log(mid / mid.shift(1)).fillna(0)

    # Synthetic OFI signal — replace with real ofi_norm in backtest_engine
    signal = pd.Series(np.random.randn(n), index=idx)

    # Fix 3: vol window = 120 bars (~20 minutes), was 20 bars (~3 minutes)
    rolling_vol = compute_rolling_vol(log_returns, window=120)

    positions_raw = vol_target_position(
        signal=signal,
        rolling_vol=rolling_vol,
        target_daily_vol=0.01,
        portfolio_value=1_000_000,
        price=mid
    )

    positions = apply_position_cap(
        positions=positions_raw,
        price=mid,
        portfolio_value=1_000_000,
        max_position_pct=0.10
    )

    position_summary(positions, mid, portfolio_value=1_000_000)

    dollar_pos = positions.abs() * mid
    assert dollar_pos.max() <= 1_000_000 * 0.10 + 1, "Cap exceeded"
    print("\nSanity check passed: cap respected")
