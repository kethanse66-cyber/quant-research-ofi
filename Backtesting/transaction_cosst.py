# transaction_costs.py
# Phase 6 Backtest — June 2
# Models three cost components: bid-ask spread crossing, market impact, commission
# All 5 issues fixed:
# 1. Market impact now multiplied by trade_abs
# 2. Spread cost comment added for future real bid/ask
# 3. Position units documented as shares, vol-scaled
# 4. Turnover metrics added to cost_summary
# 5. Division by zero protected with 1e-12 guard

import numpy as np
import pandas as pd


def spread_cost(side: pd.Series, spread: pd.Series) -> pd.Series:
    """
    Cost from crossing the bid-ask spread.
    Buy at ask, sell at bid — always pay half-spread each way.

    Parameters
    ----------
    side   : +1 for buy, -1 for sell (from signal)
    spread : bid-ask spread in price units (e.g. 0.01 for 1 cent)
             # TODO: use real ask/bid execution prices when available
             # For now assumes you always cross half-spread — conservative

    Returns
    -------
    cost_per_unit : half-spread paid per unit traded (always positive)

    Formula: cost = |side| * spread / 2
    Source: Cont, Kukanov & Stoikov (2014) — effective spread decomposition
    """
    return np.abs(side) * spread / 2


def market_impact_cost(trade_size: pd.Series,
                       adv: pd.Series,
                       realized_vol: pd.Series,
                       price: pd.Series) -> pd.Series:
    """
    Square-root market impact model — Almgren-Chriss simplified form.
    Returns impact cost PER SHARE — caller must multiply by trade_abs.

    Parameters
    ----------
    trade_size   : number of shares traded (absolute value)
    adv          : average daily volume in shares (rolling 20-day)
    realized_vol : annualized realized volatility of the asset
    price        : current mid price

    Returns
    -------
    impact_cost_per_share : price impact cost per share (multiply by trade_abs outside)

    Formula: impact_per_share = sigma_daily * price * sqrt(trade_size / adv)
    Source: Almgren & Chriss (2001) — Optimal Execution of Portfolio Transactions
    """
    daily_vol = realized_vol / np.sqrt(252)
    participation_rate = trade_size / adv.clip(lower=1)
    impact_per_share = daily_vol * price * np.sqrt(participation_rate)
    return impact_per_share


def commission_cost(trade_value: pd.Series,
                    commission_bps: float = 0.1) -> pd.Series:
    """
    Fixed commission per trade in basis points of trade value.
    0.5bp is conservative institutional estimate.

    Parameters
    ----------
    trade_value     : abs(shares * price) for each trade
    commission_bps  : commission in basis points (default 0.1bp)

    Returns
    -------
    commission : cost in dollars per trade
    """
    return trade_value * commission_bps / 10_000


def apply_transaction_costs(pnl_gross: pd.Series,
                             positions: pd.Series,
                             prices: pd.DataFrame,
                             adv: pd.Series,
                             commission_bps: float = 0.1) -> pd.DataFrame:
    """
    Apply all three cost components to a gross PnL series.

    Parameters
    ----------
    pnl_gross    : gross PnL series (no costs)
    positions    : position series in SHARES, vol-scaled, signed
                   e.g. positions = (signal / realized_vol * 100).clip(-500, 500)
    prices       : DataFrame with columns: 'mid', 'spread', 'realized_vol'
    adv          : average daily volume series in shares
    commission_bps : commission per trade in bps

    Returns
    -------
    DataFrame with columns:
        pnl_gross, cost_spread, cost_impact, cost_commission,
        total_cost, pnl_net, trade_value
    """
    trades = positions.diff().fillna(0)           # shares traded each bar
    trade_side = np.sign(trades)                   # +1 buy, -1 sell, 0 flat
    trade_abs = trades.abs()                       # absolute shares traded

    trade_value = trade_abs * prices['mid']        # dollar value per trade

    # Fix 1: impact_per_share * trade_abs — was missing trade_abs before
    impact_per_share = market_impact_cost(trade_abs, adv, prices['realized_vol'], prices['mid'])
    cost_impact = impact_per_share * trade_abs

    cost_spread = spread_cost(trade_side, prices['spread']) * trade_abs
    cost_comm = commission_cost(trade_value, commission_bps)

    total_cost = -(cost_spread + cost_impact + cost_comm)

    pnl_net = pnl_gross + total_cost

    result = pd.DataFrame({
        'pnl_gross': pnl_gross,
        'cost_spread': -cost_spread,
        'cost_impact': -cost_impact,
        'cost_commission': -cost_comm,
        'total_cost': total_cost,
        'pnl_net': pnl_net,
        'trade_value': trade_value,        # Fix 4: needed for turnover metrics
    })

    return result


def cost_summary(cost_df: pd.DataFrame) -> None:
    """
    Print clean before/after costs summary including turnover metrics.
    Interviewers ask about turnover — always have this number ready.
    """
    gross = cost_df['pnl_gross'].sum()
    net = cost_df['pnl_net'].sum()
    spread_drag = cost_df['cost_spread'].sum()
    impact_drag = cost_df['cost_impact'].sum()
    comm_drag = cost_df['cost_commission'].sum()
    trade_value = cost_df['trade_value']

    # Fix 4: turnover metrics
    total_turnover = trade_value.sum()
    n_trades = (trade_value > 0).sum()
    avg_trade_size = trade_value[trade_value > 0].mean() if n_trades > 0 else 0

    print("=" * 50)
    print("TRANSACTION COST BREAKDOWN")
    print("=" * 50)
    print(f"Gross PnL        : {gross:>12.2f}")

    # Fix 5: division by zero guard
    if abs(gross) > 1e-12:
        print(f"  Spread cost    : {spread_drag:>12.2f}  ({spread_drag/gross*100:.1f}%)")
        print(f"  Impact cost    : {impact_drag:>12.2f}  ({impact_drag/gross*100:.1f}%)")
        print(f"  Commission     : {comm_drag:>12.2f}  ({comm_drag/gross*100:.1f}%)")
        print(f"Net PnL          : {net:>12.2f}")
        print(f"Cost as % Gross  : {(gross-net)/gross*100:.1f}%")
    else:
        print(f"  Spread cost    : {spread_drag:>12.2f}  (gross~0, % undefined)")
        print(f"  Impact cost    : {impact_drag:>12.2f}  (gross~0, % undefined)")
        print(f"  Commission     : {comm_drag:>12.2f}  (gross~0, % undefined)")
        print(f"Net PnL          : {net:>12.2f}")
        print(f"Cost as % Gross  : nan  (gross PnL near zero)")

    print("-" * 50)
    print("TURNOVER METRICS")
    print(f"  Total turnover : ${total_turnover:>12,.0f}")
    print(f"  Number trades  : {n_trades:>12,}")
    print(f"  Avg trade size : ${avg_trade_size:>12,.0f}")
    print("=" * 50)


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    idx = pd.date_range("2023-01-01", periods=n, freq="10s")

    mid = pd.Series(450 + np.cumsum(np.random.randn(n) * 0.05), index=idx)
    spread = pd.Series(np.random.uniform(0.01, 0.03, n), index=idx)
    realized_vol = pd.Series(np.random.uniform(0.10, 0.25, n), index=idx)
    adv = pd.Series(np.full(n, 80_000_000), index=idx)

    prices = pd.DataFrame({'mid': mid, 'spread': spread, 'realized_vol': realized_vol})

    # Fix 3: documented — positions in shares, vol-scaled, capped at ±500
    signal = pd.Series(np.random.randn(n), index=idx)
    positions = (signal / realized_vol * 10).clip(-50, 50)

    returns = mid.pct_change().shift(-1).fillna(0)
    pnl_gross = positions.shift(1).fillna(0) * returns * mid

    cost_df = apply_transaction_costs(pnl_gross, positions, prices, adv)
    cost_summary(cost_df)

    assert cost_df['pnl_net'].sum() < cost_df['pnl_gross'].sum(), "Net must be less than gross"
    print("\nSanity check passed: net PnL < gross PnL")
