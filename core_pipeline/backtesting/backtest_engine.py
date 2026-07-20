import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, signal, prices, spread_bps=1.0,
                 commission_bps=0.5, max_position=1.0):
        self.signal = signal
        self.prices = prices
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps
        self.max_position = max_position
        # compute position once for reuse
        self.position = self.signal.shift(1).clip(
            -self.max_position, self.max_position
        )
        self.pnl = self._run()

    def _run(self):
        returns = np.log(self.prices).diff()
        gross_pnl = self.position * returns
        turnover = self.position.diff(1).abs()
        spread_cost = turnover * (self.spread_bps / 10000) / 2
        commission_cost = turnover * (self.commission_bps / 10000)
        net_pnl = gross_pnl - spread_cost - commission_cost
        return net_pnl.dropna()

    def sharpe(self):
        if self.pnl.std() == 0:
            return np.nan
        ann_factor = np.sqrt(252 * 6.5 * 360)  # 10-second bars
        return (self.pnl.mean() / self.pnl.std()) * ann_factor

    def max_drawdown(self):
        equity = (1 + self.pnl).cumprod()
        return (equity / equity.cummax() - 1).min()

    def avg_turnover(self):
        # correct — position changes not pnl changes
        return self.position.diff().abs().mean()

    def summary(self):
        equity = (1 + self.pnl).cumprod()
        print(f"Sharpe:       {self.sharpe():.2f}")
        print(f"Max Drawdown: {self.max_drawdown():.2%}")
        print(f"Final Equity: {equity.iloc[-1]:.4f}")
        print(f"Win Rate:     {(self.pnl > 0).mean():.2%}")
        print(f"Avg Turnover: {self.avg_turnover():.6f}")
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # synthetic test — random signal, random prices
    n = 1000
    prices = pd.Series(100 * np.exp(
        np.random.randn(n).cumsum() * 0.001
    ))
    signal = pd.Series(np.random.randn(n)).rolling(10).mean()
    
    engine = BacktestEngine(signal, prices)
    engine.summary()
