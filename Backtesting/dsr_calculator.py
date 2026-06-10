# ── POOLED CROSS-TICKER DSR ───────────────────────────────────────────────────
print("=" * 60)
print("POOLED DSR — ALL 12 TICKERS (84 folds)")
print("=" * 60)

all_returns = preds["sharpe_gross"].dropna()
T_pool      = len(all_returns)

mean_pool = all_returns.mean()
std_pool  = all_returns.std()
skew_pool = all_returns.skew()
kurt_pool = all_returns.kurt()

sharpe_pool_annual = (mean_pool / std_pool) * np.sqrt(12)

sr_hat_pool       = mean_pool / std_pool
var_sr_pool       = (1.0 / T_pool) * (1 - skew_pool * sr_hat_pool + (kurt_pool / 4.0) * sr_hat_pool**2)
var_sr_pool       = max(var_sr_pool, 1e-10)

dsr_pool = norm.cdf((sr_hat_pool - sr_star_per_fold) / np.sqrt(var_sr_pool))

print(f"  Total folds      : {T_pool}")
print(f"  Mean fold Sharpe : {mean_pool:.4f}")
print(f"  Std  fold Sharpe : {std_pool:.4f}")
print(f"  Skewness         : {skew_pool:.4f}")
print(f"  Excess kurtosis  : {kurt_pool:.4f}")
print(f"  Sharpe annualized: {sharpe_pool_annual:.4f}")
print(f"  DSR              : {dsr_pool:.4f}")
print()
if dsr_pool >= 0.95:
    print(f"  DSR={dsr_pool:.2f} → Strong evidence of skill across all tickers")
elif dsr_pool >= 0.90:
    print(f"  DSR={dsr_pool:.2f} → Moderate evidence of skill")
elif dsr_pool >= 0.75:
    print(f"  DSR={dsr_pool:.2f} → Weak evidence — borderline")
else:
    print(f"  DSR={dsr_pool:.2f} → Insufficient evidence")
