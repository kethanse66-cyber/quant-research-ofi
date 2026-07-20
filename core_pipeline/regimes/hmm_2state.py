import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading SPY data...")
df = pd.read_parquet('features/SPY_features.parquet')
df['ts'] = pd.to_datetime(df['ts'])
df = df.set_index('ts')

# ── 2. RESAMPLE TO 1-MINUTE BARS ─────────────────────────────────────────────
print("Resampling to 1-minute bars...")
bars = df['realized_vol'].resample('1min').mean()
bars = bars.dropna()
print(f"Total 1-min bars: {len(bars)}")

# ── 3. PREPARE FEATURE MATRIX ─────────────────────────────────────────────────
X = bars.values.reshape(-1, 1)

# ── 4. FIT 2-STATE HMM ────────────────────────────────────────────────────────
print("Fitting 2-state HMM...")
model = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
model.fit(X)

# ── 5. PREDICT STATES ─────────────────────────────────────────────────────────
states = model.predict(X)
print(f"State 0 count: {(states==0).sum()}")
print(f"State 1 count: {(states==1).sum()}")

# ── 6. LABEL STATES BY VOLATILITY ─────────────────────────────────────────────
state0_vol = X[states==0].mean()
state1_vol = X[states==1].mean()
print(f"\nState 0 mean vol: {state0_vol:.6f}")
print(f"State 1 mean vol: {state1_vol:.6f}")

if state0_vol < state1_vol:
    labels = {0: 'Low Vol', 1: 'High Vol'}
else:
    labels = {0: 'High Vol', 1: 'Low Vol'}

print(f"\nState 0 = {labels[0]}")
print(f"State 1 = {labels[1]}")

# ── 7. TRANSITION MATRIX ──────────────────────────────────────────────────────
print("\nTransition Matrix:")
print(pd.DataFrame(model.transmat_,
    index=['From State 0', 'From State 1'],
    columns=['To State 0', 'To State 1']).round(4))

# ── 8. PLOT ───────────────────────────────────────────────────────────────────
print("\nPlotting regimes...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(bars.index, bars.values, color='black', linewidth=0.5)
ax1.set_ylabel('Realized Volatility')
ax1.set_title('SPY Realized Volatility with HMM Regimes (2 States)')

colors = ['green', 'red']
for state in [0, 1]:
    mask = states == state
    ax2.fill_between(bars.index, 0, 1,
        where=mask, alpha=0.6,
        color=colors[state],
        label=labels[state])
ax2.set_ylabel('Regime')
ax2.set_title('Regime States')
ax2.legend()

plt.tight_layout()
plt.savefig('reports/hmm_2state_regimes.png', dpi=150)
print("Plot saved to reports/hmm_2state_regimes.png")
plt.show()

print("\nhmm_2state.py COMPLETE")
