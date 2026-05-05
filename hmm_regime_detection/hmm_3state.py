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

# ── 4. FIT 3-STATE HMM ────────────────────────────────────────────────────────
print("Fitting 3-state HMM...")
model = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
model.fit(X)

# ── 5. PREDICT STATES ─────────────────────────────────────────────────────────
states = model.predict(X)
print(f"State 0 count: {(states==0).sum()}")
print(f"State 1 count: {(states==1).sum()}")
print(f"State 2 count: {(states==2).sum()}")

# ── 6. LABEL STATES BY VOLATILITY ─────────────────────────────────────────────
means = {i: X[states==i].mean() for i in range(3)}
sorted_states = sorted(means, key=means.get)
labels = {
    sorted_states[0]: 'Low Vol',
    sorted_states[1]: 'Medium Vol',
    sorted_states[2]: 'High Vol'
}

print("\nState volatility means:")
for state in range(3):
    print(f"State {state} = {labels[state]}: mean vol = {means[state]:.6f}")

# ── 7. TRANSITION MATRIX ──────────────────────────────────────────────────────
print("\nTransition Matrix:")
print(pd.DataFrame(model.transmat_,
    index=['From S0', 'From S1', 'From S2'],
    columns=['To S0', 'To S1', 'To S2']).round(4))

# ── 8. REGIME STATS TABLE ─────────────────────────────────────────────────────
print("\nRegime Characterization:")
for state in range(3):
    mask = states == state
    vol_mean = X[mask].mean()
    vol_std = X[mask].std()
    pct = mask.sum() / len(states) * 100
    print(f"{labels[state]}: mean_vol={vol_mean:.4f}, std={vol_std:.4f}, % time={pct:.1f}%")

# ── 9. PLOT ───────────────────────────────────────────────────────────────────
print("\nPlotting regimes...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(bars.index, bars.values, color='black', linewidth=0.5)
ax1.set_ylabel('Realized Volatility')
ax1.set_title('SPY Realized Volatility with HMM Regimes (3 States)')

colors = {
    sorted_states[0]: 'green',
    sorted_states[1]: 'orange',
    sorted_states[2]: 'red'
}

for state in range(3):
    mask = states == state
    ax2.fill_between(bars.index, 0, 1,
        where=mask, alpha=0.6,
        color=colors[state],
        label=labels[state])

ax2.set_ylabel('Regime')
ax2.set_title('Regime States (3 States)')
ax2.legend()

plt.tight_layout()
plt.savefig('reports/hmm_3state_regimes.png', dpi=150)
print("Plot saved to reports/hmm_3state_regimes.png")
plt.show()

print("\nhmm_3state.py COMPLETE")
