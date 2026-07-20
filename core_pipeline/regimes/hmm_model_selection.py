import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading SPY data...")
df = pd.read_parquet('features/SPY_features.parquet')
df['ts'] = pd.to_datetime(df['ts'])
df = df.set_index('ts')

# ── 2. PREPARE FEATURES ───────────────────────────────────────────────────────
print("Preparing features...")
features = ['realized_vol', 'ofi_norm', 'spread']
bars = df[features].resample('1min').mean().dropna()
print(f"Total 1-min bars: {len(bars)}")

# ── 3. RANK TRANSFORM ─────────────────────────────────────────────────────────
X = bars.copy()
for col in features:
    X[col] = stats.rankdata(bars[col]) / len(bars)
X = X.values

# ── 4. FIT HMM FOR 2 TO 7 STATES — 10 seeds each ────────────────────────────
print("\nFitting HMM models (10 seeds each)...")
results = {}

for n_states in range(2, 8):
    best_model = None
    best_score = -np.inf

    for seed in range(10):
        try:
            model = GaussianHMM(n_components=n_states, covariance_type='full',
                                n_iter=100, random_state=seed)
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except:
            continue

    if best_model is None:
        print(f"States: {n_states} | FAILED all seeds")
        continue

    log_likelihood = best_score
    n_params = n_states * n_states + 2 * n_states * X.shape[1]
    n_samples = len(X)

    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(n_samples) * n_params

    results[n_states] = {
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'n_params': n_params
    }
    print(f"States: {n_states} | LogLik: {log_likelihood:.2f} | AIC: {aic:.2f} | BIC: {bic:.2f}")

# ── 5. FIND OPTIMAL STATE COUNT ───────────────────────────────────────────────
best_bic = min(results, key=lambda x: results[x]['bic'])
best_aic = min(results, key=lambda x: results[x]['aic'])
print(f"\nOptimal states by BIC: {best_bic}")
print(f"Optimal states by AIC: {best_aic}")

# ── 6. PLOT ───────────────────────────────────────────────────────────────────
print("\nPlotting BIC/AIC curves...")
states = list(results.keys())
bic_values = [results[s]['bic'] for s in states]
aic_values = [results[s]['aic'] for s in states]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(states, bic_values, 'b-o', label='BIC', linewidth=2, markersize=8)
ax.plot(states, aic_values, 'r-o', label='AIC', linewidth=2, markersize=8)
ax.axvline(x=best_bic, color='blue', linestyle='--', alpha=0.5, label=f'Best BIC = {best_bic} states')
ax.axvline(x=best_aic, color='red', linestyle='--', alpha=0.5, label=f'Best AIC = {best_aic} states')
ax.set_xlabel('Number of States')
ax.set_ylabel('Score (lower is better)')
ax.set_title('HMM Model Selection — BIC and AIC (10 seeds per state count)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/hmm_model_selection.png', dpi=150)
print("Plot saved to reports/hmm_model_selection.png")
plt.show()

print("\nhmm_model_selection.py COMPLETE")
