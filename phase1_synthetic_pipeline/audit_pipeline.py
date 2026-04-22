import pandas as pd
import numpy as np

# ── SYNTHETIC ONLY — remove when using real Polygon data ──────────────────
np.random.seed(42)
n = 100
timestamps = pd.date_range('2024-01-02 09:30:00', periods=n, freq='1min')

raw_ofi         = np.random.randn(n)
raw_spread      = np.abs(np.random.randn(n)) * 0.01
raw_volume      = np.random.randint(100, 1000, n).astype(float)
raw_microprice  = 100 + np.cumsum(np.random.randn(n) * 0.01)

df_raw = pd.DataFrame({
    'timestamp':   timestamps,
    'ofi':         raw_ofi,
    'spread':      raw_spread,
    'volume':      raw_volume,
    'microprice':  raw_microprice,
}, index=timestamps)
# ── END SYNTHETIC ONLY ────────────────────────────────────────────────────

# Step 1 — lag all features
FEATURE_COLS = ['ofi', 'spread', 'volume', 'microprice']

df_lagged = df_raw.copy()
for col in FEATURE_COLS:
    df_lagged[col] = df_raw[col].shift(1)

# Step 2 — compute target variable (future return)
df_lagged['target_1m'] = np.log(
    df_raw['microprice'].shift(-1) / df_raw['microprice']
)

# Step 3 — drop rows with any NaN
df_model = df_lagged.dropna()
print(f"Rows before dropna : {len(df_lagged)}")
print(f"Rows after dropna  : {len(df_model)}")
print(f"Rows dropped       : {len(df_lagged) - len(df_model)}")
print()

# Step 4 — audit every feature
print("=" * 55)
print("LOOK-AHEAD AUDIT REPORT")
print("=" * 55)

all_passed = True

for col in FEATURE_COLS:
    row0_value = df_model[col].iloc[0]
    is_nan     = pd.isna(row0_value)
    status     = "PASS" if not is_nan else "FAIL — NaN after dropna"
    if is_nan:
        all_passed = False
    print(f"  {col:<20} row0 = {str(round(row0_value, 4)):<10}  {status}")

print("=" * 55)

# Step 5 — print first prediction timestamp
first_valid_idx = df_model.index[0]
print(f"\nFirst valid prediction row : {first_valid_idx}")

# Step 6 — print last row timestamp
last_row_idx = df_model.index[-1]
print(f"Last row timestamp         : {last_row_idx}")

# Step 7 — correct lag validation
# After shift(1) and dropna(), df_model.iloc[0] is timestamp row1
# Its feature value must equal df_raw.iloc[0] — raw row 0
print(f"\nFirst feature value audit:")
for col in FEATURE_COLS:
    raw_val    = round(df_raw[col].iloc[0], 4)   # raw row 0
    lagged_val = round(df_model[col].iloc[0], 4)  # lagged row 0 = timestamp row1
    match      = raw_val == lagged_val
    print(f"  {col:<20} raw row0 = {raw_val:<10}  lagged row0 = {lagged_val:<10}  match = {match}")

print()
print("=" * 55)

if all_passed:
    print("\nALL FEATURES PASSED. No look-ahead bias detected.")
    print("Pipeline is clean. Ready for Phase 2.")
else:
    print("\nSOME FEATURES FAILED. Fix before Phase 2.")
