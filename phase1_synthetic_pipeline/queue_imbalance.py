import numpy as np
import pandas as pd

# SYNTHETIC ONLY — remove when using real Polygon data
np.random.seed(42)
n = 100
data = {
    'bid_size': np.random.randint(100, 1000, n),
    'ask_size': np.random.randint(100, 1000, n),
    'bid_size_l2': np.random.randint(50, 500, n),
    'ask_size_l2': np.random.randint(50, 500, n),
    'bid_size_l3': np.random.randint(50, 500, n),
    'ask_size_l3': np.random.randint(50, 500, n),
    'bid_size_l4': np.random.randint(50, 500, n),
    'ask_size_l4': np.random.randint(50, 500, n),
    'bid_size_l5': np.random.randint(50, 500, n),
    'ask_size_l5': np.random.randint(50, 500, n),
}
df = pd.DataFrame(data)

# --- BEST LEVEL QUEUE IMBALANCE ---
def queue_imbalance_best(df):
    # Formula: (bid_size - ask_size) / (bid_size + ask_size)
    # Source: standard microstructure definition
    # Example: bid=700, ask=300 → (700-300)/(700+300) = 400/1000 = 0.4
    total = df['bid_size'] + df['ask_size']
    qi = np.where(total == 0, 0.0, (df['bid_size'] - df['ask_size']) / total)
    return qi

# --- 5-LEVEL QUEUE IMBALANCE ---
def queue_imbalance_5level(df):
    # Formula: weighted sum across 5 levels
    # Weight = 1/level so best level gets most weight
    # Source: standard microstructure extension of best-level formula
    # Example: level1 bid=700 ask=300, level2 bid=400 ask=200...
    weights = [1, 0.8, 0.6, 0.4, 0.2]
    levels = [('bid_size', 'ask_size'),
              ('bid_size_l2', 'ask_size_l2'),
              ('bid_size_l3', 'ask_size_l3'),
              ('bid_size_l4', 'ask_size_l4'),
              ('bid_size_l5', 'ask_size_l5')]
    
    weighted_bid = sum(w * df[b] for w, (b, a) in zip(weights, levels))
    weighted_ask = sum(w * df[a] for w, (b, a) in zip(weights, levels))
    
    total = weighted_bid + weighted_ask
    qi5 = np.where(total == 0, 0.0, (weighted_bid - weighted_ask) / total)
    return qi5

# --- COMPUTE ---
df['queue_imbalance'] = queue_imbalance_best(df)
df['queue_imbalance_5level'] = queue_imbalance_5level(df)

# --- TEST ---
print("=== TEST: BEST LEVEL ===")
print(f"Expected when bid=1000, ask=0 : +1.0")
test1 = pd.DataFrame({'bid_size': [1000], 'ask_size': [0]})
print(f"Actual: {queue_imbalance_best(test1)[0]}")

print(f"\nExpected when bid=0, ask=1000 : -1.0")
test2 = pd.DataFrame({'bid_size': [0], 'ask_size': [1000]})
print(f"Actual: {queue_imbalance_best(test2)[0]}")

print(f"\nExpected when bid=0, ask=0 : 0.0 (no crash)")
test3 = pd.DataFrame({'bid_size': [0], 'ask_size': [0]})
print(f"Actual: {queue_imbalance_best(test3)[0]}")

print("\n=== SAMPLE OUTPUT ===")
print(df[['bid_size', 'ask_size', 'queue_imbalance', 'queue_imbalance_5level']].head(5))
