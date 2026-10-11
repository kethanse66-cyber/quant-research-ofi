import numpy as np
from scipy import stats

# Step 1 — Create 20 random strategies
np.random.seed(42)
strategies = np.random.randn(100, 20)

# Step 2 — Run t-test on each strategy
p_values = []
for i in range(20):
    t_stat, p_val = stats.ttest_1samp(strategies[:, i], 0)
    p_values.append(p_val)

p_values_array = np.array(p_values)

# Step 3 — Count significant before Bonferroni
significant_before = sum(p_values_array < 0.05)
print("Total strategies tested:", 20)
print("Significant before Bonferroni:", significant_before)

# Step 4 — Apply Bonferroni correction
bonferroni_threshold = 0.05 / 20
print("New threshold after Bonferroni:", bonferroni_threshold)

# Step 5 — Count significant after Bonferroni
significant_after = sum(p_values_array < bonferroni_threshold)
print("Significant after Bonferroni:", significant_after)

# Step 6 — The lesson
print("\nLesson: Before correction, 1 strategy looked real.")
print("After correction, 0 survived.")
print("That 1 signal was pure luck, not alpha.")
