import math

# Return Calculator
prices = [100, 102, 103, 104, 105, 106]

returns = []
for i in range(1, len(prices)):
    daily_return = (prices[i] - prices[i-1]) / prices[i-1]
    returns.append(daily_return)

print("=== Return Calculator ===")
print("Best day:      ", round(max(returns) * 100, 2), "%")
print("Worst day:     ", round(min(returns) * 100, 2), "%")
print("Average return:", round(sum(returns)/len(returns) * 100, 2), "%")

# Stats Calculator
mean = sum(returns) / len(returns)
variance = sum((r - mean) ** 2 for r in returns) / len(returns)
std = math.sqrt(variance)

print("\n=== Stats Calculator ===")
print("Mean:    ", round(mean * 100, 2), "%")
print("Variance:", round(variance, 8))
print("Std Dev: ", round(std * 100, 2), "%")
