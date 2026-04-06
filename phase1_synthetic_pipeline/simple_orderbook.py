# simple_orderbook.py

bids = {100: 500, 99: 300, 98: 200}
asks = {101: 400, 102: 600, 103: 100}

def add_order(side, price, quantity):
    if side == "buy":
        bids[price] = quantity
    elif side == "sell":
        asks[price] = quantity

def best_bid():
    return max(bids)

def best_ask():
    return min(asks)

def mid_price():
    return (best_bid() + best_ask()) / 2

def calculate_ofi():
    return sum(bids.values()) - sum(asks.values())

def queue_imbalance():
    bid = bids[best_bid()]
    ask = asks[best_ask()]
    return (bid - ask) / (bid + ask)

# test
add_order("buy", 96, 1000)

print("Bids:", bids)
print("Asks:", asks)
print("Mid Price:", mid_price())
print("OFI:", calculate_ofi())
print("Queue Imbalance:", queue_imbalance())
