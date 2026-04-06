# simple_orderbook.py
# Day 1 — Basic Order Book Structure

bids = {}
bids[100] = 500
bids[99] = 300
bids[98] = 200

asks = {}
asks[101] = 400
asks[102] = 600
asks[103] = 100

def add_order(side, price, quantity):
    if side == "buy":
        bids[price] = quantity
    elif side == "sell":
        asks[price] = quantity

def mid_price():
    return (max(bids) + min(asks)) / 2

def calculate_ofi():
    return sum(bids.values()) - sum(asks.values())

def queue_imbalance():
    return (bids[max(bids)] - asks[min(asks)]) / (bids[max(bids)] + asks[min(asks)])

# Test
add_order("buy", 96, 1000)
print("Bids:", bids)
print("Asks:", asks)
print("Mid Price:", mid_price())
print("OFI:", calculate_ofi())
print(queue_imbalance())
