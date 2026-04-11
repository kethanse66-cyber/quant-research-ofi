# simple_orderbook.py
# Simulates a basic limit order book.
# Reference: Cont, Kukanov & Stoikov (2014)
 
class OrderBook:
    """Basic order book — tracks bids, asks, and computes OFI."""
 
    def __init__(self):
        self.bids = {100: 500, 99: 300, 98: 200}
        self.asks = {101: 400, 102: 600, 103: 100}
        self._prev_best_bid_size = self.bids[max(self.bids)]
        self._prev_best_ask_size = self.asks[min(self.asks)]
        self._prev_bid_price = max(self.bids)
        self._prev_ask_price = min(self.asks)
 
    def add_order(self, side, price, quantity):
        """Add shares to a price level on the bid or ask side."""
        if side == "buy":
            self.bids[price] = self.bids.get(price, 0) + quantity
        elif side == "sell":
            self.asks[price] = self.asks.get(price, 0) + quantity
        else:
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
 
    def best_bid(self):
        """Highest price someone is willing to buy at."""
        return max(self.bids)
 
    def best_ask(self):
        """Lowest price someone is willing to sell at."""
        return min(self.asks)
 
    def mid_price(self):
        """Simple average of best bid and best ask."""
        return (self.best_bid() + self.best_ask()) / 2
 
    def weighted_mid_price(self):
        """Mid price weighted by queue sizes — more accurate than simple mid.
 
        Tilts toward the side with more volume sitting at the touch.
        Formula: (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
        Falls back to simple mid if both queues are empty.
        """
        bid_p = self.best_bid()
        ask_p = self.best_ask()
        bid_s = self.bids[bid_p]
        ask_s = self.asks[ask_p]
 
        if bid_s + ask_s == 0:
            return self.mid_price()
 
        return (bid_s * ask_p + ask_s * bid_p) / (bid_s + ask_s)
 
    def queue_imbalance(self):
        """How lopsided the book is at the best bid and ask.
 
        Formula: (bid_size - ask_size) / (bid_size + ask_size)
        Range: +1 means all size on bids, -1 means all size on asks.
        Returns 0 if both sides are empty.
        """
        bid = self.bids[self.best_bid()]
        ask = self.asks[self.best_ask()]
 
        if bid + ask == 0:
            return 0.0
 
        return (bid - ask) / (bid + ask)
 
    def calculate_ofi(self):
        """Order Flow Imbalance at the best touch — Cont et al. (2014).
 
        OFI = change in best bid size - change in best ask size.
        Positive = buying pressure. Negative = selling pressure.
 
        If the best price level changes between calls, the old queue
        is treated as fully gone (prev_size = 0) before computing the delta.
        This is critical — otherwise OFI picks up noise from level changes.
        """
        curr_bid_price = self.best_bid()
        curr_ask_price = self.best_ask()
        curr_bid_size  = self.bids[curr_bid_price]
        curr_ask_size  = self.asks[curr_ask_price]
 
        if curr_bid_price != self._prev_bid_price:
            self._prev_best_bid_size = 0
 
        if curr_ask_price != self._prev_ask_price:
            self._prev_best_ask_size = 0
 
        delta_bid = curr_bid_size - self._prev_best_bid_size
        delta_ask = curr_ask_size - self._prev_best_ask_size
 
        self._prev_best_bid_size = curr_bid_size
        self._prev_best_ask_size = curr_ask_size
        self._prev_bid_price     = curr_bid_price
        self._prev_ask_price     = curr_ask_price
 
        return delta_bid - delta_ask
 
 
if __name__ == "__main__":
    book = OrderBook()
 
    print("=== Initial State ===")
    print(f"Best bid     : {book.best_bid()}")
    print(f"Best ask     : {book.best_ask()}")
    print(f"Spread       : {book.best_ask() - book.best_bid()} ticks")
    print(f"Mid price    : {book.mid_price()}")
    print(f"Weighted mid : {book.weighted_mid_price():.4f}")
    print(f"Queue imbal  : {book.queue_imbalance():.4f}")
 
    print("\n--- Buy order at 100 for 300 shares ---")
    book.add_order("buy", 100, 300)
    print(f"OFI          : {book.calculate_ofi()}")
    print(f"Queue imbal  : {book.queue_imbalance():.4f}")
    print(f"Weighted mid : {book.weighted_mid_price():.4f}")
 
    print("\n--- Large sell at 101 for 800 shares ---")
    book.add_order("sell", 101, 800)
    print(f"OFI          : {book.calculate_ofi()}")
    print(f"Queue imbal  : {book.queue_imbalance():.4f}")
 
    print("\n--- Best bid level changes: new bid at 102 ---")
    book.add_order("buy", 102, 600)
    ofi = book.calculate_ofi()
    print(f"Best bid now : {book.best_bid()}")
    print(f"OFI          : {ofi}  (prev queue treated as 0)")
    print(f"Expected     : {book.bids[102]}")
 
    print("\n--- Zero queue test ---")
    empty_book = OrderBook()
    empty_book.bids = {100: 0}
    empty_book.asks = {101: 0}
    print(f"Weighted mid : {empty_book.weighted_mid_price():.4f}  (falls back to simple mid)")
    print(f"Queue imbal  : {empty_book.queue_imbalance():.4f}  (returns 0.0)")
