import numpy as np
import pandas as pd

np.random.seed(42)
n=100
best_bid=100+np.cumsum(np.random.uniform(-0.05,0.05,n))
best_ask=best_bid+np.random.uniform(0.01,0.03,n)
trade_price=np.where(np.random.randn(n)>0.5,best_bid,best_ask)
volume=np.random.randint(100,1000,n)
df=pd.DataFrame({
    'best_bid':best_bid,
    'best_ask':best_ask,
    'trade_price':trade_price,
    'volume':volume
})
def classify_trader(df):
    mid_price=(df['best_bid']+df['best_ask'])/2
    direction=np.where(df['trade_price']>mid_price,1,
              np.where(df['trade_price']<mid_price,-1,0))
    return direction
def trade_imbalance(df,window=10):
    df=df.copy()
    df['direction']=classify_trader(df)
    df['buy_vol']=np.where(df['direction']==1,df['volume'],0)
    df['sell_vol']=np.where(df['direction']==-1,df['volume'],0)
    rolling_buy=df['buy_vol'].rolling(window).sum()
    rolling_sell=df['sell_vol'].rolling(window).sum()
    total=rolling_buy+rolling_sell
    df['trade_imbalance']=np.where(total>0,(rolling_buy-rolling_sell)/total,0)
    return df
df=trade_imbalance(df)
print(df[["best_bid","best_ask","trade_price","direction","buy_vol","sell_vol","trade_imbalance"]].head(15))
print("\n--- TEST ---")
test = pd.DataFrame({
    'best_bid':    [100.00, 100.00, 100.00],
    'best_ask':    [100.10, 100.10, 100.10],
    'trade_price': [100.10, 100.00, 100.05],
    'volume':      [500,    500,    500]
})
test['direction'] = classify_trader(test)
print("Directions — Expected [1, -1, 0] — Got:", list(test['direction']))
    
