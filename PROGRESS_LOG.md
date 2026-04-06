# Progress Log

## D1 — Return Calculator

**Learned:**
- Python for loops, lists, basic math operations

**Built:**
- return_calculator.py
- daily returns, mean, variance, std using for loops

**Confused:**
- nothing major

---

## D2 — NumPy Foundations

**Learned:**
- NumPy arrays, vectorised returns, log returns
- Indexing, conditional filtering, win rate
- Expected value, basic probability, conditional probability
- English sentences Phase 1 all 5 done
- Mental math tricks: percentage, multiplication, EV

**Built:**
- return_calculator_numpy.py
- max, min, mean, std, win rate, log returns, no for loops

**Confused:**
- log returns math, e and natural log
- mental math multiplication still slow

---

## D3 — Pandas Basics + DataFrame Operations
**Learned:**
- Downloaded real SPY data using yfinance
- What a DataFrame is and how it differs from NumPy array
- pct_change() to calculate 1257 daily returns
- df['column'] syntax and why quotes are needed
- Plotted closing price and daily returns using matplotlib
- Saved data to CSV using to_csv()
- df.shape → rows and columns
- df.describe() → statistics of all columns
- df.isnull().sum() → count missing values
- df.loc[] → select rows by date
- df.sort_values() → sort by any column
- df.rolling(20).mean() → 20 day moving average
- df.resample('ME').sum() → monthly returns
- df.dropna() → remove missing values
- Read COVID crash March 2020 in real data
- Volatility clustering concept
- [] vs () rule in Pandas
- ## D4 — Descriptive Stats + SciPy
**Learned:**
- What scipy.stats is and why we use it
- skewness: SPY = -0.54, crashes bigger than rallies
- kurtosis: SPY = 11.44, extreme days happen far more than normal
- normality test: p-value = 0.0, SPY confirmed not normal
- histogram with normal curve overlay
- density=True to compare histogram and curve on same scale
- alpha for transparency
- np.linspace to create smooth curve points
- stats.norm.pdf to draw perfect normal bell curve

**Built:**
- stats_report.py
- histogram of SPY returns with normal curve overlay
- proved SPY is not normally distributed three ways

**Confused:**
- nothing major

**Built:**
- pandas_basics.py
- SPY closing price chart 2020 to 2024
- SPY daily returns chart
- SPY returns vs 20 day rolling average chart
- Found 5 worst and 5 best days in SPY history


**Confused:**
- nothing major

- ## D5 — Hypothesis Testing
**Learned:**
- What a null hypothesis is
- t-statistic: how far the mean is from zero in standard errors
- p-value above 0.05 means result could be luck
- ttest_1samp from scipy.stats
- f-strings and round() for printing results

**Built:**
- hypothesis_test.py
- tested if SPY mean return is different from zero
- t-stat = 1.66, p-value = 0.096, cannot confirm it is real

**Confused:**
- nothing major

- ## D6 — Correlation + OLS Regression
**Learned:**
- Correlation measures direction of relationship between two assets
- SPY vs QQQ correlation = 0.93, very strongly related
- OLS regression finds exact multiplier, not just direction
- Beta = 1.13: when SPY moves 1%, QQQ moves 1.13%
- R-squared = 0.87: SPY explains 87% of QQQ movement
- P-value = 0.0: relationship is real, not random chance
- sm.add_constant adds intercept column for alpha calculation
- Residual skew and kurtosis visible inside model.summary()

**Built:**
- regression_analysis.py
- SPY vs QQQ correlation using .corr()
- OLS regression using statsmodels
- Scatter plot of daily returns

**Confused:**
- nothing major

- ## D7 — Time Series + Stationarity
**Learned:**
- What stationarity means and why it matters
- ADF test using statsmodels adfuller()
- SPY price is NOT stationary: p-value = 0.948
- SPY returns ARE stationary: p-value = 0.0
- Rolling 20-day mean and std using .rolling(20)
- If you regress two non-stationary series you get fake results

**Built:**
- stationarity_test.py
- ADF test on SPY price and returns
- Rolling mean and std chart

**Confused:**
- nothing major

- ## D8 — Multiple Testing Problem
**Learned:**
- Why testing many strategies always produces fake signals
- Bonferroni correction: new threshold = 0.05 / number of tests
- Before correction: 1 strategy looked significant
- After correction: 0 survived
- This lesson protects the entire research project

**Built:**
- multiple_testing_demo.py
- 20 random strategies tested
- Bonferroni correction applied
- Proved the 1 signal was pure luck

**Confused:**
- nothing major
- 
Apr 3 - R1 - Returned. Reviewed all scripts. 
Data source decided: Massive..com

## D9 — Simple Order Book (Phase 1 Start)
**Learned:**
- What a limit order book is
- Bid and ask dictionaries
- Bid-ask spread calculation
- Mid price formula
- OFI = sum(bids) - sum(asks)

**Built:**
- phase1_synthetic_pipeline/simple_orderbook.py
- add_order function for buy and sell
- mid_price function
- calculate_ofi function

**Confused:**
- nothing major
