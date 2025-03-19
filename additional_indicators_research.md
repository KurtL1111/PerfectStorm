# Additional Indicators Research

Based on user feedback and the provided table, here are the additional technical indicators that need to be implemented in the Perfect Storm dashboard:

## 1. Indicators from the Table

### Parabolic SAR
- **Description**: Parabolic Stop and Reverse (SAR) is a trend-following indicator that provides entry and exit points.
- **Buy Signal**: Price crosses above the Parabolic SAR
- **Sell Signal**: Price crosses below the Parabolic SAR
- **Implementation**: Can be implemented using custom calculations or alternative libraries if TA-Lib is problematic

### Momentum
- **Description**: Measures the rate of change in price over a specified period
- **Buy Signal**: Momentum crosses above zero
- **Sell Signal**: Momentum crosses below zero
- **Implementation**: Simple calculation of current price minus price n periods ago

### KST (Know Sure Thing)
- **Description**: A complex oscillator based on the smoothed rate-of-change for different time periods
- **Buy Signal**: KST crosses above the KST Signal line
- **Sell Signal**: KST crosses below the KST Signal line
- **Implementation**: Requires multiple ROC calculations and moving averages

### ROC (Rate of Change)
- **Description**: Measures the percentage change in price over a specified period
- **Buy Signal**: ROC crosses above zero
- **Sell Signal**: ROC crosses below zero
- **Implementation**: (Current Price / Price n periods ago) * 100 - 100

### %K (Stochastic Oscillator Fast)
- **Description**: Compares a security's closing price to its price range over a specified period
- **Buy Signal**: %K crosses above %D
- **Sell Signal**: %K crosses below %D
- **Implementation**: ((Current Close - Lowest Low) / (Highest High - Lowest Low)) * 100

### %D (Stochastic Oscillator Slow)
- **Description**: Moving average of %K
- **Buy Signal**: %D crosses below 20
- **Sell Signal**: %D crosses above 80
- **Implementation**: 3-period moving average of %K

### TSI (True Strength Index)
- **Description**: A momentum oscillator based on double smoothed price changes
- **Buy Signal**: TSI crosses above zero
- **Sell Signal**: TSI crosses below zero
- **Implementation**: Complex calculation involving EMA of price changes

### SMA (Simple Moving Averages)
- **Description**: Average price over specified periods (5, 9, 20, 50, 100, 200 days)
- **Buy Signals**: 
  - Price crosses above the SMA
  - Shorter-term SMA crosses above longer-term SMA
- **Sell Signals**: 
  - Price crosses below the SMA
  - Shorter-term SMA crosses below longer-term SMA
- **Implementation**: Already implemented but need to add crossover signals

### ADL (Accumulation Distribution Line)
- **Description**: Volume-based indicator that measures the cumulative flow of money into and out of a security
- **Buy Signal**: ADL is rising
- **Sell Signal**: ADL is falling
- **Implementation**: Already implemented but need to add signals

### MFI (Money Flow Index)
- **Description**: Volume-weighted RSI that measures the flow of money into and out of a security
- **Buy Signal**: MFI crosses below 20
- **Sell Signal**: MFI crosses above 80
- **Implementation**: Requires calculation of typical price and money flow

### ADX (Average Directional Index)
- **Description**: Measures the strength of a trend
- **Buy Signal**: ADX above 20 and +DI crosses above -DI
- **Sell Signal**: ADX above 20 and +DI crosses below -DI
- **Implementation**: Already implemented but need to adjust signals

### RSI (Relative Strength Index)
- **Description**: Momentum oscillator that measures the speed and change of price movements
- **Buy Signal**: RSI crosses above 65
- **Sell Signal**: RSI crosses below 40
- **Implementation**: Standard RSI calculation

### MACD (Moving Average Convergence Divergence)
- **Description**: Trend-following momentum indicator
- **Buy Signal**: MACD crosses above the Signal line
- **Sell Signal**: MACD crosses below the Signal line
- **Implementation**: Already implemented but need to add crossover signals

### BB C/D Trigger
- **Description**: Bollinger Bands with Close/Down signal
- **Buy Signal**: Trend changes from down to up
- **Sell Signal**: Trend changes from up to down
- **Implementation**: Already implemented but need to add trend change signals

### CMF (Chaikin Money Flow)
- **Description**: Measures the amount of Money Flow Volume over a specific period
- **Buy Signal**: CMF crosses above zero
- **Sell Signal**: CMF crosses below zero
- **Implementation**: Already implemented but need to add crossover signals

### CCI (Commodity Channel Index)
- **Description**: Measures a security's variation from its statistical mean
- **Buy Signal**: CCI crosses above -100
- **Sell Signal**: CCI crosses below 100
- **Implementation**: Standard CCI calculation

## 2. ARMS Index Correction

The ARMS Index (TRIN) calculation needs to be corrected to use real market data from MarketWatch:
- Need to scrape Advancing Issues, Declining Issues, Advancing Volume, and Declining Volume from MarketWatch
- Formula: (Advancing Issues/Declining Issues)/(Advancing Volume/Declining Volume)
- URL: https://www.marketwatch.com/market-data/us?mod=market-data-center

## 3. Bulls vs Bears Ratio Update

The Bulls vs Bears Ratio should use data from the AAII Investor Sentiment Survey:
- Need to scrape or access the latest survey data from AAII
- URL: https://www.aaii.com/sentimentsurvey/sent_results
- This will provide more accurate market sentiment data than the current calculation

## 4. Buy/Sell Signals on Chart

Need to add visual indicators on the main chart for Strong Buy and Strong Sell signals:
- Use the criteria from the table to determine when these signals occur
- Never plot more than one of the same indicator in a row
- Use distinctive markers or annotations on the chart

## Implementation Approaches

Since TA-Lib installation is problematic, we have several alternatives:
1. Implement custom calculations for all indicators
2. Use pandas_ta library which is a pure Python implementation of technical analysis indicators
3. Use other libraries like finta or ta that don't require C dependencies
4. Continue trying to resolve TA-Lib installation issues

For the next steps, we'll focus on:
1. Updating data retrieval to get market data for ARMS Index and AAII sentiment
2. Implementing the missing indicators using custom calculations or alternative libraries
3. Adding buy/sell signals to the chart
4. Integrating all improvements into the dashboard
