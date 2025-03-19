# Perfect Storm Investment Strategy Indicators

Based on John R. Connelley's book "Tech Smart", the "Perfect Storm" investment strategy relies on the following key indicators:

## 1. Bollinger Bands
- **Middle Band**: 20-day simple moving average (SMA)
- **Upper Band**: 20-day SMA + (20-day standard deviation of price × 2)
- **Lower Band**: 20-day SMA - (20-day standard deviation of price × 2)
- **Formula using Bollinger Bands data**: Bellinge Bands ee / dj - bb c / d

## 2. Close/Down (C/D) Signals
- Tracks when the c/d column is trending lower (market declining) or rising (market rising)
- Used as a primary signal in the strategy
- Looking for at least 80% signal before considering

## 3. Moving Averages
- Multiple periods: 5, 9, 20, 50, 100, 200 day moving averages
- Used to determine primary trends
- Each moving average is calculated as the average price over the last n-periods

## 4. MACD (Moving Average Convergence/Divergence)
- **Formula**: 12day = (Close × 15% + 85% of yesterday's ema 12day)
- **26day** = (close × 7.5% + 92.5% of yesterday's ema 26 day)
- **MACD Line** = (12 day minus 26 day) = moving average

## 5. Volume Indicators
- **ARMS INDEX (TRIN)**: Developed in 1967 by Richard Arms
  - Ratio of advancing issues/declining issues to advancing volume/declining volume
  - Volume-based breadth indicator

- **Chaikin Money Flow Indicator**:
  - Step 1: ((Close - Low) - (High - Close)) / (High - Low) × Volume
  - Step 2: 21 Day Average of Step1 (Daily MF) / 21 Day Average of Volume

## 6. S&P Formula Spreadsheet Components
- Close values
- Totals
- Up/down signals
- Percentages
- Volume (high/medium/low compared to previous day)

## 7. Additional Components
- Accumulation/Distribution
- Directional Movement
- Bulls vs. Bears indicators
