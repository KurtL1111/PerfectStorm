# Additional Indicators Research for Perfect Storm Dashboard

## Table of Indicators from User Feedback

| Calculated Variable | Buy Signal(s) | Sell Signal(s) | Notes |
|---------------------|---------------|----------------|-------|
| Parabolic SAR | Price crosses above the Parabolic SAR | Price crosses below the Parabolic SAR | Convergence/Divergence analysis may be valuable |
| 10d Momentum | Momentum crosses above zero | Momentum crosses below zero | Maybe okay; will have to test |
| KST | KST crosses above the KST Signal line | KST crosses below the KST Signal line | Should be good; will have to test |
| ROC 12 | ROC crosses above zero | ROC crosses below zero | Good |
| %K | %K crosses above %D | %K crosses below %D | Fast stoch indicator used for VIX; same signals as below, but there might be some relationship between K crossing D so will have to investigate |
| %D | %D crosses below 20 | %D crosses above 80 | Slow stoch indicator good; should create c/d calc |
| TSI | TSI crosses above zero | TSI crosses below zero | Need to figure out what's going on here |
| 5-day SMA | Price crosses above the 5-day SMA | Price crosses below the 5-day SMA | Short-term |
| 9-day SMA | Price crosses above the 9-day SMA | Price crosses below the 9-day SMA | Short-term |
| 20-day SMA | Price crosses above the 20-day SMA | Price crosses below the 20-day SMA | Short-term |
| 5/20 SMA | 5-day SMA crosses above the 20-day SMA | 5-day SMA crosses below the 20-day SMA | Mid range |
| 9/20 SMA | 9-day SMA crosses above the 20-day SMA | 9-day SMA crosses below the 20-day SMA | Mid range |
| 20/50 SMA | 20-day SMA crosses above the 50-day SMA | 20-day SMA crosses below the 50-day SMA | Mid range |
| 50/100 SMA | 50-day SMA crosses above the 100-day SMA | 50-day SMA crosses below the 100-day SMA | Long Term |
| 100/200 SMA | 100-day SMA crosses above the 200-day SMA | 100-day SMA crosses below the 200-day SMA | Long Term |
| ADL | ADL is rising | ADL is falling | Good enough for now; book goes off IBD A-E rating |
| MFI | MFI crosses below 20 | MFI crosses above 80 | Good; need to test in comparison with above "Money Flow" manual calc |
| ADX | ADX above 20 and +DI crosses above -DI | ADX above 20 and +DI crosses below -DI | Could use ADX of 25 as well |
| RSI | RSI crosses above 65 | RSI crosses below 40 | Should plot this to track when MACD crosses 0 |
| MACD | MACD crosses above the Signal line | MACD crosses below the Signal line | Should plot this to track when MACD crosses 0 |
| BB C/D Trigger | Trend changes from down to up | Trend changes from up to down | Trend may be 4+ days increasing or decreasing |
| CMF | CMF crosses above zero | CMF crosses below zero | Could potentially tweak |
| CCI | CCI crosses above -100 | CCI crosses below 100 | Should be good; may need to tweak |

## ARMS Index (TRIN) Calculation

The ARMS Index (TRIN) needs to be calculated using real market data from MarketWatch. The correct formula is:

```
ARMS Index = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)
```

Data source: https://www.marketwatch.com/market-data/us?mod=market-data-center

This page contains the necessary market breadth data including:
- Number of advancing issues
- Number of declining issues
- Volume of advancing issues
- Volume of declining issues

## Bulls vs Bears Ratio from AAII Sentiment Survey

The Bulls vs Bears Ratio should use data from the AAII Investor Sentiment Survey:

Data source: https://www.aaii.com/sentimentsurvey/sent_results

This survey provides:
- Bullish sentiment percentage
- Bearish sentiment percentage
- Neutral sentiment percentage

The ratio can be calculated as: Bullish percentage / Bearish percentage

## Technical Indicators Implementation

Many of these indicators can be implemented using the TA-Lib library, which provides functions for technical analysis. However, there are installation issues with TA-Lib. Alternative approaches:

1. Implement the indicators manually using mathematical formulas
2. Use pandas_ta or ta libraries which are pure Python implementations
3. Use a pre-compiled version of TA-Lib if available

### Key Indicators to Implement:

1. **Parabolic SAR**: Trend following indicator that provides potential reversal points
2. **Rate of Change (ROC)**: Momentum oscillator that measures percentage change
3. **Know Sure Thing (KST)**: Momentum oscillator developed by Martin Pring
4. **True Strength Index (TSI)**: Momentum oscillator showing both trend direction and overbought/oversold conditions
5. **Stochastic Oscillator (%K and %D)**: Momentum indicator comparing closing price to price range
6. **Relative Strength Index (RSI)**: Momentum oscillator measuring speed and change of price movements
7. **Commodity Channel Index (CCI)**: Oscillator to identify cyclical trends
8. **Money Flow Index (MFI)**: Volume-weighted RSI that measures money flow

## Buy/Sell Signals Implementation

The dashboard needs to display 'Strong Buy' and 'Strong Sell' signals on the main chart based on the Perfect Storm strategy. These signals should be determined by analyzing multiple indicators:

- A 'Strong Buy' signal occurs when multiple buy indicators align
- A 'Strong Sell' signal occurs when multiple sell indicators align

The signals should be plotted as markers on the price chart, with no duplicate signals in a row.
