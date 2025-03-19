# Data Sources for Perfect Storm Dashboard

Based on the indicators required for the "Perfect Storm" investment strategy, we need reliable sources of market data. After evaluating available options, the following data sources will be used:

## Primary Data Source: Yahoo Finance API

The Yahoo Finance API provides comprehensive market data that covers most of our needs:

### 1. YahooFinance/get_stock_chart
- Provides historical price data (open, high, low, close)
- Includes volume data
- Supports different time intervals (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
- Supports different time ranges (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- Can be used to calculate:
  - Moving averages (5, 9, 20, 50, 100, 200 day)
  - Bollinger Bands
  - Price-based signals

### 2. YahooFinance/get_stock_insights
- Provides technical indicators
- Includes short/intermediate/long-term outlooks
- Contains valuation details
- Offers technical events data

### 3. YahooFinance/get_stock_profile
- Provides company information
- Includes sector and industry classification
- Contains business summary

### 4. YahooFinance/get_stock_holders
- Provides insider trading information
- Includes transaction details

### 5. YahooFinance/get_stock_sec_filing
- Provides SEC filing history
- Includes filing dates and types

## Data Availability Assessment

The Yahoo Finance API provides sufficient data for calculating all required indicators:

1. **Bollinger Bands**: Can be calculated using price data from get_stock_chart
2. **C/D Signals**: Can be derived from price data in get_stock_chart
3. **Moving Averages**: Can be calculated using price data from get_stock_chart
4. **MACD**: Can be calculated using price data from get_stock_chart
5. **Volume Indicators**: 
   - Volume data available from get_stock_chart
   - Additional market breadth data may need to be calculated

## Implementation Approach

1. Use the YahooFinance/get_stock_chart API as the primary data source
2. Implement custom calculation functions for all technical indicators
3. Cache data where appropriate to minimize API calls
4. Provide options for different time periods and stock symbols
