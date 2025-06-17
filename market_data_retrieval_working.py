import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
# Selenium imports (kept for reference, but prefer API/CSV if possible)
# ... (original selenium imports) ...
import yfinance_cache as yfc # Use yfinance_cache for robustness


class MarketDataRetriever2:
    """
    Class to retrieve market data for the Perfect Storm Dashboard.
    Prioritizes robust methods like yfinance_cache and handles API keys.
    Includes placeholders for future data sources.
    """

    def __init__(self, api_key=None):
        self.alphavantage_api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_DEFAULT_KEY") # Use env var or default
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Ensure data cache directory exists
        os.makedirs('data_cache', exist_ok=True)

    def clean_dataframe(self, df):
        # (Keep the clean_dataframe method as is)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.fillna(method='ffill').fillna(method='bfill')
        if df.isna().sum().sum() > 0:
            print("Warning: NaNs remaining after ffill/bfill, filling remaining with 0")
            df.fillna(0, inplace=True) # Fallback fill with 0
        return df

    def get_stock_history(self, symbol, interval='1d', period='1y'):
        """
        Fetches historical stock data using yfinance_cache for daily/weekly
        and AlphaVantage for intraday.
        """
        print(f"Fetching {symbol} data: interval={interval}, period={period}")
        try:
            # --- Intraday Data ---
            if interval.endswith('m') or interval.endswith('h'):
                print(f"Fetching intraday data for {symbol} via AlphaVantage...")
                # Convert interval format for AV
                if interval.endswith('m'):
                    av_interval = f"{interval.replace('m','')}min"
                    if av_interval not in ['1min', '5min', '15min', '30min', '60min']:
                        print(f"Warning: AlphaVantage interval {interval} not directly supported, may fail.")
                elif interval.endswith('h'):
                    minutes = int(interval.replace('h','')) * 60
                    if minutes == 60:
                        av_interval = '60min'
                    else:
                        print(f"Warning: Hourly interval {interval} might not be supported by AlphaVantage standard API. Requires 60min.")
                        av_interval = '60min' # Defaulting, might need premium access for others
                else:
                    print(f"Unsupported interval format for intraday: {interval}")
                    return pd.DataFrame()

                # Determine outputsize based on period (extended requires premium)
                outputsize = 'full' if period in ['3mo', '6mo', '1y', '2y', '5y'] else 'compact'

                cache_filename = os.path.join('data_cache', f"{symbol}_{interval}_{period}.csv")

                # --- Cache Check (simplified for example) ---
                fetch_new = True
                if os.path.exists(cache_filename):
                    # Add logic here to check cache age and update if necessary
                    print(f"Cache hit for {symbol}, checking age...")
                    # fetch_new = check_if_cache_is_stale(cache_filename) # Implement this check
                    try:
                        cached_data = pd.read_csv(cache_filename, index_col=0, parse_dates=True)
                        return self.clean_dataframe(cached_data) # Return cached if fresh enough for now
                    except Exception:
                        print("Error reading cache, fetching new data.")

                # Fetch from AlphaVantage API
                url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
                       f"&symbol={symbol}&interval={av_interval}&apikey={self.alphavantage_api_key}"
                       f"&outputsize={outputsize}")
                r = requests.get(url)
                data = r.json()

                if "Error Message" in data or "Note" in data: # Free API limit often hits here
                    print(f"AlphaVantage API Error/Note: {data.get('Error Message', data.get('Note', 'Unknown issue'))}")
                    # Attempt to load from cache as fallback
                    if os.path.exists(cache_filename):
                        print("Falling back to cached data due to API issue.")
                        try:
                            return self.clean_dataframe(pd.read_csv(cache_filename, index_col=0, parse_dates=True))
                        except Exception as e:
                            print(f"Error reading cache during fallback: {e}")
                            return pd.DataFrame()
                    else:
                        return pd.DataFrame() # No data

                ts_key = next((key for key in data if key.startswith("Time Series")), None)
                if not ts_key or not data[ts_key]:
                    print(f"No time series data found for {symbol} from AlphaVantage.")
                    return pd.DataFrame()

                df = pd.DataFrame.from_dict(data[ts_key], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df = df.rename(columns=lambda x: x.split('. ')[1]) # Clean column names
                df = df.sort_index()

                # Filter by period (needs conversion)
                # ... (add period filtering logic based on start/end dates) ...
                end_date = datetime.now()
                # Define start_date based on period string
                period_map = {
                    '1d': timedelta(days=1), '5d': timedelta(days=5),
                    '1mo': timedelta(days=30), '3mo': timedelta(days=90),
                    '6mo': timedelta(days=180), '1y': timedelta(days=365),
                    '2y': timedelta(days=730), '5y': timedelta(days=365*5)
                }
                if period in period_map:
                    start_date = end_date - period_map[period]
                    df = df[df.index >= start_date]

                df.to_csv(cache_filename) # Cache the fetched data
                return self.clean_dataframe(df)
            # --- Daily/Weekly Data (using yfinance_cache) ---
            else:
                print(f"Fetching daily/weekly data for {symbol} via yfinance_cache...")
                ticker = yfc.Ticker(symbol)
                df = ticker.history(period=period, interval=interval, adjust_splits=True, adjust_divs=True)
                if df.empty:
                    print(f"yfinance_cache returned no data for {symbol}.")
                    return pd.DataFrame()
                df = df.rename(columns=str.lower) # Lowercase columns
                df.index = df.index.tz_localize(None) # Remove timezone for consistency
                # Convert Date column if it's not the index
                if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

                # Add 'adj close' if missing, preferring it if present
                if 'adj close' not in df.columns and 'close' in df.columns:
                    df['adj close'] = df['close']

                # Basic check for required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Missing some OHLCV columns in yfinance data for {symbol}.")
                return self.clean_dataframe(df) # Clean final output
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching data for {symbol}: {e}")
            return pd.DataFrame() # Return empty on network error
        except ValueError as e:
            print(f"Value error processing data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred fetching data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_market_breadth_data(self, file_path="market_breadth_manual.csv"):
        # (This function remains the same, relies on manual CSV)
        if not os.path.exists(file_path):
            # print(f"Market breadth file not found: {file_path}.") # Less verbose
            return None
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            if df.empty: return None
            df = df.sort_values(by='date', ascending=False)
            latest_data = df.iloc[0].to_dict()
            # Convert values to numeric, handle potential errors
            for key in ['advancing_issues', 'declining_issues', 'advancing_volume', 'declining_volume']:
                if key in latest_data:
                    try:
                        latest_data[key] = float(latest_data[key])
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid number format for {key} in market breadth file.")
                        return None # Or handle as NaN/0? Requires decision.
                else:
                    print(f"Warning: Missing key {key} in market breadth file.")
                    return None
            return latest_data
        except Exception as e:
            print(f"Error reading market breadth file '{file_path}': {e}")
            return None

    def get_sentiment_data(self):
        """
        Placeholder for improved sentiment retrieval. Currently uses AAII static file.
        FUTURE: Implement scraping or API integration for real-time sentiment.
        """
        # --- Current Static AAII Implementation ---
        file_path = r"c:\\Users\\klamo\\Perfect_Storm_Dashboard_Enhanced\\data\\aaii_historical_sentiment_modified.csv" # Ensure path is correct
        try:
            df = pd.read_csv(file_path, parse_dates=["Reported Date"])
            # Ensure data processing handles potential issues
            for col in ['Bullish', 'Bearish', 'Neutral']:
                # Attempt to convert percentage strings, handle errors
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip('%').astype(float) / 100.0
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Use ffill/bfill to handle missing values within the historical data
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

            if df[['Bullish', 'Bearish', 'Neutral']].isna().any().any():
                print("Warning: Could not fully process AAII sentiment data, NaNs remain.")
                # Return only latest non-NaN value or default
                latest_valid_index = df[['Bullish', 'Bearish', 'Neutral']].dropna().index.max()
                if latest_valid_index is None: return {'bullish': 0.3, 'bearish': 0.3, 'neutral': 0.4} # Default neutral
                latest = df.loc[latest_valid_index]
                return {"bullish": latest["Bullish"], "bearish": latest["Bearish"], "neutral": latest["Neutral"]}

            latest = df.iloc[-1]
            return {"bullish": latest["Bullish"], "bearish": latest["Bearish"], "neutral": latest["Neutral"]}

        except FileNotFoundError:
            print(f"AAII historical data file not found at {file_path}. Returning default neutral sentiment.")
            return {"bullish": 0.3, "bearish": 0.3, "neutral": 0.4} # Example default
        except Exception as e:
            print(f"Error reading or processing AAII sentiment file: {e}")
            return {"bullish": 0.3, "bearish": 0.3, "neutral": 0.4} # Example default

    # --- Placeholder for Future: News/Social Sentiment ---
    def get_news_social_sentiment(self, symbol):
        """
        FUTURE IMPLEMENTATION: Fetch and process news/social media sentiment.
        This would involve using NLP models or third-party sentiment APIs.
        """
        print(f"Placeholder: Fetching news/social sentiment for {symbol}")
        # Example: Connect to a sentiment API, or run local NLP model
        # Requires significant external integration/setup
        # Return score between -1 (very bearish) and +1 (very bullish)
        return 0.0 # Neutral placeholder

    # --- Placeholder for Future: Options Data ---
    def get_options_data(self, symbol):
        """
        FUTURE IMPLEMENTATION: Fetch options data (IV, Put/Call ratio).
        Requires access to an options data provider API.
        """
        print(f"Placeholder: Fetching options data for {symbol}")
        # Example: Connect to CBOE data shop, Polygon, Tradier, IBKR, etc.
        # Requires subscription and API handling.
        return {
            'implied_volatility': np.nan, # Placeholder
            'iv_percentile': np.nan,      # Placeholder
            'put_call_ratio': np.nan     # Placeholder
        }

    # --- Placeholder for Future: Economic Data ---
    def get_economic_indicators(self, indicator_codes):
        """
        FUTURE IMPLEMENTATION: Fetch key economic indicators (e.g., from FRED).
        """
        print(f"Placeholder: Fetching economic indicators {indicator_codes}")
        # Example: Use pandas_datareader to get data from FRED
        # Needs mapping of relevant codes (e.g., 'UNRATE', 'CPIAUCSL', 'PAYEMS')
        # Return a DataFrame or dictionary of latest values/series
        return pd.DataFrame() # Placeholder

    # --- Helper methods remain unchanged ---
    def _parse_number(self, text):
        """Helper: Parse a numeric value from text."""
        try:
            return float(str(text).replace(',', ''))
        except:
            return None

    def _parse_percentage(self, text):
        """Helper: Parse a percentage value from text."""
        try:
            return float(str(text).replace('%', '')) / 100.0 # Return as decimal
        except:
            return None