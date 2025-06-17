import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

class MarketDataRetriever:
    def _load_btc_hourly_data(self, period='1y', interval='1h'):
        """
        Loads Gemini_BTCUSD_1h.csv from Crypto_HPs.
        Returns a DataFrame with datetime index and columns: open, high, low, close, volume.
        Filters to the requested period and interval (only '1h' supported).
        """
        import os
        file = os.path.join(os.path.dirname(__file__), 'Crypto_HPs', 'Gemini_BTCUSD_1h.csv')
        if not os.path.exists(file):
            print("No BTC hourly CSV file found in Crypto_HPs.")
            return pd.DataFrame()
        df = pd.read_csv(file, skiprows=1)  # skip first comment row
        # Standardize columns
        df = df.rename(columns={
            'date': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'Volume BTC': 'volume_btc',
            'Volume USD': 'volume_usd',
        })
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.set_index('datetime')
        # Use USD volume as 'volume' for dashboard compatibility
        df = df[['open', 'high', 'low', 'close', 'volume_usd']].copy()
        df = df.rename(columns={'volume_usd': 'volume'})
        # Convert all columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Filter by period
        end_date = df.index.max()
        if period == '1d':
            start_date = end_date - pd.Timedelta(days=1)
        elif period == '5d':
            start_date = end_date - pd.Timedelta(days=5)
        elif period == '1mo':
            start_date = end_date - pd.Timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - pd.Timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - pd.Timedelta(days=180)
        elif period == '1y':
            start_date = end_date - pd.Timedelta(days=365)
        elif period == '2y':
            start_date = end_date - pd.Timedelta(days=730)
        else:
            start_date = end_date - pd.Timedelta(days=30)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
    def _load_btc_minute_data(self, period='1y', interval='1m'):
        """
        Loads and concatenates all Gemini_BTCUSD_*_minute.csv files from Crypto_HPs.
        Returns a DataFrame with datetime index and columns: open, high, low, close, volume.
        Filters to the requested period and interval (only '1m' supported).
        """
        import glob
        folder = os.path.join(os.path.dirname(__file__), 'Crypto_HPs')
        files = sorted(glob.glob(os.path.join(folder, 'Gemini_BTCUSD_*_minute.csv')))
        if not files:
            print("No BTC minute CSV files found in Crypto_HPs.")
            return pd.DataFrame()
        dfs = []
        for f in files:
            df = pd.read_csv(f, skiprows=1)  # skip first comment row
            # Standardize columns
            df = df.rename(columns={
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'Volume BTC': 'volume_btc',
                'Volume USD': 'volume_usd',
            })
            # Parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.set_index('datetime')
        # Use USD volume as 'volume' for dashboard compatibility
        df = df[['open', 'high', 'low', 'close', 'volume_usd']].copy()
        df = df.rename(columns={'volume_usd': 'volume'})
        # Convert all columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Filter by period
        end_date = df.index.max()
        if period == '1d':
            start_date = end_date - pd.Timedelta(days=1)
        elif period == '5d':
            start_date = end_date - pd.Timedelta(days=5)
        elif period == '1mo':
            start_date = end_date - pd.Timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - pd.Timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - pd.Timedelta(days=180)
        elif period == '1y':
            start_date = end_date - pd.Timedelta(days=365)
        elif period == '2y':
            start_date = end_date - pd.Timedelta(days=730)
        else:
            start_date = end_date - pd.Timedelta(days=30)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
    """
    Class to retrieve market data for the Perfect Storm Dashboard.
    
    This unified class retrieves:
      - Historical stock prices using yfinance_cache for intervals >= 1 day.
      - Intraday stock prices using AlphaVantage for intervals < 1 day.
      - Market breadth data from a manually maintained CSV file.
      - Investor sentiment data from AAII, with a fallback to a local file.
      - (Other methods such as options and institutional flow remain as placeholders.)
    """
    
    def __init__(self, api_key=None):
        # Use the provided API key (for intraday data) or default key.
        self.api_key = api_key or "25WNVRI1YIXCDIH1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def clean_dataframe(self, df, numeric_only_for_ml=False):
        """
        Centralized method to clean dataframe and handle NaN values
        
        Parameters:
        - df: DataFrame to clean
        
        Returns:
        - Cleaned DataFrame with no NaN values
        """
        if df is None or df.empty:
            return pd.DataFrame()
        # Forward fill then backward fill (preserves time series patterns)
        df = df.ffill().bfill()
        # If cleaning for ML, only fill numeric columns and drop non-numeric columns from features
        if numeric_only_for_ml:
            # Only keep numeric columns
            df = df.select_dtypes(include=[float, int, 'number'])
            # Fill any remaining NaNs with 0
            df = df.fillna(0)
        else:
            # Check if any NaN values remain
            remaining_nans = df.isna().sum().sum()
            if remaining_nans > 0:
                nan_columns = df.columns[df.isna().any()].tolist()
                print(f"Columns still containing NaN values after ffill/bfill: {nan_columns}")
                for col in nan_columns:
                    if df[col].dtype.kind in 'ifc':
                        fill_value = df[col].mean() if not df[col].isna().all() else 0
                        df[col] = df[col].fillna(fill_value)
                    else:
                        if not df[col].empty and not df[col].isna().all():
                            fill_value = df[col].mode()[0]
                        else:
                            fill_value = "unknown"
                        df[col] = df[col].fillna(fill_value)
            assert df.isna().sum().sum() == 0, "NaN values still remain after cleaning"
        return df

    def get_stock_history(self, symbol, interval='1d', period='1y'):
        # --- BTC MINUTE DATA OVERRIDE ---
        symbol_l = symbol.lower()
        btc_names = ['btc', 'btc/usd', 'btc-usd', 'btcusd']
        if symbol_l in btc_names:
            if interval in ['1m', '1min', 'minute']:
                print(f"Loading BTC minute data from local CSVs for {symbol}...")
                df = self._load_btc_minute_data(period=period, interval=interval)
                if df is not None and not df.empty:
                    return self.clean_dataframe(df)
                else:
                    print("No BTC minute data available.")
                    return pd.DataFrame()
            elif interval in ['1h', '60m', 'hour']:
                print(f"Loading BTC hourly data from local CSV for {symbol}...")
                df = self._load_btc_hourly_data(period=period, interval=interval)
                if df is not None and not df.empty:
                    return self.clean_dataframe(df)
                else:
                    print("No BTC hourly data available.")
                    return pd.DataFrame()
        if interval.endswith('m') or interval.endswith('h'):

            # Create cache filename based on parameters
            cache_filename = os.path.join('data_cache', f"{symbol}_{interval}_{period}.csv")
            
            # Check if we already have cached data
            existing_data = None
            latest_cached_date = None
            if os.path.exists(cache_filename):
                try:
                    existing_data = pd.read_csv(cache_filename, index_col=0, parse_dates=True)
                    if not existing_data.empty:
                        # Make sure index is datetime
                        existing_data.index = pd.to_datetime(existing_data.index)
                        latest_cached_date = existing_data.index.max()
                        print(f"Found cached data for {symbol} at {interval} for {period} up to {latest_cached_date}")
                except Exception as e:
                    print(f"Error reading cached data: {e}")
                    existing_data = None

            # For intraday data, use AlphaVantage API
            function = "TIME_SERIES_INTRADAY"
            
            # Convert interval to AlphaVantage format
            interval_param = interval
            if interval.endswith('m'):
                interval_param = interval.replace('m', 'min')
            elif interval.endswith('h'):
                # Convert hours to minutes (AlphaVantage only accepts min format)
                hours = int(interval.replace('h', ''))
                interval_param = f"{hours * 60}min"
                # If it's not a supported interval, round down to nearest supported interval
                supported_intervals = [1, 5, 15, 30, 60]
                minutes = hours * 60
                if minutes not in supported_intervals:
                    # Find closest supported interval that's smaller
                    closest = max([i for i in supported_intervals if i <= minutes])
                    interval_param = f"{closest}min"
                    print(f"Warning: {interval} is not a supported AlphaVantage interval. Using {interval_param} instead.")
            
            # Determine the date range based on period
            end_date = datetime.now()
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '5d':
                start_date = end_date - timedelta(days=5)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            else:  # Default to 1 month if period is not recognized
                start_date = end_date - timedelta(days=30)
            
            # Adjust start_date if we have cached data
            if latest_cached_date is not None:
                # Set start_date to the day after the latest cached date
                # Add a small buffer (1 hour) to avoid duplicate entries
                new_start_date = latest_cached_date + timedelta(hours=1)
                
                # Only update start_date if the new one is more recent
                if new_start_date > start_date:
                    start_date = new_start_date
                    print(f"Adjusting start date to {start_date} based on cached data")
                
                # If the start_date is very close to end_date, we probably don't need to fetch new data
                if (end_date - start_date).total_seconds() < 3600:  # Less than an hour difference
                    print("Cached data is recent enough, no need to fetch new data")
                    # Fill in missing dates for existing data before returning
                    return existing_data
            
            # Initialize empty dataframe to store results
            all_data = pd.DataFrame()
            
            # AlphaVantage intraday API doesn't support month parameter
            # We'll fetch all available data and filter it later
            print(f"Fetching intraday data for {symbol} at {interval_param}")
            
            url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval_param}&apikey={self.api_key}&outputsize=full"
            r = requests.get(url)
            data = r.json()
            
            if "Error Message" in data:
                print(f"Error from AlphaVantage: {data['Error Message']}")
                # If we couldn't fetch new data but have existing data, return that
                if existing_data is not None and not existing_data.empty:
                    print(f"Using existing cached data for {symbol}")
                    return existing_data
                print(f"No data could be retrieved for {symbol}")
                return pd.DataFrame()
                
            ts_key = next((key for key in data if key.startswith("Time Series")), None)
            if ts_key is None or not data.get(ts_key):
                print(f"No data found for {symbol} or unexpected API response format")
                # If we couldn't fetch new data but have existing data, return that
                if existing_data is not None and not existing_data.empty:
                    print(f"Using existing cached data for {symbol}")
                    return existing_data
                print(f"No data could be retrieved for {symbol}")
                return pd.DataFrame()
            
            # Convert the data to a dataframe
            df = pd.DataFrame.from_dict(data[ts_key], orient='index').astype(float)
            df.index = pd.to_datetime(df.index)
            
            # Filter to the requested date range
            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
            
            # Combine with existing data
            all_data = df
            
            # If we fetched new data, process it
            if not all_data.empty:
                # Sort the combined data by date
                all_data = all_data.sort_index()
                
                # Rename columns to remove the prefixes from AlphaVantage
                all_data = all_data.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                })
                
                # Filter to the requested period
                if start_date is not None:
                    all_data = all_data[all_data.index >= pd.Timestamp(start_date)]
                
                print(f"Retrieved {len(all_data)} new intraday data points for {symbol}")
                print("New intraday data preview:\n", all_data.head())
                
                # Combine with existing data if available
                if existing_data is not None and not existing_data.empty:
                    # Ensure column names match
                    if set(existing_data.columns) != set(all_data.columns):
                        print("Warning: Column mismatch between existing and new data")
                        # Align columns
                        common_cols = set(existing_data.columns).intersection(set(all_data.columns))
                        if common_cols:
                            existing_data = existing_data[list(common_cols)]
                            all_data = all_data[list(common_cols)]
                    
                    # Combine data and remove duplicates
                    combined_data = pd.concat([existing_data, all_data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data = combined_data.sort_index()
                    
                    # Filter to requested period
                    if period != 'max':
                        # Calculate start date based on period
                        period_start_date = end_date - timedelta(days=365)  # Default to 1y
                        if period == '1d':
                            period_start_date = end_date - timedelta(days=1)
                        elif period == '5d':
                            period_start_date = end_date - timedelta(days=5)
                        elif period == '1mo':
                            period_start_date = end_date - timedelta(days=30)
                        elif period == '3mo':
                            period_start_date = end_date - timedelta(days=90)
                        elif period == '6mo':
                            period_start_date = end_date - timedelta(days=180)
                        elif period == '1y':
                            period_start_date = end_date - timedelta(days=365)
                        elif period == '2y':
                            period_start_date = end_date - timedelta(days=730)
                        
                        combined_data = combined_data[combined_data.index >= period_start_date]
                    
                    print(f"Combined data has {len(combined_data)} rows")
                    
                    # Save to cache
                    os.makedirs('data_cache', exist_ok=True)
                    combined_data.to_csv(cache_filename)
                    print(f"Saved combined data to {cache_filename}")
                    
                    # Fill in missing dates for the combined data before returning
                    return combined_data
                
                # If no existing data or empty, just use the new data
                if all_data.empty:
                    print(f"No new intraday data could be retrieved for {symbol}")
                    if existing_data is not None and not existing_data.empty:
                        return existing_data
                    return pd.DataFrame()
                
                # Save to cache
                os.makedirs('data_cache', exist_ok=True)
                all_data.to_csv(cache_filename)
                print(f"Saved new data to {cache_filename}")
                
                # Fill in missing dates for the new data before returning
                return all_data
            
            # If we couldn't fetch new data but have existing data, return that
            if existing_data is not None and not existing_data.empty:
                print(f"Using existing cached data for {symbol}")
                return existing_data
            
            print(f"No data could be retrieved for {symbol}")
            return pd.DataFrame()
        else:
            # Otherwise, fetch new data
            import yfinance_cache as yfc
            dat = yfc.Ticker(symbol)
            df = dat.history(period=period, adjust_splits=True, adjust_divs=True)
            print(df.head())
            # Check if the index is already a datetime and handle accordingly
            if isinstance(df.index, pd.DatetimeIndex):
                # For futures and other securities where the date is already the index without a 'Date' column
                # Just make a copy of the index to work with
                date_index = df.index.copy()
                # Remove timezone info
                date_index = date_index.tz_localize(None)
                
                # Create a new dataframe with the cleaned index
                df = df.copy()
                df.index = date_index
            else:
                # For securities where 'Date' might be a column
                # Reset index to ensure 'Date' is available as a column
                if 'Date' in df.columns:
                    df = df.drop(columns=['Date'])
                df.reset_index(inplace=True)
                # Set date as index
                df = df.set_index('Date')
                df.index = df.index.tz_localize(None)  # Remove timezone
            
            # Rename columns to lowercase
            df = df.rename(columns=str.lower)
            print(df.head())
            
            # Fill in missing dates for daily data
            return df

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
        Get investor sentiment data from AAII Investor Sentiment Survey online (HTML scrape).
        If online retrieval fails, falls back to local CSV file.
        Returns:
          - Dictionary with keys: 'bullish', 'bearish', 'neutral' (float values)
          - None if retrieval fails (including fallback).
        """
        import requests
        from bs4 import BeautifulSoup
        try:
            url = "https://www.aaii.com/sentimentsurvey"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Find the table with sentiment data
                table = soup.find("table")
                if table:
                    rows = table.find_all("tr")
                    sentiment = {}
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) == 2:
                            label = cells[0].get_text(strip=True).lower()
                            value = cells[1].get_text(strip=True).replace('%','').strip()
                            try:
                                value = float(value)
                            except Exception:
                                value = 0.0
                            if "bullish" in label:
                                sentiment["bullish"] = value
                            elif "bearish" in label:
                                sentiment["bearish"] = value
                            elif "neutral" in label:
                                sentiment["neutral"] = value
                    # Ensure all keys are present
                    for k in ["bullish", "bearish", "neutral"]:
                        if k not in sentiment:
                            sentiment[k] = 0.0
                    return sentiment
        except Exception:
            pass
        # Fallback: local CSV
        try:
            df = pd.read_csv(
                r"c:\\Users\\klamo\\Perfect_Storm_Dashboard_Enhanced\\data\\aaii_historical_sentiment_modified.csv",
                parse_dates=["Reported Date"]
            )
            # Forward/backward fill missing values
            for col in df.columns:
                df[col] = df[col].ffill().bfill()
            # Use the last (most recent) row
            last_row = df.iloc[-1]
            def parse_percent(val):
                if isinstance(val, str):
                    val = val.replace('%','').strip()
                try:
                    return float(val)
                except Exception:
                    return 0.0
            bullish = parse_percent(last_row['Bullish'])
            bearish = parse_percent(last_row['Bearish'])
            neutral = parse_percent(last_row['Neutral'])
            return {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral
            }
        except Exception:
            return {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}

    def get_social_sentiment_data(self, symbol="AAPL"):
        """
        Get social/news sentiment data using AlphaVantage NEWS_SENTIMENT API (per-ticker aggregation).
        Falls back to local Excel file if API fails.
        Args:
            symbol (str): Stock symbol (default 'AAPL')
        Returns:
            dict: {'bullish': float, 'bearish': float, 'neutral': float} or None
        """
        import requests
        api_key = getattr(self, 'api_key', None) or os.getenv("ALPHAVANTAGE_API_KEY", "25WNVRI1YIXCDIH1")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"AlphaVantage NEWS_SENTIMENT API request failed: {response.status_code}")
                raise Exception("API request failed")
            data = response.json()
            if not isinstance(data, dict) or 'feed' not in data or not data['feed']:
                print("AlphaVantage NEWS_SENTIMENT API returned no feed data.")
                raise Exception("No feed data")
            # Aggregate per-ticker sentiment scores from the feed
            bullish_scores = []
            bearish_scores = []
            neutral_scores = []
            symbol_upper = symbol.upper()
            for item in data['feed']:
                ticker_sentiments = item.get('ticker_sentiment', [])
                for ts in ticker_sentiments:
                    if ts.get('ticker', '').upper() == symbol_upper:
                        label = ts.get('ticker_sentiment_label', '').lower()
                        score = ts.get('ticker_sentiment_score', 0)
                        try:
                            score = float(score)
                        except Exception:
                            score = 0.0
                        if label == 'bullish':
                            bullish_scores.append(score)
                        elif label == 'bearish':
                            bearish_scores.append(abs(score))
                        elif label == 'neutral':
                            neutral_scores.append(abs(score))
                        elif label == 'somewhat-bullish':
                            bullish_scores.append(score)
                        elif label == 'somewhat-bearish':
                            bearish_scores.append(abs(score))
            # If no per-ticker sentiment found, fall back to overall_sentiment_label/score
            if not (bullish_scores or bearish_scores or neutral_scores):
                for item in data['feed']:
                    label = item.get('overall_sentiment_label', '').lower()
                    score = item.get('overall_sentiment_score', 0)
                    try:
                        score = float(score)
                    except Exception:
                        score = 0.0
                    if label == 'bullish' or label == 'somewhat-bullish':
                        bullish_scores.append(score)
                    elif label == 'bearish' or label == 'somewhat-bearish':
                        bearish_scores.append(abs(score))
                    elif label == 'neutral':
                        neutral_scores.append(abs(score))
            bullish = sum(bullish_scores) / len(bullish_scores) if bullish_scores else 0.0
            bearish = sum(bearish_scores) / len(bearish_scores) if bearish_scores else 0.0
            neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0.0
            # If all are zero, treat as no data
            if bullish == 0.0 and bearish == 0.0 and neutral == 0.0:
                print("AlphaVantage NEWS_SENTIMENT API returned only zero scores.")
                raise Exception("Zero scores")
            return {"bullish": bullish, "bearish": bearish, "neutral": neutral}
        except Exception as e:
            print(f"AlphaVantage NEWS_SENTIMENT API failed or returned no data: {e}")
            # Fallback: Local Excel file
            file_path = r'C:\Users\klamo\Downloads\historical_sentiment_20250220.xls'
            try:
                import pandas as pd
                df = pd.read_excel(file_path)
            except FileNotFoundError:
                print("Local sentiment file not found.")
                return None
            required_columns = ['Reported Date', 'Bullish', 'Neutral', 'Bearish']
            if not all(col in df.columns for col in required_columns):
                print("Required columns missing in the sentiment file.")
                return None
            last_row = df.iloc[-1]
            try:
                bullish = float(last_row['Bullish'])
                neutral = float(last_row['Neutral'])
                bearish = float(last_row['Bearish'])
            except ValueError:
                print("Invalid data format in sentiment file.")
                return None
            return {"bullish": bullish, "bearish": bearish, "neutral": neutral}
    
    def _parse_number(self, text):
        """Helper: Parse a numeric value from text."""
        try:
            return float(text.replace(',', ''))
        except:
            return None
    
    def _parse_percentage(self, text):
        """Helper: Parse a percentage value from text."""
        try:
            return float(text.replace('%', ''))
        except:
            return None

    @staticmethod
    def get_options_data(symbol, api_key):
        """
        Retrieve options data from AlphaVantage HISTORICAL_OPTIONS endpoint.
        Returns a dict with 'put_call_ratio' and 'implied_volatility'.
        Robust to API failures and unexpected data structures.
        """
        params = {'function': 'HISTORICAL_OPTIONS', 'symbol': symbol, 'apikey': api_key}
        try:
            response = requests.get('https://www.alphavantage.co/query', params=params)
            if response.status_code != 200:
                print(f"API request failed with status code: {response.status_code}")
                return {'put_call_ratio': 1.0, 'implied_volatility': 0.5}
            data = response.json()
            if not isinstance(data, dict) or 'Error Message' in data:
                print(f"API returned error: {data.get('Error Message', 'Unknown error')}")
                return {'put_call_ratio': 1.0, 'implied_volatility': 0.5}

            # AlphaVantage HISTORICAL_OPTIONS returns 'option_chain' with 'calls' and 'puts' lists
            option_chain = data.get('option_chain', {})
            calls = option_chain.get('calls', [])
            puts = option_chain.get('puts', [])

            # Put/Call ratio
            num_calls = len(calls)
            num_puts = len(puts)
            put_call_ratio = (num_puts / num_calls) if num_calls > 0 else float('inf')

            # Implied volatility: average IV from all contracts (calls and puts)
            ivs = []
            for contract in calls + puts:
                iv = contract.get('implied_volatility')
                if iv is not None:
                    try:
                        ivs.append(float(iv))
                    except Exception:
                        continue
            implied_volatility = sum(ivs) / len(ivs) if ivs else 0.5

            return {
                'put_call_ratio': put_call_ratio,
                'implied_volatility': implied_volatility
            }
        except Exception as e:
            print(f"Error in get_options_data: {str(e)}")
            return {'put_call_ratio': 1.0, 'implied_volatility': 0.5}
    
    @staticmethod
    def get_institutional_flow(symbol, api_key):
        try:
            return {"net_flow": 1000000}
        except Exception as e:
            print(f"Error in get_institutional_flow: {str(e)}")
            return {"error": str(e)}
