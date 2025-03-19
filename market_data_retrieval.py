import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time

class MarketDataRetriever:
    """
    Class to retrieve market data for the Perfect Storm Dashboard.
    
    This unified class retrieves:
      - Historical stock prices using yfinance_cache for intervals >= 1 day.
      - Intraday stock prices using AlphaVantage for intervals < 1 day.
      - Market breadth data from MarketWatch.
      - Investor sentiment data from AAII, with a fallback to a local file.
      - (Other methods such as options and institutional flow remain as placeholders.)
    """
    
    def __init__(self, api_key=None):
        # Use the provided API key (for intraday data) or default key.
        self.api_key = api_key or "25WNVRI1YIXCDIH1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_stock_history(self, symbol, interval='1d', period='1y'):
        if interval.endswith('m'):
            function = "TIME_SERIES_INTRADAY"
            interval_param = interval.replace('m', 'min')
            url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval_param}&apikey={self.api_key}&outputsize=compact"
            r = requests.get(url)
            data = r.json()
            print("AlphaVantage Response:", data)  # Debugging API response
            ts_key = next((key for key in data if key.startswith("Time Series")), None)
            if ts_key is None:
                print("No Time Series Key Found in API Response")
                return pd.DataFrame()
            df = pd.DataFrame.from_dict(data[ts_key], orient='index').astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            print("Intraday Data Preview:\n", df.head())  # Debugging output
            return df
        else:
            import yfinance_cache as yf
            df = yf.download(symbol, period=period, interval=interval)
            print("Raw yfinance_cache DataFrame Columns:", df.columns)  # Debugging output
            if not df.empty:
                df = df.rename(columns=str.lower)
                df.index = df.index.tz_localize(None)  # Remove timezone
            print("Processed DataFrame Columns:", df.columns)  # Debugging output
            return df

    def get_market_breadth_data(self):
        """
        Get market breadth data from MarketWatch.
        
        Returns:
          - Dictionary with keys: 'advancing_issues', 'declining_issues', 'advancing_volume', 'declining_volume'
          - None if retrieval fails.
        """
        try:
            url = "https://www.marketwatch.com/market-data/us?mod=market-data-center"
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to retrieve market breadth data. Status code: {response.status_code}")
                return None
            print(f"Response content: {response.text[:500]}")  # Debug: first 500 characters of response
            soup = BeautifulSoup(response.text, 'html.parser')
            # Implementation: search for table cells containing 'Advancing' and 'Declining'
            advancing_issues = declining_issues = advancing_volume = declining_volume = None
            tables = soup.find_all('table')
            for table in tables:
                headers = [th.text.strip() for th in table.find_all('th')]
                print(f"Found table headers: {headers}")  # Debug: table headers found
                if 'Advancing' in headers and 'Declining' in headers:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            cell_text = cells[0].text.strip()
                            print(f"Processing row: {cell_text}")  # Debug: current row being processed
                            if 'Issues' in cell_text:
                                advancing_issues = self._parse_number(cells[1].text)
                                declining_issues = self._parse_number(cells[2].text)
                                print(f"Parsed issues: {advancing_issues}, {declining_issues}")  # Debug: parsed issues
                            elif 'Volume' in cell_text:
                                advancing_volume = self._parse_number(cells[1].text)
                                declining_volume = self._parse_number(cells[2].text)
                                print(f"Parsed volume: {advancing_volume}, {declining_volume}")  # Debug: parsed volume
            if advancing_issues is None or declining_issues is None:
                print("Failed to find advancing/declining issues in response.")
                return None
            return {
                'advancing_issues': advancing_issues,
                'declining_issues': declining_issues,
                'advancing_volume': advancing_volume,
                'declining_volume': declining_volume
            }
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return None
    
    def get_sentiment_data(self):
        """
        Get investor sentiment data from AAII Investor Sentiment Survey.
        
        Attempts to scrape from AAII website with retry logic. If all attempts fail, falls back to local file.
        
        Returns:
          - Dictionary with keys: 'bullish', 'bearish', 'neutral'
          - None if retrieval fails.
        """
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                url = "https://www.aaii.com/sentimentsurvey/sent_results"
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    raise Exception(f"Failed to retrieve sentiment data: {response.status_code}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                bullish = bearish = neutral = None
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            if 'Bullish' in cells[0].text:
                                bullish = self._parse_percentage(cells[1].text)
                            elif 'Bearish' in cells[0].text:
                                bearish = self._parse_percentage(cells[1].text)
                            elif 'Neutral' in cells[0].text:
                                neutral = self._parse_percentage(cells[1].text)
                
                if bullish is None or bearish is None or neutral is None:
                    print("Online sentiment scrape failed; falling back to local file.")
                    return self.get_social_sentiment_data()
                
                return {
                    'bullish': bullish,
                    'bearish': bearish,
                    'neutral': neutral
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Falling back to local file.")
                    return self.get_social_sentiment_data()
    
    def get_social_sentiment_data(self):
        """
        Fallback: Get sentiment data from a local Excel file.
        
        Returns:
          - Dictionary with keys: 'bullish', 'bearish', 'neutral'
          - None if retrieval fails.
        """
        file_path = r'C:\Users\klamo\Downloads\historical_sentiment_20250220.xls'
        try:
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

    # Existing placeholder methods for options and institutional flow can remain unchanged.
    @staticmethod
    def get_options_data(symbol, api_key):
        params = {'function': 'OPTION_CHAIN', 'symbol': symbol, 'apikey': api_key}
        try:
            response = requests.get('https://www.alphavantage.co/query', params=params)
            if response.status_code != 200:
                print(f"API request failed with status code: {response.status_code}")
                return None
            data = response.json()
            if not isinstance(data, dict) or 'Error Message' in data:
                print(f"API returned error: {data.get('Error Message', 'Unknown error')}")
                return None
                
            # Extract options data
            options = data.get('options', [])
            puts = [option for option in options if option.get('type') == 'put']
            calls = [option for option in options if option.get('type') == 'call']
            
            # Calculate put-call ratio
            put_call_ratio = len(puts) / len(calls) if calls else float('inf')
            
            # Extract implied volatility
            implied_volatility = data.get('implied_volatility', None)
            
            return {
                'put_call_ratio': put_call_ratio,
                'implied_volatility': implied_volatility
            }
        except Exception as e:
            print(f"Error in get_options_data: {str(e)}")
            return None
    
    @staticmethod
    def get_institutional_flow(symbol, api_key):
        try:
            # Placeholder implementation - to be replaced with actual API calls
            return {"net_flow": 1000000}
        except Exception as e:
            print(f"Error in get_institutional_flow: {str(e)}")
            return {"error": str(e)}
