import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

class MarketDataRetriever:
    """
    Class to retrieve market data for the Perfect Storm dashboard.
    
    This class provides methods to fetch stock data, market breadth data,
    sentiment data, and placeholders for social sentiment, options data,
    and institutional flow data.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the MarketDataRetriever.
        
        Parameters:
        - api_key: AlphaVantage API key (optional). If not provided, a default key is used.
        """
        self.api_key = api_key or "25WNVRI1YIXCDIH1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_stock_history(self, symbol, interval='daily', period='1y', api_key=None):
        """
        Retrieve stock history data from AlphaVantage.
        
        Parameters:
        - symbol: Stock symbol (e.g., 'AAPL')
        - interval: Time interval ('daily', 'weekly', 'monthly')
        - period: Historical period to retrieve (e.g., '1y', '5y')
        - api_key: AlphaVantage API key (optional). If not provided, the instance's API key is used.
        
        Returns:
        - pandas.DataFrame with stock history data (columns: 'open', 'high', 'low', 'close', 'volume')
        - None if retrieval fails
        """
        function_map = {
            'daily': 'TIME_SERIES_DAILY',
            'weekly': 'TIME_SERIES_WEEKLY',
            'monthly': 'TIME_SERIES_MONTHLY'
        }
        function = function_map.get(interval, 'TIME_SERIES_DAILY')
        api_key = api_key or self.api_key
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&outputsize=full"
        r = requests.get(url)
        data = r.json()
        ts_key = next((key for key in data if key.startswith("Time Series")), None)
        if ts_key is None:
            return None
        df = pd.DataFrame.from_dict(data[ts_key], orient='index').astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns=lambda x: x.split(". ")[-1])
        # Filter based on period
        if period.endswith('y'):
            years = int(period[:-1])
            start = df.index[-1] - pd.DateOffset(years=years)
            df = df[df.index >= start]
        return df

    def get_market_breadth_data(self):
        """
        Get market breadth data from MarketWatch.
        
        Returns:
        - Dictionary with market breadth data:
          - 'advancing_issues': Number of advancing issues
          - 'declining_issues': Number of declining issues
          - 'advancing_volume': Volume of advancing issues
          - 'declining_volume': Volume of declining issues
        - None if retrieval fails
        """
        try:
            url = "https://www.marketwatch.com/market-data/us?mod=market-data-center"
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve market breadth data: {response.status_code}")
            soup = BeautifulSoup(response.text, 'html.parser')
            advancing_issues = declining_issues = advancing_volume = declining_volume = None
            tables = soup.find_all('table')
            for table in tables:
                headers = [th.text.strip() for th in table.find_all('th')]
                if 'Advancing' in headers and 'Declining' in headers:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            if 'Issues' in cells[0].text:
                                advancing_issues = self._parse_number(cells[1].text)
                                declining_issues = self._parse_number(cells[2].text)
                            elif 'Volume' in cells[0].text:
                                advancing_volume = self._parse_number(cells[1].text)
                                declining_volume = self._parse_number(cells[2].text)
            if advancing_issues is None or declining_issues is None:
                return None
            return {
                'advancing_issues': advancing_issues,
                'declining_issues': declining_issues,
                'advancing_volume': advancing_volume,
                'declining_volume': declining_volume
            }
        except Exception as e:
            print(f"Error retrieving market breadth data: {e}")
            return None
    
    def get_sentiment_data(self):
            """
            Get sentiment data from AAII Investor Sentiment Survey, first online, then from local file if online fails.
            
            Returns:
            - Dictionary with sentiment data:
            - 'bullish': Percentage of bullish sentiment
            - 'bearish': Percentage of bearish sentiment
            - 'neutral': Percentage of neutral sentiment
            - None if retrieval fails
            """
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
                if bullish is None or bearish is None:
                    return None
                return {
                    'bullish': bullish,
                    'bearish': bearish,
                    'neutral': neutral
                }
            except Exception as e:
                print(f"Error retrieving sentiment data: {e}")
                return self.get_social_sentiment_data()

    def get_social_sentiment_data(self):
        """
        Get sentiment data from a local Excel file as a fallback.
        
        Returns:
        - Dictionary with sentiment data:
          - 'bullish': Percentage of bullish sentiment
          - 'bearish': Percentage of bearish sentiment
          - 'neutral': Percentage of neutral sentiment
        - None if retrieval fails
        """
        file_path = r'C:\Users\klamo\Downloads\historical_sentiment_20250220.xls'
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            print("Error: File not found.")
            return None
        required_columns = ['Reported Date', 'Bullish', 'Neutral', 'Bearish']
        if not all(col in df.columns for col in required_columns):
            print("Error: Required columns missing in the Excel file.")
            return None
        last_row = df.iloc[-1]
        try:
            bullish = float(last_row['Bullish'])
            neutral = float(last_row['Neutral'])
            bearish = float(last_row['Bearish'])
        except ValueError:
            print("Error: Invalid data format in the Excel file.")
            return None
        return {"bullish": bullish, "bearish": bearish, "neutral": neutral}

    def _parse_number(self, text):
        """
        Parse a number from text by removing commas.
        
        Parameters:
        - text: Text to parse
        
        Returns:
        - float: Parsed number
        - None: If parsing fails
        """
        try:
            return float(text.replace(',', ''))
        except:
            return None
    
    def _parse_percentage(self, text):
        """
        Parse a percentage from text by removing the '%' symbol.
        
        Parameters:
        - text: Text to parse
        
        Returns:
        - float: Parsed percentage
        - None: If parsing fails
        """
        try:
            return float(text.replace('%', ''))
        except:
            return None

    def get_options_data(symbol, api_key):
        params = {'function': 'OPTION_CHAIN', 'symbol': symbol, 'apikey': api_key}
        response = requests.get('https://www.alphavantage.co/query', params=params)
        data = response.json()
        # Parse to calculate put/call ratio and implied volatility
        puts = [option for option in data.get('options', []) if option['type'] == 'put']
        calls = [option for option in data.get('options', []) if option['type'] == 'call']
        put_call_ratio = len(puts) / len(calls) if calls else float('inf')
        return {'put_call_ratio': put_call_ratio, 'implied_volatility': data.get('implied_volatility', 0)}

    def get_institutional_flow(symbol, api_key):
        """
        Get institutional money flow data for the given stock symbol.
        
        Note: This is a placeholder method and requires implementation with actual
              institutional flow data retrieval.
        
        Parameters:
        - symbol: Stock symbol (e.g., 'AAPL')
        
        Returns:
        - Dictionary with placeholder institutional money flow data:
          - 'net_flow': Net institutional money flow
        """
        # TODO: Implement institutional flow data retrieval
        return {"net_flow": 1000000}