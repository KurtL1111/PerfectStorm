"""
Updated data retrieval module for Perfect Storm Dashboard

This module implements functions to retrieve stock data and market data
required for the Perfect Storm investment strategy dashboard.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from bs4 import BeautifulSoup
import time
import random
from market_data_retrieval import MarketDataRetriever

class StockDataRetriever:
    """Class to retrieve stock data and market data for Perfect Storm Dashboard"""
    
    def __init__(self):
        """Initialize the data retriever"""
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_stock_history(self, symbol, interval='1d', range_period='1y'):
        """
        Get historical stock data with caching using AlphaVantage API
        
        Parameters:
        - symbol: Stock symbol
        - interval: (Ignored; AlphaVantage provides daily data) 
        - range_period: Time range (e.g., '1y') where 'y' means years
        
        Returns:
        - DataFrame with historical data
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{interval}_{range_period}.csv")
        cache_valid = False
        
        if os.path.exists(cache_file):
            # Check if cache is still valid (less than 24 hours old)
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < 86400:  # 24 hours in seconds
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        print(f"Using cached data for {symbol}")
                        cache_valid = True
                        return df
                except Exception as e:
                    print(f"Error reading cache: {e}")
        
        # If cache is not valid, fetch new data
        if not cache_valid:
            try:
                # Retrieve data using AlphaVantage API via MarketDataRetriever
                api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
                retriever = MarketDataRetriever(api_key=api_key)
                df = retriever.get_stock_data(symbol, period=range_period)
                if df is None or df.empty:
                    print(f"No data found for {symbol} using AlphaVantage. Generating sample data.")
                    df = self._generate_sample_data()
                else:
                    print(f"Retrieved data for {symbol} via AlphaVantage.")
                # Save to cache
                df.to_csv(cache_file)
                return df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return self._generate_sample_data()
    
    def get_market_breadth_data(self):
        """
        Get market breadth data from MarketWatch for ARMS Index calculation
        
        Returns:
        - Dictionary with advancing issues, declining issues, advancing volume, declining volume
        """
        try:
            # URL for MarketWatch market data
            url = "https://www.marketwatch.com/market-data/us?mod=market-data-center"
            
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            # Make the request
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Error fetching market data: Status code {response.status_code}")
                return self._generate_sample_market_breadth()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the market breadth data
            # Note: This is a simplified approach and may need adjustment based on the actual HTML structure
            market_data = {}
            
            # Look for elements containing the market breadth data
            # These selectors will need to be updated based on the actual page structure
            try:
                # Find the section with market breadth data
                breadth_section = soup.find('div', class_='element element--breadth')
                
                if breadth_section:
                    # Extract advancing and declining issues
                    adv_issues_elem = breadth_section.find('td', text='Issues advancing')
                    if adv_issues_elem:
                        adv_issues = adv_issues_elem.find_next('td').text.strip().replace(',', '')
                        market_data['advancing_issues'] = int(adv_issues)
                    
                    dec_issues_elem = breadth_section.find('td', text='Issues declining')
                    if dec_issues_elem:
                        dec_issues = dec_issues_elem.find_next('td').text.strip().replace(',', '')
                        market_data['declining_issues'] = int(dec_issues)
                    
                    # Extract advancing and declining volume
                    adv_vol_elem = breadth_section.find('td', text='Volume advancing')
                    if adv_vol_elem:
                        adv_vol = adv_vol_elem.find_next('td').text.strip().replace(',', '')
                        market_data['advancing_volume'] = int(adv_vol)
                    
                    dec_vol_elem = breadth_section.find('td', text='Volume declining')
                    if dec_vol_elem:
                        dec_vol = dec_vol_elem.find_next('td').text.strip().replace(',', '')
                        market_data['declining_volume'] = int(dec_vol)
                
                # If we couldn't find the data with the expected structure, try a more general approach
                if len(market_data) < 4:
                    # Look for any table with market breadth data
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 2:
                                cell_text = cells[0].text.strip().lower()
                                if 'advancing' in cell_text and 'issues' in cell_text:
                                    market_data['advancing_issues'] = int(cells[1].text.strip().replace(',', ''))
                                elif 'declining' in cell_text and 'issues' in cell_text:
                                    market_data['declining_issues'] = int(cells[1].text.strip().replace(',', ''))
                                elif 'advancing' in cell_text and 'volume' in cell_text:
                                    market_data['advancing_volume'] = int(cells[1].text.strip().replace(',', ''))
                                elif 'declining' in cell_text and 'volume' in cell_text:
                                    market_data['declining_volume'] = int(cells[1].text.strip().replace(',', ''))
            
            except Exception as e:
                print(f"Error parsing market breadth data: {e}")
            
            # Check if we have all the required data
            if len(market_data) < 4:
                print("Could not find all required market breadth data, using sample data")
                return self._generate_sample_market_breadth()
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching market breadth data: {e}")
            return self._generate_sample_market_breadth()
    
    def get_aaii_sentiment_data(self):
        """
        Get investor sentiment data from AAII for Bulls vs Bears ratio
        
        Returns:
        - Dictionary with bullish, bearish, and neutral percentages
        """
        try:
            # URL for AAII sentiment survey
            url = "https://www.aaii.com/sentimentsurvey/sent_results"
            
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            # Make the request
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Error fetching AAII sentiment data: Status code {response.status_code}")
                return self._generate_sample_sentiment()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the sentiment data
            sentiment_data = {}
            
            try:
                # Look for the sentiment data in the page
                # These selectors will need to be updated based on the actual page structure
                sentiment_table = soup.find('table', class_='sentimenttable')
                
                if sentiment_table:
                    rows = sentiment_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            cell_text = cells[0].text.strip().lower()
                            if 'bullish' in cell_text:
                                sentiment_data['bullish'] = float(cells[1].text.strip().replace('%', ''))
                            elif 'bearish' in cell_text:
                                sentiment_data['bearish'] = float(cells[1].text.strip().replace('%', ''))
                            elif 'neutral' in cell_text:
                                sentiment_data['neutral'] = float(cells[1].text.strip().replace('%', ''))
                
                # If we couldn't find the data with the expected structure, try a more general approach
                if len(sentiment_data) < 3:
                    # Look for any elements with sentiment percentages
                    bullish_elem = soup.find(text=lambda text: text and 'bullish' in text.lower())
                    if bullish_elem:
                        # Try to find a nearby percentage
                        for elem in bullish_elem.parent.find_all_next():
                            text = elem.text.strip()
                            if '%' in text:
                                sentiment_data['bullish'] = float(text.replace('%', ''))
                                break
                    
                    bearish_elem = soup.find(text=lambda text: text and 'bearish' in text.lower())
                    if bearish_elem:
                        for elem in bearish_elem.parent.find_all_next():
                            text = elem.text.strip()
                            if '%' in text:
                                sentiment_data['bearish'] = float(text.replace('%', ''))
                                break
                    
                    neutral_elem = soup.find(text=lambda text: text and 'neutral' in text.lower())
                    if neutral_elem:
                        for elem in neutral_elem.parent.find_all_next():
                            text = elem.text.strip()
                            if '%' in text:
                                sentiment_data['neutral'] = float(text.replace('%', ''))
                                break
            
            except Exception as e:
                print(f"Error parsing AAII sentiment data: {e}")
            
            # Check if we have all the required data
            if len(sentiment_data) < 3:
                print("Could not find all required sentiment data, using sample data")
                return self._generate_sample_sentiment()
            
            return sentiment_data
            
        except Exception as e:
            print(f"Error fetching AAII sentiment data: {e}")
            return self._generate_sample_sentiment()
    
    def _generate_sample_data(self):
        """
        Generate sample stock data for testing
        
        Returns:
        - DataFrame with sample data
        """
        print("Generating sample data for testing")
        
        # Create date range for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter out weekends
        dates = dates[dates.dayofweek < 5]
        
        # Generate random price data
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 100
        
        # Generate random daily returns
        daily_returns = np.random.normal(0.0005, 0.015, len(dates))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Calculate prices
        prices = base_price * cumulative_returns
        
        # Generate other columns
        opens = prices * np.random.uniform(0.99, 1.01, len(dates))
        highs = prices * np.random.uniform(1.01, 1.03, len(dates))
        lows = prices * np.random.uniform(0.97, 0.99, len(dates))
        volumes = np.random.randint(100000, 1000000, len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        # Rename columns to match yfinance format
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return df
    
    def _generate_sample_market_breadth(self):
        """
        Generate sample market breadth data for testing
        
        Returns:
        - Dictionary with advancing issues, declining issues, advancing volume, declining volume
        """
        print("Generating sample market breadth data for testing")
        
        # Generate random data
        advancing_issues = random.randint(1500, 2500)
        declining_issues = random.randint(1000, 2000)
        advancing_volume = random.randint(800000000, 1500000000)
        declining_volume = random.randint(500000000, 1200000000)
        
        return {
            'advancing_issues': advancing_issues,
            'declining_issues': declining_issues,
            'advancing_volume': advancing_volume,
            'declining_volume': declining_volume
        }
    
    def _generate_sample_sentiment(self):
        """
        Generate sample AAII sentiment data for testing
        
        Returns:
        - Dictionary with bullish, bearish, and neutral percentages
        """
        print("Generating sample AAII sentiment data for testing")
        
        # Generate random data
        bullish = random.uniform(30, 45)
        bearish = random.uniform(25, 40)
        neutral = 100 - bullish - bearish
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral
        }


# Example usage
#def example_usage():
#    """Example of how to use the StockDataRetriever class"""
#    
#    retriever = StockDataRetriever()
#    
#    # Get stock data
#    df = retriever.get_stock_history<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>