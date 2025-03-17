"""
Market data retrieval module for Perfect Storm Dashboard

This module handles retrieving market data from various sources,
including stock data, market breadth data, and sentiment data.
"""

import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class MarketDataRetriever:
    """Class to retrieve market data for the Perfect Storm dashboard"""
    
    def __init__(self):
        """Initialize the MarketDataRetriever"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_stock_data(self, symbol, period='1y'):
        """
        Get stock data from Yahoo Finance
        
        Parameters:
        - symbol: Stock symbol
        - period: Time period (default: '1y')
        
        Returns:
        - DataFrame with stock data or None if retrieval fails
        """
        try:
            # Get stock data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            return df
        
        except Exception as e:
            print(f"Error retrieving stock data: {e}")
            # Return None instead of fallback values
            return None
    
    def get_market_breadth_data(self):
        """
        Get market breadth data from MarketWatch
        
        Returns:
        - Dictionary with market breadth data or None if retrieval fails
        """
        try:
            # Get market breadth data from MarketWatch
            url = "https://www.marketwatch.com/market-data/us?mod=market-data-center"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve market breadth data: {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find advancing and declining issues
            advancing_issues = 0
            declining_issues = 0
            advancing_volume = 0
            declining_volume = 0
            
            # Look for the market breadth table
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
            
            # If we couldn't find the data, return None instead of fallback values
            if advancing_issues == 0 or declining_issues == 0:
                return None
            
            return {
                'advancing_issues': advancing_issues,
                'declining_issues': declining_issues,
                'advancing_volume': advancing_volume,
                'declining_volume': declining_volume
            }
        
        except Exception as e:
            print(f"Error retrieving market breadth data: {e}")
            # Return None instead of fallback values
            return None
    
    def get_sentiment_data(self):
        """
        Get sentiment data from AAII Investor Sentiment Survey
        
        Returns:
        - Dictionary with sentiment data or None if retrieval fails
        """
        try:
            # Get sentiment data from AAII Investor Sentiment Survey
            url = "https://www.aaii.com/sentimentsurvey/sent_results"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve sentiment data: {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find sentiment data
            bullish = 0
            bearish = 0
            neutral = 0
            
            # Look for the sentiment table
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
            
            # If we couldn't find the data, return None instead of fallback values
            if bullish == 0 or bearish == 0:
                return None
            
            return {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral
            }
        
        except Exception as e:
            print(f"Error retrieving sentiment data: {e}")
            # Return None instead of fallback values
            return None
    
    def _parse_number(self, text):
        """
        Parse a number from text
        
        Parameters:
        - text: Text to parse
        
        Returns:
        - Parsed number
        """
        try:
            # Remove commas and convert to float
            return float(text.replace(',', ''))
        except:
            return 0
    
    def _parse_percentage(self, text):
        """
        Parse a percentage from text
        
        Parameters:
        - text: Text to parse
        
        Returns:
        - Parsed percentage
        """
        try:
            # Remove % and convert to float
            return float(text.replace('%', ''))
        except:
            return 0
