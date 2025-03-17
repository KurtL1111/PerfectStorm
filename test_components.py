import os
import time
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from market_data_retrieval import MarketDataRetriever

# --- Daily Data Retrieval Test ---
def test_get_stock_history_daily():
    retriever = MarketDataRetriever(api_key="25WNVRI1YIXCDIH1")
    # For daily interval ("1d"), use yfinance_cache branch.
    df = retriever.get_stock_history("AAPL", interval="1d", period="1y")
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame for daily data."
    # Check that expected columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"
    # Verify that the index is datetime
    assert np.issubdtype(df.index.dtype, np.datetime64), "Index should be datetime."

# --- Intraday Data Retrieval Test ---
def test_get_stock_history_intraday():
    retriever = MarketDataRetriever(api_key="25WNVRI1YIXCDIH1")
    # For intraday intervals, e.g., "5m", using AlphaVantage.
    df = retriever.get_stock_history("AAPL", interval="5m", period="1d")
    # If no intraday data is returned, skip the test.
    if df.empty or len(df) < 2:
        pytest.skip("Intraday data not available for testing.")
    # Calculate time difference between first two rows
    delta = (df.index[1] - df.index[0]).total_seconds()
    expected_delta = 300  # 5 minutes in seconds
    # Allow a tolerance of 60 seconds
    assert abs(delta - expected_delta) < 60, f"Expected intraday delta near {expected_delta} sec, got {delta}"

# --- Market Breadth Data Test ---
def test_get_market_breadth_data():
    retriever = MarketDataRetriever()
    breadth = retriever.get_market_breadth_data()
    if breadth is not None:
        for key in ['advancing_issues', 'declining_issues', 'advancing_volume', 'declining_volume']:
            assert key in breadth, f"Market breadth data missing key: {key}"
            assert isinstance(breadth[key], (int, float)), f"Value for {key} should be numeric"
    else:
        pytest.skip("Market breadth data not available from live source.")

# --- Sentiment Data (Online) Test ---
def test_get_sentiment_data_online(monkeypatch):
    # Fake a successful response with sample sentiment percentages
    class FakeResponse:
        status_code = 200
        text = """
        <html>
          <body>
            <table>
              <tr><td>Bullish</td><td>40%</td></tr>
              <tr><td>Bearish</td><td>30%</td></tr>
              <tr><td>Neutral</td><td>30%</td></tr>
            </table>
          </body>
        </html>
        """
    def fake_get(*args, **kwargs):
        return FakeResponse()
    monkeypatch.setattr("requests.get", fake_get)
    
    retriever = MarketDataRetriever()
    sentiment = retriever.get_sentiment_data()
    assert isinstance(sentiment, dict), "Expected a dictionary for sentiment data."
    for key in ["bullish", "bearish", "neutral"]:
        assert key in sentiment, f"Sentiment data missing key: {key}"
        assert isinstance(sentiment[key], float), f"Sentiment value for {key} should be a float."
    assert sentiment["bullish"] == 40.0
    assert sentiment["bearish"] == 30.0
    assert sentiment["neutral"] == 30.0

# --- Sentiment Data Fallback Test ---
def test_get_sentiment_data_fallback(monkeypatch):
    # Simulate a failure (non-200 response) so that fallback is triggered.
    def fake_get_fail(*args, **kwargs):
        class FakeResponse:
            status_code = 500
        return FakeResponse()
    monkeypatch.setattr("requests.get", fake_get_fail)
    
    retriever = MarketDataRetriever()
    sentiment = retriever.get_social_sentiment_data()
    if sentiment is None:
        pytest.skip("Fallback sentiment data not available in test environment.")
    else:
        for key in ["bullish", "bearish", "neutral"]:
            assert key in sentiment, f"Fallback sentiment data missing key: {key}"

# --- Options Data Test ---
def test_get_options_data():
    data = MarketDataRetriever.get_options_data("AAPL", api_key="25WNVRI1YIXCDIH1")
    assert isinstance(data, dict), "Options data should be a dictionary."
    assert "put_call_ratio" in data, "Missing put_call_ratio in options data."

# --- Institutional Flow Test ---
def test_get_institutional_flow():
    data = MarketDataRetriever.get_institutional_flow("AAPL", api_key="25WNVRI1YIXCDIH1")
    assert isinstance(data, dict), "Institutional flow data should be a dictionary."
    assert "net_flow" in data, "Missing net_flow in institutional flow data."
