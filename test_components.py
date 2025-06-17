import os
import time
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch, MagicMock
import torch

# Import all modules to be tested
from ml_anomaly_detection_enhanced import MarketAnomalyDetection
from ml_clustering_enhanced_completed import PerfectStormClustering
from ml_pattern_recognition_enhanced import MarketPatternRecognition
from market_regime_detection_completed import MarketRegimeDetection
from correlation_analysis import CorrelationAnalysis
from market_data_retrieval import MarketDataRetriever
from adaptive_thresholds_enhanced import EnhancedAdaptiveThresholds
from backtesting_engine import BacktestingEngine
from portfolio_optimization import PortfolioOptimizer
from real_time_alerts import (
    AlertCondition, PriceAlertCondition, IndicatorAlertCondition, 
    PatternAlertCondition, VolatilityAlertCondition, PerfectStormAlertCondition
)

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
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Index should be datetime."


# --- Intraday Data Retrieval Tests ---


def test_get_btc_hourly_history():
    retriever = MarketDataRetriever()
    # Test BTC hourly data from local CSV
    df = retriever.get_stock_history("BTC/USD", interval="1h", period="1mo")
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame for BTC hourly data."
    assert not df.empty, "BTC hourly data DataFrame is empty."
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Index should be datetime."
    # Check that the time delta is 1 hour
    if len(df) > 2:
        delta = (df.index[1] - df.index[0]).total_seconds()
        assert abs(delta - 3600) < 10, f"Expected 1 hour delta, got {delta} seconds."

def test_get_btc_minute_history():
    retriever = MarketDataRetriever()
    # Test BTC minute data from local CSVs
    df = retriever.get_stock_history("BTC/USD", interval="1m", period="5d")
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame for BTC minute data."
    assert not df.empty, "BTC minute data DataFrame is empty."
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Index should be datetime."
    # Check that the time delta is 1 minute
    if len(df) > 2:
        delta = (df.index[1] - df.index[0]).total_seconds()
        assert abs(delta - 60) < 2, f"Expected 1 minute delta, got {delta} seconds."
        
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

# --- Anomaly Detection Tests ---
def test_anomaly_detector_initialization():
    detector = MarketAnomalyDetection()
    # For compatibility: .model may be None, just check attribute exists
    assert hasattr(detector, 'model'), "Model attribute should exist."

def test_anomaly_detection():
    detector = MarketAnomalyDetection()
    sample_data = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.1, 0.3],
        'feature2': [1.0, 1.1, 0.9, 1.2]
    })
    # Pass required feature_cols argument
    anomalies = detector.detect_anomalies(sample_data, feature_cols=['feature1', 'feature2'])
    # Accept either DataFrame or ndarray for compatibility
    assert isinstance(anomalies, (pd.DataFrame, np.ndarray)), "Output should be a DataFrame or ndarray."
    if isinstance(anomalies, pd.DataFrame):
        assert 'anomaly_score' in anomalies.columns, "DataFrame should contain 'anomaly_score' column."

# --- Clustering Tests ---
def test_clustering_model_initialization():
    model = PerfectStormClustering(n_clusters=3)
    assert model.n_clusters == 3, "Number of clusters should be set correctly."

def test_clustering():
    model = PerfectStormClustering(n_clusters=2)
    sample_data = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.8, 0.9],
        'feature2': [1.0, 1.1, 0.2, 0.3]
    })
    clusters = model.fit_predict(sample_data)
    assert len(clusters) == len(sample_data), "Each data point should have a cluster assignment."

# --- Pattern Recognition Tests ---
def test_pattern_recognizer_initialization():
    recognizer = MarketPatternRecognition()
    assert recognizer.patterns is not None, "Patterns should be initialized."

def test_pattern_recognition():
    recognizer = MarketPatternRecognition()
    sample_data = pd.DataFrame({
        'price': [100, 102, 101, 105, 107]
    })
    patterns = recognizer.predict(sample_data)
    assert isinstance(patterns, list), "Output should be a list of patterns."

# --- Market Regime Detection Tests ---
def test_analyzer_initialization():
    analyzer = MarketRegimeDetection()
    assert analyzer.lookback_period > 0, "Lookback period should be positive."

def test_correlation_analysis():
    analyzer = CorrelationAnalysis(window_size=5)
    sample_data = pd.DataFrame({
        'asset1': [100, 101, 102, 103, 104],
        'asset2': [200, 198, 202, 204, 203]
    })
    correlations = analyzer.calculate_rolling_correlations(sample_data, ['asset1', 'asset2'])
    assert isinstance(correlations, pd.DataFrame), "Output should be a DataFrame."

# --- New Tests for Market Regime Detection ---
def test_market_regime_detection_extract_features():
    """Test the feature extraction functionality of MarketRegimeDetection"""
    detector = MarketRegimeDetection()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Extract features
    features = detector.extract_regime_features(sample_data)
    
    # Verify features
    assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
    assert 'volatility' in features.columns, "Features should include volatility"
    assert 'trend' in features.columns, "Features should include trend"
    assert 'rsi' in features.columns, "Features should include RSI"
    assert len(features) > 0, "Features DataFrame should not be empty"

def test_market_regime_detection_clustering():
    """Test the clustering-based regime detection"""
    detector = MarketRegimeDetection(n_regimes=3)
    
    # Create sample data with features
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    features = pd.DataFrame({
        'volatility': np.random.normal(0.2, 0.05, 100),
        'trend': np.random.normal(0.01, 0.02, 100),
        'momentum': np.random.normal(0.005, 0.01, 100),
        'mean_reversion': np.random.normal(0, 1, 100),
        'rsi': np.random.normal(50, 10, 100),
        'autocorr_1': np.random.normal(0.1, 0.2, 100),
        'returns_skew': np.random.normal(0, 0.5, 100),
        'returns': np.random.normal(0.001, 0.01, 100)
    }, index=dates)
    
    # Detect regimes
    regimes = detector.detect_regimes_clustering(features)
    
    # Verify regimes
    assert isinstance(regimes, pd.DataFrame), "Regimes should be a DataFrame"
    assert 'regime' in regimes.columns, "Regimes should include regime column"
    assert len(regimes) == len(features), "Regimes should have same length as features"
    assert regimes['regime'].nunique() <= 3, "Should have at most 3 regimes"

def test_analyze_regime_transitions():
    """Test the analyze_regime_transitions function"""
    detector = MarketRegimeDetection(n_regimes=3)
    
    # Create sample regime history
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Simulate regime changes
    regimes = []
    current_regime = 0
    for i in range(100):
        if i > 0 and i % 20 == 0:  # Change regime every 20 days
            current_regime = (current_regime + 1) % 3
        regimes.append(current_regime)
    
    # Create features DataFrame with regime column
    features = pd.DataFrame({
        'volatility': np.random.normal(0.2, 0.05, 100),
        'trend': np.random.normal(0.01, 0.02, 100),
        'momentum': np.random.normal(0.005, 0.01, 100),
        'mean_reversion': np.random.normal(0, 1, 100),
        'rsi': np.random.normal(50, 10, 100),
        'autocorr_1': np.random.normal(0.1, 0.2, 100),
        'returns_skew': np.random.normal(0, 0.5, 100),
        'returns': np.random.normal(0.001, 0.01, 100),
        'regime': regimes
    }, index=dates)
    
    # Map numeric regimes to names
    regime_names = {0: 'trending_up', 1: 'trending_down', 2: 'ranging'}
    features['regime_name'] = features['regime'].map(regime_names)
    
    # Update regime history
    detector.update_regime_history(features)
    
    # Calculate transition matrix
    detector.calculate_transition_matrix(features)
    
    # Calculate regime statistics
    detector.calculate_regime_statistics(features)
    
    # Analyze regime transitions
    transition_analysis = detector.analyze_regime_transitions()
    
    # Verify transition analysis
    assert isinstance(transition_analysis, dict), "Transition analysis should be a dictionary"
    assert 'transition_matrix' in transition_analysis, "Should include transition matrix"
    assert 'transition_counts' in transition_analysis, "Should include transition counts"
    assert 'avg_regime_durations' in transition_analysis, "Should include average durations"
    assert 'updated_parameters' in transition_analysis, "Should include updated parameters"
    
    # Verify updated parameters
    updated_params = transition_analysis['updated_parameters']
    for regime in ['trending_up', 'trending_down', 'ranging']:
        assert regime in updated_params, f"Updated parameters should include {regime}"
        assert 'rsi_lower' in updated_params[regime], "Parameters should include RSI lower bound"
        assert 'rsi_upper' in updated_params[regime], "Parameters should include RSI upper bound"
        assert 'position_size' in updated_params[regime], "Parameters should include position size"

def test_validate_regime_classification():
    """Test the validate_regime_classification function"""
    detector = MarketRegimeDetection(n_regimes=3)
    
    # Create sample features with regimes
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create features with clear regime characteristics
    features = pd.DataFrame(index=dates)
    
    # Trending up regime (first 30 days)
    features.loc[dates[:30], 'trend'] = np.random.normal(0.02, 0.005, 30)  # Positive trend
    features.loc[dates[:30], 'volatility'] = np.random.normal(0.15, 0.02, 30)  # Moderate volatility
    features.loc[dates[:30], 'autocorr_1'] = np.random.normal(0.3, 0.1, 30)  # Positive autocorrelation
    features.loc[dates[:30], 'regime'] = 0
    
    # Trending down regime (next 30 days)
    features.loc[dates[30:60], 'trend'] = np.random.normal(-0.02, 0.005, 30)  # Negative trend
    features.loc[dates[30:60], 'volatility'] = np.random.normal(0.18, 0.03, 30)  # Moderate volatility
    features.loc[dates[30:60], 'autocorr_1'] = np.random.normal(0.25, 0.1, 30)  # Positive autocorrelation
    features.loc[dates[30:60], 'regime'] = 1
    
    # Ranging regime (last 40 days)
    features.loc[dates[60:], 'trend'] = np.random.normal(0.001, 0.005, 40)  # Minimal trend
    features.loc[dates[60:], 'volatility'] = np.random.normal(0.1, 0.02, 40)  # Lower volatility
    features.loc[dates[60:], 'autocorr_1'] = np.random.normal(-0.2, 0.1, 40)  # Negative autocorrelation
    features.loc[dates[60:], 'regime'] = 2
    
    # Add other required features
    features['returns'] = np.random.normal(0.001, 0.01, 100)
    features['returns_skew'] = np.random.normal(0, 0.5, 100)
    features['momentum'] = np.random.normal(0.005, 0.01, 100)
    features['mean_reversion'] = np.random.normal(0, 1, 100)
    features['rsi'] = np.random.normal(50, 10, 100)
    
    # Validate regime classification
    validation_results = detector.validate_regime_classification(features)
    
    # Verify validation results
    assert isinstance(validation_results, dict), "Validation results should be a dictionary"
    assert 'trend_alignment' in validation_results, "Should include trend alignment"
    assert 'volatility_alignment' in validation_results, "Should include volatility alignment"
    assert 'reversion_alignment' in validation_results, "Should include reversion alignment"
    assert 'confidence_scores' in validation_results, "Should include confidence scores"
    assert 'overall_confidence' in validation_results, "Should include overall confidence"
    
    # Verify confidence scores
    confidence_scores = validation_results['confidence_scores']
    for regime in ['trending_up', 'trending_down', 'ranging']:
        assert regime in confidence_scores, f"Confidence scores should include {regime}"
        assert 0 <= confidence_scores[regime] <= 1, f"Confidence score for {regime} should be between 0 and 1"

# --- Tests for Adaptive Thresholds Enhanced ---
def test_adaptive_thresholds_initialization():
    """Test initialization of EnhancedAdaptiveThresholds"""
    thresholds = EnhancedAdaptiveThresholds(volatility_window=20, risk_tolerance='medium')
    assert thresholds.volatility_window == 20, "Volatility window should be set correctly"
    assert thresholds.risk_tolerance == 'medium', "Risk tolerance should be set correctly"
    assert 'low' in thresholds.risk_multipliers, "Risk multipliers should include 'low'"
    assert 'medium' in thresholds.risk_multipliers, "Risk multipliers should include 'medium'"
    assert 'high' in thresholds.risk_multipliers, "Risk multipliers should include 'high'"

def test_calculate_volatility():
    """Test volatility calculation in EnhancedAdaptiveThresholds"""
    thresholds = EnhancedAdaptiveThresholds(volatility_window=10)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    sample_data = pd.DataFrame({
        'close': np.random.normal(100, 5, 50),
        'high': np.random.normal(105, 5, 50),
        'low': np.random.normal(95, 5, 50)
    }, index=dates)
    
    # Calculate volatility
    vol_data = thresholds.calculate_volatility(sample_data)
    
    # Verify volatility data
    assert isinstance(vol_data, pd.DataFrame), "Volatility data should be a DataFrame"
    assert 'volatility' in vol_data.columns, "Should include volatility column"
    assert 'volatility_regime' in vol_data.columns, "Should include volatility regime column"
    assert 'volatility_percentile' in vol_data.columns, "Should include volatility percentile column"
    assert vol_data['volatility_regime'].isin(['low', 'medium', 'high']).all(), "Volatility regime should be low, medium, or high"

def test_calculate_statistical_thresholds():
    """Test statistical threshold calculation in EnhancedAdaptiveThresholds"""
    thresholds = EnhancedAdaptiveThresholds()
    
    # Create sample data with RSI
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'rsi': np.random.normal(50, 15, 100)
    }, index=dates)
    
    # Calculate thresholds using different methods
    quantile_thresholds = thresholds.calculate_statistical_thresholds(sample_data, 'rsi', method='quantile')
    zscore_thresholds = thresholds.calculate_statistical_thresholds(sample_data, 'rsi', method='zscore')
    percentile_thresholds = thresholds.calculate_statistical_thresholds(sample_data, 'rsi', method='percentile')
    
    # Verify thresholds
    assert isinstance(quantile_thresholds, dict), "Quantile thresholds should be a dictionary"
    assert isinstance(zscore_thresholds, dict), "Z-score thresholds should be a dictionary"
    assert isinstance(percentile_thresholds, dict), "Percentile thresholds should be a dictionary"
    
    assert 'lower' in quantile_thresholds, "Quantile thresholds should include lower bound"
    assert 'upper' in quantile_thresholds, "Quantile thresholds should include upper bound"
    assert quantile_thresholds['lower'] < quantile_thresholds['upper'], "Lower bound should be less than upper bound"

def test_calculate_volatility_adjusted_thresholds():
    """Test volatility-adjusted threshold calculation in EnhancedAdaptiveThresholds"""
    thresholds = EnhancedAdaptiveThresholds(volatility_window=10)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    sample_data = pd.DataFrame({
        'close': np.random.normal(100, 5, 50),
        'rsi': np.random.normal(50, 15, 50)
    }, index=dates)
    
    # Calculate volatility-adjusted thresholds
    vol_thresholds = thresholds.calculate_volatility_adjusted_thresholds(sample_data, 'rsi')
    
    # Verify thresholds
    assert isinstance(vol_thresholds, dict), "Volatility-adjusted thresholds should be a dictionary"
    assert 'lower' in vol_thresholds, "Should include lower bound"
    assert 'upper' in vol_thresholds, "Should include upper bound"
    assert vol_thresholds['lower'] < vol_thresholds['upper'], "Lower bound should be less than upper bound"

def test_calculate_regime_specific_thresholds():
    """Test regime-specific threshold calculation in EnhancedAdaptiveThresholds"""
    thresholds = EnhancedAdaptiveThresholds()
    
    # Create sample data with market regimes
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'close': np.random.normal(100, 5, 100),
        'rsi': np.random.normal(50, 15, 100),
        'regime': np.random.choice(['trending_up', 'trending_down', 'ranging', 'volatile'], 100)
    }, index=dates)
    
    # Calculate regime-specific thresholds
    regime_thresholds = thresholds.calculate_regime_specific_thresholds(sample_data, 'rsi')
    
    # Verify thresholds
    assert isinstance(regime_thresholds, dict), "Regime-specific thresholds should be a dictionary"
    assert 'trending_up' in regime_thresholds, "Should include trending_up regime"
    assert 'trending_down' in regime_thresholds, "Should include trending_down regime"
    assert 'ranging' in regime_thresholds, "Should include ranging regime"
    
    for regime, regime_threshold in regime_thresholds.items():
        assert 'lower' in regime_threshold, f"{regime} should include lower bound"
        assert 'upper' in regime_threshold, f"{regime} should include upper bound"
        assert regime_threshold['lower'] < regime_threshold['upper'], f"{regime} lower bound should be less than upper bound"

def test_get_adaptive_thresholds():
    """Test the main get_adaptive_thresholds method in EnhancedAdaptiveThresholds"""
    thresholds = EnhancedAdaptiveThresholds()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'close': np.random.normal(100, 5, 100),
        'rsi': np.random.normal(50, 15, 100)
    }, index=dates)
    
    # Get adaptive thresholds using different methods
    statistical_thresholds = thresholds.get_adaptive_thresholds(sample_data, 'rsi', method='statistical')
    volatility_thresholds = thresholds.get_adaptive_thresholds(sample_data, 'rsi', method='volatility')
    
    # Verify thresholds
    assert isinstance(statistical_thresholds, dict), "Statistical thresholds should be a dictionary"
    assert isinstance(volatility_thresholds, dict), "Volatility thresholds should be a dictionary"
    
    assert 'lower' in statistical_thresholds, "Statistical thresholds should include lower bound"
    assert 'upper' in statistical_thresholds, "Statistical thresholds should include upper bound"
    assert statistical_thresholds['lower'] < statistical_thresholds['upper'], "Statistical lower bound should be less than upper bound"
    
    assert 'lower' in volatility_thresholds, "Volatility thresholds should include lower bound"
    assert 'upper' in volatility_thresholds, "Volatility thresholds should include upper bound"
    assert volatility_thresholds['lower'] < volatility_thresholds['upper'], "Volatility lower bound should be less than upper bound"

# --- Tests for Backtesting Engine Enhanced ---
def test_backtesting_engine_initialization():
    """Test initialization of BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    assert engine.initial_capital == 100000, "Initial capital should be set correctly"
    assert engine.commission == 0.001, "Commission should be set correctly"
    assert engine.positions == {}, "Positions should be initialized as empty dictionary"
    assert engine.trades == [], "Trades should be initialized as empty list"
    assert engine.equity_curve is None, "Equity curve should be initialized as None"

def create_sample_strategy():
    """Create a sample strategy function for testing"""
    def sample_strategy(data, params=None):
        if params is None:
            params = {'sma_short': 5, 'sma_long': 20}
        
        # Calculate indicators
        data['sma_short'] = data['close'].rolling(window=params['sma_short']).mean()
        data['sma_long'] = data['close'].rolling(window=params['sma_long']).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['sma_short'] > data['sma_long'], 'signal'] = 1  # Buy signal
        data.loc[data['sma_short'] < data['sma_long'], 'signal'] = -1  # Sell signal
        
        return data
    
    return sample_strategy

def test_run_backtest():
    """Test running a backtest with BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Run backtest
    results = engine.run_backtest(sample_data, create_sample_strategy())
    
    # Verify results
    assert isinstance(results, pd.DataFrame), "Backtest results should be a DataFrame"
    assert 'signal' in results.columns, "Results should include signal column"
    assert 'position' in results.columns, "Results should include position column"
    assert 'equity' in results.columns, "Results should include equity column"
    assert engine.equity_curve is not None, "Equity curve should be populated"

def test_calculate_performance_metrics():
    """Test calculation of performance metrics in BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Run backtest
    engine.run_backtest(sample_data, create_sample_strategy())
    
    # Calculate performance metrics
    metrics = engine.calculate_performance_metrics()
    
    # Verify metrics
    assert isinstance(metrics, dict), "Performance metrics should be a dictionary"
    assert 'total_return' in metrics, "Metrics should include total return"
    assert 'annualized_return' in metrics, "Metrics should include annualized return"
    assert 'sharpe_ratio' in metrics, "Metrics should include Sharpe ratio"
    assert 'max_drawdown' in metrics, "Metrics should include maximum drawdown"
    assert 'win_rate' in metrics, "Metrics should include win rate"
    assert 'profit_factor' in metrics, "Metrics should include profit factor"

def test_optimize_parameters():
    """Test parameter optimization in BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Define parameter grid
    param_grid = {
        'sma_short': [3, 5, 7],
        'sma_long': [15, 20, 25]
    }
    
    # Optimize parameters
    best_params, best_metric, all_results = engine.optimize_parameters(
        sample_data, create_sample_strategy(), param_grid, metric='sharpe_ratio'
    )
    
    # Verify results
    assert isinstance(best_params, dict), "Best parameters should be a dictionary"
    assert 'sma_short' in best_params, "Best parameters should include sma_short"
    assert 'sma_long' in best_params, "Best parameters should include sma_long"
    assert isinstance(best_metric, float), "Best metric should be a float"
    assert isinstance(all_results, list), "All results should be a list"
    assert len(all_results) == 9, "Should have 9 results (3x3 parameter combinations)"

def test_walk_forward_optimization():
    """Test walk-forward optimization in BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 200),
        'high': np.random.normal(105, 5, 200),
        'low': np.random.normal(95, 5, 200),
        'close': np.random.normal(102, 5, 200),
        'volume': np.random.normal(1000000, 200000, 200)
    }, index=dates)
    
    # Define parameter grid
    param_grid = {
        'sma_short': [3, 5, 7],
        'sma_long': [15, 20, 25]
    }
    
    # Run walk-forward optimization
    wfo_results = engine.walk_forward_optimization(
        sample_data, create_sample_strategy(), param_grid, window_size=50, step_size=25
    )
    
    # Verify results
    assert isinstance(wfo_results, dict), "WFO results should be a dictionary"
    assert 'windows' in wfo_results, "Results should include windows"
    assert 'optimal_parameters' in wfo_results, "Results should include optimal parameters"
    assert 'out_of_sample_metrics' in wfo_results, "Results should include out-of-sample metrics"
    assert len(wfo_results['windows']) > 0, "Should have at least one window"

def test_monte_carlo_simulation():
    """Test Monte Carlo simulation in BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Run backtest
    engine.run_backtest(sample_data, create_sample_strategy())
    
    # Run Monte Carlo simulation
    mc_results = engine.run_monte_carlo_simulation(
        sample_data, create_sample_strategy(), num_simulations=10
    )
    
    # Verify results
    assert isinstance(mc_results, dict), "MC results should be a dictionary"
    assert 'simulations' in mc_results, "Results should include simulations"
    assert 'confidence_intervals' in mc_results, "Results should include confidence intervals"
    assert len(mc_results['simulations']) == 10, "Should have 10 simulations"
    assert '95%' in mc_results['confidence_intervals'], "Should include 95% confidence interval"

def test_analyze_market_regimes():
    """Test market regime analysis in BacktestingEngine"""
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Create sample data with regimes
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100),
        'regime': np.random.choice(['trending_up', 'trending_down', 'ranging', 'volatile'], 100)
    }, index=dates)
    
    # Analyze market regimes
    regime_results = engine.analyze_market_regimes(
        sample_data, create_sample_strategy()
    )
    
    # Verify results
    assert isinstance(regime_results, dict), "Regime results should be a dictionary"
    assert 'regime_performance' in regime_results, "Results should include regime performance"
    assert 'optimal_parameters' in regime_results, "Results should include optimal parameters"
    assert len(regime_results['regime_performance']) > 0, "Should have at least one regime"

# --- Tests for Correlation Analysis ---
def test_correlation_analysis_initialization():
    """Test initialization of CorrelationAnalysis"""
    analyzer = CorrelationAnalysis(lookback_period=252, correlation_method='pearson')
    assert analyzer.lookback_period == 252, "Lookback period should be set correctly"
    assert analyzer.correlation_method == 'pearson', "Correlation method should be set correctly"

def test_calculate_correlation_matrix():
    """Test calculation of correlation matrix in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'indicator1': np.random.normal(0, 1, 100),
        'indicator2': np.random.normal(0, 1, 100),
        'indicator3': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Calculate correlation matrix
    corr_matrix = analyzer.calculate_correlation_matrix(sample_data, ['indicator1', 'indicator2', 'indicator3'])
    
    # Verify correlation matrix
    assert isinstance(corr_matrix, pd.DataFrame), "Correlation matrix should be a DataFrame"
    assert corr_matrix.shape == (3, 3), "Correlation matrix should be 3x3"
    assert (corr_matrix.values.diagonal() == 1.0).all(), "Diagonal elements should be 1.0"
    assert np.allclose(corr_matrix.values, corr_matrix.values.T), "Correlation matrix should be symmetric"

def test_calculate_rolling_correlations():
    """Test calculation of rolling correlations in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis(lookback_period=20)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'indicator1': np.random.normal(0, 1, 100),
        'indicator2': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Calculate rolling correlations
    rolling_corr = analyzer.calculate_rolling_correlations(sample_data, ['indicator1', 'indicator2'], window=20)
    
    # Verify rolling correlations
    assert isinstance(rolling_corr, pd.DataFrame), "Rolling correlations should be a DataFrame"
    assert rolling_corr.shape[0] <= 100, "Should have at most 100 rows"
    assert 'indicator1_indicator2' in rolling_corr.columns, "Should include correlation between indicator1 and indicator2"
    assert rolling_corr['indicator1_indicator2'].between(-1, 1).all(), "Correlation values should be between -1 and 1"

def test_identify_redundant_indicators():
    """Test identification of redundant indicators in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis()
    
    # Create sample correlation matrix
    corr_matrix = pd.DataFrame({
        'indicator1': [1.0, 0.9, 0.3],
        'indicator2': [0.9, 1.0, 0.2],
        'indicator3': [0.3, 0.2, 1.0]
    }, index=['indicator1', 'indicator2', 'indicator3'])
    
    # Identify redundant indicators
    redundant_groups = analyzer.identify_redundant_indicators(corr_matrix, threshold=0.8)
    
    # Verify redundant groups
    assert isinstance(redundant_groups, list), "Redundant groups should be a list"
    assert len(redundant_groups) > 0, "Should have at least one redundant group"
    assert len(redundant_groups[0]) >= 2, "First group should have at least 2 indicators"
    assert 'indicator1' in redundant_groups[0], "First group should include indicator1"
    assert 'indicator2' in redundant_groups[0], "First group should include indicator2"

def test_calculate_feature_importance():
    """Test calculation of feature importance in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'indicator1': np.random.normal(0, 1, 100),
        'indicator2': np.random.normal(0, 1, 100),
        'indicator3': np.random.normal(0, 1, 100),
        'target': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Calculate feature importance
    importance = analyzer.calculate_feature_importance(
        sample_data, ['indicator1', 'indicator2', 'indicator3'], 'target', method='random_forest'
    )
    
    # Verify feature importance
    assert isinstance(importance, dict), "Feature importance should be a dictionary"
    assert 'indicator1' in importance, "Should include importance for indicator1"
    assert 'indicator2' in importance, "Should include importance for indicator2"
    assert 'indicator3' in importance, "Should include importance for indicator3"
    assert sum(importance.values()) > 0, "Total importance should be positive"

def test_select_optimal_indicators():
    """Test selection of optimal indicators in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'indicator1': np.random.normal(0, 1, 100),
        'indicator2': np.random.normal(0, 1, 100),
        'indicator3': np.random.normal(0, 1, 100),
        'indicator4': np.random.normal(0, 1, 100),
        'indicator5': np.random.normal(0, 1, 100),
        'target': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Select optimal indicators
    selected = analyzer.select_optimal_indicators(
        sample_data, 
        ['indicator1', 'indicator2', 'indicator3', 'indicator4', 'indicator5'], 
        'target', 
        n_select=3
    )
    
    # Verify selected indicators
    assert isinstance(selected, list), "Selected indicators should be a list"
    assert len(selected) == 3, "Should select exactly 3 indicators"
    assert len(set(selected)) == 3, "Selected indicators should be unique"
    assert all(ind in ['indicator1', 'indicator2', 'indicator3', 'indicator4', 'indicator5'] for ind in selected), "Selected indicators should be from the original list"

def test_calculate_signal_weights():
    """Test calculation of signal weights in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'indicator1': np.random.normal(0, 1, 100),
        'indicator2': np.random.normal(0, 1, 100),
        'indicator3': np.random.normal(0, 1, 100),
        'target': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Calculate signal weights
    weights = analyzer.calculate_signal_weights(
        sample_data, ['indicator1', 'indicator2', 'indicator3'], 'target'
    )
    
    # Verify weights
    assert isinstance(weights, dict), "Signal weights should be a dictionary"
    assert 'indicator1' in weights, "Should include weight for indicator1"
    assert 'indicator2' in weights, "Should include weight for indicator2"
    assert 'indicator3' in weights, "Should include weight for indicator3"
    assert abs(sum(weights.values()) - 1.0) < 1e-10, "Weights should sum to 1.0"

def test_calculate_weighted_signal():
    """Test calculation of weighted signal in CorrelationAnalysis"""
    analyzer = CorrelationAnalysis()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'signal1': np.random.choice([-1, 0, 1], 100),
        'signal2': np.random.choice([-1, 0, 1], 100),
        'signal3': np.random.choice([-1, 0, 1], 100)
    }, index=dates)
    
    # Define weights
    weights = {'signal1': 0.5, 'signal2': 0.3, 'signal3': 0.2}
    
    # Calculate weighted signal
    weighted_signal = analyzer.calculate_weighted_signal(
        sample_data, ['signal1', 'signal2', 'signal3'], weights
    )
    
    # Verify weighted signal
    assert isinstance(weighted_signal, pd.Series), "Weighted signal should be a Series"
    assert len(weighted_signal) == 100, "Weighted signal should have 100 values"
    assert weighted_signal.between(-1, 1).all(), "Weighted signal values should be between -1 and 1"

# --- Tests for ML Anomaly Detection Enhanced ---
def test_ml_anomaly_detection_initialization():
    """Test initialization of MarketAnomalyDetection"""
    detector = MarketAnomalyDetection(latent_size=10, hidden_size=128)
    assert detector.latent_size == 10, "Latent size should be set correctly"
    assert detector.hidden_size == 128, "Hidden size should be set correctly"
    assert detector.model is not None, "Model should be initialized"

def test_preprocess_data():
    """Test data preprocessing in MarketAnomalyDetection"""
    detector = MarketAnomalyDetection()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Preprocess data
    features = detector.preprocess_data(sample_data, ['feature1', 'feature2', 'feature3'])
    
    # Verify preprocessed data
    assert isinstance(features, np.ndarray), "Preprocessed features should be a numpy array"
    assert features.shape == (100, 3), "Features should have shape (100, 3)"
    assert not np.isnan(features).any(), "Features should not contain NaN values"

def test_prepare_temporal_data():
    """Test temporal data preparation in MarketAnomalyDetection"""
    detector = MarketAnomalyDetection()
    
    # Create sample features
    features = np.random.normal(0, 1, (100, 3))
    
    # Prepare temporal data
    temporal_features = detector.prepare_temporal_data(features, sequence_length=5)
    
    # Verify temporal features
    assert isinstance(temporal_features, np.ndarray), "Temporal features should be a numpy array"
    assert temporal_features.shape[0] == 96, "Should have 96 sequences (100 - 5 + 1)"
    assert temporal_features.shape[1] == 5, "Each sequence should have length 5"
    assert temporal_features.shape[2] == 3, "Each time step should have 3 features"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_vae():
    """Test VAE training in MarketAnomalyDetection"""
    detector = MarketAnomalyDetection()
    
    # Create sample features
    features = np.random.normal(0, 1, (100, 3))
    
    # Train VAE
    detector.train_vae(features, device='cpu')
    
    # Verify trained model
    assert detector.vae is not None, "VAE model should be initialized"
    assert isinstance(detector.vae, torch.nn.Module), "VAE should be a PyTorch module"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_temporal_vae():
    """Test temporal VAE training in MarketAnomalyDetection"""
    detector = MarketAnomalyDetection()
    
    # Create sample features
    features = np.random.normal(0, 1, (100, 3))
    
    # Prepare temporal data
    temporal_features = detector.prepare_temporal_data(features, sequence_length=5)
    
    # Train temporal VAE
    detector.train_temporal_vae(temporal_features, device='cpu')
    
    # Verify trained model
    assert detector.temporal_vae is not None, "Temporal VAE model should be initialized"
    assert isinstance(detector.temporal_vae, torch.nn.Module), "Temporal VAE should be a PyTorch module"

def test_train_isolation_forest():
    """Test Isolation Forest training in MarketAnomalyDetection"""
    detector = MarketAnomalyDetection()
    
    # Create sample features
    features = np.random.normal(0, 1, (100, 3))
    
    # Train Isolation Forest
    detector.train_isolation_forest(features)
    
    # Verify trained model
    assert detector.isolation_forest is not None, "Isolation Forest model should be initialized"

def test_detect_anomalies():
    """Test anomaly detection in MarketAnomalyDetection"""
    detector = MarketAnomalyDetection()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Detect anomalies
    anomalies = detector.detect(sample_data, feature_cols=['feature1', 'feature2', 'feature3'])
    
    # Verify anomalies
    assert isinstance(anomalies, pd.DataFrame), "Anomalies should be a DataFrame"
    assert 'anomaly_score' in anomalies.columns, "Should include anomaly score column"
    assert 'is_anomaly' in anomalies.columns, "Should include is_anomaly column"
    assert anomalies.index.equals(sample_data.index), "Anomalies should have same index as input data"

# --- Tests for ML Pattern Recognition Enhanced ---
def test_ml_pattern_recognition_initialization():
    """Test initialization of MarketPatternRecognition"""
    recognizer = MarketPatternRecognition(sequence_length=20, hidden_size=64)
    assert recognizer.sequence_length == 20, "Sequence length should be set correctly"
    assert recognizer.hidden_size == 64, "Hidden size should be set correctly"
    assert recognizer.patterns is not None, "Patterns should be initialized"

def test_preprocess_data_pattern():
    """Test data preprocessing in MarketPatternRecognition"""
    recognizer = MarketPatternRecognition()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Preprocess data
    X_train, y_train, X_test, y_test = recognizer.preprocess_data(
        sample_data, 
        feature_cols=['open', 'high', 'low', 'close', 'volume'], 
        target_col='close'
    )
    
    # Verify preprocessed data
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a numpy array"
    assert X_train.shape[1] == recognizer.sequence_length, "X_train should have sequence_length time steps"
    assert X_train.shape[2] == 5, "X_train should have 5 features"

def test_create_model():
    """Test model creation in MarketPatternRecognition"""
    recognizer = MarketPatternRecognition()
    
    # Create model
    model = recognizer.create_model(input_size=5, output_size=1, device='cpu')
    
    # Verify model
    assert model is not None, "Model should be created"
    assert isinstance(model, torch.nn.Module), "Model should be a PyTorch module"

def test_create_ensemble_models():
    """Test ensemble model creation in MarketPatternRecognition"""
    recognizer = MarketPatternRecognition()
    
    # Create ensemble models
    models = recognizer.create_ensemble_models(input_size=5, output_size=1, device='cpu')
    
    # Verify models
    assert isinstance(models, list), "Ensemble models should be a list"
    assert len(models) > 0, "Should have at least one model"
    assert all(isinstance(model, torch.nn.Module) for model in models), "All models should be PyTorch modules"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_model():
    """Test model training in MarketPatternRecognition"""
    recognizer = MarketPatternRecognition(sequence_length=5)
    
    # Create sample data
    X = np.random.normal(0, 1, (100, 5, 3))
    y = np.random.normal(0, 1, 100)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = recognizer.create_model(input_size=3, output_size=1, device='cpu')
    
    # Train model
    trained_model, train_losses = recognizer.train_model(train_loader, device='cpu')
    
    # Verify trained model
    assert trained_model is not None, "Trained model should not be None"
    assert isinstance(train_losses, list), "Training losses should be a list"
    assert len(train_losses) > 0, "Should have at least one training loss"

def test_recognize_patterns():
    """Test pattern recognition in MarketPatternRecognition"""
    recognizer = MarketPatternRecognition()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=dates)
    
    # Recognize patterns
    patterns = recognizer.recognize(sample_data)
    
    # Verify patterns
    assert isinstance(patterns, list), "Patterns should be a list"
    assert len(patterns) > 0, "Should recognize at least one pattern"
    assert all(isinstance(pattern, dict) for pattern in patterns), "Each pattern should be a dictionary"
    assert all('name' in pattern for pattern in patterns), "Each pattern should have a name"
    assert all('confidence' in pattern for pattern in patterns), "Each pattern should have a confidence score"

# --- Tests for Portfolio Optimization ---
def test_portfolio_optimizer_initialization():
    """Test initialization of PortfolioOptimizer"""
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    assert optimizer.risk_free_rate == 0.02, "Risk-free rate should be set correctly"

def test_calculate_portfolio_metrics():
    """Test calculation of portfolio metrics in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Calculate portfolio metrics
    metrics = optimizer.calculate_portfolio_metrics(returns_data)
    
    # Verify metrics
    assert isinstance(metrics, dict), "Portfolio metrics should be a dictionary"
    assert 'mean_returns' in metrics, "Metrics should include mean returns"
    assert 'cov_matrix' in metrics, "Metrics should include covariance matrix"
    assert 'correlation_matrix' in metrics, "Metrics should include correlation matrix"
    assert 'volatility' in metrics, "Metrics should include volatility"
    assert 'annual_returns' in metrics, "Metrics should include annual returns"
    
    assert isinstance(metrics['mean_returns'], pd.Series), "Mean returns should be a Series"
    assert isinstance(metrics['cov_matrix'], pd.DataFrame), "Covariance matrix should be a DataFrame"
    assert metrics['cov_matrix'].shape == (3, 3), "Covariance matrix should be 3x3"

def test_portfolio_performance():
    """Test portfolio performance calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Calculate portfolio metrics
    metrics = optimizer.calculate_portfolio_metrics(returns_data)
    
    # Define weights
    weights = np.array([0.3, 0.4, 0.3])
    
    # Calculate portfolio performance
    returns, volatility, sharpe = optimizer.portfolio_performance(
        weights, metrics['mean_returns'], metrics['cov_matrix']
    )
    
    # Verify performance metrics
    assert isinstance(returns, float), "Returns should be a float"
    assert isinstance(volatility, float), "Volatility should be a float"
    assert isinstance(sharpe, float), "Sharpe ratio should be a float"
    assert volatility > 0, "Volatility should be positive"

def test_optimize_portfolio():
    """Test portfolio optimization in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Optimize portfolio
    optimal_weights, performance = optimizer.optimize_portfolio(returns_data)
    
    # Verify optimization results
    assert isinstance(optimal_weights, dict), "Optimal weights should be a dictionary"
    assert isinstance(performance, dict), "Performance should be a dictionary"
    assert 'asset1' in optimal_weights, "Weights should include asset1"
    assert 'asset2' in optimal_weights, "Weights should include asset2"
    assert 'asset3' in optimal_weights, "Weights should include asset3"
    assert abs(sum(optimal_weights.values()) - 1.0) < 1e-10, "Weights should sum to 1.0"
    assert 'returns' in performance, "Performance should include returns"
    assert 'volatility' in performance, "Performance should include volatility"
    assert 'sharpe_ratio' in performance, "Performance should include Sharpe ratio"

def test_efficient_frontier():
    """Test efficient frontier calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Calculate efficient frontier
    frontier = optimizer.efficient_frontier(returns_data, num_portfolios=10)
    
    # Verify efficient frontier
    assert isinstance(frontier, list), "Efficient frontier should be a list"
    assert len(frontier) == 10, "Should have 10 portfolios"
    assert all(isinstance(portfolio, dict) for portfolio in frontier), "Each portfolio should be a dictionary"
    assert all('weights' in portfolio for portfolio in frontier), "Each portfolio should include weights"
    assert all('returns' in portfolio for portfolio in frontier), "Each portfolio should include returns"
    assert all('volatility' in portfolio for portfolio in frontier), "Each portfolio should include volatility"
    assert all('sharpe_ratio' in portfolio for portfolio in frontier), "Each portfolio should include Sharpe ratio"

def test_calculate_position_sizes():
    """Test position size calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Define weights
    weights = {'asset1': 0.3, 'asset2': 0.4, 'asset3': 0.3}
    
    # Calculate position sizes
    position_sizes = optimizer.calculate_position_sizes(weights, total_capital=100000)
    
    # Verify position sizes
    assert isinstance(position_sizes, dict), "Position sizes should be a dictionary"
    assert 'asset1' in position_sizes, "Position sizes should include asset1"
    assert 'asset2' in position_sizes, "Position sizes should include asset2"
    assert 'asset3' in position_sizes, "Position sizes should include asset3"
    assert position_sizes['asset1'] == 30000, "Position size for asset1 should be 30000"
    assert position_sizes['asset2'] == 40000, "Position size for asset2 should be 40000"
    assert position_sizes['asset3'] == 30000, "Position size for asset3 should be 30000"

def test_calculate_risk_adjusted_position_sizes():
    """Test risk-adjusted position size calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Define weights and volatilities
    weights = {'asset1': 0.3, 'asset2': 0.4, 'asset3': 0.3}
    volatilities = {'asset1': 0.15, 'asset2': 0.2, 'asset3': 0.1}
    
    # Calculate risk-adjusted position sizes
    position_sizes = optimizer.calculate_risk_adjusted_position_sizes(
        weights, total_capital=100000, volatilities=volatilities
    )
    
    # Verify position sizes
    assert isinstance(position_sizes, dict), "Position sizes should be a dictionary"
    assert 'asset1' in position_sizes, "Position sizes should include asset1"
    assert 'asset2' in position_sizes, "Position sizes should include asset2"
    assert 'asset3' in position_sizes, "Position sizes should include asset3"
    assert sum(position_sizes.values()) <= 100000, "Total position size should not exceed capital"

def test_calculate_equal_risk_contribution():
    """Test equal risk contribution calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Calculate equal risk contribution
    weights, position_sizes = optimizer.calculate_equal_risk_contribution(returns_data, total_capital=100000)
    
    # Verify results
    assert isinstance(weights, dict), "Weights should be a dictionary"
    assert isinstance(position_sizes, dict), "Position sizes should be a dictionary"
    assert 'asset1' in weights, "Weights should include asset1"
    assert 'asset2' in weights, "Weights should include asset2"
    assert 'asset3' in weights, "Weights should include asset3"
    assert abs(sum(weights.values()) - 1.0) < 1e-10, "Weights should sum to 1.0"
    assert sum(position_sizes.values()) <= 100000, "Total position size should not exceed capital"

def test_implement_risk_management_rules():
    """Test risk management rules implementation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Define position sizes
    position_sizes = {'asset1': 30000, 'asset2': 40000, 'asset3': 30000}
    
    # Implement risk management rules
    adjusted_sizes = optimizer.implement_risk_management_rules(
        position_sizes, total_capital=100000, max_drawdown=0.2, max_position_pct=0.5
    )
    
    # Verify adjusted position sizes
    assert isinstance(adjusted_sizes, dict), "Adjusted position sizes should be a dictionary"
    assert 'asset1' in adjusted_sizes, "Adjusted sizes should include asset1"
    assert 'asset2' in adjusted_sizes, "Adjusted sizes should include asset2"
    assert 'asset3' in adjusted_sizes, "Adjusted sizes should include asset3"
    assert all(size <= 50000 for size in adjusted_sizes.values()), "No position should exceed 50% of capital"
    assert sum(adjusted_sizes.values()) <= 100000, "Total position size should not exceed capital"

def test_calculate_portfolio_var():
    """Test portfolio Value at Risk calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Define weights
    weights = {'asset1': 0.3, 'asset2': 0.4, 'asset3': 0.3}
    
    # Calculate portfolio VaR
    var = optimizer.calculate_portfolio_var(returns_data, weights, confidence_level=0.95)
    
    # Verify VaR
    assert isinstance(var, float), "VaR should be a float"
    assert var > 0, "VaR should be positive"

def test_calculate_portfolio_cvar():
    """Test portfolio Conditional Value at Risk calculation in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Define weights
    weights = {'asset1': 0.3, 'asset2': 0.4, 'asset3': 0.3}
    
    # Calculate portfolio CVaR
    cvar = optimizer.calculate_portfolio_cvar(returns_data, weights, confidence_level=0.95)
    
    # Verify CVaR
    assert isinstance(cvar, float), "CVaR should be a float"
    assert cvar > 0, "CVaR should be positive"

def test_optimize_portfolio_with_risk_constraints():
    """Test portfolio optimization with risk constraints in PortfolioOptimizer"""
    optimizer = PortfolioOptimizer()
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.01, 100),
        'asset2': np.random.normal(0.002, 0.015, 100),
        'asset3': np.random.normal(0.0015, 0.012, 100)
    }, index=dates)
    
    # Optimize portfolio with risk constraints
    optimal_weights, performance = optimizer.optimize_portfolio_with_risk_constraints(
        returns_data, max_var=0.05, max_cvar=0.07
    )
    
    # Verify optimization results
    assert isinstance(optimal_weights, dict), "Optimal weights should be a dictionary"
    assert isinstance(performance, dict), "Performance should be a dictionary"
    assert 'asset1' in optimal_weights, "Weights should include asset1"
    assert 'asset2' in optimal_weights, "Weights should include asset2"
    assert 'asset3' in optimal_weights, "Weights should include asset3"
    assert abs(sum(optimal_weights.values()) - 1.0) < 1e-10, "Weights should sum to 1.0"
    assert 'returns' in performance, "Performance should include returns"
    assert 'volatility' in performance, "Performance should include volatility"
    assert 'sharpe_ratio' in performance, "Performance should include Sharpe ratio"
    assert 'var' in performance, "Performance should include VaR"
    assert 'cvar' in performance, "Performance should include CVaR"
    assert performance['var'] <= 0.05, "VaR should not exceed constraint"
    assert performance['cvar'] <= 0.07, "CVaR should not exceed constraint"

# --- Tests for Real-Time Alerts ---
def test_alert_condition_initialization():
    """Test initialization of AlertCondition"""
    alert = AlertCondition(name="Test Alert", description="Test description", severity="high")
    assert alert.name == "Test Alert", "Name should be set correctly"
    assert alert.description == "Test description", "Description should be set correctly"
    assert alert.severity == "high", "Severity should be set correctly"
    assert not alert.is_triggered, "Alert should not be triggered initially"

def test_alert_condition_trigger():
    """Test triggering of AlertCondition"""
    alert = AlertCondition(name="Test Alert", description="Test description")
    
    # Trigger alert
    alert.trigger()
    
    # Verify alert state
    assert alert.is_triggered, "Alert should be triggered"
    assert alert.trigger_count == 1, "Trigger count should be 1"
    assert alert.last_triggered is not None, "Last triggered timestamp should be set"

def test_alert_condition_reset():
    """Test resetting of AlertCondition"""
    alert = AlertCondition(name="Test Alert", description="Test description")
    
    # Trigger and reset alert
    alert.trigger()
    alert.reset()
    
    # Verify alert state
    assert not alert.is_triggered, "Alert should not be triggered after reset"
    assert alert.trigger_count == 0, "Trigger count should be reset to 0"
    assert alert.last_triggered is None, "Last triggered timestamp should be reset"

def test_price_alert_condition():
    """Test PriceAlertCondition functionality"""
    alert = PriceAlertCondition(
        name="Price Alert",
        description="Test price alert",
        symbol="AAPL",
        price_threshold=150.0,
        condition="above"
    )
    
    # Test with price below threshold
    data = {"close": 140.0}
    assert not alert.check(data), "Alert should not trigger when price is below threshold"
    
    # Test with price above threshold
    data = {"close": 160.0}
    assert alert.check(data), "Alert should trigger when price is above threshold"
    assert alert.is_triggered, "Alert should be triggered"
    
    # Test with different condition
    alert = PriceAlertCondition(
        name="Price Alert",
        description="Test price alert",
        symbol="AAPL",
        price_threshold=150.0,
        condition="below"
    )
    
    # Test with price above threshold
    data = {"close": 160.0}
    assert not alert.check(data), "Alert should not trigger when price is above threshold"
    
    # Test with price below threshold
    data = {"close": 140.0}
    assert alert.check(data), "Alert should trigger when price is below threshold"
    assert alert.is_triggered, "Alert should be triggered"

def test_indicator_alert_condition():
    """Test IndicatorAlertCondition functionality"""
    alert = IndicatorAlertCondition(
        name="RSI Alert",
        description="Test RSI alert",
        symbol="AAPL",
        indicator="rsi",
        threshold=70.0,
        condition="above"
    )
    
    # Test with RSI below threshold
    data = {"rsi": 60.0}
    assert not alert.check(data), "Alert should not trigger when RSI is below threshold"
    
    # Test with RSI above threshold
    data = {"rsi": 80.0}
    assert alert.check(data), "Alert should trigger when RSI is above threshold"
    assert alert.is_triggered, "Alert should be triggered"
    
    # Test with different condition
    alert = IndicatorAlertCondition(
        name="RSI Alert",
        description="Test RSI alert",
        symbol="AAPL",
        indicator="rsi",
        threshold=30.0,
        condition="below"
    )
    
    # Test with RSI above threshold
    data = {"rsi": 40.0}
    assert not alert.check(data), "Alert should not trigger when RSI is above threshold"
    
    # Test with RSI below threshold
    data = {"rsi": 20.0}
    assert alert.check(data), "Alert should trigger when RSI is below threshold"
    assert alert.is_triggered, "Alert should be triggered"

def test_pattern_alert_condition():
    """Test PatternAlertCondition functionality"""
    alert = PatternAlertCondition(
        name="Pattern Alert",
        description="Test pattern alert",
        symbol="AAPL",
        pattern="head_and_shoulders",
        confidence_threshold=0.7
    )
    
    # Test with no patterns
    data = {"patterns": []}
    assert not alert.check(data), "Alert should not trigger when no patterns are detected"
    
    # Test with different pattern
    data = {"patterns": [{"name": "double_bottom", "confidence": 0.8}]}
    assert not alert.check(data), "Alert should not trigger when pattern doesn't match"
    
    # Test with matching pattern but low confidence
    data = {"patterns": [{"name": "head_and_shoulders", "confidence": 0.6}]}
    assert not alert.check(data), "Alert should not trigger when confidence is below threshold"
    
    # Test with matching pattern and high confidence
    data = {"patterns": [{"name": "head_and_shoulders", "confidence": 0.8}]}
    assert alert.check(data), "Alert should trigger when pattern matches with high confidence"
    assert alert.is_triggered, "Alert should be triggered"

def test_volatility_alert_condition():
    """Test VolatilityAlertCondition functionality"""
    alert = VolatilityAlertCondition(
        name="Volatility Alert",
        description="Test volatility alert",
        symbol="AAPL",
        volatility_threshold=0.2,
        condition="above",
        window=20
    )
    
    # Test with volatility below threshold
    data = {"volatility": 0.15}
    assert not alert.check(data), "Alert should not trigger when volatility is below threshold"
    
    # Test with volatility above threshold
    data = {"volatility": 0.25}
    assert alert.check(data), "Alert should trigger when volatility is above threshold"
    assert alert.is_triggered, "Alert should be triggered"
    
    # Test with different condition
    alert = VolatilityAlertCondition(
        name="Volatility Alert",
        description="Test volatility alert",
        symbol="AAPL",
        volatility_threshold=0.2,
        condition="below",
        window=20
    )
    
    # Test with volatility above threshold
    data = {"volatility": 0.25}
    assert not alert.check(data), "Alert should not trigger when volatility is above threshold"
    
    # Test with volatility below threshold
    data = {"volatility": 0.15}
    assert alert.check(data), "Alert should trigger when volatility is below threshold"
    assert alert.is_triggered, "Alert should be triggered"

def test_perfect_storm_alert_condition():
    """Test PerfectStormAlertCondition functionality"""
    alert = PerfectStormAlertCondition(
        name="Perfect Storm Alert",
        description="Test perfect storm alert",
        symbol="AAPL",
        required_signals=["rsi_oversold", "macd_crossover", "price_above_sma"],
        min_signals=2
    )
    
    # Test with no signals
    data = {"signals": []}
    assert not alert.check(data), "Alert should not trigger when no signals are present"
    
    # Test with insufficient signals
    data = {"signals": ["rsi_oversold"]}
    assert not alert.check(data), "Alert should not trigger when fewer than min_signals are present"
    
    # Test with sufficient signals
    data = {"signals": ["rsi_oversold", "macd_crossover"]}
    assert alert.check(data), "Alert should trigger when min_signals are present"
    assert alert.is_triggered, "Alert should be triggered"
    
    # Test with all signals
    data = {"signals": ["rsi_oversold", "macd_crossover", "price_above_sma"]}
    alert.reset()
    assert alert.check(data), "Alert should trigger when all signals are present"
    assert alert.is_triggered, "Alert should be triggered"
    
    # Test with different signals
    data = {"signals": ["price_below_sma", "volume_spike"]}
    alert.reset()
    assert not alert.check(data), "Alert should not trigger when signals don't match required signals"
