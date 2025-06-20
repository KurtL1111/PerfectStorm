import os
import pandas as pd
from datetime import datetime

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from market_data_retrieval import MarketDataRetriever
from ml_anomaly_detection_enhanced import MarketAnomalyDetection
from ml_clustering_enhanced_completed import PerfectStormClustering
from ml_pattern_recognition_enhanced import MarketPatternRecognition
from dashboard_utils import log_with_timestamp # Centralized logging
# from technical_indicators import TechnicalIndicators # For ensuring all features are present (commented out for now)

# Configuration for model training
TRAIN_SYMBOL = 'BTC/USD'
TRAIN_PERIOD = '2y'  # e.g., '1y', '2y', '5y'
TRAIN_INTERVAL = '1d' # e.g., '1d', '1h', '30m'

# Default feature sets (can be adjusted here)
# These should align with what the models were designed for or expect.
# The ML modules themselves might have internal feature selection/engineering.
DEFAULT_ANOMALY_FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd_line', 'stoch_k', 'cci', 'atr', 'adx', 'cmf', 'mfi', 'bb_upper', 'bb_middle', 'bb_lower']
DEFAULT_CLUSTERING_FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'bb_upper', 'bb_middle', 'bb_lower', 'adx', 'cmf', 'mfi', 'atr', 'vwap']
DEFAULT_PATTERN_FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd_line', 'stoch_k', 'cci', 'bb_width', 'atr', 'adx', 'mfi', 'kst_signal'] # Example, ensure these features are generated
DEFAULT_PATTERN_TARGET_COL = 'future_return_signal' # Example: 1 if next day's close is higher, 0 otherwise

# def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
#     # Helper to add all technical indicators if not present
#     # This ensures that the feature columns defined above are likely to exist.
#     # Assumes TechnicalIndicators class has a method like add_all_indicators
#     # or that individual indicator methods can be called.
#     # A more robust way is to have the ML modules list their required raw columns
#     # and then ensure those + indicators are present.
#     log_with_timestamp("Ensuring all technical indicators are present in the data...")
#     indicator_calculator = TechnicalIndicators()
#     # Create a dummy market_breadth and sentiment if not available for offline training
#     # or adapt calculate_all_indicators to handle their absence.
#     dummy_market_breadth = None # Or load from a default file if applicable
#     dummy_sentiment = None      # Or load from a default file if applicable

#     # Attempt to calculate all indicators.
#     # The calculate_all_indicators method might need adjustment if it strictly requires breadth/sentiment.
#     try:
#         df_with_indicators = indicator_calculator.calculate_all_indicators(df.copy(), market_breadth_data=dummy_market_breadth, sentiment_data=dummy_sentiment)
#         log_with_timestamp("Technical indicators calculation/check complete.")
#         return df_with_indicators
#     except Exception as e:
#         log_with_timestamp(f"Could not ensure all features due to error: {e}. Proceeding with available data.", "WARNING")
#         return df # Return original df if calculation fails

def main_training_logic():
    log_with_timestamp(f"Dedicated model training started for {TRAIN_SYMBOL}, {TRAIN_PERIOD}, {TRAIN_INTERVAL}.")

    # 1. Retrieve Data
    try:
        log_with_timestamp(f"Retrieving market data for {TRAIN_SYMBOL} ({TRAIN_PERIOD}, {TRAIN_INTERVAL})...")
        # Ensure API key is available if MarketDataRetriever requires it for the chosen source
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key and MarketDataRetriever.NEEDS_API_KEY_FOR_DEFAULT_SOURCE: # Assuming a class variable indicates this
             log_with_timestamp("ALPHAVANTAGE_API_KEY not found in environment variables. MarketDataRetriever might fail.", "WARNING")
        retriever = MarketDataRetriever(api_key=api_key)
        stock_data = retriever.get_stock_history(TRAIN_SYMBOL, TRAIN_PERIOD, TRAIN_INTERVAL)
        if stock_data is None or stock_data.empty:
            log_with_timestamp(f"Failed to retrieve stock data for {TRAIN_SYMBOL}. Exiting training.", "ERROR")
            return
        log_with_timestamp(f"Successfully retrieved {len(stock_data)} data points.")
    except Exception as e:
        log_with_timestamp(f"Error during data retrieval: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return

    # 2. Ensure all features are present (especially for default lists)
    # stock_data = ensure_features(stock_data) # Call the helper (Commented out as per instructions)

    # 3. Feature Engineering for Pattern Recognition Target
    try:
        log_with_timestamp("Engineering target variable for Pattern Recognition...")
        if 'close' not in stock_data.columns:
            log_with_timestamp("Critical error: 'close' column not found in stock_data. Cannot proceed.", "ERROR")
            return
        stock_data[DEFAULT_PATTERN_TARGET_COL] = (stock_data['close'].shift(-1) > stock_data['close']).astype(int)
        initial_len = len(stock_data)
        stock_data.dropna(subset=[DEFAULT_PATTERN_TARGET_COL], inplace=True)
        log_with_timestamp(f"Target variable '{DEFAULT_PATTERN_TARGET_COL}' engineered. Dropped {initial_len - len(stock_data)} rows with NaN target.")
        if stock_data.empty:
            log_with_timestamp("Stock data became empty after target engineering. Exiting.", "ERROR")
            return
    except Exception as e:
        log_with_timestamp(f"Error engineering target variable: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return

    # --- Anomaly Detection Training ---
    try:
        log_with_timestamp("Starting Anomaly Detection model training...")
        anomaly_model_path = os.path.join('models', 'Anomaly Detection Models')
        os.makedirs(anomaly_model_path, exist_ok=True) # Ensure directory exists
        anomaly_model = MarketAnomalyDetection(model_path=anomaly_model_path)
        anomaly_model.train_model(df=stock_data.copy(), feature_cols=DEFAULT_ANOMALY_FEATURE_COLS, device='cpu', symbol=TRAIN_SYMBOL, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
        anomaly_model.save_model(symbol=TRAIN_SYMBOL, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
        log_with_timestamp("Anomaly Detection model training finished.")
    except Exception as e:
        log_with_timestamp(f"Error during Anomaly Detection training: {e}", "ERROR")
        import traceback
        traceback.print_exc()

    # --- Clustering Model Training ---
    try:
        log_with_timestamp("Starting Clustering model training...")
        clustering_model_path = os.path.join('models', 'Clustering Models')
        os.makedirs(clustering_model_path, exist_ok=True) # Ensure directory exists
        clustering_model = PerfectStormClustering(model_path=clustering_model_path)
        clustering_model.train(df=stock_data.copy(), feature_columns=DEFAULT_CLUSTERING_FEATURE_COLS, add_technical_features=True, symbol=TRAIN_SYMBOL, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
        clustering_model.save_model(symbol=TRAIN_SYMBOL, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
        log_with_timestamp("Clustering model training finished.")
    except Exception as e:
        log_with_timestamp(f"Error during Clustering training: {e}", "ERROR")
        import traceback
        traceback.print_exc()

    # --- Pattern Recognition Model Training ---
    try:
        log_with_timestamp("Starting Pattern Recognition model training...")
        pattern_model_path = os.path.join('models', 'Pattern Recognition Models')
        os.makedirs(pattern_model_path, exist_ok=True) # Ensure directory exists
        pattern_model = MarketPatternRecognition(model_path=pattern_model_path)

        data_for_pattern = stock_data.copy()
        # Target column should already be in data_for_pattern due to earlier step.
        # Re-check to be absolutely sure, especially if copies were shallow or other issues.
        if DEFAULT_PATTERN_TARGET_COL not in data_for_pattern.columns or data_for_pattern[DEFAULT_PATTERN_TARGET_COL].isnull().all():
             log_with_timestamp(f"Target column {DEFAULT_PATTERN_TARGET_COL} missing or all NaN before pattern recognition. Re-engineering.", "WARNING")
             if 'close' not in data_for_pattern.columns:
                 log_with_timestamp("Critical error: 'close' column not found for pattern target re-engineering. Skipping.", "ERROR")
                 raise ValueError("'close' column needed for target re-engineering is missing.")
             data_for_pattern[DEFAULT_PATTERN_TARGET_COL] = (data_for_pattern['close'].shift(-1) > data_for_pattern['close']).astype(int)
             data_for_pattern.dropna(subset=[DEFAULT_PATTERN_TARGET_COL], inplace=True)

        if data_for_pattern.empty or len(data_for_pattern) < getattr(pattern_model, 'sequence_length', 20) * 2: # Default sequence_length if not set
            log_with_timestamp(f"Not enough data for Pattern Recognition model training ({len(data_for_pattern)} rows). Skipping.", "ERROR")
        else:
            processed_pattern_data = pattern_model.preprocess_data(df=data_for_pattern, feature_cols=DEFAULT_PATTERN_FEATURE_COLS, target_col=DEFAULT_PATTERN_TARGET_COL, add_features=True)
            if processed_pattern_data['train_loader'] is None or len(processed_pattern_data['train_loader'].dataset) == 0:
                log_with_timestamp("Pattern recognition training data is empty after preprocessing. Skipping.", "ERROR")
            else:
                pattern_model.train_model(train_loader=processed_pattern_data['train_loader'], test_loader=processed_pattern_data['test_loader'], device='cpu', symbol=TRAIN_SYMBOL, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
                pattern_model.save_model(symbol=TRAIN_SYMBOL, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
                log_with_timestamp("Pattern Recognition model training finished.")
    except Exception as e:
        log_with_timestamp(f"Error during Pattern Recognition training: {e}", "ERROR")
        import traceback
        traceback.print_exc()

    log_with_timestamp(f"Dedicated model training finished successfully for {TRAIN_SYMBOL}.")

if __name__ == '__main__':
    # Ensure model directories exist (app.py also does this, but good for standalone script)
    base_model_dir = "models"
    model_types = ["Anomaly Detection Models", "Clustering Models", "Pattern Recognition Models"]
    for model_type_dir_namepart in model_types:
        os.makedirs(os.path.join(base_model_dir, model_type_dir_namepart), exist_ok=True)
    os.makedirs("data_cache", exist_ok=True) # If data retriever uses caching

    # Call the main training logic
    main_training_logic()
