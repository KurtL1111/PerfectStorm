import pandas as pd
import numpy as np
from strategy_logic import ConvictionScoreCalculator
from technical_indicators import TechnicalIndicators # Assuming this class now primarily calculates indicators
from ml_pattern_recognition_enhanced import MarketPatternRecognition
from ml_clustering_enhanced_completed import PerfectStormClustering
from ml_anomaly_detection_enhanced import MarketAnomalyDetection
from market_regime_detection_completed import MarketRegimeDetection
from correlation_analysis import CorrelationAnalysis
from adaptive_thresholds_enhanced import EnhancedAdaptiveThresholds
import traceback

class QuantitativeStrategy:
    """
    Encapsulates the logic for the "Perfect Storm" quantitative strategy.
    Takes market data and various analysis outputs to generate a conviction score
    and trading signals.
    """

    def __init__(self, symbol, period, interval, conviction_weights=None, correlation_features=None, ml_feature_cols=None):
        """
        Initialize the strategy engine.

        Args:
            symbol (str): Stock symbol.
            period (str): Data period.
            interval (str): Data interval.
            conviction_weights (dict, optional): Weights for ConvictionScoreCalculator. Defaults to None.
            correlation_features (list, optional): Features for correlation analysis. Defaults to None.
            ml_feature_cols (list, optional): Features for ML models. Defaults to None.
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.ml_pattern_report_cache = None
        self.ml_clusters_report_cache = None # This will store the FULL output of generate_clustering_report
        self.ml_anomaly_report_cache = None
        self.regime_report_cache = None
        self.correlation_report_cache = None

        # Default feature sets if none provided
        self.correlation_features = correlation_features or ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci']
        self.ml_feature_cols = ml_feature_cols or ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'bb_upper', 'bb_lower', 'adx']

        # --- Initialize Analysis Modules ---
        # NOTE: Consider loading pre-trained models where applicable based on symbol/period/interval
        # This initialization assumes training might happen if no model is loaded.
        print("Initializing Strategy Components...")
        self.indicators_calculator = TechnicalIndicators() # Static methods, no init needed? Check class design.
        self.pattern_model = MarketPatternRecognition()
        self.clustering_model = PerfectStormClustering()
        self.anomaly_model = MarketAnomalyDetection()
        self.regime_model = MarketRegimeDetection()
        self.correlation_model = CorrelationAnalysis() # Loads/calculates as needed
        self.threshold_model = EnhancedAdaptiveThresholds() # Adaptive thresholds
        self.conviction_calculator = ConvictionScoreCalculator(weights=conviction_weights) # The core logic

        self.latest_feature_importance = pd.Series(dtype=float)
        print("Strategy Components Initialized.")

    def _get_safe_report_data(self, report, key, default=None):
        """Safely retrieve data from report dictionaries."""
        if report and isinstance(report, dict) and key in report:
            return report[key]
        return default

    def _get_latest_values(self, df, col_name, default=0):
        """Safely get the latest value from a DataFrame column."""
        if col_name in df.columns and not df[col_name].empty:
            return df[col_name].iloc[-1]
        return default

    def _calculate_features_and_factors(self, df_market_data, market_breadth, sentiment):
        """
        Calculate all technical indicators, ML predictions, regimes, etc.
        """
        print("Calculating base indicators...")
        # 1. Calculate Technical Indicators
        df_features = self.indicators_calculator.calculate_all_indicators(df_market_data, market_breadth, sentiment)
        if df_features is None or df_features.empty:
            print("Indicator calculation failed.")
            return None

        print("Running Market Regime Detection...")
        # 2. Detect Market Regime
        try:
            df_regime_features = self.regime_model.extract_regime_features(df_market_data)
            df_regimes_classified = self.regime_model.detect_regimes(df_regime_features)
            self.regime_report_cache = self.regime_model.generate_regime_report(df=df_regime_features, symbol=self.symbol, period=self.period)

            # Add regime info to main feature DataFrame
            # Need to align indices carefully
            if 'regime' in df_regimes_classified.columns:
                df_features = df_features.join(df_regimes_classified['regime'], how='left')
                df_features['regime'] = df_features['regime'].ffill().bfill() # Propagate regimes
                 # Add regime name based on mapping in the model
                regime_labels = self._get_safe_report_data(self.regime_report_cache, 'regime_labels', {})
                df_features['market_regime'] = df_features['regime'].map(regime_labels).fillna('unknown') # Add named regime
            else:
                 print("Warning: 'regime' column not found after regime detection.")
                 df_features['market_regime'] = 'unknown'

            current_regime = self._get_safe_report_data(self.regime_report_cache, 'current_regime', {'regime_name': 'unknown'})

        except Exception as e:
            print(f"Error during Market Regime Detection: {e}")
            traceback.print_exc()
            df_features['market_regime'] = 'unknown'
            current_regime = {'regime_name': 'unknown'}


        # 3. Run ML Pattern Recognition
        print("Running ML Pattern Recognition...")
        try:
             # Prepare data for ML
            df_pattern = df_features.copy()
            df_pattern['target'] = df_pattern['close'].pct_change().shift(-1).fillna(0)
            df_pattern['target'] = (df_pattern['target'] > 0).astype(int) # Example target
            valid_ml_features = [col for col in self.ml_feature_cols if col in df_pattern.columns]
            self.ml_pattern_report_cache = self.pattern_model.generate_pattern_report(df_pattern, valid_ml_features, 'target', symbol=self.symbol, period=self.period, interval=self.interval)
            # Safely join pattern predictions
            pattern_results = self._get_safe_report_data(self.ml_pattern_report_cache, 'results')
            if pattern_results is not None and not pattern_results.empty:
                df_features = df_features.join(pattern_results[['prediction', 'probability']].add_prefix('ml_pattern_'), how='left')
                df_features['ml_pattern_probability'].fillna(0.5, inplace=True) # Neutral probability if missing
                # Derive confidence from probability: scale -1 to 1 based on 0.5 neutral
                df_features['ml_pattern_confidence'] = (df_features['ml_pattern_probability'] - 0.5) * 2
            else:
                df_features['ml_pattern_confidence'] = 0.0 # Neutral if no pattern results

        except Exception as e:
            print(f"Error during ML Pattern Recognition: {e}")
            traceback.print_exc()
            df_features['ml_pattern_confidence'] = 0.0 # Default neutral score

        # 4. Run ML Clustering
        print("Running ML Clustering...")
        try:
            df_cluster_features = df_features.copy()
            valid_ml_features = [col for col in self.ml_feature_cols if col in df_cluster_features.columns]
             # Ensure target is calculated for potential use inside clustering/reporting, but not strictly needed for predict
            df_cluster_features['target'] = df_cluster_features['close'].pct_change().shift(-1).fillna(0)
            df_cluster_features['target'] = (df_cluster_features['target'] > 0).astype(int)
            print("Invoking clustering_model.generate_clustering_report from QuantitativeStrategy...")
            self.ml_clusters_report_cache = self.clustering_model.generate_clustering_report(
                df_cluster_features,
                valid_ml_features, # Ensure this is a list of feature NAMES
                price_col='close', # Pass explicitly if needed by generate_clustering_report
                symbol=self.symbol,
                period=self.period,
                interval=self.interval
            )
            cluster_results = self._get_safe_report_data(self.ml_clusters_report_cache, 'results')
            if cluster_results is not None and not cluster_results.empty:
                df_features = df_features.join(cluster_results[['cluster', 'anomaly_score']].add_prefix('ml_cluster_'), how='left')
                # Potential: Map cluster IDs to a 'cluster_modifier' score for conviction calc
            else:
                df_features['ml_cluster_cluster'] = -1 # Default invalid cluster
                df_features['ml_cluster_anomaly_score'] = 0.0

        except Exception as e:
            print(f"Error during ML Clustering: {e}")
            traceback.print_exc()
            df_features['ml_cluster_cluster'] = -1
            df_features['ml_cluster_anomaly_score'] = 0.0

        # 5. Run Anomaly Detection
        print("Running Anomaly Detection...")
        try:
            df_anomaly_features = df_features.copy()
            valid_ml_features = [col for col in self.ml_feature_cols if col in df_anomaly_features.columns]
             # Train might be needed here if no model loaded; assume predict for now
            print("-" * 20 + " ANOMALY DETECTION DEBUG " + "-" * 20)
            print(f"Features intended for Anomaly Detection: {valid_ml_features}") # Check the list passed
            print(f"Number of features intended: {len(valid_ml_features)}")
            # Make sure the dataframe has these columns before passing
            print(f"Columns available in df_anomaly_features: {df_anomaly_features.columns.tolist()}")
            # Check shape *just before* the call
            print(f"Shape of df_anomaly_features before report gen: {df_anomaly_features.shape}")

            ml_anomaly_report = self.anomaly_model.generate_anomaly_report(
                df_anomaly_features, valid_ml_features, symbol=self.symbol, period=self.period, interval=self.interval
            )
            self.ml_anomaly_report_cache = ml_anomaly_report
            anomaly_scores_df = self.anomaly_model.detect_anomalies(df_anomaly_features, valid_ml_features)

            if isinstance(anomaly_scores_df, np.ndarray): # Check if it returned an array instead of df
                anomaly_scores_df = pd.Series(anomaly_scores_df, index=df_anomaly_features.index[:len(anomaly_scores_df)]) # Assign index

            df_features['anomaly_score'] = anomaly_scores_df # Directly assign Series/Array

            # Handle potential NaN from sequence padding
            df_features['anomaly_score'] = df_features['anomaly_score'].ffill().bfill().fillna(0) # fill NaNs

        except Exception as e:
            print(f"Error during Anomaly Detection: {e}")
            traceback.print_exc()
            df_features['anomaly_score'] = 0.0 # Default neutral anomaly score

        # 6. Run Correlation Analysis (Less frequent - potentially load from cache/file)
        print("Running Correlation Analysis (or loading)...")
        try:
            valid_corr_features = [col for col in self.correlation_features if col in df_features.columns]
            # Check for 'returns' column, needed for feature importance calculation
            if 'returns' not in df_features.columns:
                df_features['returns'] = df_features['close'].pct_change()
            # Handle NaNs created by pct_change
            df_features.fillna({'returns': 0}, inplace=True)

            # Check if features and target exist
            if set(valid_corr_features).issubset(df_features.columns) and 'returns' in df_features.columns:
                 # Load existing report if available, otherwise generate
                loaded = self.correlation_model.load_correlation_analysis(symbol=self.symbol, period=self.period, interval=self.interval)
                if not loaded or not self.correlation_model.feature_importance_history:
                    correlation_report = self.correlation_model.generate_correlation_report(
                        df_features,
                        valid_corr_features,
                        target_col='returns', # Or appropriate target
                        display_dashboard=False, # Don't display automatically
                        symbol=self.symbol, period=self.period, interval=self.interval
                    )
                    self.correlation_report_cache = correlation_report
                    self.latest_feature_importance = self._get_safe_report_data(correlation_report, 'feature_importance', pd.Series(dtype=float))
                else:
                    # Get latest feature importance from loaded history
                    latest_fi_timestamp = max(self.correlation_model.feature_importance_history.keys())
                    self.latest_feature_importance = self.correlation_model.feature_importance_history[latest_fi_timestamp].get('importance', pd.Series(dtype=float))
                    # If the correlation model has a method to get the latest report, use it
                    if hasattr(self.correlation_model, 'get_latest_report'):
                        self.correlation_report_cache = self.correlation_model.get_latest_report()
                    else:
                        self.correlation_report_cache = None
            else:
                print("Warning: Missing required columns for correlation analysis.")
                self.latest_feature_importance = pd.Series(dtype=float)


        except Exception as e:
            print(f"Error during Correlation Analysis: {e}")
            traceback.print_exc()
            self.latest_feature_importance = pd.Series(dtype=float) # Ensure it's initialized

        # --- Add Adaptive Threshold Based Signals ---
        print("Calculating Adaptive Threshold Signals...")
        # Example: Calculate for RSI and Stochastic, add results as new columns
        indicator_cols_for_adaptive = ['rsi', 'stoch_k'] # Add others as needed
        for indicator in indicator_cols_for_adaptive:
             if indicator in df_features.columns:
                 try:
                     # Get thresholds (may need regime info passed)
                     thresholds = self.threshold_model.get_adaptive_thresholds(df_features, indicator) # Pass full df with regime if needed by method
                     lower_thresh = thresholds.get('lower')
                     upper_thresh = thresholds.get('upper')

                     # Generate adaptive signals based on thresholds
                     if lower_thresh is not None:
                          df_features[f'adaptive_buy_{indicator}'] = (df_features[indicator] < lower_thresh).astype(int)
                     if upper_thresh is not None:
                          df_features[f'adaptive_sell_{indicator}'] = (df_features[indicator] > upper_thresh).astype(int)

                 except Exception as e:
                     print(f"Error calculating adaptive threshold for {indicator}: {e}")
                     df_features[f'adaptive_buy_{indicator}'] = 0
                     df_features[f'adaptive_sell_{indicator}'] = 0
             else:
                 print(f"Warning: Indicator {indicator} not found for adaptive threshold calculation.")


        # Ensure essential columns exist before returning
        required_cols = ['market_regime', 'ml_pattern_confidence', 'anomaly_score']
        for col in required_cols:
            if col not in df_features:
                 print(f"Warning: Essential column '{col}' missing after feature calculation. Setting default.")
                 if col == 'market_regime':
                     df_features[col] = 'unknown'
                 else:
                    df_features[col] = 0.0


        return df_features, current_regime

    def _calculate_conviction_score(self, df_features):
        """
        Calculate the conviction score using the calculator.
        """
        print("Calculating Conviction Score...")

        # Prepare data for Conviction Score Calculation
        # Map indicator columns for the calculator
        # Use the new 'adaptive_buy/sell_*' columns generated earlier
        adaptive_buy_cols = [col for col in df_features.columns if col.startswith('adaptive_buy_')]
        adaptive_sell_cols = [col for col in df_features.columns if col.startswith('adaptive_sell_')]

        # Sum up individual adaptive signals (simple approach)
        # A more sophisticated approach might weight these based on the indicator
        df_features['individual_buy_signals_adaptive'] = df_features[adaptive_buy_cols].sum(axis=1)
        df_features['individual_sell_signals_adaptive'] = df_features[adaptive_sell_cols].sum(axis=1)

        # --- Update Weights Based on Feature Importance (Example) ---
        # You might want a more sophisticated weighting scheme
        base_weights = self.conviction_calculator.weights.copy() # Get default/initial weights
        if self.latest_feature_importance is not None and not self.latest_feature_importance.empty:
             # Example: Give higher weight to adaptive signals if important indicators triggered them
             # This is simplistic - needs refinement based on which indicators are important
             avg_importance = self.latest_feature_importance.mean() # Need a better baseline
             # Give a slight boost to adaptive signals weight based on overall importance?
             importance_factor = 1.0 + (avg_importance * 0.1) # Small boost
             base_weights['adaptive_signals'] = base_weights.get('adaptive_signals', 0.4) * importance_factor # Adjust base weight

        self.conviction_calculator.weights = base_weights # Use potentially adjusted weights

        # --- Regime-Based Weight Adjustment (Example) ---
        current_regime = df_features['market_regime'].iloc[-1] if 'market_regime' in df_features.columns else 'unknown'
        adjusted_weights = self.conviction_calculator.weights.copy()
        if current_regime == 'trending_up':
            adjusted_weights['market_regime'] = adjusted_weights.get('market_regime', 0.3) * 1.2 # Boost regime impact
            adjusted_weights['adaptive_signals'] = adjusted_weights.get('adaptive_signals', 0.4) * 1.1 # Trend signals more important
        elif current_regime == 'trending_down':
            adjusted_weights['market_regime'] = adjusted_weights.get('market_regime', 0.3) * 1.2 # Boost regime impact (negative score)
            adjusted_weights['adaptive_signals'] = adjusted_weights.get('adaptive_signals', 0.4) * 1.1
        elif current_regime == 'ranging':
             adjusted_weights['adaptive_signals'] = adjusted_weights.get('adaptive_signals', 0.4) * 0.8 # Oscillators (part of signals) might be more relevant, but trend less so
             adjusted_weights['ml_pattern'] = adjusted_weights.get('ml_pattern', 0.15) * 1.1 # Patterns like reversals might be key
        elif current_regime == 'volatile':
            # Lower weights generally, maybe increase anomaly weight
            for k in adjusted_weights: adjusted_weights[k] *= 0.8
            adjusted_weights['anomaly_score'] = adjusted_weights.get('anomaly_score', 0.15) * 1.5 # Be more sensitive to anomalies
        # --- End Regime Adjustment Example ---

        # Ensure required columns for calculator exist, use safe defaults if not
        required_input_cols = {
             'individual_buy_signals_adaptive': 0,
             'individual_sell_signals_adaptive': 0,
             'market_regime': 'unknown',
             'ml_pattern_confidence': 0.0,
             'anomaly_score': 0.0
        }
        for col, default_val in required_input_cols.items():
             if col not in df_features:
                 print(f"Warning: Required column '{col}' for conviction score not found. Using default.")
                 df_features[col] = default_val


        # Set calculator weights for this run (e.g., based on regime)
        self.conviction_calculator.weights = adjusted_weights
        print(f"Using Conviction Weights for regime '{current_regime}': {self.conviction_calculator.weights}")

        # Calculate the conviction score
        df_with_conviction = self.conviction_calculator.calculate_score(df_features)
        return df_with_conviction

    def generate_signals(self, df_with_conviction, buy_threshold=0.6, sell_threshold=-0.6, close_threshold_high=0.2, close_threshold_low=-0.2):
        """
        Generate final Buy/Sell signals based on the conviction score.
        Now uses regime-dependent thresholds for each historical day.
        """
        df_signals = df_with_conviction.copy()
        df_signals['buy_signal'] = 0
        df_signals['sell_signal'] = 0

        # Use per-row thresholds if present, else fallback to static
        buy_thresholds = df_signals['buy_threshold'] if 'buy_threshold' in df_signals else buy_threshold
        sell_thresholds = df_signals['sell_threshold'] if 'sell_threshold' in df_signals else sell_threshold

        score = df_signals['conviction_score']
        prev_score = score.shift(1)

        # --- Simple Threshold Crossing, now per-row ---
        for i in range(len(df_signals)):
            if isinstance(buy_thresholds, (pd.Series, pd.DataFrame)):
                bt = buy_thresholds.iloc[i]
            else:
                bt = buy_thresholds
            if isinstance(sell_thresholds, (pd.Series, pd.DataFrame)):
                st = sell_thresholds.iloc[i]
            else:
                st = sell_thresholds
            # Buy signal: score crosses above buy threshold
            if score.iloc[i] >= bt and prev_score.iloc[i] < bt:
                df_signals.at[df_signals.index[i], 'buy_signal'] = 1
            # Sell signal: score crosses below sell threshold
            if score.iloc[i] <= st and prev_score.iloc[i] > st:
                df_signals.at[df_signals.index[i], 'sell_signal'] = 1
        # --- Additional Conditions ---
        # Optional: Add more complex conditions based on regime or other factors
            # --- Optional: Exit signal based on returning to neutral ---
            # Exit Long: Score crosses *below* close_threshold_high (from above)
            #df_signals.loc[(score < close_threshold_high) & (prev_score >= close_threshold_high), 'sell_signal'] = 1 # Might conflict, needs careful state management

            # Exit Short: Score crosses *above* close_threshold_low (from below)
            #df_signals.loc[(score > close_threshold_low) & (prev_score <= close_threshold_low), 'buy_signal'] = 1 # Might conflict

        # --- Prevent signal spamming ---
        df_signals.loc[df_signals['buy_signal'] == 1, 'sell_signal'] = 0

        # Add position state management to prevent buy/sell on the same day or consecutive signals
        position = 0 # 0 = flat, 1 = long, -1 = short
        signals_final_buy = []
        signals_final_sell = []

        for i in range(len(df_signals)):
            buy = df_signals['buy_signal'].iloc[i]
            sell = df_signals['sell_signal'].iloc[i]
            final_buy = 0
            final_sell = 0

            # Evaluate signals based on current position state
            if buy == 1 and position <= 0: # Buy to enter long or close short
                final_buy = 1
                position = 1
            elif sell == 1 and position >= 0: # Sell to exit long or enter short
                final_sell = 1
                position = -1
            # Add logic to close positions if score returns to neutral zone if desired
            elif score.iloc[i] < close_threshold_high and position == 1:
                final_sell = 1
                position = 0
            elif score.iloc[i] > close_threshold_low and position == -1:
                final_buy = 1
                position = 0

            signals_final_buy.append(final_buy)
            signals_final_sell.append(final_sell)

        df_signals['buy_signal'] = signals_final_buy
        df_signals['sell_signal'] = signals_final_sell

        print(f"Generated {df_signals['buy_signal'].sum()} Buy signals and {df_signals['sell_signal'].sum()} Sell signals.")
        return df_signals


    def run_strategy(self, df_market_data, market_breadth, sentiment):
        """
        Execute the full strategy calculation process.
        """
        # 1. Calculate Features and Factors
        df_features, current_regime = self._calculate_features_and_factors(df_market_data, market_breadth, sentiment)
        if df_features is None:
            print("Strategy execution failed during feature calculation.")
            return pd.DataFrame(), {}, None # Return empty df, empty report dict, and None regime

        # 2. Calculate Conviction Score
        df_with_conviction = self._calculate_conviction_score(df_features)

        # 3. Assign regime-dependent thresholds for each day
        regime_threshold_map = {
            'volatile':   {'buy': 0.65, 'sell': -0.6},
            'ranging':    {'buy': 0.5, 'sell': -0.5},
            'trending_up':   {'buy': 0.5, 'sell': -0.6},
            'trending_down': {'buy': 0.6, 'sell': -0.6},
            'unknown':    {'buy': 0.6, 'sell': -0.6},
        }

        # Default to current regime for the last row
        current_regime_name = current_regime.get('regime_name', 'unknown')

        # Assign thresholds per row based on that row's regime
        def get_thresholds_for_row(regime):
            t = regime_threshold_map.get(regime, regime_threshold_map['unknown'])
            return t['buy'], t['sell']

        buy_thresholds = []
        sell_thresholds = []
        for idx, row in df_with_conviction.iterrows():
            regime = row['market_regime'] if 'market_regime' in row and pd.notnull(row['market_regime']) else 'unknown'
            # For the last row, use current regime (in case of real-time)
            if idx == df_with_conviction.index[-1]:
                regime = current_regime_name
            buy_t, sell_t = get_thresholds_for_row(regime)
            buy_thresholds.append(buy_t)
            sell_thresholds.append(sell_t)
        df_with_conviction['buy_threshold'] = buy_thresholds
        df_with_conviction['sell_threshold'] = sell_thresholds

        # 4. Generate Final Signals using per-row thresholds
        df_final = self.generate_signals(df_with_conviction)

        # 5. Consolidate Reports (Example - Adapt as needed)
        strategy_report = {
            "latest_conviction_score": df_final['conviction_score'].iloc[-1] if not df_final.empty else None,
            "current_regime": current_regime.get('regime_name', 'unknown'),
            "latest_anomaly_score": df_final['anomaly_score'].iloc[-1] if 'anomaly_score' in df_final and not df_final.empty else None,
            "latest_pattern_confidence": df_final['ml_pattern_confidence'].iloc[-1] if 'ml_pattern_confidence' in df_final and not df_final.empty else None,
            "feature_importance_summary": self.latest_feature_importance.nlargest(5).to_dict() if self.latest_feature_importance is not None and not self.latest_feature_importance.empty else {}
            # Add more key results from individual reports as needed
        }

        return df_final, strategy_report, current_regime