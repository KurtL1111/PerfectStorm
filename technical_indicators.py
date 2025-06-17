# --- Universal function for ML feature engineering ---
def add_technical_indicators(df, market_breadth_data=None, sentiment_data=None):
    """
    Add a comprehensive set of technical indicators and market features to a DataFrame for ML pattern recognition.
    This function is designed for use in ML pipelines and will add a wide range of features, including:
    - Price-based technical indicators (MA, RSI, MACD, BB, etc.)
    - Volume and volatility features
    - Breadth and sentiment (if provided)
    - Regime and trend features (if available)
    - Fills all NaNs for ML compatibility

    Parameters:
        df (pd.DataFrame): DataFrame with at least ['open', 'high', 'low', 'close', 'volume'] columns
        market_breadth_data (dict, optional): Market breadth data for ARMS, etc.
        sentiment_data (dict, optional): Sentiment data for bulls/bears ratio, etc.
    Returns:
        pd.DataFrame: DataFrame with added features, ready for ML
    """
    # Defensive copy
    df = df.copy()
    # Use the main indicator calculation pipeline
    ti = TechnicalIndicators
    # Add all standard indicators
    df = ti.calculate_moving_averages(df)
    df = ti.calculate_bollinger_bands(df)
    df = ti.calculate_macd(df)
    df = ti.calculate_rsi(df)
    df = ti.calculate_stochastic(df)
    df = ti.calculate_adx(df)
    df = ti.calculate_cci(df)
    df = ti.calculate_roc(df)
    df = ti.calculate_momentum(df)
    df = ti.calculate_cmf(df)
    df = ti.calculate_mfi(df)
    df = ti.calculate_tsi(df)
    df = ti.calculate_kst(df)
    df = ti.calculate_parabolic_sar(df)
    df = ti.calculate_cd_signal(df)
    # Add breadth and sentiment if available
    if market_breadth_data is not None:
        df = ti.calculate_arms_index(df, market_breadth_data)
    if sentiment_data is not None:
        df = ti.calculate_bulls_bears_ratio(df, sentiment_data)
    # Volatility features
    if 'close' in df.columns:
        df['volatility_5'] = df['close'].rolling(window=5).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
    # Price change features
    if 'close' in df.columns:
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)
    # Regime/trend features (optional, placeholder)
    if 'trend' in df.columns:
        df['trend_lag1'] = df['trend'].shift(1)
    # --- Additional advanced features for ML ---
    # 1. Volatility of returns (short/long window)
    if 'close' in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], 0)
        df['volatility_log_5'] = df['log_return'].rolling(window=5).std()
        df['volatility_log_20'] = df['log_return'].rolling(window=20).std()
    # 2. Price/volume ratios
    if 'close' in df.columns and 'volume' in df.columns:
        df['dollar_volume'] = df['close'] * df['volume']
        df['dollar_vol_ma_20'] = df['dollar_volume'].rolling(window=20).mean()
        df['price_to_ma_20'] = df['close'] / (df['ma_20'] + 1e-6)
    # 3. Momentum regime (bull/bear/neutral)
    if 'ma_20' in df.columns and 'ma_50' in df.columns:
        df['regime'] = np.where(df['ma_20'] > df['ma_50'], 1, np.where(df['ma_20'] < df['ma_50'], -1, 0))
    # 4. Rolling min/max for breakout features
    if 'close' in df.columns:
        df['rolling_max_20'] = df['close'].rolling(window=20).max()
        df['rolling_min_20'] = df['close'].rolling(window=20).min()
        df['breakout_up'] = (df['close'] >= df['rolling_max_20']).astype(int)
        df['breakout_down'] = (df['close'] <= df['rolling_min_20']).astype(int)
    # 5. Lagged returns (autocorrelation features)
    for lag in [1, 2, 3, 5, 10]:
        if 'return_1' in df.columns:
            df[f'return_1_lag{lag}'] = df['return_1'].shift(lag)
    # 6. Rolling skew/kurtosis (tail risk)
    if 'return_1' in df.columns:
        df['skew_20'] = df['return_1'].rolling(window=20).skew()
        df['kurt_20'] = df['return_1'].rolling(window=20).kurt()
    # 7. Drawdown features
    if 'close' in df.columns:
        roll_max = df['close'].cummax()
        df['drawdown'] = (df['close'] - roll_max) / (roll_max + 1e-6)
    # 8. Feature: Relative Strength vs. rolling mean
    if 'close' in df.columns:
        df['rel_strength_20'] = df['close'] / (df['close'].rolling(window=20).mean() + 1e-6)
    # 9. Feature: ATR (Average True Range)
    if all(col in df.columns for col in ['high', 'low', 'close']):
        tr = df[['high', 'low', 'close']].copy()
        tr['tr1'] = tr['high'] - tr['low']
        tr['tr2'] = abs(tr['high'] - tr['close'].shift(1))
        tr['tr3'] = abs(tr['low'] - tr['close'].shift(1))
        df['atr_14'] = tr[['tr1', 'tr2', 'tr3']].max(axis=1).rolling(window=14).mean()
    # 10. Feature: Rolling correlation with market index (if available)
    # (User can join market index data before calling this function)
    # 11. Feature: Sentiment score placeholder (if not present)
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
    # 12. Feature: Time of day, day of week (for intraday/daily seasonality)
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
    if hasattr(df.index, 'dayofweek'):
        df['dayofweek'] = df.index.dayofweek
    # --- Final fill for ML ---
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(0)
    return df
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TechnicalIndicators:
    """Class to calculate all technical indicators for the Perfect Storm dashboard"""

    @staticmethod
    def calculate_all_indicators(stock_data, market_breadth_data, sentiment_data):
        """
        Calculate all technical indicators needed as features or basic plots.
        Signal generation is now primarily handled by the QuantitativeStrategy class
        using the ConvictionScoreCalculator. This function focuses on calculation.

        Parameters:
        - stock_data: DataFrame with stock data (ensure required OHLCV columns).
        - market_breadth_data: Dictionary with market breadth data.
        - sentiment_data: Dictionary with sentiment data.

        Returns:
        - DataFrame with all calculated indicator values. Returns None if stock_data is invalid.
        """
        if stock_data is None or stock_data.empty or not all(col in stock_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
             print("Stock data is missing required OHLCV columns.")
             return None

        df = stock_data.copy()

        # --- Standard Indicators ---
        try: df = TechnicalIndicators.calculate_moving_averages(df)
        except Exception as e: print(f"Error MA: {e}")
        try: df = TechnicalIndicators.calculate_bollinger_bands(df)
        except Exception as e: print(f"Error BB: {e}")
        try: df = TechnicalIndicators.calculate_macd(df)                
        except Exception as e: print(f"Error MACD: {e}")
        try: df = TechnicalIndicators.calculate_rsi(df)                 
        except Exception as e: print(f"Error RSI: {e}")
        try: df = TechnicalIndicators.calculate_stochastic(df)          
        except Exception as e: print(f"Error Stoch: {e}")
        try: df = TechnicalIndicators.calculate_adx(df)                 
        except Exception as e: print(f"Error ADX: {e}")
        try: df = TechnicalIndicators.calculate_cci(df)                 
        except Exception as e: print(f"Error CCI: {e}")
        try: df = TechnicalIndicators.calculate_roc(df)                 
        except Exception as e: print(f"Error ROC: {e}")
        try: df = TechnicalIndicators.calculate_momentum(df)            
        except Exception as e: print(f"Error Mom: {e}")
        try: df = TechnicalIndicators.calculate_cmf(df)                 
        except Exception as e: print(f"Error CMF: {e}")
        try: df = TechnicalIndicators.calculate_mfi(df)                 
        except Exception as e: print(f"Error MFI: {e}")
        try: df = TechnicalIndicators.calculate_tsi(df)                 
        except Exception as e: print(f"Error TSI: {e}")
        try: df = TechnicalIndicators.calculate_kst(df)                 
        except Exception as e: print(f"Error KST: {e}")
        try: df = TechnicalIndicators.calculate_parabolic_sar(df)       
        except Exception as e: print(f"Error SAR: {e}")
        # CD Signal uses BBands, ensure BB is calculated first
        try: df = TechnicalIndicators.calculate_cd_signal(df)           
        except Exception as e: print(f"Error CD Sig: {e}")
        # --- Sentiment/Breadth Indicators ---
        try: df = TechnicalIndicators.calculate_arms_index(df, market_breadth_data)
        except Exception as e: print(f"Error ARMS: {e}")
        try: df = TechnicalIndicators.calculate_bulls_bears_ratio(df, sentiment_data)
        except Exception as e: print(f"Error B/B Ratio: {e}")

        # --- Placeholder for future news/social sentiment score ---
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0 # Initialize placeholder


        # --- Simplified/Raw Signal Triggers (Optional - mainly for conviction input) ---
        # These generate simple triggers; the Conviction Score handles the combination.
        # If using Adaptive Thresholds, these might be calculated *there* and passed.
        # df = TechnicalIndicators.generate_raw_indicator_triggers(df)

        # Final cleanup - Fill initial NaNs resulting from rolling calculations
        # Use forward fill first to carry forward calculations, then backfill for leading NaNs
        # Critical: Do this *after* all indicators are calculated.
        initial_nan_cols = df.columns[df.isna().any()].tolist()
        #print(f"Columns with NaNs before final fill: {initial_nan_cols}")
        #print(df[initial_nan_cols].isna().sum())
        df = df.fillna(method='ffill').fillna(method='bfill')
        #print(f"NaNs remaining after ffill/bfill: {df.isna().sum().sum()}")
        # Fill any remaining NaNs (e.g., if entire column was NaN) with 0 or appropriate value
        if df.isna().sum().sum() > 0:
             print("Warning: NaNs still present after ffill/bfill. Filling with 0.")
             df.fillna(0, inplace=True)


        return df

    @staticmethod
    def calculate_moving_averages(df):
        """Calculate multiple moving averages."""
        ma_periods = [5, 9, 20, 50, 100, 200]
        for period in ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        # Calculate volume MA
        if 'volume' in df.columns:
             df['volume_ma'] = df['volume'].rolling(window=20).mean() # Example: 20-day volume MA
        return df

    # --- (Keep other individual indicator calculation methods: calculate_bollinger_bands, calculate_macd, etc. as they are) ---
    # Make sure they only ADD columns and return the modified df. Example:
    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std=2):
        if 'close' not in df.columns: return df
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        df['bb_std'] = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * num_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * num_std)
        df.drop(['bb_std'], axis=1, inplace=True, errors='ignore')
        return df

    @staticmethod
    def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
        if 'close' not in df.columns: return df
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        # Don't drop ema_fast, ema_slow here if needed elsewhere, otherwise safe to drop
        # df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True, errors='ignore')
        return df

    @staticmethod
    def calculate_rsi(df, period=14):
        if 'close' not in df.columns: return df
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # Use Wilder's smoothing
        # avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        # avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.000001) # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'].fillna(50, inplace=True) # Fill initial NaNs with 50 (neutral)
        return df


    @staticmethod
    def calculate_stochastic(df, k_period=14, d_period=3): # Changed k_period to 14 as common default
        if not all(c in df.columns for c in ['low', 'high', 'close']): return df
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, 0.000001))
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        df['stoch_k'].fillna(50, inplace=True) # Fill initial NaNs
        df['stoch_d'].fillna(50, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_adx(df, period=14):
        if not all(c in df.columns for c in ['high', 'low', 'close']): return df
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()

        plus_di = 100 * (plus_dm_smooth / atr.replace(0, 0.000001))
        minus_di = 100 * (minus_dm_smooth / atr.replace(0, 0.000001))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.000001))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        df['+di'] = plus_di
        df['-di'] = minus_di
        df['adx'] = adx
        df.fillna({'adx': 20, '+di': 20, '-di': 20}, inplace=True) # Fill initial NaNs with neutral values
        return df

    @staticmethod
    def calculate_cci(df, period=20):
        if not all(c in df.columns for c in ['high', 'low', 'close']): return df
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=period).mean()
        tp_dev = abs(tp - tp_sma)
        tp_dev_sma = tp_dev.rolling(window=period).mean()
        df['cci'] = (tp - tp_sma) / (0.015 * tp_dev_sma.replace(0, 0.000001))
        df['cci'].fillna(0, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_roc(df, period=12):
        if 'close' not in df.columns: return df
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period).replace(0, 0.000001)) * 100
        df['roc'].fillna(0, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_momentum(df, period=10):
        if 'close' not in df.columns: return df
        df['momentum'] = df['close'] - df['close'].shift(period)
        df['momentum'].fillna(0, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_cmf(df, period=20):
        if not all(c in df.columns for c in ['high', 'low', 'close', 'volume']): return df
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 0.000001)
        mf_volume = mf_multiplier * df['volume']
        df['cmf'] = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum().replace(0, 0.000001)
        df['cmf'].fillna(0, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_mfi(df, period=14):
        if not all(c in df.columns for c in ['high', 'low', 'close', 'volume']): return df
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_diff = tp.diff()
        raw_money_flow = tp * df['volume']
        positive_flow = np.where(tp_diff > 0, raw_money_flow, 0)
        negative_flow = np.where(tp_diff < 0, raw_money_flow, 0)
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        money_flow_ratio = positive_mf / negative_mf.replace(0, 0.000001)
        df['mfi'] = 100 - (100 / (1 + money_flow_ratio))
        df['mfi'].fillna(50, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_tsi(df, long_period=25, short_period=13, signal_period=7):
        if 'close' not in df.columns: return df
        price_change = df['close'].diff()
        first_smooth = price_change.ewm(span=long_period, adjust=False).mean()
        second_smooth = first_smooth.ewm(span=short_period, adjust=False).mean()
        abs_price_change = abs(price_change)
        abs_first_smooth = abs_price_change.ewm(span=long_period, adjust=False).mean()
        abs_second_smooth = abs_first_smooth.ewm(span=short_period, adjust=False).mean()
        df['tsi'] = 100 * (second_smooth / abs_second_smooth.replace(0, 0.000001))
        df['tsi_signal'] = df['tsi'].ewm(span=signal_period, adjust=False).mean()
        df.fillna({'tsi': 0, 'tsi_signal': 0}, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_kst(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, signal_period=9):
        if 'close' not in df.columns: return df
        roc1 = ((df['close'] - df['close'].shift(r1)) / df['close'].shift(r1).replace(0, 0.000001))
        roc2 = ((df['close'] - df['close'].shift(r2)) / df['close'].shift(r2).replace(0, 0.000001))
        roc3 = ((df['close'] - df['close'].shift(r3)) / df['close'].shift(r3).replace(0, 0.000001))
        roc4 = ((df['close'] - df['close'].shift(r4)) / df['close'].shift(r4).replace(0, 0.000001))
        roc1_sma = roc1.rolling(window=n1).mean()
        roc2_sma = roc2.rolling(window=n2).mean()
        roc3_sma = roc3.rolling(window=n3).mean()
        roc4_sma = roc4.rolling(window=n4).mean()
        df['kst'] = (roc1_sma * 1) + (roc2_sma * 2) + (roc3_sma * 3) + (roc4_sma * 4)
        df['kst_signal'] = df['kst'].rolling(window=signal_period).mean()
        df.fillna({'kst': 0, 'kst_signal': 0}, inplace=True) # Fill initial NaNs
        return df

    @staticmethod
    def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
         # Parabolic SAR calculation needs careful iterative approach, previous version was good
        if not all(c in df.columns for c in ['high', 'low', 'close']): return df
        # Use the implementation from the original file if it works correctly
        # ... (insert iterative SAR calculation logic from original file) ...
        # Initialize SAR
        sar = df['close'].copy() * np.nan
        trend = pd.Series(index=df.index) * np.nan # 1 for uptrend, -1 for downtrend
        af = pd.Series(index=df.index) * np.nan
        ep = pd.Series(index=df.index) * np.nan   # Extreme point

        # Initial values
        sar.iloc[0] = df['low'].iloc[0]
        trend.iloc[0] = 1 # Assume initial uptrend
        ep.iloc[0] = df['high'].iloc[0]
        af.iloc[0] = af_start

        for i in range(1, len(df)):
             prev_sar = sar.iloc[i-1]
             prev_trend = trend.iloc[i-1]
             prev_af = af.iloc[i-1]
             prev_ep = ep.iloc[i-1]
             current_high = df['high'].iloc[i]
             current_low = df['low'].iloc[i]

             if prev_trend == 1: # Previous was uptrend
                  current_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                  if current_low < current_sar: # Trend reversal
                       trend.iloc[i] = -1
                       sar.iloc[i] = prev_ep # SAR reverses to the previous EP
                       ep.iloc[i] = current_low
                       af.iloc[i] = af_start
                  else: # Continue uptrend
                       trend.iloc[i] = 1
                       sar.iloc[i] = current_sar
                       new_ep = max(prev_ep, current_high)
                       ep.iloc[i] = new_ep
                       if new_ep > prev_ep: # If new high, increase AF
                            af.iloc[i] = min(prev_af + af_increment, af_max)
                       else:
                            af.iloc[i] = prev_af
             else: # Previous was downtrend
                  current_sar = prev_sar + prev_af * (prev_ep - prev_sar) # Note: EP < SAR in downtrend
                  if current_high > current_sar: # Trend reversal
                       trend.iloc[i] = 1
                       sar.iloc[i] = prev_ep # SAR reverses to the previous EP
                       ep.iloc[i] = current_high
                       af.iloc[i] = af_start
                  else: # Continue downtrend
                       trend.iloc[i] = -1
                       sar.iloc[i] = current_sar
                       new_ep = min(prev_ep, current_low)
                       ep.iloc[i] = new_ep
                       if new_ep < prev_ep: # If new low, increase AF
                            af.iloc[i] = min(prev_af + af_increment, af_max)
                       else:
                            af.iloc[i] = prev_af

        df['sar'] = sar
        df['trend'] = trend # Keep trend as it might be useful
        df.fillna({'sar': df['close'], 'trend': 1}, inplace=True) # Fill NaNs
        return df


    @staticmethod
    def calculate_cd_signal(df, period=20):
        # CD signal from book requires daily C/D action, might be related to SAR trend or simple close diff
        # If it refers to trend change from Parabolic SAR:
        if 'trend' in df.columns: # 'trend' column from calculate_parabolic_sar
            df['cd_signal_trigger'] = df['trend'].diff().fillna(0).clip(-1, 1) # 1 = broke up, -1 = broke down
            # Smooth the signal?
            df['cd_signal'] = df['cd_signal_trigger'].rolling(window=period).mean() * 100 # Example smoothing
            df.fillna({'cd_signal': 50}, inplace=True)
        else:
            df['cd_signal'] = 50.0 # Neutral default
        return df

    @staticmethod
    def calculate_arms_index(df, market_breadth_data):
        # Assigns a SINGLE value to the whole DataFrame - should probably align by date if possible
        if market_breadth_data:
            adv_i = market_breadth_data.get('advancing_issues', 0)
            dec_i = market_breadth_data.get('declining_issues', 1) # Avoid div by zero
            adv_v = market_breadth_data.get('advancing_volume', 0)
            dec_v = market_breadth_data.get('declining_volume', 1) # Avoid div by zero
            if dec_i > 0 and dec_v > 0:
                 arms = (adv_i / dec_i) / (adv_v / dec_v)
            else:
                 arms = np.nan
            df['arms_index'] = arms # Apply latest value across the df, or merge by date if breadth data has history
        else:
            df['arms_index'] = np.nan
        df['arms_index'] = df['arms_index'].ffill().bfill().fillna(1.0) # Fill NaNs, maybe 1.0 is neutral
        return df

    @staticmethod
    def calculate_bulls_bears_ratio(df, sentiment_data):
        # Similar to ARMS, assign latest ratio or merge if history is available
        if sentiment_data:
            bullish = sentiment_data.get('bullish', 0)
            bearish = sentiment_data.get('bearish', 1) # Avoid div by zero

             # Handle lists - assuming latest value is needed
            if isinstance(bullish, list): bullish = bullish[-1] if bullish else 0
            if isinstance(bearish, list): bearish = bearish[-1] if bearish else 1

             # Convert percentage strings or numbers
            try:
                bullish = float(str(bullish).strip('%')) / 100.0 if '%' in str(bullish) else float(bullish)
                bearish = float(str(bearish).strip('%')) / 100.0 if '%' in str(bearish) else float(bearish)
            except:
                bullish=0; bearish=1 # Fallback


            ratio = bullish / bearish if bearish != 0 else np.inf
            df['bulls_bears_ratio'] = ratio
        else:
            df['bulls_bears_ratio'] = np.nan
        df['bulls_bears_ratio'] = df['bulls_bears_ratio'].ffill().bfill().fillna(1.0) # Fill NaNs, 1.0 maybe neutral
        return df