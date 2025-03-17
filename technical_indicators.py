"""
Technical indicators module for Perfect Storm Dashboard

This module implements all the technical indicators needed for the
Perfect Storm investment strategy developed by John R. Connelley.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TechnicalIndicators:
    """Class to calculate all technical indicators for the Perfect Storm dashboard"""
    
    @staticmethod
    def calculate_all_indicators(stock_data, market_breadth_data, sentiment_data):
        """
        Calculate all technical indicators for the Perfect Storm dashboard
        
        Parameters:
        - stock_data: DataFrame with stock data
        - market_breadth_data: Dictionary with market breadth data
        - sentiment_data: Dictionary with sentiment data
        
        Returns:
        - DataFrame with all indicators
        """
        # Check if stock data is available
        if stock_data is None:
            return None
            
        # Make a copy of the stock data
        df = stock_data.copy()
        
        # Calculate moving averages
        ma_periods = [5, 9, 20, 50, 100, 200]
        for period in ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Calculate Bollinger Bands
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        
        # Calculate MACD
        df = TechnicalIndicators.calculate_macd(df)
        
        # Calculate RSI
        df = TechnicalIndicators.calculate_rsi(df)
        
        # Calculate Stochastic Oscillator
        df = TechnicalIndicators.calculate_stochastic(df)
        
        # Calculate ADX
        df = TechnicalIndicators.calculate_adx(df)
        
        # Calculate CCI
        df = TechnicalIndicators.calculate_cci(df)
        
        # Calculate ROC
        df = TechnicalIndicators.calculate_roc(df)
        
        # Calculate Momentum
        df = TechnicalIndicators.calculate_momentum(df)
        
        # Calculate Chaikin Money Flow
        df = TechnicalIndicators.calculate_cmf(df)
        
        # Calculate Money Flow Index
        df = TechnicalIndicators.calculate_mfi(df)
        
        # Calculate True Strength Index
        df = TechnicalIndicators.calculate_tsi(df)
        
        # Calculate Know Sure Thing
        df = TechnicalIndicators.calculate_kst(df)
        
        # Calculate Parabolic SAR
        df = TechnicalIndicators.calculate_parabolic_sar(df)
        
        # Calculate C/D Signal
        df = TechnicalIndicators.calculate_cd_signal(df)
        
        # Calculate ARMS Index
        df = TechnicalIndicators.calculate_arms_index(df, market_breadth_data)
        
        # Calculate Bulls vs Bears Ratio
        df = TechnicalIndicators.calculate_bulls_bears_ratio(df, sentiment_data)
        
        # Generate buy/sell signals
        df = TechnicalIndicators.generate_buy_sell_signals(df)
        
        return df
    
    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Parameters:
        - df: DataFrame with stock data
        - window: Window size for moving average (default: 20)
        - num_std: Number of standard deviations (default: 2)
        
        Returns:
        - DataFrame with Bollinger Bands
        """
        # Calculate middle band (20-day SMA)
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * num_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * num_std)
        
        # Clean up temporary columns
        df = df.drop(['bb_std'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Parameters:
        - df: DataFrame with stock data
        - fast_period: Fast EMA period (default: 12)
        - slow_period: Slow EMA period (default: 26)
        - signal_period: Signal EMA period (default: 9)
        
        Returns:
        - DataFrame with MACD
        """
        # Calculate fast and slow EMAs
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        
        # Calculate signal line
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        
        # Clean up temporary columns
        df = df.drop(['ema_fast', 'ema_slow'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """
        Calculate RSI (Relative Strength Index)
        
        Parameters:
        - df: DataFrame with stock data
        - period: RSI period (default: 14)
        
        Returns:
        - DataFrame with RSI
        """
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        
        # Calculate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Calculate average gains and losses
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        
        # Calculate RS
        df['rs'] = df['avg_gain'] / df['avg_loss']
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Clean up temporary columns
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_stochastic(df, k_period=5, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Parameters:
        - df: DataFrame with stock data
        - k_period: %K period (default: 5)
        - d_period: %D period (default: 3)
        
        Returns:
        - DataFrame with Stochastic Oscillator
        """
        # Calculate %K
        df['lowest_low'] = df['low'].rolling(window=k_period).min()
        df['highest_high'] = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Clean up temporary columns
        df = df.drop(['lowest_low', 'highest_high'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_adx(df, period=14):
        """
        Calculate ADX (Average Directional Movement Index)
        
        Parameters:
        - df: DataFrame with stock data
        - period: ADX period (default: 14)
        
        Returns:
        - DataFrame with ADX
        """
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        # Calculate +DM and -DM
        df['+dm'] = 0
        df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), '+dm'] = df['up_move']
        
        df['-dm'] = 0
        df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), '-dm'] = df['down_move']
        
        # Calculate smoothed +DM, -DM, and TR
        df['+dm_smooth'] = df['+dm'].rolling(window=period).sum()
        df['-dm_smooth'] = df['-dm'].rolling(window=period).sum()
        df['tr_smooth'] = df['tr'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        df['+di'] = 100 * (df['+dm_smooth'] / df['tr_smooth'])
        df['-di'] = 100 * (df['-dm_smooth'] / df['tr_smooth'])
        
        # Calculate DX
        df['dx'] = 100 * (abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']))
        
        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr', 'up_move', 'down_move', '+dm', '-dm', '+dm_smooth', '-dm_smooth', 'tr_smooth', 'dx'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_cci(df, period=20):
        """
        Calculate CCI (Commodity Channel Index)
        
        Parameters:
        - df: DataFrame with stock data
        - period: CCI period (default: 20)
        
        Returns:
        - DataFrame with CCI
        """
        # Calculate typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate SMA of typical price
        df['tp_sma'] = df['tp'].rolling(window=period).mean()
        
        # Calculate mean deviation
        df['tp_dev'] = abs(df['tp'] - df['tp_sma'])
        df['tp_dev_sma'] = df['tp_dev'].rolling(window=period).mean()
        
        # Calculate CCI
        df['cci'] = (df['tp'] - df['tp_sma']) / (0.015 * df['tp_dev_sma'])
        
        # Clean up temporary columns
        df = df.drop(['tp', 'tp_sma', 'tp_dev', 'tp_dev_sma'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_roc(df, period=12):
        """
        Calculate ROC (Rate of Change)
        
        Parameters:
        - df: DataFrame with stock data
        - period: ROC period (default: 12)
        
        Returns:
        - DataFrame with ROC
        """
        # Calculate ROC
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        return df
    
    @staticmethod
    def calculate_momentum(df, period=10):
        """
        Calculate Momentum
        
        Parameters:
        - df: DataFrame with stock data
        - period: Momentum period (default: 10)
        
        Returns:
        - DataFrame with Momentum
        """
        # Calculate Momentum
        df['momentum'] = df['close'] - df['close'].shift(period)
        
        return df
    
    @staticmethod
    def calculate_cmf(df, period=20):
        """
        Calculate CMF (Chaikin Money Flow)
        
        Parameters:
        - df: DataFrame with stock data
        - period: CMF period (default: 20)
        
        Returns:
        - DataFrame with CMF
        """
        # Calculate Money Flow Multiplier
        df['mf_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        
        # Calculate Money Flow Volume
        df['mf_volume'] = df['mf_multiplier'] * df['volume']
        
        # Calculate CMF
        df['cmf'] = df['mf_volume'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        
        # Clean up temporary columns
        df = df.drop(['mf_multiplier', 'mf_volume'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_mfi(df, period=14):
        """
        Calculate MFI (Money Flow Index)
        
        Parameters:
        - df: DataFrame with stock data
        - period: MFI period (default: 14)
        
        Returns:
        - DataFrame with MFI
        """
        # Calculate typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate raw money flow
        df['raw_money_flow'] = df['tp'] * df['volume']
        
        # Calculate money flow direction
        df['money_flow_positive'] = 0
        df['money_flow_negative'] = 0
        
        df.loc[df['tp'] > df['tp'].shift(), 'money_flow_positive'] = df['raw_money_flow']
        df.loc[df['tp'] < df['tp'].shift(), 'money_flow_negative'] = df['raw_money_flow']
        
        # Calculate money flow ratio
        df['positive_flow'] = df['money_flow_positive'].rolling(window=period).sum()
        df['negative_flow'] = df['money_flow_negative'].rolling(window=period).sum()
        df['money_flow_ratio'] = df['positive_flow'] / df['negative_flow']
        
        # Calculate MFI
        df['mfi'] = 100 - (100 / (1 + df['money_flow_ratio']))
        
        # Clean up temporary columns
        df = df.drop(['tp', 'raw_money_flow', 'money_flow_positive', 'money_flow_negative', 'positive_flow', 'negative_flow', 'money_flow_ratio'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_tsi(df, long_period=25, short_period=13, signal_period=7):
        """
        Calculate TSI (True Strength Index)
        
        Parameters:
        - df: DataFrame with stock data
        - long_period: Long EMA period (default: 25)
        - short_period: Short EMA period (default: 13)
        - signal_period: Signal EMA period (default: 7)
        
        Returns:
        - DataFrame with TSI
        """
        # Calculate price change
        df['price_change'] = df['close'].diff()
        
        # Calculate double smoothed price change
        df['first_smooth'] = df['price_change'].ewm(span=long_period, adjust=False).mean()
        df['second_smooth'] = df['first_smooth'].ewm(span=short_period, adjust=False).mean()
        
        # Calculate absolute price change
        df['abs_price_change'] = abs(df['price_change'])
        
        # Calculate double smoothed absolute price change
        df['abs_first_smooth'] = df['abs_price_change'].ewm(span=long_period, adjust=False).mean()
        df['abs_second_smooth'] = df['abs_first_smooth'].ewm(span=short_period, adjust=False).mean()
        
        # Calculate TSI
        df['tsi'] = 100 * (df['second_smooth'] / df['abs_second_smooth'])
        
        # Calculate TSI signal line
        df['tsi_signal'] = df['tsi'].ewm(span=signal_period, adjust=False).mean()
        
        # Clean up temporary columns
        df = df.drop(['price_change', 'first_smooth', 'second_smooth', 'abs_price_change', 'abs_first_smooth', 'abs_second_smooth'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_kst(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, signal_period=9):
        """
        Calculate KST (Know Sure Thing)
        
        Parameters:
        - df: DataFrame with stock data
        - r1, r2, r3, r4: ROC periods (default: 10, 15, 20, 30)
        - n1, n2, n3, n4: SMA periods (default: 10, 10, 10, 15)
        - signal_period: Signal SMA period (default: 9)
        
        Returns:
        - DataFrame with KST
        """
        # Calculate ROCs
        df['roc1'] = ((df['close'] - df['close'].shift(r1)) / df['close'].shift(r1)) * 100
        df['roc2'] = ((df['close'] - df['close'].shift(r2)) / df['close'].shift(r2)) * 100
        df['roc3'] = ((df['close'] - df['close'].shift(r3)) / df['close'].shift(r3)) * 100
        df['roc4'] = ((df['close'] - df['close'].shift(r4)) / df['close'].shift(r4)) * 100
        
        # Calculate SMAs of ROCs
        df['roc1_sma'] = df['roc1'].rolling(window=n1).mean()
        df['roc2_sma'] = df['roc2'].rolling(window=n2).mean()
        df['roc3_sma'] = df['roc3'].rolling(window=n3).mean()
        df['roc4_sma'] = df['roc4'].rolling(window=n4).mean()
        
        # Calculate KST
        df['kst'] = (df['roc1_sma'] * 1) + (df['roc2_sma'] * 2) + (df['roc3_sma'] * 3) + (df['roc4_sma'] * 4)
        
        # Calculate KST signal line
        df['kst_signal'] = df['kst'].rolling(window=signal_period).mean()
        
        # Clean up temporary columns
        df = df.drop(['roc1', 'roc2', 'roc3', 'roc4', 'roc1_sma', 'roc2_sma', 'roc3_sma', 'roc4_sma'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
        """
        Calculate Parabolic SAR
        
        Parameters:
        - df: DataFrame with stock data
        - af_start: Starting acceleration factor (default: 0.02)
        - af_increment: Acceleration factor increment (default: 0.02)
        - af_max: Maximum acceleration factor (default: 0.2)
        
        Returns:
        - DataFrame with Parabolic SAR
        """
        # Initialize columns
        df['sar'] = df['close'].copy()
        df['ep'] = df['high'].copy()  # Extreme point
        df['af'] = af_start  # Acceleration factor
        df['trend'] = 1  # 1 for uptrend, -1 for downtrend
        
        # Calculate Parabolic SAR
        for i in range(2, len(df)):
            # Get previous values
            prev_sar = df['sar'].iloc[i-1]
            prev_ep = df['ep'].iloc[i-1]
            prev_af = df['af'].iloc[i-1]
            prev_trend = df['trend'].iloc[i-1]
            
            # Current high and low
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            
            # Calculate SAR
            if prev_trend == 1:  # Uptrend
                # SAR = Previous SAR + Previous AF * (Previous EP - Previous SAR)
                curr_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Ensure SAR is not higher than the previous two lows
                curr_sar = min(curr_sar, df['low'].iloc[i-1], df['low'].iloc[i-2])
                
                # Check if trend reverses
                if curr_sar > curr_low:
                    # Trend reverses to downtrend
                    curr_trend = -1
                    curr_sar = prev_ep  # SAR becomes the previous extreme point
                    curr_ep = curr_low  # EP becomes the current low
                    curr_af = af_start  # AF resets
                else:
                    # Continue uptrend
                    curr_trend = 1
                    curr_ep = max(prev_ep, curr_high)  # EP is the highest high
                    curr_af = prev_af if curr_ep == prev_ep else min(prev_af + af_increment, af_max)
            else:  # Downtrend
                # SAR = Previous SAR - Previous AF * (Previous SAR - Previous EP)
                curr_sar = prev_sar - prev_af * (prev_sar - prev_ep)
                
                # Ensure SAR is not lower than the previous two highs
                curr_sar = max(curr_sar, df['high'].iloc[i-1], df['high'].iloc[i-2])
                
                # Check if trend reverses
                if curr_sar < curr_high:
                    # Trend reverses to uptrend
                    curr_trend = 1
                    curr_sar = prev_ep  # SAR becomes the previous extreme point
                    curr_ep = curr_high  # EP becomes the current high
                    curr_af = af_start  # AF resets
                else:
                    # Continue downtrend
                    curr_trend = -1
                    curr_ep = min(prev_ep, curr_low)  # EP is the lowest low
                    curr_af = prev_af if curr_ep == prev_ep else min(prev_af + af_increment, af_max)
            
            # Update values
            df.at[df.index[i], 'sar'] = curr_sar
            df.at[df.index[i], 'ep'] = curr_ep
            df.at[df.index[i], 'af'] = curr_af
            df.at[df.index[i], 'trend'] = curr_trend
        
        # Clean up temporary columns
        df = df.drop(['ep', 'af'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_cd_signal(df, period=20):
        """
        Calculate Close/Down (C/D) Signal
        
        Parameters:
        - df: DataFrame with stock data
        - period: C/D period (default: 20)
        
        Returns:
        - DataFrame with C/D Signal
        """
        # Calculate daily change
        df['daily_change'] = df['close'].diff()
        
        # Calculate up and down days
        df['up_day'] = df['daily_change'].apply(lambda x: 1 if x > 0 else 0)
        df['down_day'] = df['daily_change'].apply(lambda x: 1 if x < 0 else 0)
        
        # Calculate cumulative up and down days over period
        df['cum_up'] = df['up_day'].rolling(window=period).sum()
        df['cum_down'] = df['down_day'].rolling(window=period).sum()
        
        # Calculate C/D ratio
        df['cd_ratio'] = df['cum_up'] / (df['cum_up'] + df['cum_down'])
        
        # Calculate C/D signal (percentage)
        df['cd_signal'] = df['cd_ratio'] * 100
        
        # Clean up temporary columns
        df = df.drop(['daily_change', 'up_day', 'down_day', 'cum_up', 'cum_down', 'cd_ratio'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_arms_index(df, market_breadth_data):
        """
        Calculate ARMS Index (TRIN)
        
        Parameters:
        - df: DataFrame with stock data
        - market_breadth_data: Dictionary with market breadth data
        
        Returns:
        - DataFrame with ARMS Index
        """
        # Check if market breadth data is available
        if market_breadth_data is None:
            # Add NaN values for ARMS Index
            df['arms_index'] = np.nan
            return df
        
        # Extract market breadth data
        advancing_issues = market_breadth_data['advancing_issues']
        declining_issues = market_breadth_data['declining_issues']
        advancing_volume = market_breadth_data['advancing_volume']
        declining_volume = market_breadth_data['declining_volume']
        
        # Calculate ARMS Index
        # ARMS Index = (Advancing Issues/Declining Issues)/(Advancing Volume/Declining Volume)
        if declining_issues > 0 and declining_volume > 0:
            arms_index = (advancing_issues / declining_issues) / (advancing_volume / declining_volume)
        else:
            arms_index = np.nan
        
        # Add ARMS Index to DataFrame
        df['arms_index'] = arms_index
        
        return df
    
    @staticmethod
    def calculate_bulls_bears_ratio(df, sentiment_data):
        """
        Calculate Bulls vs Bears Ratio
        
        Parameters:
        - df: DataFrame with stock data
        - sentiment_data: Dictionary with sentiment data
        
        Returns:
        - DataFrame with Bulls vs Bears Ratio
        """
        # Check if sentiment data is available
        if sentiment_data is None:
            # Add NaN values for Bulls vs Bears Ratio
            df['bulls_bears_ratio'] = np.nan
            return df
        
        # Extract sentiment data
        bullish = sentiment_data['bullish']
        bearish = sentiment_data['bearish']
        
        # Calculate Bulls vs Bears Ratio
        if bearish > 0:
            bulls_bears_ratio = bullish / bearish
        else:
            bulls_bears_ratio = np.nan
        
        # Add Bulls vs Bears Ratio to DataFrame
        df['bulls_bears_ratio'] = bulls_bears_ratio
        
        return df
    
    @staticmethod
    def generate_buy_sell_signals(df):
        """
        Generate buy/sell signals based on technical indicators
        
        Parameters:
        - df: DataFrame with stock data and indicators
        
        Returns:
        - DataFrame with buy/sell signals
        """
        # Initialize buy/sell signals
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        
        # Check if all required indicators are available
        required_indicators = ['rsi', 'macd_line', 'macd_signal', 'stoch_k', 'stoch_d', 'cci', 'bb_upper', 'bb_lower']
        for indicator in required_indicators:
            if indicator not in df.columns:
                return df
        
        # Generate buy signals
        buy_conditions = (
            (df['rsi'] < 40) &  # RSI below 40
            (df['macd_line'] > df['macd_signal']) &  # MACD line above signal line
            (df['stoch_k'] < 20) &  # Stochastic %K below 20 (oversold)
            (df['cci'] < -100) &  # CCI below -100 (oversold)
            (df['close'] < df['bb_lower'])  # Price below lower Bollinger Band
        )
        
        # Generate sell signals
        sell_conditions = (
            (df['rsi'] > 65) &  # RSI above 65
            (df['macd_line'] < df['macd_signal']) &  # MACD line below signal line
            (df['stoch_k'] > 80) &  # Stochastic %K above 80 (overbought)
            (df['cci'] > 100) &  # CCI above 100 (overbought)
            (df['close'] > df['bb_upper'])  # Price above upper Bollinger Band
        )
        
        # Apply buy/sell signals
        df.loc[buy_conditions, 'buy_signal'] = 1
        df.loc[sell_conditions, 'sell_signal'] = 1
        
        # Ensure we don't have consecutive signals of the same type
        for i in range(1, len(df)):
            if df['buy_signal'].iloc[i] == 1 and df['buy_signal'].iloc[i-1] == 1:
                df.at[df.index[i], 'buy_signal'] = 0
            if df['sell_signal'].iloc[i] == 1 and df['sell_signal'].iloc[i-1] == 1:
                df.at[df.index[i], 'sell_signal'] = 0
        
        return df

def compute_adaptive_thresholds(prices, risk_tolerance=0.5):
    """
    Compute adaptive thresholds based on volatility and risk tolerance.
    Returns lower and upper thresholds for an indicator.
    """
    returns = prices.pct_change().dropna()
    volatility = returns.std()
    # For simplicity use quantile-based thresholds (could be replaced by more advanced stats)
    lower = prices.quantile(0.25) * (1 - risk_tolerance * volatility)
    upper = prices.quantile(0.75) * (1 + risk_tolerance * volatility)
    return lower, upper
