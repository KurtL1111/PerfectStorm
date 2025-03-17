"""
Updated indicator calculation module for Perfect Storm Dashboard

This module implements calculation functions for all technical indicators
required by the Perfect Storm investment strategy, including additional
indicators discovered during validation.
"""

import pandas as pd
import numpy as np


class PerfectStormIndicators:
    """Class to calculate all indicators for the Perfect Storm strategy"""
    
    @staticmethod
    def calculate_moving_averages(df, periods=[5, 9, 20, 50, 100, 200]):
        """
        Calculate simple moving averages for specified periods
        
        Parameters:
        - df: DataFrame with price data (must have 'close' column)
        - periods: List of periods for moving averages
        
        Returns:
        - DataFrame with original data plus moving averages
        """
        result_df = df.copy()
        
        for period in periods:
            column_name = f'ma_{period}'
            result_df[column_name] = result_df['close'].rolling(window=period).mean()
            
        return result_df
    
    @staticmethod
    def calculate_bollinger_bands(df, period=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Parameters:
        - df: DataFrame with price data (must have 'close' column)
        - period: Period for moving average (default: 20)
        - num_std: Number of standard deviations (default: 2)
        
        Returns:
        - DataFrame with original data plus Bollinger Bands
        """
        result_df = df.copy()
        
        # Calculate middle band (SMA)
        result_df['bb_middle'] = result_df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        result_df['bb_std'] = result_df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result_df['bb_upper'] = result_df['bb_middle'] + (result_df['bb_std'] * num_std)
        result_df['bb_lower'] = result_df['bb_middle'] - (result_df['bb_std'] * num_std)
        
        # Calculate %B (position within bands)
        result_df['bb_percent_b'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        # Calculate bandwidth
        result_df['bb_bandwidth'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        
        return result_df
    
    @staticmethod
    def calculate_cd_signal(df):
        """
        Calculate Close/Down (C/D) signal as described in Perfect Storm strategy
        
        Parameters:
        - df: DataFrame with price data (must have 'close' column)
        
        Returns:
        - DataFrame with original data plus C/D signal
        """
        result_df = df.copy()
        
        # Calculate daily price change
        result_df['price_change'] = result_df['close'].diff()
        
        # Determine if price is trending up or down
        result_df['trend'] = np.where(result_df['price_change'] > 0, 'up', 'down')
        
        # Calculate C/D signal (simplified version based on available information)
        # Using a 10-day window to determine trend strength
        window_size = 10
        result_df['cd_signal'] = 0.0
        
        for i in range(window_size, len(result_df)):
            window = result_df['trend'].iloc[i-window_size:i]
            down_count = (window == 'down').sum()
            result_df.loc[result_df.index[i], 'cd_signal'] = (down_count / window_size) * 100
        
        # Signal strength (80% threshold mentioned in the book)
        result_df['cd_signal_strong'] = result_df['cd_signal'] >= 80
        
        return result_df
    
    @staticmethod
    def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence/Divergence)
        
        Parameters:
        - df: DataFrame with price data (must have 'close' column)
        - fast_period: Period for fast EMA (default: 12)
        - slow_period: Period for slow EMA (default: 26)
        - signal_period: Period for signal line (default: 9)
        
        Returns:
        - DataFrame with original data plus MACD indicators
        """
        result_df = df.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = result_df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result_df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        result_df['macd_line'] = fast_ema - slow_ema
        
        # Calculate signal line
        result_df['macd_signal'] = result_df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
        
        # Calculate convergence/divergence
        result_df['macd_convergence'] = result_df['macd_line'].diff() > 0
        
        return result_df
    
    @staticmethod
    def calculate_chaikin_money_flow(df, period=21):
        """
        Calculate Chaikin Money Flow
        
        Parameters:
        - df: DataFrame with price data (must have 'close', 'high', 'low', 'volume' columns)
        - period: Period for calculation (default: 21 as mentioned in the book)
        
        Returns:
        - DataFrame with original data plus Chaikin Money Flow
        """
        result_df = df.copy()
        
        # Step 1: Calculate Money Flow Multiplier
        result_df['mf_multiplier'] = ((result_df['close'] - result_df['low']) - 
                                     (result_df['high'] - result_df['close'])) / (result_df['high'] - result_df['low'])
        
        # Handle division by zero
        result_df['mf_multiplier'] = result_df['mf_multiplier'].replace([np.inf, -np.inf], 0)
        result_df['mf_multiplier'] = result_df['mf_multiplier'].fillna(0)
        
        # Step 2: Calculate Money Flow Volume
        result_df['mf_volume'] = result_df['mf_multiplier'] * result_df['volume']
        
        # Step 3: Calculate Chaikin Money Flow
        result_df['cmf'] = result_df['mf_volume'].rolling(window=period).sum() / result_df['volume'].rolling(window=period).sum()
        
        return result_df
    
    @staticmethod
    def calculate_arms_index(df, advancing_issues=None, declining_issues=None, advancing_volume=None, declining_volume=None):
        """
        Calculate ARMS Index (TRIN)
        
        Parameters:
        - df: DataFrame with price data
        - advancing_issues: Number of advancing issues (optional)
        - declining_issues: Number of declining issues (optional)
        - advancing_volume: Volume of advancing issues (optional)
        - declining_volume: Volume of declining issues (optional)
        
        Returns:
        - DataFrame with original data plus ARMS Index
        """
        result_df = df.copy()
        
        # If market breadth data is not provided, estimate from price movements
        if advancing_issues is None or declining_issues is None or advancing_volume is None or declining_volume is None:
            # Calculate daily price changes
            result_df['price_change'] = result_df['close'].diff()
            
            # Count advancing and declining days in a 10-day window
            window_size = 10
            result_df['arms_index'] = np.nan
            
            for i in range(window_size, len(result_df)):
                window = result_df['price_change'].iloc[i-window_size:i]
                adv_issues = (window > 0).sum()
                dec_issues = (window < 0).sum()
                
                # Calculate volume for advancing and declining days
                adv_vol = result_df.loc[result_df['price_change'] > 0, 'volume'].iloc[i-window_size:i].sum()
                dec_vol = result_df.loc[result_df['price_change'] < 0, 'volume'].iloc[i-window_size:i].sum()
                
                # Avoid division by zero
                if dec_issues == 0 or dec_vol == 0:
                    result_df.loc[result_df.index[i], 'arms_index'] = 1.0
                    continue
                
                # Calculate ARMS Index
                adv_dec_ratio = adv_issues / dec_issues
                adv_dec_vol_ratio = adv_vol / dec_vol
                
                result_df.loc[result_df.index[i], 'arms_index'] = adv_dec_ratio / adv_dec_vol_ratio
        
        return result_df
    
    @staticmethod
    def calculate_adx(df, period=14):
        """
        Calculate Average Directional Movement Index (ADX)
        
        Parameters:
        - df: DataFrame with price data (must have 'high', 'low', 'close' columns)
        - period: Period for calculation (default: 14)
        
        Returns:
        - DataFrame with original data plus ADX indicators
        """
        result_df = df.copy()
        
        # Calculate True Range (TR)
        result_df['tr1'] = abs(result_df['high'] - result_df['low'])
        result_df['tr2'] = abs(result_df['high'] - result_df['close'].shift())
        result_df['tr3'] = abs(result_df['low'] - result_df['close'].shift())
        result_df['tr'] = result_df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement (DM)
        result_df['up_move'] = result_df['high'] - result_df['high'].shift()
        result_df['down_move'] = result_df['low'].shift() - result_df['low']
        
        # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
        result_df['+dm'] = np.where((result_df['up_move'] > result_df['down_move']) & (result_df['up_move'] > 0), 
                                   result_df['up_move'], 0)
        result_df['-dm'] = np.where((result_df['down_move'] > result_df['up_move']) & (result_df['down_move'] > 0), 
                                   result_df['down_move'], 0)
        
        # Calculate smoothed TR, +DM, and -DM
        result_df['smoothed_tr'] = result_df['tr'].rolling(window=period).sum()
        result_df['smoothed_+dm'] = result_df['+dm'].rolling(window=period).sum()
        result_df['smoothed_-dm'] = result_df['-dm'].rolling(window=period).sum()
        
        # Calculate Directional Indicators (+DI and -DI)
        result_df['+di'] = 100 * (result_df['smoothed_+dm'] / result_df['smoothed_tr'])
        result_df['-di'] = 100 * (result_df['smoothed_-dm'] / result_df['smoothed_tr'])
        
        # Calculate Directional Index (DX)
        result_df['dx'] = 100 * (abs(result_df['+di'] - result_df['-di']) / (result_df['+di'] + result_df['-di']))
        
        # Calculate Average Directional Index (ADX)
        result_df['adx'] = result_df['dx'].rolling(window=period).mean()
        
        # Clean up intermediate columns
        cols_to_drop = ['tr1', 'tr2', 'tr3', 'up_move', 'down_move', 'smoothed_tr', 
                        'smoothed_+dm', 'smoothed_-dm', 'dx']
        result_df = result_df.drop(columns=cols_to_drop)
        
        return result_df
    
    @staticmethod
    def calculate_bulls_bears_ratio(df, period=52):
        """
        Calculate Bulls vs Bears Ratio
        
        Parameters:
        - df: DataFrame with price data
        - period: Period for calculation (default: 52 weeks as mentioned in the book)
        
        Returns:
        - DataFrame with original data plus Bulls vs Bears ratio
        """
        result_df = df.copy()
        
        # Calculate 52-week high and low (or specified period)
        result_df['period_high'] = result_df['high'].rolling(window=period).max()
        result_df['period_low'] = result_df['low'].rolling(window=period).min()
        
        # Calculate Bulls vs Bears ratio (simplified version based on available information)
        # Using the formula mentioned in the book: bull / (bull + bear) * 100
        result_df['bulls_ratio'] = (result_df['close'] - result_df['period_low']) / (result_df['period_high'] - result_df['period_low'])
        result_df['bears_ratio'] = 1 - result_df['bulls_ratio']
        
        # Calculate Bulls vs Bears ratio
        result_df['bulls_bears_ratio'] = result_df['bulls_ratio'] / (result_df['bulls_ratio'] + result_df['bears_ratio'])
        
        # Clean up intermediate columns
        result_df = result_df.drop(columns=['period_high', 'period_low'])
        
        return result_df
    
    @staticmethod
    def calculate_accumulation_distribution(df):
        """
        Calculate Accumulation/Distribution Line
        
        Parameters:
        - df: DataFrame with price data (must have 'close', 'high', 'low', 'volume' columns)
        
        Returns:
        - DataFrame with original data plus Accumulation/Distribution Line
        """
        result_df = df.copy()
        
        # Calculate Money Flow Multiplier
        result_df['mf_multiplier'] = ((result_df['close'] - result_df['low']) - 
                                     (result_df['high'] - result_df['close'])) / (result_df['high'] - result_df['low'])
        
        # Handle division by zero
        result_df['mf_multiplier'] = result_df['mf_multiplier'].replace([np.inf, -np.inf], 0)
        result_df['mf_multiplier'] = result_df['mf_multiplier'].fillna(0)
        
        # Calculate Money Flow Volume
        result_df['mf_volume'] = result_df['mf_multiplier'] * result_df['volume']
        
        # Calculate Accumulation/Distribution Line
        result_df['acc_dist_line'] = result_df['mf_volume'].cumsum()
        
        # Clean up intermediate columns
        result_df = result_df.drop(columns=['mf_multiplier', 'mf_volume'])
        
        return result_df
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all Perfect Storm indicators in one function
        
        Parameters:
        - df: DataFrame with price data (must have 'open', 'high', 'low', 'close', 'volume' columns)
        
        Returns:
        - DataFrame with all indicators calculated
        """
        result_df = df.copy()
        
        # Calculate moving averages
        result_df = PerfectStormIndicators.calculate_moving_averages(result_df)
        
        # Calculate Bollinger Bands
        result_df = PerfectStormIndicators.calculate_bollinger_bands(result_df)
        
        # Calculate C/D signal
        result_df = PerfectStormIndicators.calculate_cd_signal(result_df)
        
        # Calculate MACD
        result_df = PerfectStormIndicators.calculate_macd(result_df)
        
        # Calculate Chaikin Money Flow
        result_df = PerfectStormIndicators.calculate_chaikin_money_flow(result_df)
        
        # Calculate ARMS Index
        result_df = PerfectStormIndicators.calculate_arms_index(result_df)
        
        # Calculate ADX (Average Directional Movement)
        result_df = PerfectStormIndicators.calculate_adx(result_df)
        
        # Calculate Bulls vs Bears Ratio
        result_df = PerfectStormIndicators.calculate_bulls_bears_ratio(result_df)
        
        # Calculate Accumulation/Distribution Line
        result_df = PerfectStormIndicators.calculate_accumulation_distribution(result_df)
        
        return result_df


# Example usage
#def example_usage():
#    """Example of how to use the PerfectStormIndicators class"""
#    
#    # Create sample data
#    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
#    data = {
#        'open': np.random.normal(100, 5, 100),
#        'high': np.random.normal(105, 5, 100),
#        'low': np.random.normal(95, 5, 100),
#        'close': np.random.normal(100, 5, 100),
#        'volume': np.random.normal(1000000, 200000, 100)
#    }
#   
#    # Ensure high is always highest and low is always lowest
#    for i in range(len(data['open'])):
#        values = [data['open'][i], data['close'][i]]
#        data['high'][i] = max(values) + abs(np.random.normal(0, 1))
#        data['low'][i] = min(values) - abs(np.random.normal(0, 1))
#    
#    df = pd.DataFrame(data, index=<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>