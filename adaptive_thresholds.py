"""
Adaptive Indicator Thresholds Module for Perfect Storm Dashboard

This module implements dynamic thresholds for technical indicators that adjust based on:
1. Market volatility
2. Statistical methods for optimal threshold determination
3. Risk tolerance personalization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
import os
import pickle
from technical_indicators import compute_adaptive_thresholds  # new import

class AdaptiveThresholds:
    """Class for adaptive indicator thresholds"""
    
    def __init__(self, volatility_window=20, lookback_period=252, risk_tolerance='medium'):
        """
        Initialize the AdaptiveThresholds class
        
        Parameters:
        - volatility_window: Window size for volatility calculation (default: 20)
        - lookback_period: Period for historical analysis (default: 252, i.e., 1 year)
        - risk_tolerance: Risk tolerance level ('low', 'medium', 'high', default: 'medium')
        """
        self.volatility_window = volatility_window
        self.lookback_period = lookback_period
        self.risk_tolerance = risk_tolerance
        
        # Risk tolerance multipliers
        self.risk_multipliers = {
            'low': 0.8,      # More conservative thresholds
            'medium': 1.0,   # Standard thresholds
            'high': 1.2      # More aggressive thresholds
        }
        
        # Initialize threshold history
        self.threshold_history = {}

    def compute_thresholds(self, df, risk_tolerance=0.05):
        """
        Compute dynamic thresholds based on market volatility and statistical quantiles.
        Parameters:
         - df: DataFrame with market data and indicators
         - risk_tolerance: Personal risk tolerance factor
        Returns:
         - thresholds: Dict with adaptive thresholds
        """
        # Calculate rolling volatility of close price returns
        vol = df['close'].pct_change().rolling(window=20).std()
        # Adaptive RSI threshold around 50 adjusted by the median volatility
        rsi_threshold = 50 + (vol.median() * 100 * risk_tolerance)
        # Adaptive MACD threshold based on the median of MACD histogram if available
        macd_threshold = df['macd_hist'].quantile(0.5) if 'macd_hist' in df.columns else 0
        thresholds = {
            'rsi_threshold': rsi_threshold,
            'macd_threshold': macd_threshold,
            # ...other thresholds can be added similarly...
        }
        return thresholds

    def calculate_volatility(self, df, price_col='close'):
        """
        Calculate market volatility
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        
        Returns:
        - df_vol: DataFrame with volatility metrics
        """
        # Make a copy of the DataFrame
        df_vol = df.copy()
        
        # Calculate returns
        df_vol['returns'] = df_vol[price_col].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        df_vol['volatility'] = df_vol['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)  # Annualized
        
        # Calculate normalized volatility (z-score)
        rolling_vol_mean = df_vol['volatility'].rolling(window=self.lookback_period).mean()
        rolling_vol_std = df_vol['volatility'].rolling(window=self.lookback_period).std()
        df_vol['volatility_z'] = (df_vol['volatility'] - rolling_vol_mean) / rolling_vol_std
        
        # Calculate volatility regime (low, medium, high)
        df_vol['volatility_regime'] = 'medium'
        df_vol.loc[df_vol['volatility_z'] < -0.5, 'volatility_regime'] = 'low'
        df_vol.loc[df_vol['volatility_z'] > 0.5, 'volatility_regime'] = 'high'
        
        # Calculate volatility percentile
        df_vol['volatility_percentile'] = df_vol['volatility'].rolling(window=self.lookback_period).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100
        )
        
        return df_vol
    
    def detect_market_regime(self, df, price_col='close', n_clusters=3):
        """
        Detect market regime using clustering
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - n_clusters: Number of clusters/regimes (default: 3)
        
        Returns:
        - df_regime: DataFrame with market regime
        """
        # Make a copy of the DataFrame
        df_regime = df.copy()
        
        # Calculate returns
        df_regime['returns'] = df_regime[price_col].pct_change()
        
        # Calculate features for regime detection
        df_regime['volatility'] = df_regime['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        df_regime['trend'] = df_regime[price_col].rolling(window=self.volatility_window).mean().pct_change(self.volatility_window)
        df_regime['momentum'] = df_regime['returns'].rolling(window=self.volatility_window).mean()
        
        # Drop NaN values
        df_features = df_regime.dropna()[['volatility', 'trend', 'momentum']]
        
        # Standardize features
        features_std = (df_features - df_features.mean()) / df_features.std()
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_regime.loc[df_features.index, 'regime_cluster'] = kmeans.fit_predict(features_std)
        
        # Determine regime characteristics
        regime_stats = df_regime.groupby('regime_cluster')[['volatility', 'trend', 'momentum']].mean()
        
        # Label regimes
        regime_labels = [''] * n_clusters
        
        # Identify trending regime (highest absolute trend)
        trending_idx = abs(regime_stats['trend']).idxmax()
        if regime_stats.loc[trending_idx, 'trend'] > 0:
            regime_labels[trending_idx] = 'trending_up'
        else:
            regime_labels[trending_idx] = 'trending_down'
        
        # Identify volatile regime (highest volatility)
        volatile_idx = regime_stats['volatility'].idxmax()
        if volatile_idx != trending_idx:
            regime_labels[volatile_idx] = 'volatile'
        
        # Identify ranging regime (lowest volatility and trend)
        remaining_idx = [i for i in range(n_clusters) if i != trending_idx and i != volatile_idx]
        if remaining_idx:
            regime_labels[remaining_idx[0]] = 'ranging'
        
        # Map cluster numbers to regime labels
        cluster_to_regime = {i: label for i, label in enumerate(regime_labels)}
        df_regime['market_regime'] = df_regime['regime_cluster'].map(cluster_to_regime)
        
        # Forward fill regime for NaN values
        df_regime['market_regime'] = df_regime['market_regime'].ffill()
        
        return df_regime
    
    def calculate_adaptive_rsi_thresholds(self, df, base_lower=30, base_upper=70):
        """
        Calculate adaptive RSI thresholds
        
        Parameters:
        - df: DataFrame with market data and volatility metrics
        - base_lower: Base lower threshold (default: 30)
        - base_upper: Base upper threshold (default: 70)
        
        Returns:
        - df_thresholds: DataFrame with adaptive RSI thresholds
        """
        # Make a copy of the DataFrame
        df_thresholds = df.copy()
        
        # Ensure volatility metrics are available
        if 'volatility_percentile' not in df_thresholds.columns:
            df_vol = self.calculate_volatility(df_thresholds)
            df_thresholds['volatility_percentile'] = df_vol['volatility_percentile']
            df_thresholds['volatility_regime'] = df_vol['volatility_regime']
        
        # Ensure market regime is available
        if 'market_regime' not in df_thresholds.columns:
            df_regime = self.detect_market_regime(df_thresholds)
            df_thresholds['market_regime'] = df_regime['market_regime']
        
        # Apply risk tolerance multiplier
        risk_mult = self.risk_multipliers.get(self.risk_tolerance, 1.0)
        
        # Calculate adaptive thresholds based on volatility
        # Higher volatility -> wider thresholds
        # Lower volatility -> narrower thresholds
        volatility_adjustment = (df_thresholds['volatility_percentile'] - 0.5) * 20  # Scale to [-10, 10]
        
        # Calculate regime adjustment
        regime_adjustment = pd.Series(0, index=df_thresholds.index)
        
        # Adjust thresholds based on market regime
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_up'] = -5  # Lower thresholds in uptrend
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_down'] = 5  # Higher thresholds in downtrend
        regime_adjustment.loc[df_thresholds['market_regime'] == 'volatile'] = 10     # Wider thresholds in volatile regime
        regime_adjustment.loc[df_thresholds['market_regime'] == 'ranging'] = -5      # Narrower thresholds in ranging regime
        
        # Calculate final thresholds
        df_thresholds['rsi_lower_threshold'] = base_lower - (volatility_adjustment + regime_adjustment) * risk_mult
        df_thresholds['rsi_upper_threshold'] = base_upper + (volatility_adjustment + regime_adjustment) * risk_mult
        
        # Ensure thresholds are within valid range [5, 95]
        df_thresholds['rsi_lower_threshold'] = df_thresholds['rsi_lower_threshold'].clip(5, 45)
        df_thresholds['rsi_upper_threshold'] = df_thresholds['rsi_upper_threshold'].clip(55, 95)
        
        # Store threshold history
        self.threshold_history['rsi_lower'] = df_thresholds['rsi_lower_threshold'].copy()
        self.threshold_history['rsi_upper'] = df_thresholds['rsi_upper_threshold'].copy()
        
        return df_thresholds
    
    def calculate_adaptive_stochastic_thresholds(self, df, base_lower=20, base_upper=80):
        """
        Calculate adaptive Stochastic thresholds
        
        Parameters:
        - df: DataFrame with market data and volatility metrics
        - base_lower: Base lower threshold (default: 20)
        - base_upper: Base upper threshold (default: 80)
        
        Returns:
        - df_thresholds: DataFrame with adaptive Stochastic thresholds
        """
        # Make a copy of the DataFrame
        df_thresholds = df.copy()
        
        # Ensure volatility metrics are available
        if 'volatility_percentile' not in df_thresholds.columns:
            df_vol = self.calculate_volatility(df_thresholds)
            df_thresholds['volatility_percentile'] = df_vol['volatility_percentile']
            df_thresholds['volatility_regime'] = df_vol['volatility_regime']
        
        # Ensure market regime is available
        if 'market_regime' not in df_thresholds.columns:
            df_regime = self.detect_market_regime(df_thresholds)
            df_thresholds['market_regime'] = df_regime['market_regime']
        
        # Apply risk tolerance multiplier
        risk_mult = self.risk_multipliers.get(self.risk_tolerance, 1.0)
        
        # Calculate adaptive thresholds based on volatility
        volatility_adjustment = (df_thresholds['volatility_percentile'] - 0.5) * 20  # Scale to [-10, 10]
        
        # Calculate regime adjustment
        regime_adjustment = pd.Series(0, index=df_thresholds.index)
        
        # Adjust thresholds based on market regime
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_up'] = -5
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_down'] = 5
        regime_adjustment.loc[df_thresholds['market_regime'] == 'volatile'] = 10
        regime_adjustment.loc[df_thresholds['market_regime'] == 'ranging'] = -5
        
        # Calculate final thresholds
        df_thresholds['stoch_lower_threshold'] = base_lower - (volatility_adjustment + regime_adjustment) * risk_mult
        df_thresholds['stoch_upper_threshold'] = base_upper + (volatility_adjustment + regime_adjustment) * risk_mult
        
        # Ensure thresholds are within valid range [5, 95]
        df_thresholds['stoch_lower_threshold'] = df_thresholds['stoch_lower_threshold'].clip(5, 45)
        df_thresholds['stoch_upper_threshold'] = df_thresholds['stoch_upper_threshold'].clip(55, 95)
        
        # Store threshold history
        self.threshold_history['stoch_lower'] = df_thresholds['stoch_lower_threshold'].copy()
        self.threshold_history['stoch_upper'] = df_thresholds['stoch_upper_threshold'].copy()
        
        return df_thresholds
    
    def calculate_adaptive_cci_thresholds(self, df, base_lower=-100, base_upper=100):
        """
        Calculate adaptive CCI thresholds
        
        Parameters:
        - df: DataFrame with market data and volatility metrics
        - base_lower: Base lower threshold (default: -100)
        - base_upper: Base upper threshold (default: 100)
        
        Returns:
        - df_thresholds: DataFrame with adaptive CCI thresholds
        """
        # Make a copy of the DataFrame
        df_thresholds = df.copy()
        
        # Ensure volatility metrics are available
        if 'volatility_percentile' not in df_thresholds.columns:
            df_vol = self.calculate_volatility(df_thresholds)
            df_thresholds['volatility_percentile'] = df_vol['volatility_percentile']
            df_thresholds['volatility_regime'] = df_vol['volatility_regime']
        
        # Ensure market regime is available
        if 'market_regime' not in df_thresholds.columns:
            df_regime = self.detect_market_regime(df_thresholds)
            df_thresholds['market_regime'] = df_regime['market_regime']
        
        # Apply risk tolerance multiplier
        risk_mult = self.risk_multipliers.get(self.risk_tolerance, 1.0)
        
        # Calculate adaptive thresholds based on volatility
        volatility_adjustment = (df_thresholds['volatility_percentile'] - 0.5) * 100  # Scale to [-50, 50]
        
        # Calculate regime adjustment
        regime_adjustment = pd.Series(0, index=df_thresholds.index)
        
        # Adjust thresholds based on market regime
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_up'] = -25
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_down'] = 25
        regime_adjustment.loc[df_thresholds['market_regime'] == 'volatile'] = 50
        regime_adjustment.loc[df_thresholds['market_regime'] == 'ranging'] = -25
        
        # Calculate final thresholds
        df_thresholds['cci_lower_threshold'] = base_lower - (volatility_adjustment + regime_adjustment) * risk_mult
        df_thresholds['cci_upper_threshold'] = base_upper + (volatility_adjustment + regime_adjustment) * risk_mult
        
        # Ensure thresholds are within valid range [-300, 300]
        df_thresholds['cci_lower_threshold'] = df_thresholds['cci_lower_threshold'].clip(-300, -50)
        df_thresholds['cci_upper_threshold'] = df_thresholds['cci_upper_threshold'].clip(50, 300)
        
        # Store threshold history
        self.threshold_history['cci_lower'] = df_thresholds['cci_lower_threshold'].copy()
        self.threshold_history['cci_upper'] = df_thresholds['cci_upper_threshold'].copy()
        
        return df_thresholds
    
    def calculate_adaptive_bollinger_thresholds(self, df, base_std=2.0):
        """
        Calculate adaptive Bollinger Bands thresholds
        
        Parameters:
        - df: DataFrame with market data and volatility metrics
        - base_std: Base standard deviation multiplier (default: 2.0)
        
        Returns:
        - df_thresholds: DataFrame with adaptive Bollinger Bands thresholds
        """
        # Make a copy of the DataFrame
        df_thresholds = df.copy()
        
        # Ensure volatility metrics are available
        if 'volatility_percentile' not in df_thresholds.columns:
            df_vol = self.calculate_volatility(df_thresholds)
            df_thresholds['volatility_percentile'] = df_vol['volatility_percentile']
            df_thresholds['volatility_regime'] = df_vol['volatility_regime']
        
        # Ensure market regime is available
        if 'market_regime' not in df_thresholds.columns:
            df_regime = self.detect_market_regime(df_thresholds)
            df_thresholds['market_regime'] = df_regime['market_regime']
        
        # Apply risk tolerance multiplier
        risk_mult = self.risk_multipliers.get(self.risk_tolerance, 1.0)
        
        # Calculate adaptive thresholds based on volatility
        volatility_adjustment = (df_thresholds['volatility_percentile'] - 0.5) * 1.0  # Scale to [-0.5, 0.5]
        
        # Calculate regime adjustment
        regime_adjustment = pd.Series(0, index=df_thresholds.index)
        
        # Adjust thresholds based on market regime
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_up'] = -0.2
        regime_adjustment.loc[df_thresholds['market_regime'] == 'trending_down'] = 0.2
        regime_adjustment.loc[df_thresholds['market_regime'] == 'volatile'] = 0.5
        regime_adjustment.loc[df_thresholds['market_regime'] == 'ranging'] = -0.2
        
        # Calculate final threshold
        df_thresholds['bb_std_threshold'] = base_std + (volatility_adjustment + regime_adjustment) * risk_mult
        
        # Ensure threshold is within valid range [1.0, 4.0]
        df_thresholds['bb_std_threshold'] = df_thresholds['bb_std_threshold'].clip(1.0, 4.0)
        
        # Store threshold history
        self.threshold_history['bb_std'] = df_thresholds['bb_std_threshold'].copy()
        
        return df_thresholds
    
    def calculate_all_adaptive_thresholds(self, df):
        """
        Calculate all adaptive thresholds
        
        Parameters:
        - df: DataFrame with market data
        
        Returns:
        - df_thresholds: DataFrame with all adaptive thresholds
        """
        # Calculate volatility metrics
        df_vol = self.calculate_volatility(df)
        
        # Detect market regime
        df_regime = self.detect_market_regime(df_vol)
        
        # Calculate adaptive thresholds for each indicator
        df_thresholds = self.calculate_adaptive_rsi_thresholds(df_regime)
        df_thresholds = self.calculate_adaptive_stochastic_thresholds(df_thresholds)
        df_thresholds = self.calculate_adaptive_cci_thresholds(df_thresholds)
        df_thresholds = self.calculate_adaptive_bollinger_thresholds(df_thresholds)
        
        return df_thresholds
    
    def set_risk_tolerance(self, risk_tolerance):
        """
        Set risk tolerance level
        
        Parameters:
        - risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
        """
        if risk_tolerance not in self.risk_multipliers:
            raise ValueError(f"Invalid risk tolerance level: {risk_tolerance}. Must be one of {list(self.risk_multipliers.keys())}")
        
        self.risk_tolerance = risk_tolerance
        print(f"Risk tolerance set to: {risk_tolerance}")
    
    def plot_adaptive_thresholds(self, df, indicator='rsi', figsize=(12, 8)):
        """
        Plot adaptive thresholds
        
        Parameters:
        - df: DataFrame with market data and adaptive thresholds
        - indicator: Indicator to plot ('rsi', 'stoch', 'cci', 'bb', default: 'rsi')
        - figsize: Figure size (default: (12, 8))
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot indicator and thresholds
        if indicator == 'rsi':
            if 'rsi' not in df.columns:
                raise ValueError("RSI column not found in DataFrame")
            
            ax1.plot(df.index, df['rsi'], label='RSI')
            ax1.plot(df.index, df['rsi_lower_threshold'], label='Lower Threshold', linestyle='--', color='green')
            ax1.plot(df.index, df['rsi_upper_threshold'], label='Upper Threshold', linestyle='--', color='red')
            ax1.set_ylabel('RSI')
            ax1.set_title('Adaptive RSI Thresholds')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight overbought/oversold regions
            ax1.fill_between(df.index, df['rsi_upper_threshold'], 100, alpha=0.2, color='red')
            ax1.fill_between(df.index, 0, df['rsi_lower_threshold'], alpha=0.2, color='green')
        
        elif indicator == 'stoch':
            if 'stoch_k' not in df.columns:
                raise ValueError("Stochastic %K column not found in DataFrame")
            
            ax1.plot(df.index, df['stoch_k'], label='Stochastic %K')
            if 'stoch_d' in df.columns:
                ax1.plot(df.index, df['stoch_d'], label='Stochastic %D', alpha=0.7)
            
            ax1.plot(df.index, df['stoch_lower_threshold'], label='Lower Threshold', linestyle='--', color='green')
            ax1.plot(df.index, df['stoch_upper_threshold'], label='Upper Threshold', linestyle='--', color='red')
            ax1.set_ylabel('Stochastic')
            ax1.set_title('Adaptive Stochastic Thresholds')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight overbought/oversold regions
            ax1.fill_between(df.index, df['stoch_upper_threshold'], 100, alpha=0.2, color='red')
            ax1.fill_between(df.index, 0, df['stoch_lower_threshold'], alpha=0.2, color='green')
        
        elif indicator == 'cci':
            if 'cci' not in df.columns:
                raise ValueError("CCI column not found in DataFrame")
            
            ax1.plot(df.index, df['cci'], label='CCI')
            ax1.plot(df.index, df['cci_lower_threshold'], label='Lower Threshold', linestyle='--', color='green')
            ax1.plot(df.index, df['cci_upper_threshold'], label='Upper Threshold', linestyle='--', color='red')
            ax1.set_ylabel('CCI')
            ax1.set_title('Adaptive CCI Thresholds')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight overbought/oversold regions
            ax1.fill_between(df.index, df['cci_upper_threshold'], 300, alpha=0.2, color='red')
            ax1.fill_between(df.index, -300, df['cci_lower_threshold'], alpha=0.2, color='green')
        
        elif indicator == 'bb':
            if 'close' not in df.columns:
                raise ValueError("Close price column not found in DataFrame")
            
            # Calculate Bollinger Bands with adaptive threshold
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + df['bb_std_threshold'] * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - df['bb_std_threshold'] * df['bb_std']
            
            ax1.plot(df.index, df['close'], label='Close Price')
            ax1.plot(df.index, df['bb_middle'], label='Middle Band', linestyle='-', color='blue', alpha=0.7)
            ax1.plot(df.index, df['bb_upper'], label='Upper Band', linestyle='--', color='red')
            ax1.plot(df.index, df['bb_lower'], label='Lower Band', linestyle='--', color='green')
            ax1.set_ylabel('Price')
            ax1.set_title('Adaptive Bollinger Bands')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight area between bands
            ax1.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.1, color='blue')
        
        else:
            raise ValueError(f"Invalid indicator: {indicator}. Must be one of ['rsi', 'stoch', 'cci', 'bb']")
        
        # Plot volatility and market regime
        ax2.plot(df.index, df['volatility_percentile'], label='Volatility Percentile', color='purple')
        ax2.set_ylabel('Volatility Percentile')
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left')
        
        # Add market regime as background color
        regime_colors = {
            'trending_up': 'lightgreen',
            'trending_down': 'lightcoral',
            'volatile': 'lightsalmon',
            'ranging': 'lightblue'
        }
        
        # Get unique regimes and their start/end dates
        regime_changes = df['market_regime'].ne(df['market_regime'].shift()).cumsum()
        regimes = df.groupby(regime_changes)['market_regime'].first()
        
        for i, regime in enumerate(regimes):
            if regime in regime_colors:
                # Get start and end dates for this regime
                regime_df = df[regime_changes == i+1]
                if not regime_df.empty:
                    start_date = regime_df.index[0]
                    end_date = regime_df.index[-1]
                    
                    # Add colored background
                    ax2.axvspan(start_date, end_date, alpha=0.3, color=regime_colors[regime])
        
        # Add regime labels
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3) for color in regime_colors.values()]
        labels = list(regime_colors.keys())
        ax2.legend(handles, labels, loc='upper right', title='Market Regime')
        
        plt.tight_layout()
        plt.show()
    
    def plot_threshold_distribution(self, indicator='rsi', figsize=(12, 6)):
        """
        Plot distribution of adaptive thresholds
        
        Parameters:
        - indicator: Indicator to plot ('rsi', 'stoch', 'cci', 'bb', default: 'rsi')
        - figsize: Figure size (default: (12, 6))
        """
        if not self.threshold_history:
            raise ValueError("No threshold history available. Calculate adaptive thresholds first.")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        if indicator == 'rsi':
            if 'rsi_lower' not in self.threshold_history or 'rsi_upper' not in self.threshold_history:
                raise ValueError("RSI threshold history not found")
            
            plt.hist(self.threshold_history['rsi_lower'], bins=20, alpha=0.7, label='Lower Threshold', color='green')
            plt.hist(self.threshold_history['rsi_upper'], bins=20, alpha=0.7, label='Upper Threshold', color='red')
            plt.xlabel('RSI Threshold Value')
            plt.title('Distribution of Adaptive RSI Thresholds')
        
        elif indicator == 'stoch':
            if 'stoch_lower' not in self.threshold_history or 'stoch_upper' not in self.threshold_history:
                raise ValueError("Stochastic threshold history not found")
            
            plt.hist(self.threshold_history['stoch_lower'], bins=20, alpha=0.7, label='Lower Threshold', color='green')
            plt.hist(self.threshold_history['stoch_upper'], bins=20, alpha=0.7, label='Upper Threshold', color='red')
            plt.xlabel('Stochastic Threshold Value')
            plt.title('Distribution of Adaptive Stochastic Thresholds')
        
        elif indicator == 'cci':
            if 'cci_lower' not in self.threshold_history or 'cci_upper' not in self.threshold_history:
                raise ValueError("CCI threshold history not found")
            
            plt.hist(self.threshold_history['cci_lower'], bins=20, alpha=0.7, label='Lower Threshold', color='green')
            plt.hist(self.threshold_history['cci_upper'], bins=20, alpha=0.7, label='Upper Threshold', color='red')
            plt.xlabel('CCI Threshold Value')
            plt.title('Distribution of Adaptive CCI Thresholds')
        
        elif indicator == 'bb':
            if 'bb_std' not in self.threshold_history:
                raise ValueError("Bollinger Bands threshold history not found")
            
            plt.hist(self.threshold_history['bb_std'], bins=20, alpha=0.7, label='Standard Deviation Multiplier', color='blue')
            plt.xlabel('Bollinger Bands Std Multiplier')
            plt.title('Distribution of Adaptive Bollinger Bands Thresholds')
        
        else:
            raise ValueError(f"Invalid indicator: {indicator}. Must be one of ['rsi', 'stoch', 'cci', 'bb']")
        
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, file_path='models/adaptive_thresholds.pkl'):
        """
        Save the adaptive thresholds model
        
        Parameters:
        - file_path: Path to save the model (default: 'models/adaptive_thresholds.pkl')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model
        with open(file_path, 'wb') as f:
            pickle.dump({
                'volatility_window': self.volatility_window,
                'lookback_period': self.lookback_period,
                'risk_tolerance': self.risk_tolerance,
                'risk_multipliers': self.risk_multipliers,
                'threshold_history': self.threshold_history
            }, f)
        
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path='models/adaptive_thresholds.pkl'):
        """
        Load the adaptive thresholds model
        
        Parameters:
        - file_path: Path to load the model from (default: 'models/adaptive_thresholds.pkl')
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load model
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Set attributes
        self.volatility_window = model_data['volatility_window']
        self.lookback_period = model_data['lookback_period']
        self.risk_tolerance = model_data['risk_tolerance']
        self.risk_multipliers = model_data['risk_multipliers']
        self.threshold_history = model_data['threshold_history']
        
        print(f"Model loaded from {file_path}")

class PerfectStormAdaptiveStrategy:
    """Class for Perfect Storm trading strategy with adaptive thresholds"""
    
    @staticmethod
    def generate_signals(df, min_signals_buy=3, min_signals_sell=3):
        """
        Generate buy/sell signals based on Perfect Storm strategy with adaptive thresholds
        
        Parameters:
        - df: DataFrame with market data, indicators, and adaptive thresholds
        - min_signals_buy: Minimum number of buy signals required (default: 3)
        - min_signals_sell: Minimum number of sell signals required (default: 3)
        
        Returns:
        - df: DataFrame with buy/sell signals
        """
        # Make a copy of the DataFrame
        df_signals = df.copy()
        
        # Initialize buy/sell signals
        df_signals['buy_signal'] = 0
        df_signals['sell_signal'] = 0
        
        # Check if all required indicators and thresholds are available
        required_indicators = ['rsi', 'macd_line', 'macd_signal', 'stoch_k', 'cci']
        required_thresholds = ['rsi_lower_threshold', 'rsi_upper_threshold', 'stoch_lower_threshold', 
                              'stoch_upper_threshold', 'cci_lower_threshold', 'cci_upper_threshold']
        
        missing_indicators = [ind for ind in required_indicators if ind not in df_signals.columns]
        missing_thresholds = [thr for thr in required_thresholds if thr not in df_signals.columns]
        
        if missing_indicators or missing_thresholds:
            raise ValueError(f"Missing indicators: {missing_indicators}, Missing thresholds: {missing_thresholds}")
        
        # Generate individual signals using adaptive thresholds
        df_signals['rsi_buy'] = df_signals['rsi'] < df_signals['rsi_lower_threshold']
        df_signals['rsi_sell'] = df_signals['rsi'] > df_signals['rsi_upper_threshold']
        
        df_signals['macd_buy'] = df_signals['macd_line'] > df_signals['macd_signal']
        df_signals['macd_sell'] = df_signals['macd_line'] < df_signals['macd_signal']
        
        df_signals['stoch_buy'] = df_signals['stoch_k'] < df_signals['stoch_lower_threshold']
        df_signals['stoch_sell'] = df_signals['stoch_k'] > df_signals['stoch_upper_threshold']
        
        df_signals['cci_buy'] = df_signals['cci'] < df_signals['cci_lower_threshold']
        df_signals['cci_sell'] = df_signals['cci'] > df_signals['cci_upper_threshold']
        
        # Check for Bollinger Bands thresholds
        if 'bb_std_threshold' in df_signals.columns and 'close' in df_signals.columns:
            # Calculate Bollinger Bands with adaptive threshold
            df_signals['bb_middle'] = df_signals['close'].rolling(window=20).mean()
            df_signals['bb_std'] = df_signals['close'].rolling(window=20).std()
            df_signals['bb_upper'] = df_signals['bb_middle'] + df_signals['bb_std_threshold'] * df_signals['bb_std']
            df_signals['bb_lower'] = df_signals['bb_middle'] - df_signals['bb_std_threshold'] * df_signals['bb_std']
            
            df_signals['bb_buy'] = df_signals['close'] < df_signals['bb_lower']
            df_signals['bb_sell'] = df_signals['close'] > df_signals['bb_upper']
            
            # Count buy and sell signals
            buy_columns = ['rsi_buy', 'macd_buy', 'stoch_buy', 'cci_buy', 'bb_buy']
            sell_columns = ['rsi_sell', 'macd_sell', 'stoch_sell', 'cci_sell', 'bb_sell']
        else:
            # Count buy and sell signals without Bollinger Bands
            buy_columns = ['rsi_buy', 'macd_buy', 'stoch_buy', 'cci_buy']
            sell_columns = ['rsi_sell', 'macd_sell', 'stoch_sell', 'cci_sell']
        
        df_signals['buy_count'] = df_signals[buy_columns].sum(axis=1)
        df_signals['sell_count'] = df_signals[sell_columns].sum(axis=1)
        
        # Generate final signals
        df_signals.loc[df_signals['buy_count'] >= min_signals_buy, 'buy_signal'] = 1
        df_signals.loc[df_signals['sell_count'] >= min_signals_sell, 'sell_signal'] = 1
        
        # Ensure we don't have buy and sell signals on the same day
        df_signals.loc[df_signals['buy_signal'] & df_signals['sell_signal'], 'sell_signal'] = 0
        
        # Ensure we don't have consecutive signals of the same type
        for i in range(1, len(df_signals)):
            if df_signals['buy_signal'].iloc[i] == 1 and df_signals['buy_signal'].iloc[i-1] == 1:
                df_signals.loc[df_signals.index[i], 'buy_signal'] = 0
            if df_signals['sell_signal'].iloc[i] == 1 and df_signals['sell_signal'].iloc[i-1] == 1:
                df_signals.loc[df_signals.index[i], 'sell_signal'] = 0
        
        return df_signals

    def __init__(self, risk_tolerance=0.5):
        self.adaptive_threshold = AdaptiveThresholds(risk_tolerance)

    def adjust_signals(self, df):
        """
        Adjust buy/sell signals dynamically based on computed thresholds.
        """
        thresholds = self.adaptive_threshold.compute_thresholds(df)
        # Use adaptive thresholds to update RSI-based signals.
        df['buy_signal'] = (df['rsi'] < thresholds['rsi_lower']).astype(int)
        df['sell_signal'] = (df['rsi'] > thresholds['rsi_upper']).astype(int)
        return df

# Example usage
def example_usage():
    """Example of how to use the AdaptiveThresholds class"""
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = {
        'open': np.random.normal(100, 5, 500),
        'high': np.random.normal(105, 5, 500),
        'low': np.random.normal(95, 5, 500),
        'close': np.random.normal(100, 5, 500),
        'volume': np.random.normal(1000000, 200000, 500)
    }
    
    # Ensure high is always highest and low is always lowest
    for i in range(len(data['open'])):
        values = [data['open'][i], data['close'][i]]
        data['high'][i] = max(values) + abs(np.random.normal(0, 1))
        data['low'][i] = min(values) - abs(np.random.normal(0, 1))
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some technical indicators
    df['rsi'] = np.random.normal(50, 15, 500)  # Simplified RSI
    df['macd_line'] = np.random.normal(0, 1, 500)  # Simplified MACD
    df['macd_signal'] = np.random.normal(0, 1, 500)  # Simplified MACD signal
    df['stoch_k'] = np.random.normal(50, 20, 500)  # Simplified Stochastic
    df['stoch_d'] = np.random.normal(50, 20, 500)  # Simplified Stochastic
    df['cci'] = np.random.normal(0, 100, 500)  # Simplified CCI
    
    # Create an AdaptiveThresholds instance
    at = AdaptiveThresholds(risk_tolerance='medium')
    
    # Calculate adaptive thresholds
    df_thresholds = at.calculate_all_adaptive_thresholds(df)
    
    # Plot adaptive thresholds
    at.plot_adaptive_thresholds(df_thresholds, indicator='rsi')
    at.plot_adaptive_thresholds(df_thresholds, indicator='stoch')
    at.plot_adaptive_thresholds(df_thresholds, indicator='cci')
    at.plot_adaptive_thresholds(df_thresholds, indicator='bb')
    
    # Plot threshold distributions
    at.plot_threshold_distribution(indicator='rsi')
    
    # Generate signals with adaptive thresholds
    df_signals = PerfectStormAdaptiveStrategy.generate_signals(df_thresholds)
    
    # Count buy and sell signals
    buy_signals = df_signals['buy_signal'].sum()
    sell_signals = df_signals['sell_signal'].sum()
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Change risk tolerance and recalculate
    at.set_risk_tolerance('high')
    df_thresholds_high = at.calculate_all_adaptive_thresholds(df)
    
    # Generate signals with new risk tolerance
    df_signals_high = PerfectStormAdaptiveStrategy.generate_signals(df_thresholds_high)
    
    # Count buy and sell signals
    buy_signals_high = df_signals_high['buy_signal'].sum()
    sell_signals_high = df_signals_high['sell_signal'].sum()
    
    print(f"Buy signals (high risk): {buy_signals_high}")
    print(f"Sell signals (high risk): {sell_signals_high}")

if __name__ == '__main__':
    example_usage()
