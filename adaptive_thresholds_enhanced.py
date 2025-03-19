"""
Enhanced Adaptive Indicator Thresholds Module for Perfect Storm Dashboard

This module implements dynamic thresholds for technical indicators that adjust based on:
1. Market volatility
2. Statistical methods for optimal threshold determination
3. Risk tolerance personalization
4. Machine learning-based threshold optimization
5. Regime-specific threshold adaptation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class ThresholdDataset(Dataset):
    """Dataset class for threshold optimization"""
    
    def __init__(self, features, targets):
        """
        Initialize the dataset
        
        Parameters:
        - features: Feature tensor
        - targets: Target tensor
        """
        self.features = features
        self.targets = targets
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        return self.features[idx], self.targets[idx]

class ThresholdOptimizer(nn.Module):
    """Neural network for threshold optimization"""
    
    def __init__(self, input_size, hidden_size=64, output_size=1):
        """
        Initialize the threshold optimizer model
        
        Parameters:
        - input_size: Number of input features
        - hidden_size: Size of hidden layers (default: 64)
        - output_size: Number of output thresholds (default: 1)
        """
        super(ThresholdOptimizer, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - output: Predicted thresholds
        """
        return self.model(x)

class EnhancedAdaptiveThresholds:
    """Enhanced class for adaptive indicator thresholds"""
    
    def __init__(self, volatility_window=20, lookback_period=252, risk_tolerance='medium',
                statistical_method='quantile', optimization_method='ml',
                model_path='models', use_ensemble=True):
        """
        Initialize the EnhancedAdaptiveThresholds class
        
        Parameters:
        - volatility_window: Window size for volatility calculation (default: 20)
        - lookback_period: Period for historical analysis (default: 252, i.e., 1 year)
        - risk_tolerance: Risk tolerance level ('low', 'medium', 'high', default: 'medium')
        - statistical_method: Method for statistical threshold determination 
                             ('quantile', 'zscore', 'percentile', 'bootstrap', default: 'quantile')
        - optimization_method: Method for threshold optimization 
                              ('ml', 'grid_search', 'bayesian', default: 'ml')
        - model_path: Path to save/load models (default: 'models')
        - use_ensemble: Whether to use ensemble methods (default: True)
        """
        self.volatility_window = volatility_window
        self.lookback_period = lookback_period
        self.risk_tolerance = risk_tolerance
        self.statistical_method = statistical_method
        self.optimization_method = optimization_method
        self.model_path = model_path
        self.use_ensemble = use_ensemble
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Risk tolerance multipliers
        self.risk_multipliers = {
            'low': 0.8,      # More conservative thresholds
            'medium': 1.0,   # Standard thresholds
            'high': 1.2      # More aggressive thresholds
        }
        
        # Initialize threshold history
        self.threshold_history = {}
        
        # Initialize models
        self.threshold_models = {}
        
        # Initialize threshold statistics
        self.threshold_stats = {}
        
        # Initialize regime-specific thresholds
        self.regime_thresholds = {}
    
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
        
        # Calculate GARCH volatility (simplified)
        df_vol['garch_volatility'] = np.sqrt(
            0.01 * df_vol['returns'].pow(2) + 
            0.89 * df_vol['volatility'].shift(1).fillna(0)
        )
        
        # Calculate volatility of volatility
        df_vol['vol_of_vol'] = df_vol['volatility'].pct_change().rolling(window=self.volatility_window).std()
        
        # Calculate volatility trend
        df_vol['vol_trend'] = df_vol['volatility'].pct_change(self.volatility_window)
        
        # Calculate volatility acceleration
        df_vol['vol_acceleration'] = df_vol['vol_trend'].diff(self.volatility_window)
        
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
        
        # Add more features for better regime detection
        df_regime['mean_reversion'] = (df_regime[price_col] - df_regime[price_col].rolling(window=self.volatility_window).mean()) / df_regime[price_col].rolling(window=self.volatility_window).std()
        df_regime['volume_trend'] = df_regime['volume'].pct_change(self.volatility_window) if 'volume' in df_regime.columns else 0
        
        # Drop NaN values
        feature_cols = ['volatility', 'trend', 'momentum', 'mean_reversion']
        if 'volume_trend' in df_regime.columns:
            feature_cols.append('volume_trend')
        
        df_features = df_regime.dropna()[feature_cols]
        
        # Standardize features
        features_std = (df_features - df_features.mean()) / df_features.std()
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_regime.loc[df_features.index, 'regime_cluster'] = kmeans.fit_predict(features_std)
        
        # Determine regime characteristics
        regime_stats = df_regime.groupby('regime_cluster')[feature_cols].mean()
        
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
    
    def calculate_statistical_thresholds(self, df, indicator_col, method=None, confidence_level=0.95):
        """
        Calculate statistical thresholds for an indicator
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - method: Statistical method to use (default: None, use self.statistical_method)
        - confidence_level: Confidence level for statistical methods (default: 0.95)
        
        Returns:
        - thresholds: Dictionary with lower and upper thresholds
        """
        if method is None:
            method = self.statistical_method
        
        # Get indicator values
        indicator_values = df[indicator_col].dropna()
        
        if len(indicator_values) < 30:
            # Not enough data, use default thresholds
            if indicator_col == 'rsi':
                return {'lower': 30, 'upper': 70}
            elif indicator_col == 'stoch':
                return {'lower': 20, 'upper': 80}
            elif indicator_col == 'cci':
                return {'lower': -100, 'upper': 100}
            elif indicator_col == 'macd_hist':
                return {'lower': -0.5, 'upper': 0.5}
            else:
                # Generic thresholds
                return {'lower': indicator_values.mean() - indicator_values.std(),
                        'upper': indicator_values.mean() + indicator_values.std()}
        
        if method == 'quantile':
            # Use quantiles
            lower_quantile = (1 - confidence_level) / 2
            upper_quantile = 1 - lower_quantile
            
            lower_threshold = indicator_values.quantile(lower_quantile)
            upper_threshold = indicator_values.quantile(upper_quantile)
        
        elif method == 'zscore':
            # Use z-scores
            z_critical = stats.norm.ppf(confidence_level)
            
            mean = indicator_values.mean()
            std = indicator_values.std()
            
            lower_threshold = mean - z_critical * std
            upper_threshold = mean + z_critical * std
        
        elif method == 'percentile':
            # Use percentiles
            lower_percentile = (1 - confidence_level) * 100 / 2
            upper_percentile = 100 - lower_percentile
            
            lower_threshold = np.percentile(indicator_values, lower_percentile)
            upper_threshold = np.percentile(indicator_values, upper_percentile)
        
        elif method == 'bootstrap':
            # Use bootstrap method
            n_bootstrap = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(indicator_values, size=len(indicator_values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            lower_percentile = (1 - confidence_level) * 100 / 2
            upper_percentile = 100 - lower_percentile
            
            lower_threshold = np.percentile(bootstrap_means, lower_percentile)
            upper_threshold = np.percentile(bootstrap_means, upper_percentile)
        
        else:
            # Unknown method, use quantiles
            lower_threshold = indicator_values.quantile(0.25)
            upper_threshold = indicator_values.quantile(0.75)
        
        # Store threshold statistics
        self.threshold_stats[indicator_col] = {
            'mean': indicator_values.mean(),
            'std': indicator_values.std(),
            'min': indicator_values.min(),
            'max': indicator_values.max(),
            'median': indicator_values.median(),
            'q1': indicator_values.quantile(0.25),
            'q3': indicator_values.quantile(0.75)
        }
        
        return {'lower': lower_threshold, 'upper': upper_threshold}
    
    def personalize_thresholds(self, thresholds, risk_tolerance=None):
        """
        Personalize thresholds based on risk tolerance
        
        Parameters:
        - thresholds: Dictionary with lower and upper thresholds
        - risk_tolerance: Risk tolerance level (default: None, use self.risk_tolerance)
        
        Returns:
        - personalized_thresholds: Dictionary with personalized thresholds
        """
        if risk_tolerance is None:
            risk_tolerance = self.risk_tolerance
        
        # Get risk multiplier
        risk_multiplier = self.risk_multipliers.get(risk_tolerance, 1.0)
        
        # Personalize thresholds
        personalized_thresholds = {}
        
        for indicator, threshold_values in thresholds.items():
            if isinstance(threshold_values, dict) and 'lower' in threshold_values and 'upper' in threshold_values:
                # Adjust thresholds based on risk tolerance
                lower = threshold_values['lower']
                upper = threshold_values['upper']
                
                # Calculate midpoint
                midpoint = (lower + upper) / 2
                
                # Adjust thresholds
                if risk_tolerance == 'low':
                    # More conservative: narrow the range
                    new_lower = midpoint - (midpoint - lower) * risk_multiplier
                    new_upper = midpoint + (upper - midpoint) * risk_multiplier
                elif risk_tolerance == 'high':
                    # More aggressive: widen the range
                    new_lower = midpoint - (midpoint - lower) * risk_multiplier
                    new_upper = midpoint + (upper - midpoint) * risk_multiplier
                else:
                    # Medium: keep th<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>