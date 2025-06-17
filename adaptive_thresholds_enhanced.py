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

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights properly to avoid NaN issues
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        """Forward pass with NaN handling"""
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("NaN detected in AdaptiveThreshold input")
            print(f"ThresholdOptimizer foward x: {x}")
            # Replace NaNs with zeros
            x = torch.nan_to_num(x, nan=0.0)
        
        # Continue with regular forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x

class EnhancedAdaptiveThresholds:
    """Enhanced class for adaptive indicator thresholds"""
    
    def __init__(self, volatility_window=20, lookback_period=252, risk_tolerance='high',
                statistical_method='bootstrap', optimization_method='ensemble',
                model_path='models\\Adaptive Threshold Models', use_ensemble=True):
        """
        Initialize the EnhancedAdaptiveThresholds class
        
        Parameters:
        - volatility_window: Window size for volatility calculation (default: 20)
        - lookback_period: Period for historical analysis (default: 252, i.e., 1 year)
        - risk_tolerance: Risk tolerance level ('low', 'medium', 'high', default: 'medium')
        - statistical_method: Method for statistical threshold determination 
                             ('quantile', 'zscore', 'percentile', 'bootstrap', default: 'quantile')
        - optimization_method: Method for threshold optimization 
                              ('ml', 'grid_search', 'bayesian', 'ensemble' default: 'ml')
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
        rolling_vol_mean = df_vol['volatility'].rolling(window=self.lookback_period, min_periods=1).mean()
        rolling_vol_std = df_vol['volatility'].rolling(window=self.lookback_period, min_periods=1).std()
        df_vol['volatility_z'] = (df_vol['volatility'] - rolling_vol_mean) / rolling_vol_std
        
        # Calculate volatility regime (low, medium, high)
        df_vol['volatility_regime'] = 'medium'
        df_vol.loc[df_vol['volatility_z'] < -0.5, 'volatility_regime'] = 'low'
        df_vol.loc[df_vol['volatility_z'] > 0.5, 'volatility_regime'] = 'high'
        
        # Calculate volatility percentile
        df_vol['volatility_percentile'] = df_vol['volatility'].rolling(window=self.lookback_period, min_periods=1).apply(
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
        
        # Identify feature columns
        feature_cols = ['volatility', 'trend', 'momentum', 'mean_reversion']
        if 'volume_trend' in df_regime.columns:
            feature_cols.append('volume_trend')
            
        # Handle NaN values in each feature column
        for col in feature_cols:
            # First try forward and backward fill
            df_regime[col] = df_regime[col].fillna(method='ffill').fillna(method='bfill')
            
            # If NaNs still remain, use mean or 0
            if df_regime[col].isna().any():
                if df_regime[col].dtype.kind in 'ifc':  # If numeric
                    fill_value = df_regime[col].mean() if not df_regime[col].isna().all() else 0
                    df_regime[col] = df_regime[col].fillna(fill_value)
                else:  # If non-numeric
                    df_regime[col] = df_regime[col].fillna(0)

        # Extract features for clustering, excluding NaN values
        df_features = df_regime[feature_cols].copy()
        
        # If no valid features are available, assign default regime and return
        if df_features.empty or df_features.isna().all().all():
            df_regime['regime_cluster'] = -1
            df_regime['market_regime'] = 'unknown'
            return df_regime
            
        # Do a final check for any NaN values and fill them
        df_features = df_features.fillna(0)
        
        # Standardize features
        features_mean = df_features.mean()
        features_std = df_features.std()
        
        # Handle zero standard deviation (avoid division by zero)
        for col in feature_cols:
            if features_std[col] == 0:
                features_std[col] = 1
                
        # Standardize the features
        features_standardized = (df_features - features_mean) / features_std
        
        # Convert to numpy array and ensure no NaNs remain
        features_array = features_standardized.values
        if np.isnan(features_array).any():
            # Replace any remaining NaNs with 0
            features_array = np.nan_to_num(features_array, nan=0.0)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_regime.loc[df_features.index, 'regime_cluster'] = kmeans.fit_predict(features_array)
        
        # Initialize market_regime column to avoid the KeyError
        df_regime['market_regime'] = None
        
        # Determine regime characteristics
        regime_stats = df_regime.groupby('regime_cluster')[feature_cols].mean()
        
        # Label regimes
        regime_labels = [''] * n_clusters
        
        # Identify trending regime (highest absolute trend)
        trending_idx = int(abs(regime_stats['trend']).idxmax())
        if regime_stats.loc[trending_idx, 'trend'] > 0:
            regime_labels[trending_idx] = 'trending_up'
        else:
            regime_labels[trending_idx] = 'trending_down'
        
        # Identify volatile regime (highest volatility)
        volatile_idx = int(regime_stats['volatility'].idxmax())
        if volatile_idx != trending_idx:
            regime_labels[volatile_idx] = 'volatile'
        
        # Identify ranging regime (lowest volatility and trend)
        remaining_idx = [i for i in range(n_clusters) if i != trending_idx and i != volatile_idx]
        if remaining_idx:
            regime_labels[remaining_idx[0]] = 'ranging'
        
        # Map cluster numbers to regime labels
        cluster_to_regime = {i: label for i, label in enumerate(regime_labels)}
        
        # Apply regime labels to the DataFrame
        for cluster, label in cluster_to_regime.items():
            df_regime.loc[df_regime['regime_cluster'] == cluster, 'market_regime'] = label
        
        # Forward fill regime values to handle NaN periods
        df_regime['market_regime'] = df_regime['market_regime'].ffill()
        
        return df_regime
    
    def calculate_statistical_thresholds(self, df, indicator_col, method=None, confidence_level=0.95):
        """
        Calculate statistical thresholds for an indicator
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - method: Statistical method to use (default: None, uses self.statistical_method)
        - confidence_level: Confidence level for threshold calculation (default: 0.95)
        
        Returns:
        - thresholds: Dictionary with upper and lower thresholds
        """
        if method is None:
            method = self.statistical_method
        
        # Get indicator values
        indicator_values = df[indicator_col].dropna()
        
        # Apply risk tolerance multiplier
        risk_multiplier = self.risk_multipliers.get(self.risk_tolerance, 1.0)
        
        # Calculate thresholds based on the specified method
        if method == 'quantile':
            lower_threshold = indicator_values.quantile(0.05) * risk_multiplier
            upper_threshold = indicator_values.quantile(0.95) * risk_multiplier
        
        elif method == 'zscore':
            mean = indicator_values.mean()
            std = indicator_values.std()
            z_value = stats.norm.ppf(confidence_level)
            
            lower_threshold = (mean - z_value * std) * risk_multiplier
            upper_threshold = (mean + z_value * std) * risk_multiplier
        
        elif method == 'percentile':
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            
            lower_threshold = np.percentile(indicator_values, lower_percentile) * risk_multiplier
            upper_threshold = np.percentile(indicator_values, upper_percentile) * risk_multiplier
        
        elif method == 'bootstrap':
            # Bootstrap sampling
            n_bootstrap = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(indicator_values, size=len(indicator_values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            lower_threshold = np.percentile(bootstrap_means, 2.5) * risk_multiplier
            upper_threshold = np.percentile(bootstrap_means, 97.5) * risk_multiplier
        
        else:
            # Default to simple mean +/- std
            mean = indicator_values.mean()
            std = indicator_values.std()
            
            lower_threshold = (mean - 2 * std) * risk_multiplier
            upper_threshold = (mean + 2 * std) * risk_multiplier
        
        # Store threshold statistics
        self.threshold_stats[indicator_col] = {
            'method': method,
            'mean': indicator_values.mean(),
            'std': indicator_values.std(),
            'min': indicator_values.min(),
            'max': indicator_values.max(),
            'median': indicator_values.median(),
            'skew': stats.skew(indicator_values),
            'kurtosis': stats.kurtosis(indicator_values)
        }
        
        return {'lower': lower_threshold, 'upper': upper_threshold}
    
    def calculate_volatility_adjusted_thresholds(self, df, indicator_col, price_col='close'):
        """
        Calculate volatility-adjusted thresholds for an indicator
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        
        Returns:
        - thresholds: Dictionary with upper and lower thresholds
        """
        # Calculate volatility
        df_vol = self.calculate_volatility(df, price_col)
        
        # Get the latest volatility percentile
        latest_vol_percentile = df_vol['volatility_percentile'].iloc[-1]
        
        # Calculate base thresholds
        base_thresholds = self.calculate_statistical_thresholds(df, indicator_col)
        
        # Adjust thresholds based on volatility
        if latest_vol_percentile > 0.7:  # High volatility
            # Widen thresholds in high volatility
            volatility_factor = 1.2
        elif latest_vol_percentile < 0.3:  # Low volatility
            # Narrow thresholds in low volatility
            volatility_factor = 0.8
        else:  # Medium volatility
            volatility_factor = 1.0
        
        # Apply volatility adjustment
        lower_threshold = base_thresholds['lower'] * volatility_factor
        upper_threshold = base_thresholds['upper'] * volatility_factor
        
        return {'lower': lower_threshold, 'upper': upper_threshold}
    
    def calculate_regime_specific_thresholds(self, df, indicator_col, price_col='close'):
        """
        Calculate regime-specific thresholds for an indicator.
        Prioritizes an existing 'market_regime' column in df.
        
        Parameters:
        - df: DataFrame with market data, potentially including a 'market_regime' column.
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        
        Returns:
        - regime_thresholds: Dictionary with thresholds for each regime
        """
        df_with_regime = df.copy()

        # Check if market_regime column already exists and is populated
        if 'market_regime' not in df_with_regime.columns or df_with_regime['market_regime'].isnull().all():
            print("Market regime not found in input DataFrame or is all NaN. Detecting internally.")
            df_with_regime = self.detect_market_regime(df_with_regime, price_col)
        else:
            print("Using pre-existing market_regime column from input DataFrame.")
            # Ensure the pre-existing column is forward-filled if it has some NaNs
            if df_with_regime['market_regime'].isnull().any():
                df_with_regime['market_regime'] = df_with_regime['market_regime'].ffill()
                # If still NaN at the beginning, bfill
                if df_with_regime['market_regime'].isnull().any():
                     df_with_regime['market_regime'] = df_with_regime['market_regime'].bfill()
                # If still NaN (e.g. all were NaN), assign a default
                if df_with_regime['market_regime'].isnull().all():
                    print("Warning: Pre-existing market_regime column was all NaN. Defaulting to 'unknown'.")
                    df_with_regime['market_regime'] = 'unknown'


        # Calculate thresholds for each regime
        regime_thresholds_map = {}
        
        unique_regimes = df_with_regime['market_regime'].unique()
        if len(unique_regimes) == 0 or (len(unique_regimes) == 1 and pd.isna(unique_regimes[0])) :
             print(f"Warning: No valid market regimes found for {indicator_col}. Using overall statistical thresholds.")
             return self.calculate_statistical_thresholds(df, indicator_col)

        for regime in unique_regimes:
            if pd.isna(regime) or regime == '' or regime == 'unknown': # Skip if regime is NaN, empty or 'unknown' after processing
                # For 'unknown' or truly empty regimes, we might want to use overall stats later as fallback
                # but not calculate specific thresholds for an 'unknown' category unless it's meaningful
                print(f"Skipping threshold calculation for undefined or unknown regime: '{regime}'")
                continue
                
            # Get data for this regime
            regime_data = df_with_regime[df_with_regime['market_regime'] == regime]
            
            # Calculate thresholds for this regime
            if len(regime_data) > 30:  # Ensure enough data points
                print(f"Calculating thresholds for regime '{regime}' with {len(regime_data)} data points for {indicator_col}.")
                regime_thresholds_map[regime] = self.calculate_statistical_thresholds(regime_data, indicator_col)
            else:
                # Use overall thresholds if not enough data for this specific regime
                print(f"Not enough data ({len(regime_data)} points) for regime '{regime}' for {indicator_col}. Using overall statistical thresholds for this regime.")
                regime_thresholds_map[regime] = self.calculate_statistical_thresholds(df, indicator_col) # Use original df for overall stats
        
        # Store regime-specific thresholds
        self.regime_thresholds[indicator_col] = regime_thresholds_map
        
        # Get current regime from the (potentially modified) df_with_regime
        current_regime = df_with_regime['market_regime'].iloc[-1] if not df_with_regime.empty else 'unknown'
        
        # Return thresholds for current regime
        if current_regime in regime_thresholds_map and regime_thresholds_map[current_regime] is not None:
            print(f"Returning thresholds for current regime '{current_regime}' for {indicator_col}.")
            return regime_thresholds_map[current_regime]
        elif 'unknown' in regime_thresholds_map and regime_thresholds_map['unknown'] is not None: # Fallback if current_regime was not processed
            print(f"Current regime '{current_regime}' thresholds not found or None, falling back to 'unknown' regime thresholds for {indicator_col}.")
            return regime_thresholds_map['unknown']
        else:
            # Ultimate fallback to overall statistical thresholds if no specific or 'unknown' regime thresholds are suitable
            print(f"No specific or 'unknown' regime thresholds available for current regime '{current_regime}' for {indicator_col}. Falling back to overall statistical thresholds.")
            return self.calculate_statistical_thresholds(df, indicator_col)
    
    def prepare_training_data(self, df, indicator_col, price_col='close', target_horizon=10):
        """
        Prepare training data for ML-based threshold optimization
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        - target_horizon: Horizon for future returns (default: 10)
        
        Returns:
        - X: Feature matrix
        - y: Target vector
        """
        # Make a copy of the DataFrame
        df_ml = df.copy()
        # Calculate future returns
        df_ml['future_return'] = df_ml[price_col].pct_change(target_horizon).shift(-target_horizon)
        # Calculate volatility features
        df_vol = self.calculate_volatility(df_ml, price_col)
        df_ml = pd.concat([df_ml, df_vol.drop(columns=[col for col in df_vol.columns if col in df_ml.columns])], axis=1)

        # Calculate indicator features
        df_ml[f'{indicator_col}_zscore'] = (df_ml[indicator_col] - df_ml[indicator_col].rolling(window=self.lookback_period, min_periods=1).mean()) / df_ml[indicator_col].rolling(window=self.lookback_period, min_periods=1).std()
        # Handle NaN values in percentile calculation
        df_ml[f'{indicator_col}_percentile'] = df_ml[indicator_col].rolling(window=self.lookback_period, min_periods=1).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if not x.dropna().empty else 0.5
        )
        
        df_ml[f'{indicator_col}_trend'] = df_ml[indicator_col].pct_change(self.volatility_window)
        
        # Create target variable (1 if future return is positive, 0 otherwise)
        df_ml['target'] = (df_ml['future_return'] > 0).astype(int)
        # Select features
        feature_cols = [
            indicator_col,
            f'{indicator_col}_zscore',
            f'{indicator_col}_percentile',
            f'{indicator_col}_trend',
            'volatility',
            'volatility_percentile',
            'vol_trend'
        ]
        # Check for NaN values before filling
        nan_cols = [col for col in feature_cols if col in df_ml.columns and df_ml[col].isna().any()]
        if nan_cols:
            print(f"Columns with NaN values before filling: {nan_cols}")
            print(f"NaN counts: {df_ml[nan_cols].isna().sum()}")
        
        # Fill NaN values using forward and backward fill
        for col in feature_cols:
            if col in df_ml.columns:
                # First try forward and backward fill
                df_ml[col] = df_ml[col].fillna(method='ffill').fillna(method='bfill')
                
                # If NaNs still remain, use mean or median
                if df_ml[col].isna().any():
                    print(f"NaNs still remain in {col} after ffill/bfill")
                    if df_ml[col].dtype.kind in 'ifc':  # If numeric
                        fill_value = df_ml[col].mean() if not df_ml[col].isna().all() else 0
                        df_ml[col] = df_ml[col].fillna(fill_value)
                        print(f"Filled remaining NaNs in {col} with mean: {fill_value}")
                    else:  # If non-numeric
                        fill_value = df_ml[col].mode()[0] if not df_ml[col].empty and not df_ml[col].isna().all() else "unknown"
                        df_ml[col] = df_ml[col].fillna(fill_value)
                        print(f"Filled remaining NaNs in {col} with mode: {fill_value}")

        # Check if any NaNs remain
        remaining_nans = df_ml[feature_cols].isna().sum().sum()
        if remaining_nans > 0:
            print(f"WARNING: {remaining_nans} NaN values still remain after cleaning")
            print(f"Columns with remaining NaNs: {[col for col in feature_cols if col in df_ml.columns and df_ml[col].isna().any()]}")
            # Last resort: replace any remaining NaNs with 0
            df_ml[feature_cols] = df_ml[feature_cols].fillna(0)
            print("Filled all remaining NaNs with 0")

        # Extract features and target
        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        
        # Final check for NaNs in numpy arrays
        if np.isnan(X).any():
            print("WARNING: NaNs still present in X after conversion to numpy array")
            print(f"NaN count in X: {np.isnan(X).sum()}")
            # Replace NaNs with 0
            X = np.nan_to_num(X, nan=0.0)
            print("Replaced all NaNs in X with 0")
        
        if np.isnan(y).any():
            print("WARNING: NaNs present in y after conversion to numpy array")
            print(f"NaN count in y: {np.isnan(y).sum()}")
            # Replace NaNs with 0
            y = np.nan_to_num(y, nan=0.0)
            print("Replaced all NaNs in y with 0")
        
        print("AdaptiveThreshold X Shape:", X.shape)
        print("AdaptiveThreshold y Shape:", y.shape)
        return X, y
    
    def train_threshold_model(self, df, indicator_col, price_col='close', target_horizon=10, epochs=100, batch_size=32, symbol=None, method=None):
        """
        Train ML model for threshold optimization
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        - target_horizon: Horizon for future returns (default: 10)
        - epochs: Number of training epochs (default: 100)
        - batch_size: Batch size for training (default: 32)
        - symbol: Symbol for the asset (default: None)
        - method: Method for threshold optimization (default: None)
        
        Returns:
        - model: Trained model
        """
        # Prepare training data
        print("Preparing adaptive threshold training data...")
        X, y = self.prepare_training_data(df, indicator_col, price_col, target_horizon)
        
        print("Splitting adaptive threshold training data...")
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Converting adaptive threshold training data to tensors...")
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        print("Creating adaptive threshold datasets...")
        # Create datasets and dataloaders
        train_dataset = ThresholdDataset(X_train_tensor, y_train_tensor)
        val_dataset = ThresholdDataset(X_val_tensor, y_val_tensor)
        print("Creating adaptive threshold dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        print("Initializing adaptive threshold model...")
        # Initialize model
        input_size = X.shape[1]
        model = ThresholdOptimizer(input_size)
        print("AdaptiveThreshold model initialized.")
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Starting adaptive threshold training...")
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    print("AdaptiveThreshold loss:", loss)
                    val_loss += loss.item()
                    print("AdaptiveThreshold val_loss:", val_loss)
            
            # Print progress every 10 epochs
            #if (epoch + 1) % 10 == 0:
            #    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save model
        if method is None:
            method = self.optimization_method
        symbol_str = f"_{symbol}" if symbol else ""
        method_str = f"_{method}" if method else ""
        model_filename = os.path.join(self.model_path, f'threshold_model_{indicator_col}{method_str}{symbol_str}.pth')
        torch.save(model.state_dict(), model_filename)
        
        # Store model in memory
        self.threshold_models[(indicator_col, method, symbol)] = model
        
        return model
    
    def predict_optimal_threshold(self, df, indicator_col, price_col='close', symbol=None, method=None):
        """
        Predict optimal threshold using trained ML model
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        - symbol: Symbol for the asset (default: None)
        - method: Method for threshold optimization (default: None)
        
        Returns:
        - threshold: Optimal threshold value
        """
        # Check if model exists
        if method is None:
            method = self.optimization_method
        symbol_str = f"_{symbol}" if symbol else ""
        method_str = f"_{method}" if method else ""
        model_key = (indicator_col, method, symbol)
        if model_key not in self.threshold_models:
            # Try to load model from disk
            model_filename = os.path.join(self.model_path, f'threshold_model_{indicator_col}{method_str}{symbol_str}.pth')
            if os.path.exists(model_filename):
                input_size = 7  # Number of features used in training
                model = ThresholdOptimizer(input_size)
                model.load_state_dict(torch.load(model_filename))
                self.threshold_models[model_key] = model
            else:
                # Train new model
                self.train_threshold_model(df, indicator_col, price_col, symbol=symbol, method=method)
        
        # Prepare features for prediction
        X, _ = self.prepare_training_data(df.iloc[-self.lookback_period:], indicator_col, price_col)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X[-1:])  # Use the latest data point
        
        # Make prediction
        model = self.threshold_models[model_key]
        model.eval()
        with torch.no_grad():
            threshold_pred = model(X_tensor).item()
        
        # Scale prediction to indicator range
        indicator_min = df[indicator_col].min()
        indicator_max = df[indicator_col].max()
        scaled_threshold = indicator_min + threshold_pred * (indicator_max - indicator_min)
        
        return scaled_threshold
    
    def optimize_thresholds_grid_search(self, df, indicator_col, price_col='close', n_thresholds=20):
        """
        Optimize thresholds using grid search
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        - n_thresholds: Number of threshold values to test (default: 20)
        
        Returns:
        - optimal_thresholds: Dictionary with optimal upper and lower thresholds
        """
        # Make a copy of the DataFrame
        df_opt = df.copy()
        
        # Calculate future returns
        df_opt['future_return'] = df_opt[price_col].pct_change(10).shift(-10)
        
        # Get indicator range
        indicator_min = df_opt[indicator_col].min()
        indicator_max = df_opt[indicator_col].max()
        
        # Generate threshold values
        threshold_values = np.linspace(indicator_min, indicator_max, n_thresholds)
        
        # Evaluate each threshold
        results = []
        
        for lower_idx, lower_threshold in enumerate(threshold_values):
            for upper_idx, upper_threshold in enumerate(threshold_values[lower_idx+1:], lower_idx+1):
                # Generate signals
                df_opt['signal'] = 0
                df_opt.loc[df_opt[indicator_col] < lower_threshold, 'signal'] = 1  # Buy signal
                df_opt.loc[df_opt[indicator_col] > upper_threshold, 'signal'] = -1  # Sell signal
                
                # Calculate strategy returns
                df_opt['strategy_return'] = df_opt['signal'] * df_opt['future_return']
                
                # Calculate performance metrics
                total_return = df_opt['strategy_return'].sum()
                win_rate = (df_opt['strategy_return'] > 0).mean()
                sharpe_ratio = df_opt['strategy_return'].mean() / df_opt['strategy_return'].std() if df_opt['strategy_return'].std() > 0 else 0
                
                # Store results
                results.append({
                    'lower_threshold': lower_threshold,
                    'upper_threshold': upper_threshold,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe_ratio
                })
        
        # Find optimal thresholds
        results_df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        # Get optimal thresholds
        optimal_thresholds = {
            'lower': results_df.iloc[0]['lower_threshold'],
            'upper': results_df.iloc[0]['upper_threshold']
        }
        
        return optimal_thresholds
    
    def get_adaptive_thresholds(self, df, indicator_col, price_col='close', method=None):
        """
        Get adaptive thresholds for an indicator
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        - method: Method to use for threshold calculation (default: None, uses self.optimization_method)
        
        Returns:
        - thresholds: Dictionary with upper and lower thresholds
        """
        if method is None:
            method = self.optimization_method
        
        # Calculate thresholds based on the specified method
        if method == 'ml':
            # ML-based threshold optimization
            symbol = df.attrs.get('symbol', None) if hasattr(df, 'attrs') and 'symbol' in df.attrs else None
            symbol_str = f"_{symbol}" if symbol else ""
            method_str = f"_{method}" if method else ""
            model_key = (indicator_col, method, symbol)
            if model_key not in self.threshold_models:
                model_filename = os.path.join(self.model_path, f'threshold_model_{indicator_col}{method_str}{symbol_str}.pth')
                if os.path.exists(model_filename):
                    input_size = 7
                    model = ThresholdOptimizer(input_size)
                    model.load_state_dict(torch.load(model_filename))
                    self.threshold_models[model_key] = model
            optimal_threshold = self.predict_optimal_threshold(df, indicator_col, price_col, symbol=symbol, method=method)
            
            # Calculate statistical thresholds as a baseline
            stat_thresholds = self.calculate_statistical_thresholds(df, indicator_col)
            
            # Use ML prediction for the threshold that's closer to the current value
            current_value = df[indicator_col].iloc[-1]
            
            if abs(current_value - stat_thresholds['lower']) < abs(current_value - stat_thresholds['upper']):
                # Current value is closer to lower threshold
                thresholds = {
                    'lower': optimal_threshold,
                    'upper': stat_thresholds['upper']
                }
            else:
                # Current value is closer to upper threshold
                thresholds = {
                    'lower': stat_thresholds['lower'],
                    'upper': optimal_threshold
                }
        
        elif method == 'grid_search':
            # Grid search optimization
            thresholds = self.optimize_thresholds_grid_search(df, indicator_col, price_col)
        
        elif method == 'volatility_adjusted':
            # Volatility-adjusted thresholds
            thresholds = self.calculate_volatility_adjusted_thresholds(df, indicator_col, price_col)
        
        elif method == 'regime_specific':
            # Regime-specific thresholds
            # The df passed here might already contain 'market_regime' from callbacks.py
            thresholds = self.calculate_regime_specific_thresholds(df, indicator_col, price_col)
        
        elif method == 'ensemble':
            # Ensemble of methods
            methods = ['volatility_adjusted', 'regime_specific', 'grid_search']
            ensemble_thresholds = []
            
            for m in methods:
                ensemble_thresholds.append(self.get_adaptive_thresholds(df, indicator_col, price_col, method=m))
            
            # Average the thresholds
            thresholds = {
                'lower': np.mean([t['lower'] for t in ensemble_thresholds]),
                'upper': np.mean([t['upper'] for t in ensemble_thresholds])
            }
        
        else:
            # Default to statistical thresholds
            thresholds = self.calculate_statistical_thresholds(df, indicator_col)
        
        # Store thresholds in history
        if indicator_col not in self.threshold_history:
            self.threshold_history[indicator_col] = []
        
        self.threshold_history[indicator_col].append({
            'date': df.index[-1],
            'method': method,
            'lower': thresholds['lower'],
            'upper': thresholds['upper'],
            'current_value': df[indicator_col].iloc[-1]
        })
        
        return thresholds
    
    def plot_adaptive_thresholds(self, df, indicator_col, price_col='close', method=None, figsize=(12, 8)):
        """
        Plot indicator with adaptive thresholds
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        - method: Method to use for threshold calculation (default: None, uses self.optimization_method)
        - figsize: Figure size (default: (12, 8))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Calculate adaptive thresholds
        thresholds = self.get_adaptive_thresholds(df, indicator_col, price_col, method)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price
        axes[0].plot(df.index, df[price_col], label=price_col.capitalize())
        axes[0].set_title(f'{price_col.capitalize()} Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot indicator with thresholds
        axes[1].plot(df.index, df[indicator_col], label=indicator_col)
        axes[1].axhline(y=thresholds['lower'], color='g', linestyle='--', label=f'Lower Threshold ({thresholds["lower"]:.2f})')
        axes[1].axhline(y=thresholds['upper'], color='r', linestyle='--', label=f'Upper Threshold ({thresholds["upper"]:.2f})')
        
        # Highlight buy/sell regions
        axes[1].fill_between(df.index, df[indicator_col], thresholds['lower'], where=df[indicator_col] < thresholds['lower'], color='g', alpha=0.3)
        axes[1].fill_between(df.index, df[indicator_col], thresholds['upper'], where=df[indicator_col] > thresholds['upper'], color='r', alpha=0.3)
        
        axes[1].set_title(f'{indicator_col} with Adaptive Thresholds (Method: {method or self.optimization_method})')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        fig_out = fig
        plt.close(fig)
        return fig_out
    
    def save_thresholds(self, filename=None, symbol=None, period=None, interval=None):
        """
        Save adaptive thresholds to file
        
        Parameters:
        - filename: Filename to save thresholds (default: None, use default filename)
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - filename: Filename of saved thresholds
        """
        from dashboard_utils import get_standardized_model_filename
        
        if filename is None:
            # Use the standardized filename format
            filename = get_standardized_model_filename(
                model_type="adaptive_thresholds",
                model_name=self.optimization_method,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".json"
        
        # Create dictionary with threshold data
        threshold_data = {
            'thresholds': self.thresholds,
            'threshold_history': self.threshold_history,
            'threshold_stats': self.threshold_stats,
            'model_data': {
                'model_type': self.model_type if hasattr(self, 'model_type') else None,
                'model_path': self.model_path
            },
            'config': {
                'volatility_window': self.volatility_window,
                'lookback_period': self.lookback_period,
                'risk_tolerance': self.risk_tolerance,
                'statistical_method': self.statistical_method,
                'optimization_method': self.optimization_method
            },
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(threshold_data, f)
        
        print(f"Thresholds saved to {filename}")
        return filename
    
    def load_thresholds(self, filename=None, symbol=None, period=None, interval=None):
        """
        Load threshold history from file
        
        Parameters:
        - filename: Filename to load thresholds from (default: None, use default filename)
        - symbol: Symbol to include in filename search (default: None)
        - period: Time period to include in filename search (default: None)
        - interval: Time interval to include in filename search (default: None)
        
        Returns:
        - success: Whether loading was successful
        """
        from dashboard_utils import get_standardized_model_filename
        
        if filename is None:
            # Use the standardized filename format
            filename = get_standardized_model_filename(
                model_type="adaptive_thresholds",
                model_name=self.optimization_method,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".json"
        
        if not os.path.exists(filename):
            print(f"Threshold file {filename} not found")
            return False
        
        try:
            # Load from file
            with open(filename, 'r') as f:
                threshold_data = json.load(f)
            
            # Load configuration if available
            if 'config' in threshold_data:
                config = threshold_data['config']
                self.volatility_window = config.get('volatility_window', self.volatility_window)
                self.lookback_period = config.get('lookback_period', self.lookback_period)
                self.risk_tolerance = config.get('risk_tolerance', self.risk_tolerance)
                self.statistical_method = config.get('statistical_method', self.statistical_method)
                self.optimization_method = config.get('optimization_method', self.optimization_method)
            
            # Load thresholds
            if 'thresholds' in threshold_data:
                self.thresholds = threshold_data['thresholds']
            
            # Load threshold history
            if 'threshold_history' in threshold_data:
                threshold_history_json = threshold_data['threshold_history']
                
                # Convert strings to dates in threshold history
                self.threshold_history = {}
                
                for indicator, history in threshold_history_json.items():
                    self.threshold_history[indicator] = []
                    
                    for entry in history:
                        entry_copy = entry.copy()
                        entry_copy['date'] = pd.to_datetime(entry_copy['date'])
                        self.threshold_history[indicator].append(entry_copy)
            
            # Load threshold statistics
            if 'threshold_stats' in threshold_data:
                self.threshold_stats = threshold_data['threshold_stats']
            
            print(f"Thresholds loaded from {filename}")
            return True
        
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return False
    
    def get_threshold_statistics(self, indicator_col):
        """
        Get statistics for an indicator's thresholds
        
        Parameters:
        - indicator_col: Column name of the indicator
        
        Returns:
        - stats: Dictionary with threshold statistics
        """
        if indicator_col not in self.threshold_stats:
            return None
        
        return self.threshold_stats[indicator_col]
    
    def get_threshold_recommendations(self, df, indicator_col, price_col='close'):
        """
        Get trading recommendations based on adaptive thresholds
        
        Parameters:
        - df: DataFrame with market data
        - indicator_col: Column name of the indicator
        - price_col: Column to use for price (default: 'close')
        
        Returns:
        - recommendation: Dictionary with trading recommendation
        """
        # Get adaptive thresholds
        thresholds = self.get_adaptive_thresholds(df, indicator_col, price_col)
        
        # Get current value
        current_value = df[indicator_col].iloc[-1]
        
        # Determine signal
        if current_value < thresholds['lower']:
            signal = 'buy'
            strength = (thresholds['lower'] - current_value) / (thresholds['lower'] - df[indicator_col].min())
            strength = min(max(strength, 0), 1)  # Ensure between 0 and 1
        elif current_value > thresholds['upper']:
            signal = 'sell'
            strength = (current_value - thresholds['upper']) / (df[indicator_col].max() - thresholds['upper'])
            strength = min(max(strength, 0), 1)  # Ensure between 0 and 1
        else:
            signal = 'neutral'
            # Calculate relative position within the neutral zone
            zone_size = thresholds['upper'] - thresholds['lower']
            if zone_size > 0:
                relative_pos = (current_value - thresholds['lower']) / zone_size
                # Strength is 0 at the middle, increases towards the edges
                strength = abs(relative_pos - 0.5) * 2
            else:
                strength = 0
        
        # Create recommendation
        recommendation = {
            'signal': signal,
            'strength': strength,
            'current_value': current_value,
            'lower_threshold': thresholds['lower'],
            'upper_threshold': thresholds['upper'],
            'indicator': indicator_col,
            'timestamp': df.index[-1]
        }
        
        return recommendation
