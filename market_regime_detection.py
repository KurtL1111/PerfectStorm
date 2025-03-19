"""
Market Regime Detection Module for Perfect Storm Dashboard

This module implements algorithms to:
1. Identify different market regimes (trending, ranging, volatile)
2. Adjust strategy parameters based on the current regime
3. Provide different signal interpretations based on market context
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDataset(Dataset):
    """Dataset class for market regime detection"""
    
    def __init__(self, features, labels=None):
        """
        Initialize the dataset
        
        Parameters:
        - features: Feature tensor
        - labels: Label tensor (optional)
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

class MarketRegimeClassifier(nn.Module):
    """Neural network for market regime classification"""
    
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        """
        Initialize the market regime classifier model
        
        Parameters:
        - input_size: Number of input features
        - hidden_size: Size of hidden layers (default: 64)
        - num_classes: Number of market regime classes (default: 3)
        """
        super(MarketRegimeClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - output: Predicted regime probabilities
        """
        return self.model(x)

class MarketRegimeDetection:
    """Class for market regime detection"""
    
    def __init__(self, lookback_period=252, volatility_window=20, trend_window=50, 
                 n_regimes=3, detection_method='clustering', model_path='models'):
        """
        Initialize the MarketRegimeDetection class
        
        Parameters:
        - lookback_period: Period for historical analysis (default: 252, i.e., 1 year)
        - volatility_window: Window size for volatility calculation (default: 20)
        - trend_window: Window size for trend calculation (default: 50)
        - n_regimes: Number of market regimes to detect (default: 3)
        - detection_method: Method for regime detection 
                           ('clustering', 'hmm', 'ml', default: 'clustering')
        - model_path: Path to save/load models (default: 'models')
        """
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.n_regimes = n_regimes
        self.detection_method = detection_method
        self.model_path = model_path
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize regime history
        self.regime_history = {}
        
        # Initialize regime models
        self.regime_models = {}
        
        # Initialize regime statistics
        self.regime_stats = {}
        
        # Initialize regime transition matrix
        self.transition_matrix = None
        
        # Initialize regime labels
        self.regime_labels = {
            0: 'trending_up',
            1: 'trending_down',
            2: 'ranging',
            3: 'volatile'
        }
        
        # Initialize strategy parameters for each regime
        self.strategy_parameters = {
            'trending_up': {
                'rsi_lower': 40,
                'rsi_upper': 80,
                'macd_threshold': 0,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'position_size': 1.0
            },
            'trending_down': {
                'rsi_lower': 20,
                'rsi_upper': 60,
                'macd_threshold': 0,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'position_size': 0.5
            },
            'ranging': {
                'rsi_lower': 30,
                'rsi_upper': 70,
                'macd_threshold': 0,
                'stop_loss': 0.03,
                'take_profit': 0.08,
                'position_size': 0.7
            },
            'volatile': {
                'rsi_lower': 20,
                'rsi_upper': 80,
                'macd_threshold': 0,
                'stop_loss': 0.08,
                'take_profit': 0.20,
                'position_size': 0.3
            }
        }
        
        # Initialize signal interpretations for each regime
        self.signal_interpretations = {
            'trending_up': {
                'rsi_overbought': 'Potential for continued upward momentum despite overbought conditions',
                'rsi_oversold': 'Strong buy signal - potential reversal to uptrend',
                'macd_positive': 'Confirmation of uptrend strength',
                'macd_negative': 'Potential temporary pullback within uptrend',
                'macd_crossover': 'Entry signal with trend',
                'macd_crossunder': 'Caution signal - monitor for trend change'
            },
            'trending_down': {
                'rsi_overbought': 'Potential shorting opportunity - resistance level',
                'rsi_oversold': 'Potential for continued downward momentum despite oversold conditions',
                'macd_positive': 'Potential temporary bounce within downtrend',
                'macd_negative': 'Confirmation of downtrend strength',
                'macd_crossover': 'Caution signal - monitor for trend change',
                'macd_crossunder': 'Entry signal for short positions'
            },
            'ranging': {
                'rsi_overbought': 'Strong sell/short signal at range resistance',
                'rsi_oversold': 'Strong buy signal at range support',
                'macd_positive': 'Approaching upper range - prepare to sell',
                'macd_negative': 'Approaching lower range - prepare to buy',
                'macd_crossover': 'Buy signal near support level',
                'macd_crossunder': 'Sell signal near resistance level'
            },
            'volatile': {
                'rsi_overbought': 'Extreme caution - high risk of sharp reversal',
                'rsi_oversold': 'Extreme caution - potential value trap',
                'macd_positive': 'Unreliable signal - confirm with other indicators',
                'macd_negative': 'Unreliable signal - confirm with other indicators',
                'macd_crossover': 'Potential false signal - use smaller position size',
                'macd_crossunder': 'Potential false signal - use smaller position size'
            }
        }
    
    def extract_regime_features(self, df, price_col='close', volume_col='volume'):
        """
        Extract features for regime detection
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - volume_col: Column to use for volume (default: 'volume')
        
        Returns:
        - df_features: DataFrame with regime features
        """
        # Make a copy of the DataFrame
        df_features = df.copy()
        
        # Calculate returns
        df_features['returns'] = df_features[price_col].pct_change()
        df_features['log_returns'] = np.log(df_features[price_col] / df_features[price_col].shift(1))
        
        # Calculate volatility features
        df_features['volatility'] = df_features['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)  # Annualized
        df_features['volatility_change'] = df_features['volatility'].pct_change(self.volatility_window)
        df_features['high_low_range'] = (df_features['high'] - df_features['low']) / df_features[price_col]
        
        # Calculate trend features
        df_features['trend'] = df_features[price_col].pct_change(self.trend_window)
        df_features['trend_strength'] = abs(df_features['trend'])
        df_features['sma_20'] = df_features[price_col].rolling(window=20).mean()
        df_features['sma_50'] = df_features[price_col].rolling(window=50).mean()
        df_features['sma_ratio'] = df_features['sma_20'] / df_features['sma_50']
        
        # Calculate momentum features
        df_features['momentum'] = df_features['returns'].rolling(window=self.volatility_window).mean()
        df_features['momentum_change'] = df_features['momentum'].pct_change(self.volatility_window)
        
        # Calculate mean reversion features
        df_features['mean_reversion'] = (df_features[price_col] - df_features[price_col].rolling(window=self.volatility_window).mean()) / df_features[price_col].rolling(window=self.volatility_window).std()
        
        # Calculate volume features if volume column exists
        if volume_col in df_features.columns:
            df_features['volume_change'] = df_features[volume_col].pct_change()
            df_features['volume_ma'] = df_features[volume_col].rolling(window=self.volatility_window).mean()
            df_features['relative_volume'] = df_features[volume_col] / df_features['volume_ma']
            df_features['volume_trend'] = df_features[volume_col].pct_change(self.trend_window)
        
        # Calculate additional technical features
        df_features['rsi'] = self.calculate_rsi(df_features[price_col])
        df_features['rsi_trend'] = df_features['rsi'].pct_change(self.volatility_window)
        
        # Calculate autocorrelation features
        df_features['autocorr_1'] = df_features['returns'].rolling(window=self.lookback_period).apply(lambda x: x.autocorr(lag=1), raw=False)
        df_features['autocorr_5'] = df_features['returns'].rolling(window=self.lookback_period).apply(lambda x: x.autocorr(lag=5), raw=False)
        
        # Calculate skewness and kurtosis
        df_features['returns_skew'] = df_features['returns'].rolling(window=self.lookback_period).skew()
        df_features['returns_kurt'] = df_features['returns'].rolling(window=self.lookback_period).kurt()
        
        # Drop NaN values
        df_features = df_features.dropna()
        
        return df_features
    
    def calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Parameters:
        - prices: Series of prices
        - window: Window size for RSI calculation (default: 14)
        
        Returns:
        - rsi: Series with RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_regimes_clustering(self, df_features, n_regimes=None, method='kmeans'):
        """
        Detect market regimes using clustering
        
        Parameters:
        - df_features: DataFrame with regime features
        - n_regimes: Number of regimes to detect (default: None, use self.n_regimes)
        - method: Clustering method ('kmeans', 'gmm', 'dbscan', default: 'kmeans')
        
        Returns:
        - df_regimes: DataFrame with regime labels
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        # Select features for clustering
        feature_cols = [
            'volatility', 'trend', 'momentum', 'mean_reversion', 
            'rsi', 'autocorr_1', 'returns_skew'
        ]
        
        # Add volume features if available
        if 'relative_volume' in df_features.columns:
            feature_cols.extend(['relative_volume', 'volume_trend'])
        
        # Prepare data for clustering
        X = df_features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(len(feature_cols), 5))
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply clustering based on method
        if method == 'kmeans':
            # Use KMeans clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            
            # Store cluster centers
            self.regime_models['kmeans'] = kmeans
            self.regime_models['pca'] = pca
            self.regime_models['scaler'] = scaler
        
        elif method == 'gmm':
            # Use Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
            labels = gmm.fit_predict(X_pca)
            
            # Store model
            self.regime_models['gmm'] = gmm
            self.regime_models['pca'] = pca
            self.regime_models['scaler'] = scaler
        
        elif method == 'dbscan':
            # Use DBSCAN with volatility-adjusted parameters
            eps = 0.5 * (1 + df_features['volatility'].iloc[-1])
            min_samples = max(5, int(len(X_pca) * 0.01))
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_pca)
            
            # Handle noise points (label -1)
            if -1 in labels:
                noise_indices = np.where(labels == -1)[0]
                if len(noise_indices) < len(X_pca):
                    non_noise_indices = np.where(labels != -1)[0]
                    non_noise_labels = labels[non_noise_indices]
                    non_noise_points = X_pca[non_noise_indices]
                    
                    for idx in noise_indices:
                        distances = np.linalg.norm(X_pca[idx] - non_noise_points, axis=1)
                        nearest_idx = np.argmin(distances)
                        labels[idx] = non_noise_labels[nearest_idx]
            
            # Validate cluster count and fallback if needed
            unique_labels = np.unique(labels)
            if len(unique_labels) != n_regimes:
                kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_pca)
                self.regime_models.update({
                    'kmeans': kmeans,
                    'pca': pca,
                    'scaler': scaler
                })
            else:
                # Store all components for DBSCAN
                self.regime_models.update({
                    'dbscan': dbscan,
                    'pca': pca,
                    'scaler': scaler
                })
        
        return df_features.assign(regime=labels)

    def save_regime_model(self, model_name):
        """Save the current regime detection model to disk"""
        if not self.regime_models:
            raise ValueError("No trained model available to save")
            
        model_path = os.path.join(self.model_path, f"{model_name}_regime_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.regime_models,
                'scaler': self.regime_models['scaler'],
                'pca': self.regime_models['pca'],
                'config': {
                    'lookback_period': self.lookback_period,
                    'volatility_window': self.volatility_window,
                    'trend_window': self.trend_window
                }
            }, f)

    def load_regime_model(self, model_name):
        """Load a previously saved regime detection model"""
        model_path = os.path.join(self.model_path, f"{model_name}_regime_model.pkl")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.regime_models = saved_data['models']
            # Update configuration parameters
            self.lookback_period = saved_data['config']['lookback_period']
            self.volatility_window = saved_data['config']['volatility_window']
            self.trend_window = saved_data['config']['trend_window']

    def analyze_regime_transitions(self):
        """Analyze historical regime transitions and update strategy parameters"""
        # Implementation would go here
        pass

    def validate_regime_classification(self, df_features):
        """Validate regime classification against fundamental market data"""
        # Implementation would go here
        pass

# Rest of the file remains unchanged...
