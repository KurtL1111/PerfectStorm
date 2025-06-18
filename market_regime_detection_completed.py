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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.tools as tls
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
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
from datetime import datetime
from dashboard_utils import get_standardized_model_filename
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('agg')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

class MarketRegimeDataset(Dataset):
    """Dataset class for market regime detection"""
    
    def __init__(self, features, labels=None):
        """
        Initialize the dataset
        
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
    
    def __init__(self, input_size, hidden_size=64, num_classes=4, device='cpu'):
        """
        Initialize the market regime classifier model
        
        Parameters:
        - input_size: Number of input features
        - hidden_size: Size of hidden layers (default: 64)
        - num_classes: Number of market regime classes (default: 4)
        - device: Device to run the model on (default: 'cpu')
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
        self.to(device)
    
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
                 n_regimes=4, detection_method='clustering', model_path='models\Market Regime Models'):
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
        
        # Determine device for GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for MarketRegimeDetection: {self.device}")
        
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
        df_features['log_returns'] = df_features['log_returns'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate volatility features
        df_features['volatility'] = df_features['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)  # Annualized
        df_features['volatility_change'] = df_features['volatility'].pct_change(self.volatility_window)
        df_features['volatility_change'] = df_features['volatility_change'].replace([np.inf, -np.inf], np.nan)
        df_features['high_low_range'] = (df_features['high'] - df_features['low']) / df_features[price_col]
        df_features['high_low_range'] = df_features['high_low_range'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate trend features
        df_features['trend'] = df_features[price_col].pct_change(self.trend_window)
        df_features['trend'] = df_features['trend'].replace([np.inf, -np.inf], np.nan)
        df_features['trend_strength'] = abs(df_features['trend'])
        df_features['sma_20'] = df_features[price_col].rolling(window=20).mean()
        df_features['sma_50'] = df_features[price_col].rolling(window=50).mean()
        df_features['sma_ratio'] = df_features['sma_20'] / df_features['sma_50']
        df_features['sma_ratio'] = df_features['sma_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate momentum features
        df_features['momentum'] = df_features['returns'].rolling(window=self.volatility_window).mean()
        df_features['momentum_change'] = df_features['momentum'].pct_change(self.volatility_window)
        df_features['momentum_change'] = df_features['momentum_change'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate mean reversion features
        std_dev = df_features[price_col].rolling(window=self.volatility_window).std()
        df_features['mean_reversion'] = (df_features[price_col] - df_features[price_col].rolling(window=self.volatility_window).mean()) / std_dev
        df_features['mean_reversion'] = df_features['mean_reversion'].where(std_dev > 1e-10).replace([np.inf, -np.inf], np.nan)
        
        # Calculate volume features if volume column exists
        if volume_col in df_features.columns:
            df_features['volume_change'] = df_features[volume_col].pct_change()
            df_features['volume_change'] = df_features['volume_change'].replace([np.inf, -np.inf], np.nan)
            df_features['volume_ma'] = df_features[volume_col].rolling(window=self.volatility_window).mean()
            df_features['relative_volume'] = df_features[volume_col] / df_features['volume_ma']
            df_features['relative_volume'] = df_features['relative_volume'].replace([np.inf, -np.inf], np.nan)
            df_features['volume_trend'] = df_features[volume_col].pct_change(self.trend_window)
            df_features['volume_trend'] = df_features['volume_trend'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate autocorrelation features
        df_features['autocorr_1'] = df_features['returns'].rolling(window=self.volatility_window).apply(lambda x: x.autocorr(1), raw=False)
        df_features['autocorr_5'] = df_features['returns'].rolling(window=self.volatility_window).apply(lambda x: x.autocorr(5), raw=False)
        
        # Calculate statistical features
        df_features['returns_skew'] = df_features['returns'].rolling(window=self.volatility_window).skew()
        df_features['returns_kurt'] = df_features['returns'].rolling(window=self.volatility_window).kurt()
        
        # Calculate RSI
        df_features['rsi'] = self.calculate_rsi(df_features[price_col])
        
        # Fill NaN values with reasonable defaults or drop them
        df_features = df_features.fillna(0).replace([np.inf, -np.inf], 0)
        
        return df_features
    
    def calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Parameters:
        - prices: Series of prices
        - window: RSI window (default: 14)
        
        Returns:
        - rsi: Series of RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS
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
        
        # Clean data before scaling: replace infinities and very large values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        X = np.clip(X, -1e10, 1e10)
        
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
    
    def detect_regimes_hmm(self, df_features, n_regimes=None):
        """
        Detect market regimes using Hidden Markov Model
        
        Parameters:
        - df_features: DataFrame with regime features
        - n_regimes: Number of regimes to detect (default: None, use self.n_regimes)
        
        Returns:
        - df_regimes: DataFrame with regime labels
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        # Import HMM implementation
        from hmmlearn import hmm
        
        # Select features for HMM
        feature_cols = ['returns', 'volatility', 'trend']
        
        # Prepare data for HMM
        X = df_features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train HMM
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_scaled)
        
        # Predict hidden states
        labels = model.predict(X_scaled)
        
        # Store model
        self.regime_models['hmm'] = model
        self.regime_models['scaler'] = scaler
        
        return df_features.assign(regime=labels)
    
    def detect_regimes_ml(self, df_features, df_target=None, n_regimes=None):
        """
        Detect market regimes using machine learning
        
        Parameters:
        - df_features: DataFrame with regime features
        - df_target: DataFrame with target labels (default: None, use unsupervised approach)
        - n_regimes: Number of regimes to detect (default: None, use self.n_regimes)
        
        Returns:
        - df_regimes: DataFrame with regime labels
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        # Select features for ML
        feature_cols = [
            'volatility', 'trend', 'momentum', 'mean_reversion', 
            'rsi', 'autocorr_1', 'returns_skew', 'returns_kurt',
            'sma_ratio', 'volatility_change', 'momentum_change'
        ]
        
        # Add volume features if available
        if 'relative_volume' in df_features.columns:
            feature_cols.extend(['relative_volume', 'volume_trend'])
        
        # Prepare data for ML
        X = df_features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # If target labels are provided, use supervised learning
        if df_target is not None:
            y = df_target.values
            
            # Use Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train with time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
            
            # Predict labels
            labels = model.predict(X_scaled)
            
            # Store model
            self.regime_models['rf'] = model
            self.regime_models['scaler'] = scaler
        
        else:
            # Use unsupervised learning (clustering)
            return self.detect_regimes_clustering(df_features, n_regimes)
        
        return df_features.assign(regime=labels)
    
    def detect_regimes_nn(self, df_features, df_target=None, n_regimes=None):
        """
        Detect market regimes using neural network
        
        Parameters:
        - df_features: DataFrame with regime features
        - df_target: DataFrame with target labels (default: None, use unsupervised approach)
        - n_regimes: Number of regimes to detect (default: None, use self.n_regimes)
        
        Returns:
        - df_regimes: DataFrame with regime labels
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        # Select features for NN
        feature_cols = [
            'volatility', 'trend', 'momentum', 'mean_reversion', 
            'rsi', 'autocorr_1', 'returns_skew', 'returns_kurt',
            'sma_ratio', 'volatility_change', 'momentum_change'
        ]
        
        # Add volume features if available
        if 'relative_volume' in df_features.columns:
            feature_cols.extend(['relative_volume', 'volume_trend'])
        
        # Prepare data for NN
        X = df_features[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use the class-level device
        device = self.device
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # If target labels are provided, use supervised learning
        if df_target is not None:
            y = df_target.values
            y_tensor = torch.LongTensor(y).to(device)
            
            # Create dataset
            dataset = MarketRegimeDataset(X_tensor, y_tensor)
            
            # Create dataloader with multi-core processing
            batch_size = min(32, len(dataset))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4 if device.type == "cpu" else 0)
            
            # Create model
            input_size = X_tensor.shape[1]
            model = MarketRegimeClassifier(input_size, hidden_size=64, num_classes=n_regimes, device=device)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            num_epochs = 100
            for epoch in range(num_epochs):
                for batch_X, batch_y in dataloader:
                    # Ensure data is on the correct device
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Predict labels
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                labels = predicted.cpu().numpy()
            
            # Store model
            self.regime_models['nn'] = model
            self.regime_models['scaler'] = scaler
        
        else:
            # Use unsupervised learning (clustering)
            return self.detect_regimes_clustering(df_features, n_regimes)
        
        return df_features.assign(regime=labels)
    
    def detect_regimes(self, df, price_col='close', volume_col='volume', method=None):
        """
        Detect market regimes
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - volume_col: Column to use for volume (default: 'volume')
        - method: Method for regime detection (default: None, use self.detection_method)
        
        Returns:
        - df_regimes: DataFrame with regime labels
        """
        if method is None:
            method = self.detection_method
        
        # Extract features
        df_features = self.extract_regime_features(df, price_col, volume_col)
        
        # Detect regimes based on method
        if method == 'clustering':
            df_regimes = self.detect_regimes_clustering(df_features)
        elif method == 'hmm':
            df_regimes = self.detect_regimes_hmm(df_features)
        elif method == 'ml':
            df_regimes = self.detect_regimes_ml(df_features)
        elif method == 'nn':
            df_regimes = self.detect_regimes_nn(df_features)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Map numeric labels to regime names
        regime_map = {i: name for i, name in self.regime_labels.items() if i < self.n_regimes}
        df_regimes['regime_name'] = df_regimes['regime'].map(regime_map)
        
        # Update regime history
        self.update_regime_history(df_regimes)
        
        # Calculate regime statistics
        self.calculate_regime_statistics(df_regimes)
        
        return df_regimes
    
    def update_regime_history(self, df_regimes):
        """
        Update regime history
        
        Parameters:
        - df_regimes: DataFrame with regime labels
        """
        # Extract regime information
        regime_data = df_regimes[['regime', 'regime_name']].copy()
        
        # Add timestamp
        regime_data['timestamp'] = df_regimes.index
        
        # Update regime history
        for _, row in regime_data.iterrows():
            self.regime_history[row['timestamp']] = {
                'regime': int(row['regime']),
                'regime_name': row['regime_name']
            }
    
    def calculate_regime_statistics(self, df_regimes):
        """
        Calculate regime statistics
        
        Parameters:
        - df_regimes: DataFrame with regime labels
        """
        # Group by regime
        grouped = df_regimes.groupby('regime')
        
        # Calculate statistics for each regime
        for regime, group in grouped:
            regime_name = self.regime_labels.get(regime, f"regime_{regime}")
            
            # Calculate return statistics
            returns = group['returns']
            
            self.regime_stats[regime_name] = {
                'count': len(group),
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'min_return': returns.min(),
                'max_return': returns.max(),
                'skew': returns.skew(),
                'kurtosis': returns.kurt(),
                'mean_volatility': group['volatility'].mean(),
                'mean_trend': group['trend'].mean(),
                'mean_rsi': group['rsi'].mean()
            }
            
            # Add volume statistics if available
            if 'relative_volume' in group.columns:
                self.regime_stats[regime_name].update({
                    'mean_relative_volume': group['relative_volume'].mean(),
                    'mean_volume_trend': group['volume_trend'].mean()
                })
        
        # Calculate transition matrix
        self.calculate_transition_matrix(df_regimes)
    
    def calculate_transition_matrix(self, df_regimes):
        """
        Calculate regime transition matrix
        
        Parameters:
        - df_regimes: DataFrame with regime labels
        """
        # Get regime sequence
        regimes = df_regimes['regime'].values
        
        # Initialize transition matrix
        n_regimes = max(self.n_regimes, max(regimes) + 1)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize by row sums
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                     out=np.zeros_like(transition_matrix), where=row_sums!=0)
        
        # Store transition matrix
        self.transition_matrix = transition_matrix
    
    def get_current_regime(self, df_regimes):
        """
        Get current market regime
        
        Parameters:
        - df_regimes: DataFrame with regime labels
        
        Returns:
        - current_regime: Current regime information
        """
        # Get last row
        last_row = df_regimes.iloc[-1]
        
        # Get regime information
        regime = int(last_row['regime'])
        regime_name = last_row['regime_name']
        
        # Get strategy parameters for current regime
        strategy_params = self.strategy_parameters.get(regime_name, {})
        
        # Get signal interpretations for current regime
        signal_interpretations = self.signal_interpretations.get(regime_name, {})
        
        # Create current regime dictionary
        current_regime = {
            'regime': regime,
            'regime_name': regime_name,
            'strategy_parameters': strategy_params,
            'signal_interpretations': signal_interpretations
        }
        
        return current_regime
    
    def predict_next_regime(self, current_regime):
        """
        Predict next market regime
        
        Parameters:
        - current_regime: Current regime information
        
        Returns:
        - next_regime_probs: Probabilities of next regimes
        """
        if self.transition_matrix is None:
            return None
        
        # Get current regime
        regime = current_regime['regime']
        
        # Get transition probabilities
        transition_probs = self.transition_matrix[regime]
        
        # Create dictionary of next regime probabilities
        next_regime_probs = {
            self.regime_labels.get(i, f"regime_{i}"): prob 
            for i, prob in enumerate(transition_probs)
        }
        
        return next_regime_probs
    
    def plot_regimes(self, df, df_regimes, price_col='close', figsize=(12, 8)):
        """
        Plot market regimes
        
        Parameters:
        - df: DataFrame with market data
        - df_regimes: DataFrame with regime labels
        - price_col: Column to use for price (default: 'close')
        - figsize: Figure size (default: (12, 8))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(df.index, df[price_col], label=price_col.capitalize())
        
        # Color background by regime
        for regime in range(self.n_regimes):
            regime_name = self.regime_labels.get(regime, f"regime_{regime}")
            regime_data = df_regimes[df_regimes['regime'] == regime]
            
            if len(regime_data) > 0:
                # Get color based on regime
                if 'up' in regime_name:
                    color = 'lightgreen'
                elif 'down' in regime_name:
                    color = 'lightcoral'
                elif 'ranging' in regime_name:
                    color = 'lightskyblue'
                elif 'volatile' in regime_name:
                    color = 'lightyellow'
                else:
                    color = f"C{regime}"
                
                # Highlight regime periods
                for i in range(len(regime_data) - 1):
                    start_idx = regime_data.index[i]
                    end_idx = regime_data.index[i + 1]
                    ax1.axvspan(start_idx, end_idx, alpha=0.3, color=color)
        
        # Set labels and title for price chart
        ax1.set_title(f'{price_col.capitalize()} with Market Regimes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create a categorical y-axis for regimes with names instead of numbers
        regime_mapping = {regime: i for i, regime in enumerate(self.regime_labels.values())}
        df_regimes['regime_numeric'] = df_regimes['regime_name'].map(regime_mapping)
        
        # Plot regime as scatter with regime names
        scatter = ax2.scatter(df_regimes.index, df_regimes['regime_numeric'], c=df_regimes['regime'], cmap='viridis')
        
        # Set labels and title for regime chart
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Regime')
        ax2.set_title('Market Regime')
        ax2.grid(True, alpha=0.3)
        
        # Set y-ticks to regime names
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_yticklabels([self.regime_labels.get(i, f"regime_{i}") for i in range(self.n_regimes)])
        
        # Adjust layout
        plt.tight_layout()
        fig_plotly = self.convert_mpl_to_plotly(fig)
        plt.close(fig)
        return fig_plotly
    
    def plot_transition_matrix(self, figsize=(8, 6)):
        """
        Plot regime transition matrix
        
        Parameters:
        - figsize: Figure size (default: (8, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        if self.transition_matrix is None:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get regime names for labels
        regime_names = [self.regime_labels.get(i, f"regime_{i}") for i in range(len(self.transition_matrix))]
        
        # Plot transition matrix
        im = ax.imshow(self.transition_matrix, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels and title
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        ax.set_title('Regime Transition Matrix')
        
        # Set ticks to regime names
        ax.set_xticks(range(len(regime_names)))
        ax.set_yticks(range(len(regime_names)))
        ax.set_xticklabels(regime_names, rotation=45, ha='right')
        ax.set_yticklabels(regime_names)
        
        # Add text annotations
        for i in range(len(self.transition_matrix)):
            for j in range(len(self.transition_matrix)):
                text = ax.text(j, i, f"{self.transition_matrix[i, j]:.2f}",
                              ha="center", va="center", color="w" if self.transition_matrix[i, j] > 0.5 else "black")
        
        # Adjust layout
        plt.tight_layout()
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def save_model(self, filename=None, symbol=None, period=None, interval=None):
        """
        Save model to file

        Parameters:
        - filename: Filename to save model (default: None, use default filename)
        - symbol: Symbol to incorporate into filename (default: None)
        - period: Time period to incorporate into filename (default: None)
        - interval: Time interval to incorporate into filename (default: None)
        
        Returns:
        - filename: Filename of saved model
        """
        from dashboard_utils import get_standardized_model_filename
        
        if filename is None:
            # Use the standardized filename format
            filename = get_standardized_model_filename(
                model_type="regime_detection",
                model_name=self.detection_method,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pkl"
        
        # Create dictionary with model data
        model_data = {
            'config': {
                'lookback_period': self.lookback_period,
                'volatility_window': self.volatility_window,
                'trend_window': self.trend_window,
                'n_regimes': self.n_regimes,
                'detection_method': self.detection_method
            },
            'regime_labels': self.regime_labels,
            'strategy_parameters': self.strategy_parameters,
            'signal_interpretations': self.signal_interpretations,
            'regime_history': self.regime_history,
            'regime_stats': self.regime_stats,
            'transition_matrix': self.transition_matrix,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save model data
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save regime models separately if they exist 
        if self.regime_models:
            # Use the same naming convention for consistency
            models_filename = os.path.splitext(filename)[0] + "_models.pkl"
            with open(models_filename, 'wb') as f:
                pickle.dump(self.regime_models, f)
        
        print(f"Model saved to {filename}")
        return filename

    def load_model(self, filename=None, symbol=None, period=None, interval=None):
        """
        Load model from file
        
        Parameters:
        - filename: Filename to load model from (default: None, use default filename)
        - symbol: Symbol to incorporate into filename (default: None)
        - period: Time period to incorporate into filename (default: None)
        - interval: Interval to incorporate into filename (default: None)
        Returns:
        - success: Whether loading was successful
        """
        from dashboard_utils import get_standardized_model_filename

        symbol_str = f"_{symbol}" if symbol else ""
        period_str = f"_{period}" if period else ""
        interval_str = f"_{interval}" if interval else ""

        if filename is None:
            # Use get_standardized_model_filename to get the base (without timestamp)
            # Remove timestamp for loading so we can search for all matching files
            filename_base = get_standardized_model_filename(
                model_type="regime_detection",
                model_name=self.detection_method,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path,
                include_timestamp=False,
                extension="pkl"
            )
            # filename_base will be like .../regime_detection_method_symbol_period_interval.pkl
            # But saved files may have _YYYYMMDD before .pkl, so search for all matching
            base_name = os.path.splitext(os.path.basename(filename_base))[0]
            search_dir = os.path.dirname(filename_base)
            # Find all files that start with base_name and end with .pkl
            potential_files = [f for f in os.listdir(search_dir) if f.startswith(base_name) and f.endswith(".pkl")]
            if not potential_files:
                print(f"No model files found for prefix {base_name} in {search_dir}")
                return False
            # Sort by timestamp if present, else by name
            def extract_timestamp(f):
                import re
                m = re.search(r"_(\d{8})\.pkl$", f)
                if m:
                    return m.group(1)
                return "00000000"  # Put undated files first
            potential_files.sort(key=extract_timestamp, reverse=True)
            filename = os.path.join(search_dir, potential_files[0])
            print(f"Attempting to load latest found model: {filename}")
        
        try:
            # Load model data
            with open(filename, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Update model attributes
            self.regime_labels = saved_data['regime_labels']
            self.strategy_parameters = saved_data['strategy_parameters']
            self.signal_interpretations = saved_data['signal_interpretations']
            self.regime_history = saved_data['regime_history']
            self.regime_stats = saved_data['regime_stats']
            self.transition_matrix = saved_data['transition_matrix']
            
            # Update configuration parameters
            self.lookback_period = saved_data['config']['lookback_period']
            self.volatility_window = saved_data['config']['volatility_window']
            self.trend_window = saved_data['config']['trend_window']
            self.n_regimes = saved_data['config']['n_regimes']
            self.detection_method = saved_data['config']['detection_method']
            
            # Optionally, load regime models if needed (try to match interval as well)
            models_base = f"market_regime_models_{self.detection_method}{symbol_str}{period_str}{interval_str}"
            models_files = [f for f in os.listdir(self.model_path) if f.startswith(models_base) and f.endswith(".pkl")]
            if models_files:
                # Pick the latest by timestamp if present
                def extract_models_timestamp(f):
                    import re
                    m = re.search(r"_(\d{8})\.pkl$", f)
                    if m:
                        return m.group(1)
                    return "00000000"
                models_files.sort(key=extract_models_timestamp, reverse=True)
                models_filename = os.path.join(self.model_path, models_files[0])
                with open(models_filename, 'rb') as f:
                    self.regime_models = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_regime_transitions(self):
        """
        Analyze historical regime transitions and update strategy parameters
        
        This function analyzes the historical transitions between different market regimes,
        calculates performance metrics for each transition type, and updates strategy parameters
        based on the analysis to optimize trading performance.
        
        Returns:
        - transition_analysis: Dictionary with transition analysis results
        """
        if not self.regime_history or self.transition_matrix is None:
            return None
        
        # Convert regime history to DataFrame for analysis
        history_data = []
        for timestamp, data in self.regime_history.items():
            history_data.append({
                'timestamp': timestamp,
                'regime': data['regime'],
                'regime_name': data['regime_name']
            })
        
        history_df = pd.DataFrame(history_data)
        history_df = history_df.sort_values('timestamp')
        
        # Identify regime transitions
        history_df['prev_regime'] = history_df['regime'].shift(1)
        history_df['prev_regime_name'] = history_df['regime_name'].shift(1)
        history_df['is_transition'] = history_df['regime'] != history_df['prev_regime']
        
        # Filter to only transition points
        transitions_df = history_df[history_df['is_transition']].copy()
        
        # Calculate transition frequencies
        transition_counts = {}
        for from_regime in range(self.n_regimes):
            from_name = self.regime_labels.get(from_regime, f"regime_{from_regime}")
            transition_counts[from_name] = {}
            
            for to_regime in range(self.n_regimes):
                to_name = self.regime_labels.get(to_regime, f"regime_{to_regime}")
                count = len(transitions_df[(transitions_df['prev_regime'] == from_regime) & 
                                          (transitions_df['regime'] == to_regime)])
                transition_counts[from_name][to_name] = count
        
        # Calculate average duration of each regime
        regime_durations = {}
        current_regime = None
        current_start = None
        durations = []
        
        for i, row in history_df.iterrows():
            if row['regime_name'] != current_regime:
                if current_regime is not None:
                    duration = (row['timestamp'] - current_start).total_seconds() / (60 * 60 * 24)  # in days
                    durations.append(duration)
                    
                    if current_regime not in regime_durations:
                        regime_durations[current_regime] = []
                    
                    regime_durations[current_regime].append(duration)
                
                current_regime = row['regime_name']
                current_start = row['timestamp']
        
        # Calculate average durations
        avg_durations = {regime: np.mean(durations) for regime, durations in regime_durations.items()}
        
        # Analyze performance during transitions
        # This would require price data which we don't have in this context
        # Instead, we'll use the regime statistics we already have
        
        # Update strategy parameters based on transition analysis
        updated_parameters = {}
        
        for regime_name, params in self.strategy_parameters.items():
            # Make a copy of the original parameters
            updated_params = params.copy()
            
            # Get regime statistics
            regime_stats = self.regime_stats.get(regime_name, {})
            
            if regime_stats:
                # Adjust RSI thresholds based on regime statistics
                mean_rsi = regime_stats.get('mean_rsi', 50)
                if 'trending_up' in regime_name:
                    # In uptrends, RSI tends to stay higher
                    updated_params['rsi_lower'] = max(30, min(45, mean_rsi - 20))
                    updated_params['rsi_upper'] = max(70, min(85, mean_rsi + 20))
                elif 'trending_down' in regime_name:
                    # In downtrends, RSI tends to stay lower
                    updated_params['rsi_lower'] = max(20, min(35, mean_rsi - 15))
                    updated_params['rsi_upper'] = max(55, min(70, mean_rsi + 15))
                elif 'ranging' in regime_name:
                    # In ranging markets, use standard RSI thresholds
                    updated_params['rsi_lower'] = 30
                    updated_params['rsi_upper'] = 70
                elif 'volatile' in regime_name:
                    # In volatile markets, widen the RSI thresholds
                    updated_params['rsi_lower'] = 20
                    updated_params['rsi_upper'] = 80
                
                # Adjust position sizing based on volatility
                mean_volatility = regime_stats.get('mean_volatility', 0.2)
                if mean_volatility > 0.3:  # High volatility
                    updated_params['position_size'] = min(0.5, params['position_size'])
                elif mean_volatility < 0.1:  # Low volatility
                    updated_params['position_size'] = min(1.0, params['position_size'] * 1.2)
                
                # Adjust stop loss and take profit based on volatility
                updated_params['stop_loss'] = max(0.02, min(0.1, mean_volatility * 0.5))
                updated_params['take_profit'] = max(0.05, min(0.25, mean_volatility * 1.5))
            
            updated_parameters[regime_name] = updated_params
        
        # Update strategy parameters
        self.strategy_parameters = updated_parameters
        
        # Create transition analysis dictionary
        transition_analysis = {
            'transition_matrix': self.transition_matrix.tolist(),
            'transition_counts': transition_counts,
            'avg_regime_durations': avg_durations,
            'updated_parameters': updated_parameters
        }
        
        return transition_analysis
    
    def validate_regime_classification(self, df_features):
        """
        Validate regime classification against fundamental market data
        
        This function evaluates the accuracy and effectiveness of the regime classification
        by comparing it with fundamental market indicators, calculating performance metrics,
        and providing confidence scores for the classification.
        
        Parameters:
        - df_features: DataFrame with regime features and classifications
        
        Returns:
        - validation_results: Dictionary with validation results
        """
        if 'regime' not in df_features.columns:
            return None
        
        # Calculate silhouette score to measure clustering quality
        if self.detection_method == 'clustering':
            # Select features for validation
            feature_cols = [
                'volatility', 'trend', 'momentum', 'mean_reversion', 
                'rsi', 'autocorr_1', 'returns_skew'
            ]
            
            # Add volume features if available
            if 'relative_volume' in df_features.columns:
                feature_cols.extend(['relative_volume', 'volume_trend'])
            
            # Prepare data for validation
            X = df_features[feature_cols].values
            labels = df_features['regime'].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate silhouette score
            try:
                silhouette_avg = silhouette_score(X_scaled, labels)
            except:
                silhouette_avg = 0
        else:
            silhouette_avg = None
        
        # Validate against market fundamentals
        # 1. Check if trending regimes align with actual price trends
        trend_alignment = {}
        for regime in range(self.n_regimes):
            regime_name = self.regime_labels.get(regime, f"regime_{regime}")
            regime_data = df_features[df_features['regime'] == regime]
            
            if len(regime_data) > 0:
                # Calculate average trend
                avg_trend = regime_data['trend'].mean()
                
                # Check alignment
                if 'trending_up' in regime_name and avg_trend > 0:
                    trend_alignment[regime_name] = True
                elif 'trending_down' in regime_name and avg_trend < 0:
                    trend_alignment[regime_name] = True
                elif 'ranging' in regime_name and abs(avg_trend) < 0.05:
                    trend_alignment[regime_name] = True
                elif 'volatile' in regime_name and regime_data['volatility'].mean() > df_features['volatility'].mean():
                    trend_alignment[regime_name] = True
                else:
                    trend_alignment[regime_name] = False
        
        # 2. Check if volatility regimes align with actual volatility
        volatility_alignment = {}
        for regime in range(self.n_regimes):
            regime_name = self.regime_labels.get(regime, f"regime_{regime}")
            regime_data = df_features[df_features['regime'] == regime]
            
            if len(regime_data) > 0:
                # Calculate average volatility
                avg_volatility = regime_data['volatility'].mean()
                overall_avg_volatility = df_features['volatility'].mean()
                
                # Check alignment
                if 'volatile' in regime_name and avg_volatility > overall_avg_volatility:
                    volatility_alignment[regime_name] = True
                elif 'ranging' in regime_name and avg_volatility < overall_avg_volatility:
                    volatility_alignment[regime_name] = True
                elif 'trending' in regime_name:
                    # Trending markets can have varying volatility
                    volatility_alignment[regime_name] = True
                else:
                    volatility_alignment[regime_name] = False
        
        # Calculate overall confidence score
        confidence_scores = {}
        for regime in range(self.n_regimes):
            regime_name = self.regime_labels.get(regime, f"regime_{regime}")
            
            # Base confidence on alignment with fundamentals
            trend_score = 1 if trend_alignment.get(regime_name, False) else 0
            volatility_score = 1 if volatility_alignment.get(regime_name, False) else 0
            
            # Add silhouette score if available
            if silhouette_avg is not None:
                confidence = (trend_score + volatility_score + max(0, silhouette_avg)) / 3
            else:
                confidence = (trend_score + volatility_score) / 2
            
            confidence_scores[regime_name] = confidence
        
        # Create validation results dictionary
        validation_results = {
            'silhouette_score': silhouette_avg,
            'trend_alignment': trend_alignment,
            'volatility_alignment': volatility_alignment,
            'confidence_scores': confidence_scores
        }
        
        return validation_results
    
    def train(self, df, price_col='close', volume_col='volume', method=None):
        """
        Train the market regime detection model
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - volume_col: Column to use for volume (default: 'volume')
        - method: Method for regime detection (default: None, use self.detection_method)
        
        Returns:
        - df_regimes: DataFrame with regime labels
        """
        # Detect regimes
        df_regimes = self.detect_regimes(df, price_col, volume_col, method)
        
        # Validate regime classification
        validation_results = self.validate_regime_classification(df_regimes)
        
        # Analyze regime transitions
        transition_analysis = self.analyze_regime_transitions()
        
        # Save model
        self.save_model()
        
        return df_regimes
    
    def predict(self, df, price_col='close', volume_col='volume'):
        """
        Predict market regimes for new data
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - volume_col: Column to use for volume (default: 'volume')
        
        Returns:
        - df_regimes: DataFrame with regime labels
        - current_regime: Current regime information
        - next_regime_probs: Probabilities of next regimes
        """
        # Check if model is trained
        if not self.regime_models:
            raise ValueError("Model not trained. Call train() first.")
        
        # Detect regimes
        df_regimes = self.detect_regimes(df, price_col, volume_col)
        
        # Get current regime
        current_regime = self.get_current_regime(df_regimes)
        
        # Predict next regime
        next_regime_probs = self.predict_next_regime(current_regime)
        
        return {
            'df_regimes': df_regimes,
            'current_regime': current_regime,
            'next_regime_probs': next_regime_probs
        }
    
    def evaluate(self, df, price_col='close', volume_col='volume'):
        """
        Evaluate market regime detection performance
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - volume_col: Column to use for volume (default: 'volume')
        
        Returns:
        - evaluation_results: Dictionary with evaluation results
        """
        # Detect regimes
        df_regimes = self.detect_regimes(df, price_col, volume_col)
        
        # Validate regime classification
        validation_results = self.validate_regime_classification(df_regimes)
        
        # Calculate regime statistics
        regime_stats = self.regime_stats.copy()
        
        # Calculate transition matrix
        transition_matrix = self.transition_matrix.copy() if self.transition_matrix is not None else None
        
        # Create evaluation results dictionary
        evaluation_results = {
            'validation_results': validation_results,
            'regime_stats': regime_stats,
            'transition_matrix': transition_matrix
        }
        
        return evaluation_results
    
    def plot_regime_statistics(self, figsize=(12, 8)):
        """
        Plot regime statistics
        
        Parameters:
        - figsize: Figure size (default: (12, 8))
        
        Returns:
        - fig: Matplotlib figure
        """
        if not self.regime_stats:
            return None
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data for plotting
        regimes = list(self.regime_stats.keys())
        mean_returns = [stats['mean_return'] for stats in self.regime_stats.values()]
        sharpe_ratios = [stats['sharpe'] for stats in self.regime_stats.values()]
        volatilities = [stats['mean_volatility'] for stats in self.regime_stats.values()]
        counts = [stats['count'] for stats in self.regime_stats.values()]
        
        # Plot mean returns
        axs[0, 0].bar(regimes, mean_returns)
        axs[0, 0].set_title('Mean Returns by Regime')
        axs[0, 0].set_ylabel('Mean Return')
        axs[0, 0].grid(True, alpha=0.3)
        plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Plot Sharpe ratios
        axs[0, 1].bar(regimes, sharpe_ratios)
        axs[0, 1].set_title('Sharpe Ratio by Regime')
        axs[0, 1].set_ylabel('Sharpe Ratio')
        axs[0, 1].grid(True, alpha=0.3)
        plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Plot volatilities
        axs[1, 0].bar(regimes, volatilities)
        axs[1, 0].set_title('Volatility by Regime')
        axs[1, 0].set_ylabel('Volatility')
        axs[1, 0].grid(True, alpha=0.3)
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Plot counts
        axs[1, 1].bar(regimes, counts)
        axs[1, 1].set_title('Frequency by Regime')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].grid(True, alpha=0.3)
        plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def plot_regime_returns_distribution_plotly(self, df_regimes, n_regimes=None, regime_labels=None):
        """
        Plot returns distribution for each market regime using Plotly
        
        Parameters:
        - df_regimes: DataFrame with regime labels
        - n_regimes: Number of regimes (default: None, use self.n_regimes)
        - regime_labels: Dictionary of regime labels (default: None, use self.regime_labels)
        
        Returns:
        - fig: Plotly figure
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        if regime_labels is None:
            regime_labels = self.regime_labels
        
        # Create subplots: one row per regime
        fig = make_subplots(rows=n_regimes, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=[f"Returns Distribution - {regime_labels.get(i, f'regime_{i}')}" for i in range(n_regimes)])
        
        # Loop through each regime
        for regime in range(n_regimes):
            regime_name = regime_labels.get(regime, f"regime_{regime}")
            regime_data = df_regimes[df_regimes['regime'] == regime]
            if not regime_data.empty:
                # Histogram trace
                hist = go.Histogram(
                    x=regime_data['returns'],
                    nbinsx=30,
                    histnorm='probability density',
                    marker_color='skyblue',
                    opacity=0.6,
                    name=f"{regime_name} Hist"
                )
                fig.add_trace(hist, row=regime+1, col=1)
                
                # KDE trace if possible
                try:
                    counts, bins = np.histogram(regime_data['returns'], bins=30, density=True)
                    kde = gaussian_kde(regime_data['returns'])
                    x_vals = np.linspace(bins[0], bins[-1], 100)
                    y_vals = kde(x_vals)
                    
                    kde_trace = go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        line=dict(color='darkblue', width=2),
                        name=f"{regime_name} KDE"
                    )
                    fig.add_trace(kde_trace, row=regime+1, col=1)
                except Exception as e:
                    print(f"Failed to compute KDE for regime {regime_name}: {e}")
                
                # Vertical lines for mean and std
                mean_return = regime_data['returns'].mean()
                std_return = regime_data['returns'].std()
                
                for x, dash, color, label in [
                    (mean_return, "dash", "red", f"Mean: {mean_return:.4f}"),
                    (mean_return + std_return, "dot", "green", f"Mean+Std: {mean_return+std_return:.4f}"),
                    (mean_return - std_return, "dot", "green", f"Mean-Std: {mean_return-std_return:.4f}")
                ]:
                    fig.add_shape(
                        type="line",
                        x0=x, x1=x,
                        y0=0, y1=max(y_vals) if 'y_vals' in locals() else 1,
                        line=dict(color=color, dash=dash),
                        row=regime+1, col=1
                    )
                    
                # Update axis
                fig.update_xaxes(title_text="Returns", row=regime+1, col=1)
                fig.update_yaxes(title_text="Density", row=regime+1, col=1)
        
        fig.update_layout(title="Returns Distribution by Regime",
                        barmode='overlay', showlegend=True,
                        height=400*n_regimes)
        return fig

    @staticmethod
    def convert_mpl_to_plotly(fig):
        import plotly.tools as tls
        import plotly.graph_objects as go

        # Convert Matplotlib figure to Plotly figure
        plotly_fig = tls.mpl_to_plotly(fig)

        # Post-process layout shapes: if any shape’s line dash is "circle", set it to "solid"
        if hasattr(plotly_fig.layout, "to_plotly_json"):
            layout_dict = plotly_fig.layout.to_plotly_json()
        else:
            layout_dict = dict(plotly_fig.layout)
        if "shapes" in layout_dict:
            for shape in layout_dict["shapes"]:
                if "line" in shape and shape["line"].get("dash") == "circle":
                    shape["line"]["dash"] = "solid"
            plotly_fig.update_layout(shapes=layout_dict["shapes"])

        return plotly_fig.to_dict()
    
    def generate_regime_report(self, df, price_col='close', volume_col='volume', method=None, filename=None, symbol=None, period=None):
        """
        Generate comprehensive market regime report
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column to use for price (default: 'close')
        - volume_col: Column to use for volume (default: 'volume')
        - method: Method for regime detection (default: None, use self.detection_method)
        - filename: Filename to save the report (default: None, displays the report)
        - symbol: Symbol to incorporate into filenames (default: None)
        - period: Time period to incorporate into filenames (default: None)
        
        Returns:
        - report: Dictionary with report components
        """
        # Create formatted strings for symbol and period
        symbol_str = f"_{symbol}" if symbol else ""
        period_str = f"_{period}" if period else ""
        filename_suffix = f"{symbol_str}{period_str}"
        
        # Build default model filename which includes detection_method, symbol, and period
        model_filename = os.path.join(self.model_path, f'market_regime_detection_{self.detection_method}{filename_suffix}.pkl')
        
        # Check if a saved model is available and load it; if not, train a new model.
        if os.path.exists(model_filename):
            print(f"Found existing model at {model_filename}, attempting to load...")
            load_success = self.load_model(model_filename, symbol=symbol, period=period)
            if not load_success:
                print("Loading model failed, will detect regimes from scratch")
                df_regimes = self.detect_regimes(df, price_col, volume_col, method)
        else:
            print("No existing model found, detecting regimes from scratch...")
            df_regimes = self.detect_regimes(df, price_col, volume_col, method)
        
        # Get current regime
        print("Generating regime report...")
        current_regime = self.get_current_regime(df_regimes)
        
        # Predict next regime
        print("Predicting next regime probabilities...")
        next_regime_probs = self.predict_next_regime(current_regime)
        
        # Validate regime classification
        print("Validating regime classification...")
        validation_results = self.validate_regime_classification(df_regimes)
        
        # Analyze regime transitions
        print("Analyzing regime transitions...")
        transition_analysis = self.analyze_regime_transitions()
        
        # Create report components
        report = {
            'df_regimes': df_regimes,
            'current_regime': current_regime,
            'next_regime_probs': next_regime_probs,
            'validation_results': validation_results,
            'transition_analysis': transition_analysis,
            'regime_stats': self.regime_stats
        }
        
        # Generate visualizations
        figs = {}
        
        # Regimes over time
        regimes_fig = self.plot_regimes(df, df_regimes, price_col)
        figs['regimes'] = regimes_fig
        
        # Transition matrix
        transition_fig = self.plot_transition_matrix()
        if transition_fig is not None:
            figs['transition_matrix'] = transition_fig
        
        # Regime statistics
        stats_fig = self.plot_regime_statistics()
        if stats_fig is not None:
            figs['regime_statistics'] = stats_fig
        
        # Returns distribution by regime
        returns_fig = self.plot_regime_returns_distribution_plotly(df_regimes)
        if returns_fig is not None:
            figs['returns_distribution'] = returns_fig
        
        # Add visualizations to report
        report['visualizations'] = figs
        
        # Save report if filename provided
        if filename is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Save visualizations
            for name, fig in figs.items():
                fig.savefig(f"{filename}_{name}.png", dpi=300, bbox_inches='tight')
            
            # Save model (include symbol and period)
            self.save_model(f"{filename}_model.pkl", symbol=symbol, period=period)
            
            # Save report data
            report_data = {k: v for k, v in report.items() if k != 'visualizations' and k != 'df_regimes'}
            with open(f"{filename}_data.json", 'w') as f:
                json.dump(report_data, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
            
            # Save regimes data
            df_regimes.to_csv(f"{filename}_regimes.csv")
        
        return report
