"""
Enhanced Machine Learning Anomaly Detection Module for Perfect Storm Dashboard

This module implements advanced anomaly detection algorithms to identify unusual
market conditions that could represent opportunities or risks.

Enhancements:
1. Improved Variational Autoencoder (VAE) with attention mechanism
2. Temporal anomaly detection for time series data
3. Ensemble methods combining multiple anomaly detection techniques
4. Optimized ONNX Runtime integration
5. Advanced visualization capabilities
6. Adaptive thresholding based on market volatility
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxruntime as ort
import os
import pickle
import json
import time
from scipy import stats

class MarketAnomalyDataset(Dataset):
    """Dataset class for market anomaly detection"""
    
    def __init__(self, features):
        """
        Initialize the dataset
        
        Parameters:
        - features: Feature tensor
        """
        self.features = features
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        return self.features[idx]

class VariationalAutoencoder(nn.Module):
    """Enhanced Variational Autoencoder model for anomaly detection"""
    
    def __init__(self, input_size, latent_size=10, hidden_size=128):
        """
        Initialize the VAE model
        
        Parameters:
        - input_size: Number of input features
        - latent_size: Size of the latent space (default: 10)
        - hidden_size: Size of hidden layers (default: 128)
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Mean and variance for the latent space
        self.fc_mu = nn.Linear(hidden_size // 2, latent_size)
        self.fc_var = nn.Linear(hidden_size // 2, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Assuming features are normalized to [0, 1]
        )
    
    def encode(self, x):
        """
        Encode input to latent space
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        """
        # Encode
        x = self.encoder(x)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        
        Parameters:
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        
        Returns:
        - z: Sampled latent vector
        """
        # Calculate standard deviation
        std = torch.exp(0.5 * log_var)
        
        # Sample from standard normal distribution
        eps = torch.randn_like(std)
        
        # Reparameterize
        z = mu + eps * std
        
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction
        
        Parameters:
        - z: Latent vector
        
        Returns:
        - reconstruction: Reconstructed input
        """
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - reconstruction: Reconstructed input
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var

class TemporalVariationalAutoencoder(nn.Module):
    """Temporal Variational Autoencoder model for time series anomaly detection"""
    
    def __init__(self, input_size, sequence_length, latent_size=10, hidden_size=64):
        """
        Initialize the Temporal VAE model
        
        Parameters:
        - input_size: Number of input features
        - sequence_length: Length of input sequence
        - latent_size: Size of the latent space (default: 10)
        - hidden_size: Size of hidden layers (default: 64)
        """
        super(TemporalVariationalAutoencoder, self).__init__()
        
        # Encoder (LSTM-based)
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Mean and variance for the latent space
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        
        # Decoder (LSTM-based)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.decoder_fc = nn.Linear(hidden_size, input_size)
    
    def encode(self, x):
        """
        Encode input to latent space
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        """
        # Encode with LSTM
        outputs, (h_n, _) = self.encoder_lstm(x)
        
        # Apply attention
        attention_weights = self.attention(outputs)
        context_vector = torch.sum(outputs * attention_weights, dim=1)
        
        # Get mean and log variance
        mu = self.fc_mu(context_vector)
        log_var = self.fc_var(context_vector)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick
        
        Parameters:
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        
        Returns:
        - z: Sampled latent vector
        """
        # Calculate standard deviation
        std = torch.exp(0.5 * log_var)
        
        # Sample from standard normal distribution
        eps = torch.randn_like(std)
        
        # Reparameterize
        z = mu + eps * std
        
        return z
    
    def decode(self, z, sequence_length):
        """
        Decode latent vector to reconstruction
        
        Parameters:
        - z: Latent vector
        - sequence_length: Length of output sequence
        
        Returns:
        - reconstruction: Reconstructed input
        """
        # Repeat latent vector for each time step
        z_repeated = z.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Decode with LSTM
        outputs, _ = self.decoder_lstm(z_repeated)
        
        # Apply fully connected layer
        reconstruction = self.decoder_fc(outputs)
        
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - reconstruction: Reconstructed input
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decode(z, x.size(1))
        
        return reconstruction, mu, log_var

class MarketAnomalyDetection:
    """Enhanced class for market anomaly detection"""
    
    def __init__(self, latent_size=10, hidden_size=128, learning_rate=0.001, 
                 batch_size=32, num_epochs=100, model_path='models',
                 anomaly_method='vae', use_temporal=False, sequence_length=10,
                 use_ensemble=False):
        """
        Initialize the MarketAnomalyDetection class
        
        Parameters:
        - latent_size: Size of the latent space (default: 10)
        - hidden_size: Size of hidden layers (default: 128)
        - learning_rate: Learning rate for optimization (default: 0.001)
        - batch_size: Batch size for training (default: 32)
        - num_epochs: Number of training epochs (default: 100)
        - model_path: Path to save/load models (default: 'models')
        - anomaly_method: Method for anomaly detection ('vae', 'isolation_forest', 'lof', 'svm', default: 'vae')
        - use_temporal: Whether to use temporal anomaly detection (default: False)
        - sequence_length: Length of sequence for temporal anomaly detection (default: 10)
        - use_ensemble: Whether to use ensemble anomaly detection (default: False)
        """
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.anomaly_method = anomaly_method
        self.use_temporal = use_temporal
        self.sequence_length = sequence_length
        self.use_ensemble = use_ensemble
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Initialize model
        self.model = None
        self.input_size = None
        
        # Initialize ensemble models
        self.ensemble_models = []
        
        # Initialize ONNX session
        self.onnx_session = None
        
        # Initialize metrics history
        self.metrics_history = {}
        
        # Initialize anomaly thresholds
        self.anomaly_threshold = None
        self.dynamic_thresholds = {}
    
    def _prepare_data(self, df, feature_columns=None, add_technical_features=True):
        """
        Prepare data for anomaly detection
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, use all numeric columns)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        
        Returns:
        - X: Feature matrix
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Add technical features if requested
        if add_technical_features:
            # Add returns
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                if col != 'volume' and 'return' not in col:
                    df_copy[f'{col}_return_1d'] = df_copy[col].pct_change()
                    df_copy[f'{col}_return_5d'] = df_copy[col].pct_change(5)
            
            # Add volatility
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                if col != 'volume' and 'return' not in col and 'volatility' not in col:
                    df_copy[f'{col}_volatility_5d'] = df_copy[col].pct_change().rolling(5).std()
                    df_copy[f'{col}_volatility_10d'] = df_copy[col].pct_change().rolling(10).std()
            
            # Add moving averages
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                if col != 'volume' and 'ma' not in col and 'return' not in col and 'volatility' not in col:
                    df_copy[f'{col}_ma_5d'] = df_copy[col].rolling(5).mean()
                    df_copy[f'{col}_ma_10d'] = df_copy[col].rolling(10).mean()
            
            # Add moving average crossovers
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                if col != 'volume' and 'ma' not in col and 'return' not in col and 'volatility' not in col and 'ratio' not in col:
                    ma_5d = f'{col}_ma_5d'
                    ma_10d = f'{col}_ma_10d'
                    if ma_5d in df_copy.columns and ma_10d in df_copy.columns:
                        df_copy[f'{col}_ma_ratio'] = df_copy[ma_5d] / df_copy[ma_10d]
            
            # Add volume features
            if 'volume' in df_copy.columns:
                df_copy['volume_ma_5d'] = df_copy['volume'].rolling(5).mean()
                df_copy['volume_ma_10d'] = df_copy['volume'].rolling(10).mean()
                df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma_5d']
        
        # Drop rows with NaN values
        df_copy = df_copy.dropna()
        
        # Select features
        if feature_columns is None:
            # Use all numeric columns
            feature_columns = df_copy.select_dtypes(include=[np.number]).columns
        
        # Extract features
        X = df_copy[feature_columns].values
        
        return X, df_copy.index, feature_columns
    
    def _prepare_temporal_data(self, df, feature_columns=None, add_technical_features=True):
        """
        Prepare temporal data for anomaly detection
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, use all numeric columns)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        
        Returns:
        - X: Feature sequences
        """
        # Prepare regular data
        X, indices, feature_columns = self._prepare_data(df, feature_columns, add_technical_features)
        
        # Create sequences
        X_sequences = []
        sequence_indices = []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_sequences.append(X[i:i+self.sequence_length])
            sequence_indices.append(indices[i+self.sequence_length-1])
        
        return np.array(X_sequences), sequence_indices, feature_columns
    
    def _create_model(self, input_size):
        """
        Create a new model
        
        Parameters:
        - input_size: Number of input features
        
        Returns:
        - model: PyTorch model or scikit-learn model
        """
        if self.anomaly_method == 'vae':
            if self.use_temporal:
                return TemporalVariationalAutoencoder(input_size, self.sequence_length, self.latent_size, self.hidden_size)
            else:
                return VariationalAutoencoder(input_size, self.latent_size, self.hidden_size)
        elif self.anomaly_method == 'isolat<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>