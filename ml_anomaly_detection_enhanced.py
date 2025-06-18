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
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#import onnx
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
    
    def __init__(self, input_size, latent_size=14, hidden_size=128, device='cpu'):
        """
        Initialize the VAE model
        
        Parameters:
        - input_size: Number of input features
        - latent_size: Size of the latent space (default: 14)
        - hidden_size: Size of hidden layers (default: 128)
        - device: Device to run the model on (default: 'cpu')
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
        # Move model to the specified device
        self.to(device)
    
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
    
    def __init__(self, input_size, sequence_length, latent_size=14, hidden_size=64, device='cpu'):
        """
        Initialize the Temporal VAE model
        
        Parameters:
        - input_size: Number of input features
        - sequence_length: Length of input sequence
        - latent_size: Size of the latent space (default: 14)
        - hidden_size: Size of hidden layers (default: 64)
        - device: Device to run the model on (default: 'cpu')
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
    
    def __init__(self, latent_size=14, hidden_size=128, learning_rate=0.001,
                 batch_size=32, num_epochs=100, model_path='models\\Anomaly Detection Models',
                 anomaly_method='ensemble', use_temporal=True, sequence_length=14,
                 use_ensemble=True):
        """
        Initialize the MarketAnomalyDetection class
        
        Parameters:
        - latent_size: Size of the latent space for VAE (default: 14)
        - hidden_size: Size of hidden layers (default: 128)
        - learning_rate: Learning rate for training (default: 0.001)
        - batch_size: Batch size for training (default: 32)
        - num_epochs: Number of training epochs (default: 100)
        - model_path: Path to save/load models (default: 'models')
        - anomaly_method: Method for anomaly detection 
                         ('vae', 'isolation_forest', 'lof', 'svm', 'ensemble', default: 'vae')
        - use_temporal: Whether to use temporal models for time series data (default: False)
        - sequence_length: Length of input sequence for temporal models (default: 14)
        - use_ensemble: Whether to use ensemble methods (default: False)
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

        # Initialize models
        self.vae_model = None
        self.temporal_vae_model = None
        self.isolation_forest_model = None
        self.lof_model = None
        self.svm_model = None

        # For test compatibility: provide a .model attribute (None by default)
        self.model = None

        # Initialize scalers
        self.scaler = None

        # Initialize anomaly scores
        self.anomaly_scores = None
        self.anomaly_thresholds = None

        # Initialize ONNX models
        self.onnx_model_path = None
        self.onnx_session = None

        # Track feature columns used for scaler
        self.feature_cols_fitted = None
    
    def preprocess_data(self, df, feature_cols):
        """
        Preprocess data for anomaly detection
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        
        Returns:
        - features: Preprocessed features
        """
        # Only keep numeric columns from the provided list
        numeric_cols = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        # Save the feature columns used for fitting
        if self.scaler is None:
            self.feature_cols_fitted = numeric_cols.copy()
        # Always use the same columns and order as when scaler was fit
        if self.feature_cols_fitted is not None:
            # Warn if columns are missing
            missing = [col for col in self.feature_cols_fitted if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in input data required by scaler: {missing}")
            features = df[self.feature_cols_fitted].fillna(0).values
        else:
            features = df[numeric_cols].fillna(0).values
        print(f"Anomaly Preprocessing: Initial features shape: {features.shape}")
        # Initialize scaler if not already
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        return features
    
    def prepare_temporal_data(self, features, sequence_length=None):
        """
        Prepare temporal data for time series anomaly detection
        
        Parameters:
        - features: Preprocessed features
        - sequence_length: Length of input sequence (default: None, use self.sequence_length)
        
        Returns:
        - temporal_features: Temporal features
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Create sequences
        temporal_features = []
        
        for i in range(len(features) - sequence_length + 1):
            temporal_features.append(features[i:i+sequence_length])
        
        return np.array(temporal_features)
    
    def train_vae(self, features, device=None, symbol=None, period=None, interval=None):
        """
        Train Variational Autoencoder for anomaly detection
        
        Parameters:
        - features: Preprocessed features
        - device: Device to use for training ('cpu' or 'cuda', default: None, auto-detects GPU if available)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model: Trained VAE model
        """
        # Auto-detect GPU if device is not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device auto-detected for VAE training: {device}")
        from dashboard_utils import get_standardized_model_filename
        
        # Create standardized filename
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name="vae",
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        model_filename = f"{base_filename}.pth"
        
        # Check if model file exists
        if os.path.exists(model_filename):
            input_size = features.shape[1]
            model = VariationalAutoencoder(input_size, self.latent_size, self.hidden_size)
            model.load_state_dict(torch.load(model_filename, map_location=device))
            model.to(device)
            model.eval()
            self.vae_model = model
            return model
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features)
        
        # Create dataset and dataloader
        dataset = MarketAnomalyDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = features.shape[1]
        model = VariationalAutoencoder(input_size, self.latent_size, self.hidden_size, device=device)
        
        # Define loss function and optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                reconstruction, mu, log_var = model(batch)
                
                # Calculate loss
                reconstruction_loss = F.mse_loss(reconstruction, batch, reduction='sum')
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruction_loss + kl_divergence
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
        # Save model
        torch.save(model.state_dict(), model_filename)
        
        # Store model
        self.vae_model = model
        
        return model

    def train_temporal_vae(self, features, device=None, symbol=None, period=None, interval=None):
        """
        Train Temporal Variational Autoencoder for time series anomaly detection
        
        Parameters:
        - features: Preprocessed features
        - device: Device to use for training ('cpu' or 'cuda', default: None, auto-detects GPU if available)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model: Trained Temporal VAE model
        """
        # Auto-detect GPU if device is not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device auto-detected for Temporal VAE training: {device}")
        from dashboard_utils import get_standardized_model_filename
        
        # Create standardized filename
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name="temporal_vae",
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        model_filename = f"{base_filename}.pth"
        
        # Check if model file exists
        if os.path.exists(model_filename):
            input_size = features.shape[1]
            model = TemporalVariationalAutoencoder(input_size, self.sequence_length, self.latent_size, self.hidden_size)
            model.load_state_dict(torch.load(model_filename, map_location=device))
            model.to(device)
            model.eval()
            self.temporal_vae_model = model
            return model
        
        # Prepare temporal data
        temporal_features = self.prepare_temporal_data(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(temporal_features)
        
        # Create dataset and dataloader
        dataset = MarketAnomalyDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = features.shape[1]
        model = TemporalVariationalAutoencoder(input_size, self.sequence_length, self.latent_size, self.hidden_size)
        model.to(device)
        
        # Define loss function and optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                reconstruction, mu, log_var = model(batch)
                
                # Calculate loss
                reconstruction_loss = F.mse_loss(reconstruction, batch, reduction='sum')
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruction_loss + kl_divergence
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
        # Save model
        torch.save(model.state_dict(), model_filename)
        
        # Store model
        self.temporal_vae_model = model
        
        return model

    def train_isolation_forest(self, features, symbol=None, period=None, interval=None):
        """
        Train Isolation Forest for anomaly detection
        
        Parameters:
        - features: Preprocessed features
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model: Trained Isolation Forest model
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Create standardized filename
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name="isolation_forest",
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        model_filename = f"{base_filename}.pkl"
        
        # Check if model file exists
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            self.isolation_forest_model = model
            return model
        
        # Initialize model
        model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,  # Assuming 5% of data are anomalies
            random_state=42
        )
        
        # Train model
        model.fit(features)
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Store model
        self.isolation_forest_model = model
        
        return model

    def train_lof(self, features, symbol=None, period=None, interval=None):
        """
        Train Local Outlier Factor for anomaly detection
        
        Parameters:
        - features: Preprocessed features
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model: Trained LOF model
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Create standardized filename
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name="lof",
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        model_filename = f"{base_filename}.pkl"
        
        # Check if model file exists
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            self.lof_model = model
            return model
        
        # Initialize model
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,  # Assuming 5% of data are anomalies
            novelty=True
        )
        
        # Train model
        model.fit(features)
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Store model
        self.lof_model = model
        
        return model

    def train_svm(self, features, symbol=None, period=None, interval=None):
        """
        Train One-Class SVM for anomaly detection
        
        Parameters:
        - features: Preprocessed features
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model: Trained SVM model
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Create standardized filename
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name="svm",
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        model_filename = f"{base_filename}.pkl"
        
        # Check if model file exists
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            self.svm_model = model
            return model
        
        # Initialize model
        model = OneClassSVM(
            nu=0.05,  # Assuming 5% of data are anomalies
            kernel='rbf',
            gamma='scale'
        )
        
        # Train model
        model.fit(features)
        
        # Save model
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Store model
        self.svm_model = model
        
        return model

    def train_model(self, df, feature_cols, method=None, device='cpu', symbol=None, period=None, interval=None):
        """
        Train anomaly detection model
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        - method: Method for anomaly detection (default: None, use self.anomaly_method)
        - device: Device to use for training ('cpu' or 'cuda', default: 'cpu')
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model: Trained model
        """
        if method is None:
            method = self.anomaly_method
        
        # Preprocess data
        features = self.preprocess_data(df, feature_cols)
        
        # Train model based on method
        if method == 'vae':
            if self.use_temporal:
                model = self.train_temporal_vae(features, device, symbol, period, interval)
            else:
                model = self.train_vae(features, device, symbol, period, interval)
        
        elif method == 'isolation_forest':
            model = self.train_isolation_forest(features, symbol, period, interval)
        
        elif method == 'lof':
            model = self.train_lof(features, symbol, period, interval)
        
        elif method == 'svm':
            model = self.train_svm(features, symbol, period, interval)
        
        elif method == 'ensemble':
            # Train all models
            if self.use_temporal:
                vae_model = self.train_temporal_vae(features, device, symbol, period, interval)
            else:
                vae_model = self.train_vae(features, device, symbol, period, interval)
            
            isolation_forest_model = self.train_isolation_forest(features, symbol, period, interval)
            lof_model = self.train_lof(features, symbol, period, interval)
            svm_model = self.train_svm(features, symbol, period, interval)
            
            # Return dictionary of models
            model = {
                'vae': vae_model,
                'isolation_forest': isolation_forest_model,
                'lof': lof_model,
                'svm': svm_model
            }
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        return model
    
    def calculate_vae_anomaly_scores(self, features, device='cpu'):
        """
        Calculate anomaly scores using VAE
        
        Parameters:
        - features: Preprocessed features
        - device: Device to use for inference ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features).to(device)
        
        # Set model to evaluation mode
        self.vae_model.eval()
        
        # Calculate anomaly scores
        anomaly_scores = []
        
        with torch.no_grad():
            for i in range(len(features)):
                # Get single sample
                sample = features_tensor[i:i+1]
                
                # Forward pass
                reconstruction, mu, log_var = self.vae_model(sample)
                
                # Calculate reconstruction error
                reconstruction_error = F.mse_loss(reconstruction, sample, reduction='sum').item()
                
                # Calculate KL divergence
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).item()
                
                # Anomaly score is a combination of reconstruction error and KL divergence
                anomaly_score = reconstruction_error + kl_divergence
                
                anomaly_scores.append(anomaly_score)
        
        return np.array(anomaly_scores)
    
    def calculate_temporal_vae_anomaly_scores(self, features, device='cpu'):
        """
        Calculate anomaly scores using Temporal VAE
        
        Parameters:
        - features: Preprocessed features
        - device: Device to use for inference ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Prepare temporal data
        temporal_features = self.prepare_temporal_data(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(temporal_features).to(device)
        
        # Set model to evaluation mode
        self.temporal_vae_model.eval()
        
        # Calculate anomaly scores
        anomaly_scores = []
        
        with torch.no_grad():
            for i in range(len(temporal_features)):
                # Get single sample
                sample = features_tensor[i:i+1]
                
                # Forward pass
                reconstruction, mu, log_var = self.temporal_vae_model(sample)
                
                # Calculate reconstruction error
                reconstruction_error = F.mse_loss(reconstruction, sample, reduction='sum').item()
                
                # Calculate KL divergence
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).item()
                
                # Anomaly score is a combination of reconstruction error and KL divergence
                anomaly_score = reconstruction_error + kl_divergence
                
                anomaly_scores.append(anomaly_score)
        
        # Pad the beginning with NaN values
        padding = np.full(self.sequence_length - 1, np.nan)
        anomaly_scores = np.concatenate([padding, anomaly_scores])
        
        return anomaly_scores
    
    def calculate_isolation_forest_anomaly_scores(self, features):
        """
        Calculate anomaly scores using Isolation Forest
        
        Parameters:
        - features: Preprocessed features
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Calculate anomaly scores
        # Note: decision_function returns the negative of the anomaly score
        anomaly_scores = -self.isolation_forest_model.decision_function(features)
        
        return anomaly_scores
    
    def calculate_lof_anomaly_scores(self, features):
        """
        Calculate anomaly scores using Local Outlier Factor
        
        Parameters:
        - features: Preprocessed features
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Calculate anomaly scores
        # Note: decision_function returns the negative of the anomaly score
        anomaly_scores = -self.lof_model.decision_function(features)
        
        return anomaly_scores
    
    def calculate_svm_anomaly_scores(self, features):
        """
        Calculate anomaly scores using One-Class SVM
        
        Parameters:
        - features: Preprocessed features
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Calculate anomaly scores
        # Note: decision_function returns the negative of the anomaly score
        anomaly_scores = -self.svm_model.decision_function(features)
        
        return anomaly_scores
    
    def calculate_ensemble_anomaly_scores(self, features, device='cpu'):
        """
        Calculate anomaly scores using ensemble methods
        
        Parameters:
        - features: Preprocessed features
        - device: Device to use for inference ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Calculate anomaly scores for each method
        if self.use_temporal:
            vae_scores = self.calculate_temporal_vae_anomaly_scores(features, device)
        else:
            vae_scores = self.calculate_vae_anomaly_scores(features, device)
        
        isolation_forest_scores = self.calculate_isolation_forest_anomaly_scores(features)
        lof_scores = self.calculate_lof_anomaly_scores(features)
        svm_scores = self.calculate_svm_anomaly_scores(features)
        
        # Normalize scores
        vae_scores = (vae_scores - np.nanmean(vae_scores)) / np.nanstd(vae_scores)
        isolation_forest_scores = (isolation_forest_scores - np.mean(isolation_forest_scores)) / np.std(isolation_forest_scores)
        lof_scores = (lof_scores - np.mean(lof_scores)) / np.std(lof_scores)
        svm_scores = (svm_scores - np.mean(svm_scores)) / np.std(svm_scores)
        
        # Combine scores (weighted average)
        weights = {
            'vae': 0.4,
            'isolation_forest': 0.2,
            'lof': 0.2,
            'svm': 0.2
        }
        
        # Handle NaN values in VAE scores (for temporal models)
        mask = ~np.isnan(vae_scores)
        ensemble_scores = np.zeros_like(isolation_forest_scores)
        
        # For non-NaN values, use weighted average
        ensemble_scores[mask] = (
            weights['vae'] * vae_scores[mask] +
            weights['isolation_forest'] * isolation_forest_scores[mask] +
            weights['lof'] * lof_scores[mask] +
            weights['svm'] * svm_scores[mask]
        )
        
        # For NaN values, use weighted average of other methods
        if not np.all(mask):
            nan_weights = {
                'isolation_forest': weights['isolation_forest'] / (1 - weights['vae']),
                'lof': weights['lof'] / (1 - weights['vae']),
                'svm': weights['svm'] / (1 - weights['vae'])
            }
            
            ensemble_scores[~mask] = (
                nan_weights['isolation_forest'] * isolation_forest_scores[~mask] +
                nan_weights['lof'] * lof_scores[~mask] +
                nan_weights['svm'] * svm_scores[~mask]
            )
        
        return ensemble_scores
    
    def detect_anomalies(self, df, feature_cols, method=None, device=None):
        """
        Detect anomalies in market data
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        - method: Method for anomaly detection (default: None, use self.anomaly_method)
        - device: Device to use for inference ('cpu' or 'cuda', default: None, auto-detects GPU if available)
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        # Auto-detect GPU if device is not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device auto-detected for anomaly detection: {device}")
        if method is None:
            method = self.anomaly_method
        
        # Preprocess data
        features = self.preprocess_data(df, feature_cols)
        
        # Calculate anomaly scores based on method
        if method == 'vae':
            if self.use_temporal:
                if self.temporal_vae_model is None:
                    raise ValueError("Temporal VAE model not trained. Call train_model first.")
                anomaly_scores = self.calculate_temporal_vae_anomaly_scores(features, device)
            else:
                if self.vae_model is None:
                    raise ValueError("VAE model not trained. Call train_model first.")
                anomaly_scores = self.calculate_vae_anomaly_scores(features, device)
        
        elif method == 'isolation_forest':
            if self.isolation_forest_model is None:
                raise ValueError("Isolation Forest model not trained. Call train_model first.")
            anomaly_scores = self.calculate_isolation_forest_anomaly_scores(features)
        
        elif method == 'lof':
            if self.lof_model is None:
                raise ValueError("LOF model not trained. Call train_model first.")
            anomaly_scores = self.calculate_lof_anomaly_scores(features)
        
        elif method == 'svm':
            if self.svm_model is None:
                raise ValueError("SVM model not trained. Call train_model first.")
            anomaly_scores = self.calculate_svm_anomaly_scores(features)
        
        elif method == 'ensemble':
            if not all([
                (self.use_temporal and self.temporal_vae_model is not None) or 
                (not self.use_temporal and self.vae_model is not None),
                self.isolation_forest_model is not None,
                self.lof_model is not None,
                self.svm_model is not None
            ]):
                raise ValueError("Not all ensemble models are trained. Call train_model first.")
            
            anomaly_scores = self.calculate_ensemble_anomaly_scores(features, device)
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        # Store anomaly scores
        self.anomaly_scores = anomaly_scores
        
        return anomaly_scores
    
    def calculate_adaptive_threshold(self, anomaly_scores, df=None, volatility_col=None, 
                                    base_percentile=95, min_percentile=90, max_percentile=99):
        """
        Calculate adaptive threshold for anomaly detection
        
        Parameters:
        - anomaly_scores: Anomaly scores
        - df: DataFrame with market data (default: None)
        - volatility_col: Column name for volatility (default: None)
        - base_percentile: Base percentile for threshold (default: 95)
        - min_percentile: Minimum percentile for threshold (default: 90)
        - max_percentile: Maximum percentile for threshold (default: 99)
        
        Returns:
        - threshold: Adaptive threshold
        """
        # Calculate base threshold
        base_threshold = np.nanpercentile(anomaly_scores, base_percentile)
        
        # If volatility column not provided, return base threshold
        if df is None or volatility_col is None or volatility_col not in df.columns:
            self.anomaly_thresholds = {
                'base': base_threshold,
                'adaptive': base_threshold
            }
            return base_threshold
        
        # Calculate volatility percentile
        volatility = df[volatility_col].values
        volatility_percentile = np.zeros_like(volatility)
        
        for i in range(len(volatility)):
            if i < 20:  # Not enough history for first few points
                volatility_percentile[i] = 0.5  # Default to middle
            else:
                # Calculate percentile of current volatility relative to history
                volatility_percentile[i] = stats.percentileofscore(volatility[:i], volatility[i]) / 100
        
        # Calculate adaptive threshold
        adaptive_threshold = np.zeros_like(anomaly_scores)
        
        for i in range(len(anomaly_scores)):
            if np.isnan(anomaly_scores[i]):
                adaptive_threshold[i] = np.nan
            else:
                # Adjust percentile based on volatility
                adjusted_percentile = min_percentile + volatility_percentile[i] * (max_percentile - min_percentile)
                
                # Calculate threshold
                if i < 20:  # Not enough history for first few points
                    adaptive_threshold[i] = base_threshold
                else:
                    adaptive_threshold[i] = np.nanpercentile(anomaly_scores[:i], adjusted_percentile)
        
        # Store thresholds
        self.anomaly_thresholds = {
            'base': base_threshold,
            'adaptive': adaptive_threshold
        }
        
        return adaptive_threshold
    
    def export_to_onnx(self, input_size, device='cpu'):
        """
        Export model to ONNX format
        
        Parameters:
        - input_size: Input size for the model
        - device: Device to use for export ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - onnx_model_path: Path to the exported ONNX model
        """
        if self.anomaly_method == 'vae':
            if self.use_temporal:
                if self.temporal_vae_model is None:
                    raise ValueError("Temporal VAE model not trained. Call train_model first.")
                
                # Create dummy input
                dummy_input = torch.randn(1, self.sequence_length, input_size).to(device)
                
                # Export model
                onnx_model_path = os.path.join(self.model_path, 'temporal_vae_model.onnx')
                torch.onnx.export(
                    self.temporal_vae_model,
                    dummy_input,
                    onnx_model_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['reconstruction', 'mu', 'log_var'],
                    dynamic_axes={
                        'input': {0: 'batch_size', 1: 'sequence_length'},
                        'reconstruction': {0: 'batch_size', 1: 'sequence_length'},
                        'mu': {0: 'batch_size'},
                        'log_var': {0: 'batch_size'}
                    }
                )
            else:
                if self.vae_model is None:
                    raise ValueError("VAE model not trained. Call train_model first.")
                
                # Create dummy input
                dummy_input = torch.randn(1, input_size).to(device)
                
                # Export model
                onnx_model_path = os.path.join(self.model_path, 'vae_model.onnx')
                torch.onnx.export(
                    self.vae_model,
                    dummy_input,
                    onnx_model_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['reconstruction', 'mu', 'log_var'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'reconstruction': {0: 'batch_size'},
                        'mu': {0: 'batch_size'},
                        'log_var': {0: 'batch_size'}
                    }
                )
        else:
            raise ValueError(f"ONNX export not supported for method: {self.anomaly_method}")
        
        # Store ONNX model path
        self.onnx_model_path = onnx_model_path
        
        return onnx_model_path
    
    def load_onnx_model(self, onnx_model_path=None):
        """
        Load ONNX model
        
        Parameters:
        - onnx_model_path: Path to the ONNX model (default: None, use self.onnx_model_path)
        
        Returns:
        - session: ONNX Runtime session
        """
        if onnx_model_path is None:
            if self.onnx_model_path is None:
                raise ValueError("ONNX model path not specified. Call export_to_onnx first.")
            onnx_model_path = self.onnx_model_path
        # Load ONNX model
        session = ort.InferenceSession(onnx_model_path)
        # Store ONNX session
        self.onnx_session = session
        
        return session
    
    def detect_anomalies_onnx(self, df, feature_cols):
        """
        Detect anomalies using ONNX model
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        
        Returns:
        - anomaly_scores: Anomaly scores
        """
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded. Call load_onnx_model first.")
        
        # Preprocess data
        features = self.preprocess_data(df, feature_cols)
        
        # Calculate anomaly scores
        anomaly_scores = []
        
        if self.use_temporal:
            # Prepare temporal data
            temporal_features = self.prepare_temporal_data(features)
            
            for i in range(len(temporal_features)):
                # Get single sample
                sample = temporal_features[i:i+1].astype(np.float32)
                
                # Run inference
                outputs = self.onnx_session.run(
                    None,
                    {'input': sample}
                )
                
                # Extract outputs
                reconstruction = outputs[0]
                mu = outputs[1]
                log_var = outputs[2]
                
                # Calculate reconstruction error
                reconstruction_error = np.mean((reconstruction - sample) ** 2)
                
                # Calculate KL divergence
                kl_divergence = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
                
                # Anomaly score is a combination of reconstruction error and KL divergence
                anomaly_score = reconstruction_error + kl_divergence
                
                anomaly_scores.append(anomaly_score)
            
            # Pad the beginning with NaN values
            padding = np.full(self.sequence_length - 1, np.nan)
            anomaly_scores = np.concatenate([padding, anomaly_scores])
        else:
            for i in range(len(features)):
                # Get single sample
                sample = features[i:i+1].astype(np.float32)
                
                # Run inference
                outputs = self.onnx_session.run(
                    None,
                    {'input': sample}
                )
                
                # Extract outputs
                reconstruction = outputs[0]
                mu = outputs[1]
                log_var = outputs[2]
                
                # Calculate reconstruction error
                reconstruction_error = np.mean((reconstruction - sample) ** 2)
                
                # Calculate KL divergence
                kl_divergence = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
                
                # Anomaly score is a combination of reconstruction error and KL divergence
                anomaly_score = reconstruction_error + kl_divergence
                
                anomaly_scores.append(anomaly_score)
        
        # Store anomaly scores
        self.anomaly_scores = np.array(anomaly_scores)
        
        return self.anomaly_scores
    
    def plot_anomaly_scores(self, df, anomaly_scores=None, threshold=None, figsize=(12, 6)):
        """
        Plot anomaly scores
        
        Parameters:
        - df: DataFrame with market data
        - anomaly_scores: Anomaly scores (default: None, use self.anomaly_scores)
        - threshold: Anomaly threshold (default: None, use self.anomaly_thresholds)
        - figsize: Figure size (default: (12, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        if anomaly_scores is None:
            if self.anomaly_scores is None:
                raise ValueError("Anomaly scores not calculated. Call detect_anomalies first.")
            anomaly_scores = self.anomaly_scores
        
        if threshold is None:
            if self.anomaly_thresholds is None:
                # Calculate threshold
                threshold = np.nanpercentile(anomaly_scores, 95)
            else:
                threshold = self.anomaly_thresholds['base']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot anomaly scores
        ax.plot(df.index, anomaly_scores, label='Anomaly Score')
        
        # Plot threshold
        if isinstance(threshold, (int, float)):
            ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        else:
            ax.plot(df.index, threshold, 'r--', label='Adaptive Threshold')
        
        # Highlight anomalies
        if isinstance(threshold, (int, float)):
            anomalies = anomaly_scores > threshold
        else:
            anomalies = anomaly_scores > threshold
        
        ax.scatter(
            df.index[anomalies],
            anomaly_scores[anomalies],
            color='red',
            label='Anomalies'
        )
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Market Anomaly Detection')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        fig_plotly = self.convert_mpl_to_plotly(fig)
        plt.close(fig)
        return fig_plotly
    
    def plot_anomalies_on_price(self, df, price_col='close', anomaly_scores=None, threshold=None, figsize=(12, 8)):
        """
        Plot anomalies on price chart
        
        Parameters:
        - df: DataFrame with market data
        - price_col: Column name for price (default: 'close')
        - anomaly_scores: Anomaly scores (default: None, use self.anomaly_scores)
        - threshold: Anomaly threshold (default: None, use self.anomaly_thresholds)
        - figsize: Figure size (default: (12, 8))
        
        Returns:
        - fig: Matplotlib figure
        """
        if anomaly_scores is None:
            if self.anomaly_scores is None:
                raise ValueError("Anomaly scores not calculated. Call detect_anomalies first.")
            anomaly_scores = self.anomaly_scores
        
        if threshold is None:
            if self.anomaly_thresholds is None:
                # Calculate threshold
                threshold = np.nanpercentile(anomaly_scores, 95)
            else:
                threshold = self.anomaly_thresholds['base']
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        axes[0].plot(df.index, df[price_col], label=price_col.capitalize())
        
        # Identify anomalies
        if isinstance(threshold, (int, float)):
            anomalies = anomaly_scores > threshold
        else:
            anomalies = anomaly_scores > threshold
        
        # Highlight anomalies on price chart
        axes[0].scatter(
            df.index[anomalies],
            df[price_col][anomalies],
            color='red',
            label='Anomalies'
        )
        
        # Set labels and title for price chart
        axes[0].set_title(f'{price_col.capitalize()} with Anomalies')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot anomaly scores
        axes[1].plot(df.index, anomaly_scores, label='Anomaly Score')
        
        # Plot threshold
        if isinstance(threshold, (int, float)):
            axes[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        else:
            axes[1].plot(df.index, threshold, 'r--', label='Adaptive Threshold')
        
        # Highlight anomalies on anomaly score chart
        axes[1].scatter(
            df.index[anomalies],
            anomaly_scores[anomalies],
            color='red',
            label='Anomalies'
        )
        
        # Set labels and title for anomaly score chart
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Anomaly Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def save_model(self, filename=None, symbol=None, period=None, interval=None):
        """
        Save model to file
        
        Parameters:
        - filename: Filename to save model (default: None, use default filename)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - filename: Filename of saved model
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Use the standardized filename format
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name=self.anomaly_method,
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        
        # Create dictionary with model parameters
        model_dict = {
            'latent_size': self.latent_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'model_path': self.model_path,
            'anomaly_method': self.anomaly_method,
            'use_temporal': self.use_temporal,
            'sequence_length': self.sequence_length,
            'use_ensemble': self.use_ensemble,
            'scaler': self.scaler,
            'anomaly_thresholds': self.anomaly_thresholds,
            'feature_cols_fitted': self.feature_cols_fitted,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save PyTorch models separately
        if self.vae_model is not None:
            # Generate a standardized filename for the VAE model
            vae_filename = get_standardized_model_filename(
                model_type="anomaly_detection",
                model_name="vae",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            )
            vae_path = f"{vae_filename}.pth"
            torch.save(self.vae_model.state_dict(), vae_path)
            model_dict['vae_model_path'] = vae_path
        
        if self.temporal_vae_model is not None:
            # Generate a standardized filename for the temporal VAE model
            temporal_vae_filename = get_standardized_model_filename(
                model_type="anomaly_detection",
                model_name="temporal_vae",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            )
            temporal_vae_path = f"{temporal_vae_filename}.pth"
            torch.save(self.temporal_vae_model.state_dict(), temporal_vae_path)
            model_dict['temporal_vae_model_path'] = temporal_vae_path
        
        # Save scikit-learn models
        if self.isolation_forest_model is not None:
            # Generate a standardized filename for the isolation forest model
            if_filename = get_standardized_model_filename(
                model_type="anomaly_detection",
                model_name="isolation_forest",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            )
            if_path = f"{if_filename}.pkl"
            with open(if_path, 'wb') as f:
                pickle.dump(self.isolation_forest_model, f)
            model_dict['isolation_forest_model_path'] = if_path
        
        if self.lof_model is not None:
            # Generate a standardized filename for the LOF model
            lof_filename = get_standardized_model_filename(
                model_type="anomaly_detection",
                model_name="lof",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            )
            lof_path = f"{lof_filename}.pkl"
            with open(lof_path, 'wb') as f:
                pickle.dump(self.lof_model, f)
            model_dict['lof_model_path'] = lof_path
        
        if self.svm_model is not None:
            # Generate a standardized filename for the SVM model
            svm_filename = get_standardized_model_filename(
                model_type="anomaly_detection",
                model_name="svm",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            )
            svm_path = f"{svm_filename}.pkl"
            with open(svm_path, 'wb') as f:
                pickle.dump(self.svm_model, f)
            model_dict['svm_model_path'] = svm_path
        
        # Save model dictionary
        config_path = f"{base_filename}_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"Model saved to {base_filename}")
        return base_filename
    
    def load_model(self, filename=None, device='cpu', symbol=None, period=None, interval=None):
        """
        Load model from file
        
        Parameters:
        - filename: Filename to load model from (default: None, use default filename)
        - device: Device to load PyTorch models to ('cpu' or 'cuda', default: 'cpu')
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - success: Whether loading was successful
        """
        from dashboard_utils import get_standardized_model_filename
        
        if filename is None:
            # Use the standardized filename format
            base_filename = get_standardized_model_filename(
                model_type="anomaly_detection",
                model_name=self.anomaly_method,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            )
            filename = f"{base_filename}_config.pkl"
        
        try:
            # Load model dictionary
            with open(filename, 'rb') as f:
                model_dict = pickle.load(f)
            
            # Set model parameters
            self.latent_size = model_dict['latent_size']
            self.hidden_size = model_dict['hidden_size']
            self.learning_rate = model_dict['learning_rate']
            self.batch_size = model_dict['batch_size']
            self.num_epochs = model_dict['num_epochs']
            self.model_path = model_dict['model_path']
            self.anomaly_method = model_dict['anomaly_method']
            self.use_temporal = model_dict['use_temporal']
            self.sequence_length = model_dict['sequence_length']
            self.use_ensemble = model_dict['use_ensemble']
            self.scaler = model_dict['scaler']
            self.anomaly_thresholds = model_dict['anomaly_thresholds']
            self.feature_cols_fitted = model_dict.get('feature_cols_fitted', None)
            
            # Load PyTorch models
            if 'vae_model_path' in model_dict:
                # Initialize model
                input_size = model_dict.get('input_size', 14)  # Default to 14 if not available
                self.vae_model = VariationalAutoencoder(input_size, self.latent_size, self.hidden_size)
                self.vae_model.load_state_dict(torch.load(model_dict['vae_model_path'], map_location=device))
                self.vae_model.to(device)
                self.vae_model.eval()
            
            if 'temporal_vae_model_path' in model_dict:
                # Initialize model
                input_size = model_dict.get('input_size', 14)  # Default to 14 if not available
                self.temporal_vae_model = TemporalVariationalAutoencoder(
                    input_size, self.sequence_length, self.latent_size, self.hidden_size
                )
                self.temporal_vae_model.load_state_dict(torch.load(model_dict['temporal_vae_model_path'], map_location=device))
                self.temporal_vae_model.to(device)
                self.temporal_vae_model.eval()
            
            # Load scikit-learn models
            if 'isolation_forest_model_path' in model_dict:
                with open(model_dict['isolation_forest_model_path'], 'rb') as f:
                    self.isolation_forest_model = pickle.load(f)
            
            if 'lof_model_path' in model_dict:
                with open(model_dict['lof_model_path'], 'rb') as f:
                    self.lof_model = pickle.load(f)
            
            if 'svm_model_path' in model_dict:
                with open(model_dict['svm_model_path'], 'rb') as f:
                    self.svm_model = pickle.load(f)
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    @staticmethod
    def convert_mpl_to_plotly(fig):
        # Convert and return as a figure dictionary
        plotly_fig = tls.mpl_to_plotly(fig)
        return plotly_fig.to_dict()
    
    def generate_anomaly_report(self, df, feature_cols, price_col='close', method=None, device='cpu', filename=None, symbol=None, period=None, interval=None):
        """
        Generate comprehensive anomaly detection report
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        - price_col: Column name for price (default: 'close')
        - method: Method for anomaly detection (default: None, use self.anomaly_method)
        - device: Device to use for inference ('cpu' or 'cuda', default: 'cpu')
        - filename: Filename to save the report (default: None, displays the report)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - report: Dictionary with report components
        """
        # Detect anomalies
        from dashboard_utils import get_standardized_model_filename
        
        # Use standardized filename function for consistency
        base_filename = get_standardized_model_filename(
            model_type="anomaly_detection",
            model_name=self.anomaly_method,
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        )
        
        config_file = f"{base_filename}_config.pkl"
        model_loaded = False
        
        if os.path.exists(config_file):
            model_loaded = self.load_model(filename=config_file, device=device, symbol=symbol, period=period, interval=interval)
        
        if self.vae_model is None and self.temporal_vae_model is None and not model_loaded:
            self.train_model(df, feature_cols, method, device, symbol, period, interval)
            self.save_model(symbol=symbol, period=period, interval=interval)
        
        anomaly_scores = self.detect_anomalies(df, feature_cols, method, device)
        
        # Calculate adaptive threshold
        if 'volatility' in df.columns:
            threshold = self.calculate_adaptive_threshold(anomaly_scores, df, 'volatility')
        else:
            threshold = self.calculate_adaptive_threshold(anomaly_scores)
        
        # Identify anomalies
        if isinstance(threshold, (int, float)):
            anomalies = anomaly_scores > threshold
        else:
            anomalies = anomaly_scores > threshold
        
        # Create report components
        report = {
            'anomaly_scores': anomaly_scores,
            'threshold': threshold,
            'anomalies': anomalies,
            'anomaly_dates': df.index[anomalies],
            'anomaly_count': np.sum(anomalies),
            'anomaly_percentage': np.mean(anomalies) * 100
        }
        
        # Generate visualizations
        figs = {}
        
        # Anomaly scores plot
        figs['anomaly_scores'] = self.plot_anomaly_scores(df, anomaly_scores, threshold)
        
        # Anomalies on price chart
        figs['price_anomalies'] = self.plot_anomalies_on_price(df, price_col, anomaly_scores, threshold)
        
        # Add visualizations to report
        report['visualizations'] = figs
        
        # Save report if filename provided
        if filename is not None:
            # Save visualizations
            for name, fig in figs.items():
                fig.savefig(f"{filename}_{name}.png", dpi=300, bbox_inches='tight')
            
            # Save model
            self.save_model(f"{filename}_model.pkl", symbol=symbol, period=period, interval=interval)
            
            # Save report data
            report_data = {k: v for k, v in report.items() if k != 'visualizations'}
            with open(f"{filename}_data.pkl", 'wb') as f:
                pickle.dump(report_data, f)
        
        return report
