"""
Enhanced Machine Learning Clustering Module for Perfect Storm Dashboard

This module implements advanced clustering algorithms to identify when multiple
indicators align in a "perfect storm" configuration that could signal
significant market movements.

Enhancements:
1. Improved clustering with DBSCAN and HDBSCAN for density-based clustering
2. Hierarchical clustering for multi-level pattern detection
3. Temporal clustering to identify time-dependent patterns
4. Optimized ONNX Runtime integration
5. Ensemble clustering methods
6. Advanced visualization capabilities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxruntime as ort
import os
import pickle
import json
import time
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

class MarketClusteringDataset(Dataset):
    """Dataset class for market clustering"""
    
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

class Autoencoder(nn.Module):
    """Autoencoder model for dimensionality reduction and anomaly detection"""
    
    def __init__(self, input_size, encoding_size=10):
        """
        Initialize the autoencoder model
        
        Parameters:
        - input_size: Number of input features
        - encoding_size: Size of the encoded representation (default: 10)
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()  # Assuming features are normalized to [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - x_reconstructed: Reconstructed input
        - encoded: Encoded representation
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        x_reconstructed = self.decoder(encoded)
        
        return x_reconstructed, encoded

class TemporalAutoencoder(nn.Module):
    """Temporal autoencoder model for time-dependent clustering"""
    
    def __init__(self, input_size, sequence_length, encoding_size=10):
        """
        Initialize the temporal autoencoder model
        
        Parameters:
        - input_size: Number of input features
        - sequence_length: Length of input sequence
        - encoding_size: Size of the encoded representation (default: 10)
        """
        super(TemporalAutoencoder, self).__init__()
        
        # Encoder (LSTM-based)
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.encoder_fc = nn.Linear(64, encoding_size)
        
        # Decoder (LSTM-based)
        self.decoder_lstm = nn.LSTM(
            input_size=encoding_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.decoder_fc = nn.Linear(64, input_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - x_reconstructed: Reconstructed input
        - encoded: Encoded representation
        """
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)
        # Get the output from the last layer
        encoded_seq = h_n[-1]
        # Apply fully connected layer
        encoded = self.encoder_fc(encoded_seq)
        
        # Repeat encoded vector for each time step
        encoded_repeated = encoded.unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decode
        decoded_seq, _ = self.decoder_lstm(encoded_repeated)
        x_reconstructed = self.decoder_fc(decoded_seq)
        
        return x_reconstructed, encoded

class PerfectStormClustering:
    """Enhanced class for clustering market data to identify 'perfect storm' configurations"""
    
    def __init__(self, n_clusters=5, encoding_size=10, learning_rate=0.001, 
                 batch_size=32, num_epochs=50, model_path='models',
                 clustering_method='kmeans', use_temporal=False, sequence_length=10,
                 use_ensemble=False):
        """
        Initialize the PerfectStormClustering class
        
        Parameters:
        - n_clusters: Number of clusters (default: 5)
        - encoding_size: Size of the encoded representation (default: 10)
        - learning_rate: Learning rate for optimization (default: 0.001)
        - batch_size: Batch size for training (default: 32)
        - num_epochs: Number of training epochs (default: 50)
        - model_path: Path to save/load models (default: 'models')
        - clustering_method: Method for clustering ('kmeans', 'dbscan', 'hdbscan', 'gmm', 'hierarchical', default: 'kmeans')
        - use_temporal: Whether to use temporal clustering (default: False)
        - sequence_length: Length of sequence for temporal clustering (default: 10)
        - use_ensemble: Whether to use ensemble clustering (default: False)
        """
        self.n_clusters = n_clusters
        self.encoding_size = encoding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.clustering_method = clustering_method
        self.use_temporal = use_temporal
        self.sequence_length = sequence_length
        self.use_ensemble = use_ensemble
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Initialize autoencoder
        self.autoencoder = None
        self.input_size = None
        
        # Initialize clustering model
        self.clustering_model = None
        
        # Initialize ensemble models
        self.ensemble_models = []
        
        # Initialize ONNX session
        self.onnx_session = None
        
        # Initialize metrics history
        self.metrics_history = {}
        
        # Initialize cluster centers and labels
        self.cluster_centers = None
        self.labels = None
        
        # Initialize PCA for visualization
        self.pca = None
    
    def _prepare_data(self, df, feature_columns=None, add_technical_features=True):
        """
        Prepare data for clustering
        
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
            df_copy['vwap'] = (df_copy['close'] * df_copy['volume']).cumsum() / df_copy['volume'].cumsum()

        # Add MACD
        df_copy['macd_12_26_9'], df_copy['macd_signal_12_26_9'], df_copy['macd_hist_12_26_9'] = self._calculate_macd(df_copy['close'], fast=12, slow=26, signal=9)
        df_copy['macd_20_50_10'], df_copy['macd_signal_20_50_10'], df_copy['macd_hist_20_50_10'] = self._calculate_macd(df_copy['close'], fast=20, slow=50, signal=10)

        # Add Bollinger Bands
        df_copy['bb_upper_20_2'], df_copy['bb_middle_20_2'], df_copy['bb_lower_20_2'] = self._calculate_bollinger_bands(df_copy['close'], window=20, window_dev=2)
        df_copy['bb_upper_14_2'], df_copy['bb_middle_14_2'], df_copy['bb_lower_14_2'] = self._calculate_bollinger_bands(df_copy['close'], window=14, window_dev=2)

        # Add lagged features
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            for lag in [1, 2, 3]:
                df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)

        # Add rolling window calculations
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            for window in [5, 10, 20]:
                df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window).mean()
                df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window).std()
                
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
        Prepare temporal data for clustering
        
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
    
    def _create_autoencoder(self, input_size):
        """
        Create a new autoencoder model
        
        Parameters:
        - input_size: Number of input features
        
        Returns:
        - autoencoder: PyTorch autoencoder model
        """
        if self.use_temporal:
            return TemporalAutoencoder(input_size, self.sequence_length, self.encoding_size)
        else:
            return Autoencoder(input_size, self.encoding_size)
    
    def _create_clustering_model(self):
        """
        Create a new clustering model
        
        Returns:
        - clustering_model: Clustering model
        """
        if self.clustering_method == 'kmeans':
            return KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.clustering_method == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
        elif self.clustering_method == 'hdbscan':
            if HDBSCAN_AVAILABLE:
                return hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
            else:
                print("HDBSCAN not available, falling back to DBSCAN")
                return DBSCAN(eps=0.5, min_samples=5)
        elif self.clustering_method == 'gmm':
            return GaussianMixture(n_components=self.n_clusters, random_state=42)
        elif self.clustering_method == 'hierarchical':
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.clustering_method == 'affinity':
            from sklearn.cluster import AffinityPropagation
            return AffinityPropagation(random_state=42)
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
    
    def train(self, df, feature_columns=None, add_technical_features=True, validation_split=0.2):
        """
        Train the clustering model using the provided market data.
        
        Parameters:
        - df (DataFrame): DataFrame containing market data with features and timestamps
        - feature_columns (List[str], optional): List of columns to use as features. If None, all numeric columns will be used (default: None)
        - add_technical_features (bool, optional): Whether to add engineered technical indicators as features (default: True)
        - validation_split (float, optional): Fraction of data to reserve for validation (between 0 and 1, default: 0.2)
        
        Returns:
        - Dict: Dictionary containing training history with 'train_loss' and 'val_loss' lists
        """
        # Prepare data
        if self.use_temporal:
            X, indices, feature_columns = self._prepare_temporal_data(df, feature_columns, add_technical_features)
        else:
            X, indices, feature_columns = self._prepare_data(df, feature_columns, add_technical_features)
        
        # Scale features
        if self.use_temporal:
            X_reshaped = X.reshape(-1, X.shape[-1])
            self.feature_scaler.fit(X_reshaped)
            X_reshaped = self.feature_scaler.transform(X_reshaped)
            X = X_reshaped.reshape(X.shape)
        else:
            self.feature_scaler.fit(X)
            X = self.feature_scaler.transform(X)
        
        # Save feature scaler
        scaler_file = os.path.join(self.model_path, 'feature_scaler_clustering.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Set input size
        if self.use_temporal:
            self.input_size = X.shape[2]
        else:
            self.input_size = X.shape[1]
        
        # Create autoencoder
        self.autoencoder = self._create_autoencoder(self.input_size)
        
        # Convert to PyTorch tensors
        if self.use_temporal:
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = torch.FloatTensor(X)
        
        # Split data into train and validation sets
        train_size = int((1 - validation_split) * len(X_tensor))
        train_dataset = MarketClusteringDataset(X_tensor[:train_size])
        val_dataset = MarketClusteringDataset(X_tensor[train_size:])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.autoencoder.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                x_reconstructed, _ = self.autoencoder(batch)
                loss = criterion(x_reconstructed, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # Validation phase
            self.autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_reconstructed, _ = self.autoencoder(batch)
                    loss = criterion(x_reconstructed, batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            # Store losses
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Print epoch summary
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the trained autoencoder model
        model_file = os.path.join(self.model_path, 'autoencoder_clustering.pth')
        torch.save(self.autoencoder.state_dict(), model_file)
        
        # Create clustering model
        self.clustering_model = self._create_clustering_model()
        
        # Encode data using the trained autoencoder
        self.autoencoder.eval()
        with torch.no_grad():
            if self.use_temporal:
                encoded_features = []
                for batch in train_loader:
                    _, encoded = self.autoencoder(batch)
                    encoded_features.append(encoded.detach().cpu().numpy())
                encoded_features = np.concatenate(encoded_features, axis=0)
            else:
                encoded_features = []
                for batch in train_loader:
                    _, encoded = self.autoencoder(batch)
                    encoded_features.append(encoded.detach().cpu().numpy())
                encoded_features = np.concatenate(encoded_features, axis=0)
        
        # Fit clustering model
        if self.clustering_method == 'hdbscan' and HDBSCAN_AVAILABLE:
            self.labels = self.clustering_model.fit_predict(encoded_features)
        else:
            self.clustering_model.fit(encoded_features)
            self.labels = self.clustering_model.labels_
        
        # Save clustering model
        if self.clustering_method == 'hdbscan' and HDBSCAN_AVAILABLE:
            model_file = os.path.join(self.model_path, 'clustering_model_clustering.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(self.clustering_model, f)
        else:
            model_file = os.path.join(self.model_path, 'clustering_model_clustering.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(self.clustering_model, f)
        
        # Optionally, perform PCA for visualization
        if self.clustering_method != 'hdbscan' or not HDBSCAN_AVAILABLE:
            self.pca = PCA(n_components=2)
            self.cluster_centers = self.pca.fit_transform(self.clustering_model.cluster_centers_)
        else:
            self.pca = PCA(n_components=2)
            self.cluster_centers = self.pca.fit_transform(encoded_features)
        
        # Save PCA model
        pca_file = os.path.join(self.model_path, 'pca_clustering.pkl')
        with open(pca_file, 'wb') as f:
            pickle.dump(self.pca, f)
        
        return history
