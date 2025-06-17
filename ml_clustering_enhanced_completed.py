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
from indicator_calculations import PerfectStormIndicators 
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
            dropout=0.1
        )
        
        self.encoder_fc = nn.Linear(64, encoding_size)
        
        # Decoder (LSTM-based)
        self.decoder_lstm = nn.LSTM(
            input_size=encoding_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
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
                 batch_size=32, num_epochs=50, model_path='models\Clustering Models',
                 clustering_method='ensemble', use_temporal=True, sequence_length=10,
                 use_ensemble=True):
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

    def calculate_macd(self, df_copy):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Parameters:
        - df_copy: DataFrame with stock data
        
        Returns:
        - Copy DataFrame with MACD
        """
        # Calculate fast and slow EMAs
        df_copy['ema_fast1'] = df_copy['close'].ewm(span=12, adjust=False).mean()
        df_copy['ema_slow1'] = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy['ema_fast2'] = df_copy['close'].ewm(span=20, adjust=False).mean()
        df_copy['ema_slow2'] = df_copy['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate MACD line
        df_copy['macd_12_26_9'] = df_copy['ema_fast1'] - df_copy['ema_slow1']
        df_copy['macd_20_50_10']= df_copy['ema_fast2'] - df_copy['ema_slow2']
        
        # Calculate signal line
        df_copy['macd_signal_12_26_9'] = df_copy['macd_12_26_9'].ewm(span=9, adjust=False).mean()
        df_copy['macd_signal_20_50_10']= df_copy['macd_20_50_10'].ewm(span=10, adjust=False).mean()

        # Calculate histogram
        df_copy['macd_hist_12_26_9'] = df_copy['macd_12_26_9'] - df_copy['macd_signal_12_26_9']
        df_copy['macd_hist_20_50_10'] = df_copy['macd_20_50_10'] - df_copy['macd_signal_20_50_10']

        # Clean up temporary columns
        df_copy.drop(['ema_fast1', 'ema_slow1', 'ema_fast2', 'ema_slow2'], axis=1, inplace=True)

        #return df_copy['macd_12_26_9'], df_copy['macd_signal_12_26_9'], df_copy['macd_hist_12_26_9'], df_copy['macd_20_50_10'], df_copy['macd_signal_20_50_10'], df_copy['macd_hist_20_50_10']
        return df_copy
    
    def calculate_bollinger_bands(self, df_copy, num_std=2):
        """
        Calculate Bollinger Bands
        
        Parameters:
        - df_copy: DataFrame with stock data
        - window: Window size for moving average (default: 20)
        - num_std: Number of standard deviations (default: 2)
        
        Returns:
        - DataFrame with Bollinger Bands
        """
        # Calculate middle band (20-day and 14-day SMA)
        df_copy['bb_middle_20_2'] = df_copy['close'].rolling(window=20).mean()
        df_copy['bb_middle_14_2'] = df_copy['close'].rolling(window=14).mean()
        
        # Calculate standard deviation
        df_copy['bb_std20'] = df_copy['close'].rolling(window=20).std()
        df_copy['bb_std14'] = df_copy['close'].rolling(window=14).std()
        
        # Calculate upper and lower bands
        df_copy['bb_upper_20_2'] = df_copy['bb_middle_20_2'] + (df_copy['bb_std20'] * num_std)
        df_copy['bb_lower_20_2'] = df_copy['bb_middle_20_2'] - (df_copy['bb_std20'] * num_std)
        df_copy['bb_upper_14_2'] = df_copy['bb_middle_14_2'] + (df_copy['bb_std14'] * num_std)
        df_copy['bb_lower_14_2'] = df_copy['bb_middle_14_2'] - (df_copy['bb_std14'] * num_std)
        
        # Clean up temporary columns
        df_copy.drop(['bb_std20', 'bb_std14'], axis=1, inplace=True)
        
        return df_copy

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
        print("Normal preparation of clustering data")
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
        self.calculate_macd(df_copy)

        # Add Bollinger Bands
        self.calculate_bollinger_bands(df_copy)

        # Add lagged features
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            for lag in [1, 2, 3]:
                df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)

        # Add rolling window calculations
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            for window in [5, 10, 20]:
                df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window).mean()
                df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window).std()
        
        # Select features
        if feature_columns is None:
            # Use all numeric columns
            feature_columns = [col for col in df_copy.select_dtypes(include=[np.number]).columns]
            X = df_copy[feature_columns]
        else:
            # Only keep numeric columns from the provided list
            numeric_cols = [col for col in feature_columns if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col])]
            X = df_copy[numeric_cols]
            feature_columns = numeric_cols
        
        # Fill NaN values instead of dropping
        X = X.fillna(method='ffill').fillna(method='bfill')

        # Replace infinite values with NaN, then fill those NaNs
        X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

        # As a last resort, replace any remaining NaNs with 0
        X = X.fillna(0)
        
        # Scale features
        if len(X) > 0:  # Add this check
            X = StandardScaler().fit_transform(X)
            return X, df_copy.index, feature_columns
        else:
            print("WARNING: Empty dataframe after preparation")
            # Return a dummy array with at least one sample to avoid StandardScaler error
            dummy = np.zeros((1, X.shape[1] if X.ndim>1 else 1))
            return dummy, df_copy.index, feature_columns
    
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
        print("Preparing temporal data")
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
    
    def train(self, df, feature_columns=None, add_technical_features=True, validation_split=0.2, path=None, symbol=None, X_scaled_for_encoding=None):
        """
        Train the clustering model using the provided market data.
        
        Parameters:
        - df (DataFrame): DataFrame containing market data with features and timestamps
        - feature_columns (List[str], optional): List of columns to use as features. If None, all numeric columns will be used (default: None)
        - add_technical_features (bool, optional): Whether to add engineered technical indicators as features (default: True)
        - validation_split (float, optional): Fraction of data to reserve for validation (between 0 and 1, default: 0.2)
        - path (str, optional): Path to save the trained model and scaler. If None, uses the default model path (default: None)
        
        Returns:
        - Dict: Dictionary containing training history with 'train_loss' and 'val_loss' lists
        """
        # Set path if not provided
        if path is None:
            path = self.model_path

        # Add symbol to filenames if provided
        symbol_str = f"_{symbol}" if symbol else ""

        # Check if model exists and load if available
        config_file = os.path.join(path, f'config_clustering{symbol_str}.json')
        if os.path.exists(config_file):
            self.load_model(path=path, symbol=symbol)
            if self.autoencoder is not None and self.clustering_model is not None:
                print("Model already trained and loaded. Skipping retraining.")
                return {'train_loss': [], 'val_loss': []}
        
        # Prepare or use pre-scaled data
        if X_scaled_for_encoding is not None:
            X_tensor = X_scaled_for_encoding
            if self.use_temporal:
                self.input_size = X_tensor.shape[2]
            else:
                self.input_size = X_tensor.shape[1]
            indices = df.index
        else:
            if self.use_temporal:
                X, indices, feature_columns = self._prepare_temporal_data(df, feature_columns, add_technical_features)
                X_reshaped = X.reshape(-1, X.shape[-1])
                self.feature_scaler.fit(X_reshaped)
                X_reshaped = self.feature_scaler.transform(X_reshaped)
                X = X_reshaped.reshape(X.shape)
            else:
                X, indices, feature_columns = self._prepare_data(df, feature_columns, add_technical_features)
                self.feature_scaler.fit(X)
                X = self.feature_scaler.transform(X)
            X_tensor = torch.FloatTensor(X)
            if self.use_temporal:
                self.input_size = X.shape[2]
            else:
                self.input_size = X.shape[1]
        # Save feature scaler
        scaler_file = os.path.join(path, f'feature_scaler_clustering{symbol_str}.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        # Create autoencoder
        self.autoencoder = self._create_autoencoder(self.input_size)
        
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
            #print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
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
            print("Debug: About to call clustering algorithm")
            self.clustering_model.fit(encoded_features)
            self.labels = self.clustering_model.labels_
            print(f"Debug: Clustering result type: {type(self.labels)}, length if tuple: {len(self.labels) if isinstance(self.labels, tuple) else 'not a tuple'}")
        
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

    def predict(self, df=None, feature_columns=None, add_technical_features=True, encoded_features=None, indices=None, X_original_scaled=None):
        """
        Predict cluster labels and anomaly scores for the given data.
        If pre-encoded features and indices are provided, use them directly to avoid redundant computation.
        Otherwise, prepare and encode features as usual.
        
        Parameters:
        - df: DataFrame with market data (optional if encoded_features and X_original_scaled are provided)
        - feature_columns: List of columns to use as features (default: None)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        - encoded_features: Optional, pre-encoded features (numpy array)
        - indices: Optional, indices corresponding to the encoded features
        - X_original_scaled: Optional, original scaled features (numpy array) for anomaly score calculation
        
        Returns:
        - results: DataFrame with cluster labels and anomaly scores
        """
        if encoded_features is not None and indices is not None:
            ef = encoded_features
            idx = indices
            anomaly_scores = np.full(len(ef), np.nan)  # No reconstruction possible, so set to NaN
        else:
            raise ValueError("You must provide encoded_features and indices for prediction.")
        # Use clustering_model to predict cluster labels
        if self.clustering_method == 'hdbscan' and HDBSCAN_AVAILABLE:
            labels, _ = hdbscan.approximate_predict(self.clustering_model, ef)
        else:
            labels = self.clustering_model.predict(ef)
        results = pd.DataFrame({
            'cluster': labels,
            'anomaly_score': anomaly_scores
        }, index=idx[:len(labels)])
        return results
    
    def evaluate(self, encoded_features, labels, df=None, feature_columns=None, add_technical_features=True, indices=None):
        """
        Evaluate clustering performance.
        Returns a dictionary of evaluation metrics.
        Requires pre-encoded features and labels.
        """
        if encoded_features is None or labels is None:
            raise ValueError("You must provide both encoded_features and labels for evaluation.")
        metrics = {}
        metrics['silhouette_score'] = silhouette_score(encoded_features, labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(encoded_features, labels)
        # Add other metrics as needed
        return metrics
    
    def visualize(self, encoded_features, labels, plot_type='scatter', save_path=None, indices=None, df=None, feature_columns=None, add_technical_features=True):
        """
        Visualize clustering results
        
        Parameters:
        - plot_type: Type of plot ('scatter', 'tsne', 'umap', default: 'scatter')
        - save_path: Path to save the visualization (default: None, display only)
        - encoded_features: Pre-encoded features (required)
        - labels: Cluster labels (required)
        
        Returns:
        - fig: Matplotlib figure
        """
        if encoded_features is None or labels is None:
            raise ValueError("You must provide both encoded_features and labels for visualization.")
        ef = encoded_features
        lbls = labels
        fig, ax = plt.subplots(figsize=(12, 8))
        if plot_type == 'scatter':
            if self.pca is None:
                self.pca = PCA(n_components=2)
                self.pca.fit(ef)
            X_pca = self.pca.transform(ef)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lbls, cmap='viridis', alpha=0.7)
            if not self.use_ensemble and hasattr(self.clustering_model, 'cluster_centers_') and self.clustering_method != 'hdbscan':
                centers_pca = self.pca.transform(self.clustering_model.cluster_centers_)
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.8, marker='X')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('Clustering Results (PCA)')
        elif plot_type == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(ef)
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=lbls, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('Clustering Results (t-SNE)')
        elif plot_type == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(random_state=42)
                X_umap = reducer.fit_transform(ef)
                scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=lbls, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Cluster')
                ax.set_xlabel('UMAP Component 1')
                ax.set_ylabel('UMAP Component 2')
                ax.set_title('Clustering Results (UMAP)')
            except ImportError:
                print("UMAP not available, falling back to PCA")
                if self.pca is None:
                    self.pca = PCA(n_components=2)
                    self.pca.fit(ef)
                X_pca = self.pca.transform(ef)
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lbls, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Cluster')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('Clustering Results (PCA)')
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        fig_plotly = self.convert_mpl_to_plotly(fig)
        plt.close(fig)
        return fig_plotly
    
    def export_to_onnx(self, input_size=None):
        """
        Export the trained autoencoder model to ONNX format
        
        Parameters:
        - input_size: Input size for the model (default: None, use self.input_size)
        
        Returns:
        - onnx_path: Path to the exported ONNX model
        """
        # Check if model is trained
        if self.autoencoder is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Set input size if not provided
        if input_size is None:
            input_size = self.input_size
        
        # Set model to evaluation mode
        self.autoencoder.eval()
        
        # Create dummy input
        if self.use_temporal:
            dummy_input = torch.randn(1, self.sequence_length, input_size)
        else:
            dummy_input = torch.randn(1, input_size)
        
        # Set ONNX export path
        onnx_path = os.path.join(self.model_path, 'autoencoder_clustering.onnx')
        
        # Export model to ONNX
        torch.onnx.export(
            self.autoencoder,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['reconstructed', 'encoded'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'reconstructed': {0: 'batch_size'},
                'encoded': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Initialize ONNX Runtime session
        self.onnx_session = None
        
        return onnx_path
    
    def load_onnx_model(self, onnx_path):
        """
        Load an ONNX model for inference
        
        Parameters:
        - onnx_path: Path to the ONNX model
        """
        # Check if the ONNX model exists
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        # Initialize ONNX Runtime session
        self.onnx_session = ort.InferenceSession(onnx_path)
    
    def predict_with_onnx(self, df, feature_columns=None, add_technical_features=True):
        """
        Make predictions using the ONNX model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, use all numeric columns)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        
        Returns:
        - labels: Cluster labels for each data point
        - anomaly_scores: Anomaly scores for each data point
        """
        # Check if ONNX session is initialized
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded. Call load_onnx_model() first.")
        
        # Check if clustering model is trained
        if self.clustering_model is None:
            raise ValueError("Clustering model not trained. Call train() first.")
        
        # Prepare data
        if self.use_temporal:
            X, indices, _ = self._prepare_temporal_data(df, feature_columns, add_technical_features)
        else:
            X, indices, _ = self._prepare_data(df, feature_columns, add_technical_features)
        
        # Scale features
        if self.use_temporal:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_reshaped = self.feature_scaler.transform(X_reshaped)
            X = X_reshaped.reshape(X.shape)
        else:
            X = self.feature_scaler.transform(X)
        
        # Run inference with ONNX
        encoded_features = []
        reconstructed_features = []
        
        # Process in batches
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i+self.batch_size]
            
            # Prepare input
            if self.use_temporal:
                onnx_input = {self.onnx_session.get_inputs()[0].name: batch.astype(np.float32)}
            else:
                onnx_input = {self.onnx_session.get_inputs()[0].name: batch.astype(np.float32)}
            
            # Run inference
            onnx_outputs = self.onnx_session.run(None, onnx_input)
            
            # Extract outputs
            reconstructed = onnx_outputs[0]
            encoded = onnx_outputs[1]
            
            reconstructed_features.append(reconstructed)
            encoded_features.append(encoded)
        
        # Concatenate batches
        reconstructed_features = np.concatenate(reconstructed_features, axis=0)
        encoded_features = np.concatenate(encoded_features, axis=0)
        
        # Calculate reconstruction error (anomaly score)
        if self.use_temporal:
            anomaly_scores = np.mean(np.square(X - reconstructed_features), axis=(1, 2))
        else:
            anomaly_scores = np.mean(np.square(X - reconstructed_features), axis=1)
        
        # Predict cluster labels
        if self.clustering_method == 'hdbscan' and HDBSCAN_AVAILABLE:
            # For HDBSCAN, use approximate_predict
            labels, _ = hdbscan.approximate_predict(self.clustering_model, encoded_features)
        else:
            # For other methods, use predict
            labels = self.clustering_model.predict(encoded_features)
        
        # Create a DataFrame with results
        results = pd.DataFrame({
            'cluster': labels,
            'anomaly_score': anomaly_scores
        }, index=indices[:len(labels)])
        
        return results
    
    def create_ensemble_models(self):
        """
        Create ensemble of clustering models
        
        Returns:
        - ensemble_models: List of clustering models
        """
        # Create different clustering models
        ensemble_models = []
        
        # KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        ensemble_models.append(('kmeans', kmeans))
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        ensemble_models.append(('dbscan', dbscan))
        
        # GMM
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        ensemble_models.append(('gmm', gmm))
        
        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
        ensemble_models.append(('hierarchical', hierarchical))
        
        # HDBSCAN if available
        if HDBSCAN_AVAILABLE:
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
            ensemble_models.append(('hdbscan', hdbscan_model))
        
        return ensemble_models
    
    def train_ensemble(self, df, feature_columns=None, add_technical_features=True, validation_split=0.2, path=None, symbol=None, X_scaled_for_encoding=None):
        """
        Train ensemble of clustering models
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, use all numeric columns)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        - validation_split: Fraction of data to reserve for validation (default: 0.2)
        - X_scaled_for_encoding: Pre-scaled features to use for autoencoder and clustering (default: None)
        
        Returns:
        - history: Dictionary with training history
        """
        # Set path if not provided
        if path is None:
            path = self.model_path
        
        # Add symbol to filenames if provided
        symbol_str = f"_{symbol}" if symbol else ""

        # Set use_ensemble flag
        self.use_ensemble = True
        
        # Train autoencoder as usual
        #history = self.train_ensemble(df, feature_columns, add_technical_features, validation_split)
        
        # Prepare or use pre-scaled data
        if X_scaled_for_encoding is not None:
            X_tensor = torch.FloatTensor(X_scaled_for_encoding)
            if self.use_temporal:
                self.input_size = X_tensor.shape[2]
            else:
                self.input_size = X_tensor.shape[1]
            indices = df.index
            X = X_scaled_for_encoding  # For later use in clustering
        else:
            if self.use_temporal:
                X, indices, feature_columns = self._prepare_temporal_data(df, feature_columns, add_technical_features)
                X_reshaped = X.reshape(-1, X.shape[-1])
                self.feature_scaler.fit(X_reshaped)
                X_reshaped = self.feature_scaler.transform(X_reshaped)
                X = X_reshaped.reshape(X.shape)
            else:
                X, indices, feature_columns = self._prepare_data(df, feature_columns, add_technical_features)
                self.feature_scaler.fit(X)
                X = self.feature_scaler.transform(X)
            X_tensor = torch.FloatTensor(X)
            if self.use_temporal:
                self.input_size = X.shape[2]
            else:
                self.input_size = X.shape[1]
        # Save feature scaler
        scaler_file = os.path.join(path, f'feature_scaler_clustering{symbol_str}.pkl')
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
            #print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the trained autoencoder model
        model_file = os.path.join(self.model_path, 'autoencoder_clustering.pth')
        torch.save(self.autoencoder.state_dict(), model_file)
        
        # Create ensemble models
        self.ensemble_models = self.create_ensemble_models()
        
        # Prepare data for clustering
        if self.use_temporal:
            X, indices, _ = self._prepare_temporal_data(df, feature_columns, add_technical_features)
        else:
            X, indices, _ = self._prepare_data(df, feature_columns, add_technical_features)
        
        # Scale features as usual
        if self.use_temporal:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X = self.feature_scaler.transform(X_reshaped).reshape(X.shape)
        else:
            X = self.feature_scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Create dataloader (or use your _prepare_dataloader helper if you have it)
        dataset = MarketClusteringDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Encode features using the trained autoencoder
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = []
            for batch in dataloader:
                _, encoded = self.autoencoder(batch)
                encoded_features.append(encoded.detach().cpu().numpy())
            encoded_features = np.concatenate(encoded_features, axis=0)
        
        # Now train each ensemble model and obtain labels
        ensemble_labels = []
        saved_models = set()
        for name, model in self.ensemble_models:
            if name == 'hdbscan':
                labels = model.fit_predict(encoded_features)
            elif name == 'gmm':
                model.fit(encoded_features)
                labels = model.predict(encoded_features)
            elif name in ['dbscan', 'hierarchical', 'affinity']:
                labels = model.fit_predict(encoded_features)
            else:
                model.fit(encoded_features)
                labels = model.labels_  # normally available for KMeans
            ensemble_labels.append(labels)
            # Save each model only once
            # model_file = os.path.join(self.model_path, f'clustering_model_{name}.pkl')
            # if name not in saved_models:
            #     with open(model_file, 'wb') as f:
            #         pickle.dump(model, f)
            #     saved_models.add(name)
        
        # Combine ensemble predictions using consensus clustering
        self.labels = self.consensus_clustering(ensemble_labels)
        
        return history
    
    def consensus_clustering(self, ensemble_labels):
        """
        Combine ensemble predictions using consensus clustering
        
        Parameters:
        - ensemble_labels: List of cluster label arrays from different models
        
        Returns:
        - consensus_labels: Consensus cluster labels
        """
        # Create co-association matrix
        n_samples = len(ensemble_labels[0])
        co_association = np.zeros((n_samples, n_samples))
        
        # Fill co-association matrix
        for labels in ensemble_labels:
            for i in range(n_samples):
                for j in range(i, n_samples):
                    if labels[i] == labels[j]:
                        co_association[i, j] += 1
                        co_association[j, i] += 1
        
        # Normalize co-association matrix
        co_association /= len(ensemble_labels)
        
        # Apply hierarchical clustering to co-association matrix
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='average')
        consensus_labels = hierarchical.fit_predict(1 - co_association)
        
        return consensus_labels
    
    def predict_ensemble(self, df=None, feature_columns=None, add_technical_features=True, encoded_features=None, indices=None, X_original_scaled=None):
        """
        Make predictions with ensemble of clustering models
        
        Parameters:
        - df: DataFrame with market data (optional if encoded_features and X_original_scaled are provided)
        - feature_columns: List of columns to use as features (default: None)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        - encoded_features: Optional, pre-encoded features (numpy array)
        - indices: Optional, indices corresponding to the encoded features
        - X_original_scaled: Optional, original scaled features (numpy array) for anomaly score calculation
        
        Returns:
        - results: DataFrame with cluster labels and anomaly scores (with individual model predictions as extra columns)
        """
        if not self.ensemble_models:
            raise ValueError("Ensemble models not trained. Call train_ensemble() first.")
        if encoded_features is not None and indices is not None:
            ef = encoded_features
            idx = indices
            # Use X_original_scaled for anomaly score if provided, else fallback to NaN
            if X_original_scaled is not None:
                # If use_temporal, X_original_scaled shape: (n_samples, seq_len, n_features)
                # If not, shape: (n_samples, n_features)
                # For now, assume reconstructed_features is not available, so set anomaly_scores to NaN
                anomaly_scores = np.full(len(ef), np.nan)
            else:
                anomaly_scores = np.full(len(ef), np.nan)
        else:
            # Prepare and encode features as usual
            if self.use_temporal:
                X, idx, _ = self._prepare_temporal_data(df, feature_columns, add_technical_features)
                X_reshaped = X.reshape(-1, X.shape[-1])
                X = self.feature_scaler.transform(X_reshaped).reshape(X.shape)
            else:
                X, idx, _ = self._prepare_data(df, feature_columns, add_technical_features)
                X = self.feature_scaler.transform(X)
            X_tensor = torch.FloatTensor(X)
            dataset = MarketClusteringDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            self.autoencoder.eval()
            with torch.no_grad():
                reconstructed_features = []
                ef = []
                for batch in dataloader:
                    reconstructed, encoded = self.autoencoder(batch)
                    reconstructed_features.append(reconstructed.detach().cpu().numpy())
                    ef.append(encoded.detach().cpu().numpy())
                reconstructed_features = np.concatenate(reconstructed_features, axis=0)
                ef = np.concatenate(ef, axis=0)
            if self.use_temporal:
                anomaly_scores = np.mean(np.square(X_tensor.numpy() - reconstructed_features), axis=(1, 2))
            else:
                anomaly_scores = np.mean(np.square(X_tensor.numpy() - reconstructed_features), axis=1)
        # Loop over ensemble models to obtain cluster predictions
        ensemble_labels = []
        for name, model in self.ensemble_models:
            if name == 'hdbscan':
                labels, _ = hdbscan.approximate_predict(model, ef)
            elif name in ['dbscan', 'hierarchical', 'affinity']:
                labels = model.fit_predict(ef)
            elif name == 'gmm':
                model.fit(ef)
                labels = model.predict(ef)
            else:
                labels = model.predict(ef)
            ensemble_labels.append(labels)
        # Combine predictions via consensus clustering
        consensus_labels = self.consensus_clustering(ensemble_labels)
        # Assemble results in a DataFrame, adding individual model predictions if desired
        results = pd.DataFrame({
            'cluster': consensus_labels,
            'anomaly_score': anomaly_scores
        }, index=idx[:len(consensus_labels)])
        for i, (name, _) in enumerate(self.ensemble_models):
            results[f'cluster_{name}'] = ensemble_labels[i]
        return results
    
    def save_model(self, path=None, symbol=None, period=None, interval=None):
        """
        Save model to file
        
        Parameters:
        - path: Path to save model (default: None, use self.model_path)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None) 
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - model_path: Path to saved model
        """
        from dashboard_utils import get_standardized_model_filename
        
        if path is None:
            path = self.model_path
            
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Use the standardized filename format for the main model file
        base_filename = get_standardized_model_filename(
            model_type="clustering",
            model_name=self.clustering_method,
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=path
        )
        
        # Save autoencoder model
        if self.autoencoder is not None:
            torch.save(self.autoencoder.state_dict(), f"{base_filename}_autoencoder.pth")
        
        # Save clustering model if not an ensemble
        if not self.use_ensemble and self.clustering_model is not None:
            with open(f"{base_filename}_clustering_model.pkl", 'wb') as f:
                pickle.dump(self.clustering_model, f)
        
        # Save ensemble models if using ensemble
        if self.use_ensemble and self.ensemble_models:
            with open(f"{base_filename}_ensemble_models.pkl", 'wb') as f:
                pickle.dump(self.ensemble_models, f)
        
        # Save feature scaler
        with open(f"{base_filename}_feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Save PCA model if available
        if self.pca is not None:
            with open(f"{base_filename}_pca.pkl", 'wb') as f:
                pickle.dump(self.pca, f)
        
        # Save configuration
        config = {
            'n_clusters': self.n_clusters,
            'encoding_size': self.encoding_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'clustering_method': self.clustering_method,
            'use_temporal': self.use_temporal,
            'sequence_length': self.sequence_length,
            'use_ensemble': self.use_ensemble,
            'input_size': self.input_size,
            'metrics_history': self.metrics_history,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{base_filename}_config.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            for key, value in config.items():
                if isinstance(value, np.ndarray):
                    config[key] = value.tolist()
            
            json.dump(config, f)
        
        print(f"Model saved to {base_filename}")
        return base_filename
    
    def load_model(self, path=None, symbol=None, period=None, interval=None):
        """
        Load a trained model from disk
        
        Parameters:
        - path: Path to load the model from (default: None, use self.model_path)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Set path if not provided
        if path is None:
            path = self.model_path
        
        # Use standardized filename function for consistency with save_model
        base_filename = get_standardized_model_filename(
            model_type="clustering",
            model_name=self.clustering_method,
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=path
        )
        
        # Load configuration
        config_file = f"{base_filename}_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            self.n_clusters = config['n_clusters']
            self.encoding_size = config['encoding_size']
            self.learning_rate = config['learning_rate']
            self.batch_size = config['batch_size']
            self.num_epochs = config['num_epochs']
            self.clustering_method = config['clustering_method']
            self.use_temporal = config['use_temporal']
            self.sequence_length = config['sequence_length']
            self.use_ensemble = config['use_ensemble']
            self.input_size = config['input_size']
        
        # Load feature scaler
        scaler_file = f"{base_filename}_feature_scaler.pkl"
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        # Load PCA model
        pca_file = f"{base_filename}_pca.pkl"
        if os.path.exists(pca_file):
            with open(pca_file, 'rb') as f:
                self.pca = pickle.load(f)
        
        # Create and load autoencoder model
        if self.input_size is not None:
            self.autoencoder = self._create_autoencoder(self.input_size)
            model_file = f"{base_filename}_autoencoder.pth"
            if os.path.exists(model_file):
                self.autoencoder.load_state_dict(torch.load(model_file))
                self.autoencoder.eval()
        
        # Load clustering model
        model_file = f"{base_filename}_clustering_model.pkl"
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.clustering_model = pickle.load(f)
        
        # Load ensemble models if use_ensemble is True
        if self.use_ensemble:
            self.ensemble_models = []
            for name in ['kmeans', 'dbscan', 'gmm', 'hierarchical', 'hdbscan']:
                model_file = f"{base_filename}_{name}.pkl"
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                        self.ensemble_models.append((name, model))

    def generate_clustering_report(self, df, feature_columns=None, add_technical_features=True, price_col='close', filename=None, symbol=None, period=None, interval=None):
        """
        Generate comprehensive clustering report
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, use all numeric columns)
        - add_technical_features: Whether to add engineered technical indicators as features (default: True)
        - price_col: Column name for price (default: 'close')
        - filename: Base filename for saving the report (default: None)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (default: None)
        - interval: Time interval to include in filenames (default: None)
        
        Returns:
        - report: Dictionary with report components (including visuals)
        """
        import torch
        import numpy as np
        from dashboard_utils import get_standardized_model_filename
        model_path = self.model_path
        base_filename = get_standardized_model_filename(
            model_type="clustering",
            model_name=self.clustering_method,
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=model_path
        )
        config_file = f"{base_filename}_config.json"
        model_loaded = False

        if os.path.exists(config_file):
            self.load_model(path=model_path, symbol=symbol, period=period, interval=interval)
            model_loaded = True

        # --- Prepare features ONCE ---
        if self.use_temporal:
            X_prepared, indices, _ = self._prepare_temporal_data(df, feature_columns=feature_columns, add_technical_features=add_technical_features)
            X_reshaped = X_prepared.reshape(-1, X_prepared.shape[-1])
            if self.feature_scaler is None or not hasattr(self.feature_scaler, 'mean_'):
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(X_reshaped)
            X_scaled_reshaped = self.feature_scaler.transform(X_reshaped)
            X_scaled = X_scaled_reshaped.reshape(X_prepared.shape)
        else:
            X_prepared, indices, _ = self._prepare_data(df, feature_columns=feature_columns, add_technical_features=add_technical_features)
            if self.feature_scaler is None or not hasattr(self.feature_scaler, 'mean_'):
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(X_prepared)
            X_scaled = self.feature_scaler.transform(X_prepared)
        X_tensor_scaled = torch.FloatTensor(X_scaled)

        # --- Train if necessary, passing pre-scaled features ---
        essential_model_present = (self.autoencoder is not None and 
                               (self.clustering_model is not None or 
                                (self.use_ensemble and self.ensemble_models)))
        if not model_loaded or not essential_model_present:
            if self.use_ensemble:
                self.train_ensemble(df, feature_columns=feature_columns, add_technical_features=add_technical_features, X_scaled_for_encoding=X_tensor_scaled)
            else:
                self.train(df, feature_columns=feature_columns, add_technical_features=add_technical_features, X_scaled_for_encoding=X_tensor_scaled)
            self.save_model(path=model_path, symbol=symbol, period=period, interval=interval)

        # --- Encode features ONCE using the (now definitely trained/loaded) autoencoder ---
        self.autoencoder.eval()
        dataset_for_encoding = MarketClusteringDataset(X_tensor_scaled)
        dataloader_for_encoding = DataLoader(dataset_for_encoding, batch_size=self.batch_size)
        with torch.no_grad():
            reconstructed_list = []
            encoded_list = []
            for batch_scaled in dataloader_for_encoding:
                reconstructed, encoded = self.autoencoder(batch_scaled)
                reconstructed_list.append(reconstructed.detach().cpu().numpy())
                encoded_list.append(encoded.detach().cpu().numpy())
        reconstructed_features_final = np.concatenate(reconstructed_list, axis=0)
        encoded_features_final = np.concatenate(encoded_list, axis=0)

        # --- Predict, Evaluate, Visualize using the single set of prepared/encoded features ---
        if self.use_ensemble:
            results = self.predict_ensemble(
                encoded_features=encoded_features_final, X_original_scaled=X_tensor_scaled.numpy(), indices=indices
            )
        else:
            results = self.predict(
                encoded_features=encoded_features_final, X_original_scaled=X_tensor_scaled.numpy(), indices=indices
            )

        metrics = self.evaluate(
            encoded_features=encoded_features_final, labels=results['cluster'].values if 'cluster' in results else None
        )

        report = {
            'results': results,
            'metrics': metrics
        }

        figs = {}
        cluster_fig = self.visualize(
            encoded_features=encoded_features_final, labels=results['cluster'].values if 'cluster' in results else None, plot_type='scatter'
        )
        figs['cluster_scatter'] = cluster_fig
        try:
            tsne_fig = self.visualize(
                encoded_features=encoded_features_final, labels=results['cluster'].values if 'cluster' in results else None, plot_type='tsne'
            )
            figs['cluster_tsne'] = tsne_fig
        except Exception as e:
            print(f"t-SNE visualization failed: {e}")
        try:
            umap_fig = self.visualize(
                encoded_features=encoded_features_final, labels=results['cluster'].values if 'cluster' in results else None, plot_type='umap'
            )
            figs['cluster_umap'] = umap_fig
        except Exception as e:
            print(f"UMAP visualization failed: {e}")

        time_fig = self.plot_clusters_over_time(df.loc[results.index], results, price_col)
        figs['clusters_time_series'] = time_fig
        anomaly_fig = self.plot_anomaly_scores(df.loc[results.index], results, price_col)
        figs['anomaly_scores'] = anomaly_fig

        report['visualizations'] = figs

        if filename is not None:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            for name, fig in figs.items():
                fig.savefig(f"{filename}_{name}.png", dpi=300, bbox_inches='tight')
            report_data = {k: v for k, v in report.items() if k != 'visualizations'}
            with open(f"{filename}_data.pkl", 'wb') as f:
                pickle.dump(report_data, f)

        return report
    
    def plot_clusters_over_time(self, df, results, price_col='close'):
        """
        Plot clusters over time with price
        
        Parameters:
        - df: DataFrame with market data
        - results: DataFrame with clustering results
        - price_col: Column name for price (default: 'close')
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col.capitalize(), color='blue')
        ax1.set_ylabel(price_col.capitalize())
        ax1.set_title('Price and Clusters Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot clusters on bottom subplot
        scatter = ax2.scatter(results.index, [0.5] * len(results), c=results['cluster'], cmap='viridis', 
                             s=50, alpha=0.7)
        ax2.set_ylabel('Clusters')
        ax2.set_yticks([])
        ax2.grid(True)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Cluster')
        
        # Adjust layout
        plt.tight_layout()
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def plot_anomaly_scores(self, df, results, price_col='close'):
        """
        Plot anomaly scores with price
        
        Parameters:
        - df: DataFrame with market data
        - results: DataFrame with clustering results
        - price_col: Column name for price (default: 'close')
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col.capitalize(), color='blue')
        ax1.set_ylabel(price_col.capitalize())
        ax1.set_title('Price and Anomaly Scores Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot anomaly scores on bottom subplot
        ax2.plot(results.index, results['anomaly_score'], color='red', label='Anomaly Score')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Highlight high anomaly scores
        threshold = results['anomaly_score'].mean() + 2 * results['anomaly_score'].std()
        high_anomalies = results[results['anomaly_score'] > threshold]
        ax2.scatter(high_anomalies.index, high_anomalies['anomaly_score'], color='darkred', s=50, zorder=5)
        
        # Highlight high anomalies on price chart
        for date in high_anomalies.index:
            if date in df.index:
                ax1.axvline(x=date, color='red', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def convert_mpl_to_plotly(self, fig_mpl):
        """
        Convert Matplotlib figure to Plotly figure
        
        Parameters:
        - fig_mpl: Matplotlib figure
        
        Returns:
        - fig_plotly: Plotly figure
        """
        try:
            import plotly.tools as tls
            import plotly.graph_objects as go
            
            # Convert to Plotly figure
            fig_plotly = tls.mpl_to_plotly(fig_mpl)
            
            # Update layout
            fig_plotly.update_layout(
                template='plotly_white',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig_plotly
        except ImportError:
            print("Plotly not available, returning Matplotlib figure")
            return fig_mpl
    
    def _encode_features(self, df, feature_columns, add_technical_features):
        """
        Prepare and encode features using the trained autoencoder.
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features
        - add_technical_features: Whether to add engineered technical indicators
        
        Returns:
        - encoded_features: NumPy array of encoded features
        """
        # Prepare data
        if self.use_temporal:
            X, indices, _ = self._prepare_temporal_data(df, feature_columns, add_technical_features)
            X_reshaped = X.reshape(-1, X.shape[-1])
            X = self.feature_scaler.transform(X_reshaped).reshape(X.shape)
        else:
            X, indices, _ = self._prepare_data(df, feature_columns, add_technical_features)
            X = self.feature_scaler.transform(X)
        
        # Convert to tensor and create dataloader
        X_tensor = torch.FloatTensor(X)
        dataset = MarketClusteringDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Encode data using the autoencoder
        self.autoencoder.eval()
        encoded_list = []
        with torch.no_grad():
            for batch in dataloader:
                _, encoded = self.autoencoder(batch)
                encoded_list.append(encoded.detach().cpu().numpy())
        encoded_features = np.concatenate(encoded_list, axis=0)
        return encoded_features
