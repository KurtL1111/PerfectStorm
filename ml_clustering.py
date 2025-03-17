"""
Machine Learning Clustering Module for Perfect Storm Dashboard

This module implements clustering algorithms to identify when multiple
indicators align in a "perfect storm" configuration that could signal
significant market movements.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle

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
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
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

class PerfectStormClustering:
    """Class for clustering market data to identify 'perfect storm' configurations"""
    
    def __init__(self, n_clusters=5, encoding_size=10, learning_rate=0.001, 
                 batch_size=32, num_epochs=50, model_path='models'):
        """
        Initialize the PerfectStormClustering class
        
        Parameters:
        - n_clusters: Number of clusters for KMeans (default: 5)
        - encoding_size: Size of the encoded representation for autoencoder (default: 10)
        - learning_rate: Learning rate for autoencoder training (default: 0.001)
        - batch_size: Batch size for autoencoder training (default: 32)
        - num_epochs: Number of training epochs for autoencoder (default: 50)
        - model_path: Path to save/load models (default: 'models')
        """
        self.n_clusters = n_clusters
        self.encoding_size = encoding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize models
        self.kmeans = None
        self.dbscan = None
        self.autoencoder = None
        self.pca = None
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Initialize cluster performance metrics
        self.silhouette_avg = None
        
        # Initialize cluster characteristics
        self.cluster_characteristics = None
        self.cluster_returns = None
    
    def _prepare_features(self, df, feature_columns=None):
        """
        Prepare features for clustering
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - features: Numpy array of features
        """
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.fit_transform(features)
        
        return features
    
    def train_kmeans(self, df, feature_columns=None):
        """
        Train KMeans clustering model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - self: The trained model
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Train KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Calculate silhouette score
        self.silhouette_avg = silhouette_score(features, cluster_labels)
        print(f"Silhouette Score: {self.silhouette_avg:.4f}")
        
        # Save the model
        with open(os.path.join(self.model_path, 'kmeans_model.pkl'), 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Analyze cluster characteristics
        self._analyze_clusters(df, cluster_labels, feature_columns)
        
        return self
    
    def train_dbscan(self, df, feature_columns=None, eps=0.5, min_samples=5):
        """
        Train DBSCAN clustering model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other (default: 0.5)
        - min_samples: The number of samples in a neighborhood for a point to be considered as a core point (default: 5)
        
        Returns:
        - self: The trained model
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Train DBSCAN
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan.fit_predict(features)
        
        # Calculate silhouette score if there are at least 2 clusters (excluding noise)
        if len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) >= 2:
            # Filter out noise points (label -1)
            mask = cluster_labels != -1
            if np.sum(mask) > 1:  # Ensure there are at least 2 points after filtering
                self.silhouette_avg = silhouette_score(features[mask], cluster_labels[mask])
                print(f"Silhouette Score (excluding noise): {self.silhouette_avg:.4f}")
        
        # Save the model
        with open(os.path.join(self.model_path, 'dbscan_model.pkl'), 'wb') as f:
            pickle.dump(self.dbscan, f)
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Analyze cluster characteristics
        self._analyze_clusters(df, cluster_labels, feature_columns)
        
        return self
    
    def train_autoencoder(self, df, feature_columns=None):
        """
        Train autoencoder for dimensionality reduction and anomaly detection
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - history: Training history
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features)
        
        # Create dataset and dataloader
        dataset = MarketClusteringDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create autoencoder
        input_size = features.shape[1]
        self.autoencoder = Autoencoder(input_size, self.encoding_size)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        
        # Training loop
        history = {'loss': []}
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                # Forward pass
                outputs, _ = self.autoencoder(batch)
                loss = criterion(outputs, batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            history['loss'].append(epoch_loss)
            
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.6f}')
        
        # Save the model
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'input_size': input_size,
            'encoding_size': self.encoding_size
        }, os.path.join(self.model_path, 'autoencoder_model.pth'))
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        return history
    
    def train_pca(self, df, feature_columns=None, n_components=2):
        """
        Train PCA for dimensionality reduction
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        - n_components: Number of principal components (default: 2)
        
        Returns:
        - self: The trained model
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Train PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(features)
        
        # Save the model
        with open(os.path.join(self.model_path, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(self.pca, f)
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Print explained variance
        explained_variance = self.pca.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")
        
        return self
    
    def _analyze_clusters(self, df, cluster_labels, feature_columns=None):
        """
        Analyze cluster characteristics
        
        Parameters:
        - df: DataFrame with market data
        - cluster_labels: Cluster labels
        - feature_columns: List of columns used as features
        """
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add cluster labels to DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster characteristics
        self.cluster_characteristics = df_with_clusters.groupby('cluster')[feature_columns].mean()
        
        # Calculate future returns for each cluster (if 'close' column exists)
        if 'close' in df.columns:
            # Calculate 5-day future returns
            df_with_clusters['future_return_5d'] = df_with_clusters['close'].pct_change(5).shift(-5)
            
            # Calculate 10-day future returns
            df_with_clusters['future_return_10d'] = df_with_clusters['close'].pct_change(10).shift(-10)
            
            # Calculate 20-day future returns
            df_with_clusters['future_return_20d'] = df_with_clusters['close'].pct_change(20).shift(-20)
            
            # Calculate average future returns for each cluster
            self.cluster_returns = df_with_clusters.groupby('cluster')[
                ['future_return_5d', 'future_return_10d', 'future_return_20d']
            ].mean()
            
            # Print cluster returns
            print("\nCluster Future Returns:")
            print(self.cluster_returns)
    
    def load_kmeans(self, model_file=None, scaler_file=None):
        """
        Load a trained KMeans model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'kmeans_model.pkl')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.kmeans = pickle.load(f)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"KMeans model loaded from {model_file}")
    
    def load_dbscan(self, model_file=None, scaler_file=None):
        """
        Load a trained DBSCAN model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'dbscan_model.pkl')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.dbscan = pickle.load(f)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"DBSCAN model loaded from {model_file}")
    
    def load_autoencoder(self, model_file=None, scaler_file=None):
        """
        Load a trained autoencoder model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'autoencoder_model.pth')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        checkpoint = torch.load(model_file)
        
        # Create the autoencoder
        input_size = checkpoint['input_size']
        self.encoding_size = checkpoint['encoding_size']
        self.autoencoder = Autoencoder(input_size, self.encoding_size)
        
        # Load the state dictionary
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode
        self.autoencoder.eval()
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"Autoencoder model loaded from {model_file}")
    
    def load_pca(self, model_file=None, scaler_file=None):
        """
        Load a trained PCA model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'pca_model.pkl')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.pca = pickle.load(f)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"PCA model loaded from {model_file}")
    
    def predict_kmeans(self, df, feature_columns=None):
        """
        Predict clusters using KMeans
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - cluster_labels: Predicted cluster labels
        """
        # Check if model is loaded
        if self.kmeans is None:
            raise ValueError("KMeans model not loaded. Please train or load a model first.")
        
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Predict clusters
        cluster_labels = self.kmeans.predict(features)
        
        return cluster_labels
    
    def predict_dbscan(self, df, feature_columns=None):
        """
        Predict clusters using DBSCAN
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - cluster_labels: Predicted cluster labels
        """
        # Check if model is loaded
        if self.dbscan is None:
            raise ValueError("DBSCAN model not loaded. Please train or load a model first.")
        
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Predict clusters
        cluster_labels = self.dbscan.fit_predict(features)
        
        return cluster_labels
    
    def encode_autoencoder(self, df, feature_columns=None):
        """
        Encode data using autoencoder
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - encoded_features: Encoded features
        """
        # Check if model is loaded
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not loaded. Please train or load a model first.")
        
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features)
        
        # Encode data
        with torch.no_grad():
            _, encoded_features = self.autoencoder(features_tensor)
        
        return encoded_features.numpy()
    
    def reconstruct_autoencoder(self, df, feature_columns=None):
        """
        Reconstruct data using autoencoder
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - reconstructed_features: Reconstructed features
        - reconstruction_error: Mean squared error between original and reconstructed features
        """
        # Check if model is loaded
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not loaded. Please train or load a model first.")
        
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features)
        
        # Reconstruct data
        with torch.no_grad():
            reconstructed_features, _ = self.autoencoder(features_tensor)
        
        # Calculate reconstruction error
        reconstruction_error = torch.mean((reconstructed_features - features_tensor) ** 2, dim=1).numpy()
        
        return reconstructed_features.numpy(), reconstruction_error
    
    def transform_pca(self, df, feature_columns=None):
        """
        Transform data using PCA
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - transformed_features: PCA-transformed features
        """
        # Check if model is loaded
        if self.pca is None:
            raise ValueError("PCA model not loaded. Please train or load a model first.")
        
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Transform data
        transformed_features = self.pca.transform(features)
        
        return transformed_features
    
    def identify_perfect_storm(self, df, feature_columns=None, method='kmeans', threshold=0.9):
        """
        Identify 'perfect storm' configurations in market data
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        - method: Method to use for identification ('kmeans', 'dbscan', 'autoencoder', default: 'kmeans')
        - threshold: Threshold for autoencoder reconstruction error (default: 0.9, 90th percentile)
        
        Returns:
        - df_storm: DataFrame with perfect storm signals
        """
        # Create a copy of the DataFrame
        df_storm = df.copy()
        
        if method == 'kmeans':
            # Predict clusters using KMeans
            cluster_labels = self.predict_kmeans(df, feature_columns)
            
            # Add cluster labels to DataFrame
            df_storm['cluster'] = cluster_labels
            
            # Identify the most profitable cluster (if cluster_returns is available)
            if self.cluster_returns is not None:
                # Find the cluster with the highest 5-day return
                best_cluster = self.cluster_returns['future_return_5d'].idxmax()
                
                # Add perfect storm signal
                df_storm['perfect_storm'] = (df_storm['cluster'] == best_cluster).astype(int)
            else:
                # If cluster_returns is not available, use the rarest cluster as the perfect storm
                cluster_counts = np.bincount(cluster_labels)
                rarest_cluster = np.argmin(cluster_counts)
                
                # Add perfect storm signal
                df_storm['perfect_storm'] = (df_storm['cluster'] == rarest_cluster).astype(int)
        
        elif method == 'dbscan':
            # Predict clusters using DBSCAN
            cluster_labels = self.predict_dbscan(df, feature_columns)
            
            # Add cluster labels to DataFrame
            df_storm['cluster'] = cluster_labels
            
            # Identify the anomaly cluster (-1)
            df_storm['perfect_storm'] = (df_storm['cluster'] == -1).astype(int)
        
        elif method == 'autoencoder':
            # Reconstruct data using autoencoder
            _, reconstruction_error = self.reconstruct_autoencoder(df, feature_columns)
            
            # Add reconstruction error to DataFrame
            df_storm['reconstruction_error'] = reconstruction_error
            
            # Calculate threshold based on percentile
            error_threshold = np.percentile(reconstruction_error, threshold * 100)
            
            # Add perfect storm signal
            df_storm['perfect_storm'] = (df_storm['reconstruction_error'] > error_threshold).astype(int)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return df_storm
    
    def plot_clusters(self, df, feature_columns=None, method='kmeans', plot_type='pca'):
        """
        Plot clusters
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        - method: Method used for clustering ('kmeans', 'dbscan', default: 'kmeans')
        - plot_type: Type of plot ('pca', 'tsne', default: 'pca')
        """
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Get cluster labels
        if method == 'kmeans':
            cluster_labels = self.predict_kmeans(df, feature_columns)
        elif method == 'dbscan':
            cluster_labels = self.predict_dbscan(df, feature_columns)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Reduce dimensionality for visualization
        if plot_type == 'pca':
            # Use PCA for dimensionality reduction
            if self.pca is None:
                self.pca = PCA(n_components=2)
                reduced_features = self.pca.fit_transform(features)
            else:
                reduced_features = self.pca.transform(features)
            
            x_label = 'Principal Component 1'
            y_label = 'Principal Component 2'
        
        elif plot_type == 'tsne':
            # Use t-SNE for dimensionality reduction
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)
            
            x_label = 't-SNE Component 1'
            y_label = 't-SNE Component 2'
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('tab10', len(unique_clusters))
        
        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            # Get indices for current cluster
            idx = (cluster_labels == cluster)
            
            # Plot points
            plt.scatter(
                reduced_features[idx, 0],
                reduced_features[idx, 1],
                s=50, c=[cmap(i)],
                label=f'Cluster {cluster}'
            )
        
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Cluster Visualization using {plot_type.upper()}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def plot_perfect_storm(self, df_storm, price_column='close'):
        """
        Plot perfect storm signals along with price
        
        Parameters:
        - df_storm: DataFrame with perfect storm signals
        - price_column: Column to plot (default: 'close')
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot price
        ax1.plot(df_storm.index, df_storm[price_column], label=price_column.capitalize())
        ax1.set_ylabel(price_column.capitalize())
        ax1.set_title('Price and Perfect Storm Signals')
        ax1.legend()
        
        # Highlight perfect storm signals
        signal_dates = df_storm[df_storm['perfect_storm'] == 1].index
        for date in signal_dates:
            ax1.axvline(x=date, color='r', linestyle='--', alpha=0.3)
        
        # Plot cluster or reconstruction error
        if 'cluster' in df_storm.columns:
            ax2.scatter(df_storm.index, df_storm['cluster'], label='Cluster', alpha=0.7)
            ax2.set_ylabel('Cluster')
        elif 'reconstruction_error' in df_storm.columns:
            ax2.plot(df_storm.index, df_storm['reconstruction_error'], label='Reconstruction Error', color='purple')
            ax2.set_ylabel('Reconstruction Error')
        
        ax2.set_xlabel('Date')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
def example_usage():
    """Example of how to use the PerfectStormClustering class"""
    
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
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = np.random.normal(50, 15, 500)  # Simplified RSI
    
    # Drop NaN values
    df = df.dropna()
    
    # Create a PerfectStormClustering instance
    psc = PerfectStormClustering(n_clusters=5, num_epochs=5)
    
    # Train KMeans model
    psc.train_kmeans(df)
    
    # Identify perfect storm configurations
    df_storm = psc.identify_perfect_storm(df, method='kmeans')
    
    # Plot perfect storm signals
    psc.plot_perfect_storm(df_storm)
    
    # Plot clusters
    psc.plot_clusters(df, method='kmeans')

if __name__ == '__main__':
    example_usage()
