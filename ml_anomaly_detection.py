"""
Machine Learning Anomaly Detection Module for Perfect Storm Dashboard

This module implements anomaly detection algorithms to identify unusual
market conditions that could represent opportunities or risks.
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle

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
    """Variational Autoencoder model for anomaly detection"""
    
    def __init__(self, input_size, latent_size=10):
        """
        Initialize the VAE model
        
        Parameters:
        - input_size: Number of input features
        - latent_size: Size of the latent space (default: 10)
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Mean and variance for the latent space
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_var = nn.Linear(64, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
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
        - x_reconstructed: Reconstructed input
        """
        # Decode
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - x_reconstructed: Reconstructed input
        - mu: Mean of the latent distribution
        - log_var: Log variance of the latent distribution
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, mu, log_var

class MarketAnomalyDetection:
    """Class for anomaly detection in market data"""
    
    def __init__(self, model_type='isolation_forest', contamination=0.05, 
                 latent_size=10, learning_rate=0.001, batch_size=32, num_epochs=50,
                 model_path='models'):
        """
        Initialize the MarketAnomalyDetection class
        
        Parameters:
        - model_type: Type of model to use ('isolation_forest', 'lof', 'one_class_svm', 'vae', default: 'isolation_forest')
        - contamination: Expected proportion of outliers in the data (default: 0.05)
        - latent_size: Size of the latent space for VAE (default: 10)
        - learning_rate: Learning rate for VAE training (default: 0.001)
        - batch_size: Batch size for VAE training (default: 32)
        - num_epochs: Number of training epochs for VAE (default: 50)
        - model_path: Path to save/load models (default: 'models')
        """
        self.model_type = model_type
        self.contamination = contamination
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize models
        self.isolation_forest = None
        self.lof = None
        self.one_class_svm = None
        self.vae = None
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Initialize anomaly scores
        self.anomaly_scores = None
        self.anomaly_threshold = None
    
    def _prepare_features(self, df, feature_columns=None):
        """
        Prepare features for anomaly detection
        
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
    
    def train_isolation_forest(self, df, feature_columns=None):
        """
        Train Isolation Forest model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - self: The trained model
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.isolation_forest.fit(features)
        
        # Calculate anomaly scores
        # Note: Isolation Forest returns negative scores, where lower values indicate anomalies
        # We convert to positive scores where higher values indicate anomalies
        raw_scores = -self.isolation_forest.decision_function(features)
        self.anomaly_scores = raw_scores
        
        # Calculate anomaly threshold (based on contamination)
        self.anomaly_threshold = np.percentile(raw_scores, (1 - self.contamination) * 100)
        
        # Save the model
        with open(os.path.join(self.model_path, 'isolation_forest_model.pkl'), 'wb') as f:
            pickle.dump(self.isolation_forest, f)
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        return self
    
    def train_lof(self, df, feature_columns=None):
        """
        Train Local Outlier Factor model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - self: The trained model
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Train Local Outlier Factor
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True
        )
        self.lof.fit(features)
        
        # Calculate anomaly scores
        # Note: LOF returns negative scores, where lower values indicate anomalies
        # We convert to positive scores where higher values indicate anomalies
        raw_scores = -self.lof.decision_function(features)
        self.anomaly_scores = raw_scores
        
        # Calculate anomaly threshold (based on contamination)
        self.anomaly_threshold = np.percentile(raw_scores, (1 - self.contamination) * 100)
        
        # Save the model
        with open(os.path.join(self.model_path, 'lof_model.pkl'), 'wb') as f:
            pickle.dump(self.lof, f)
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        return self
    
    def train_one_class_svm(self, df, feature_columns=None):
        """
        Train One-Class SVM model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - self: The trained model
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Train One-Class SVM
        self.one_class_svm = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='scale'
        )
        self.one_class_svm.fit(features)
        
        # Calculate anomaly scores
        # Note: One-Class SVM returns negative scores, where lower values indicate anomalies
        # We convert to positive scores where higher values indicate anomalies
        raw_scores = -self.one_class_svm.decision_function(features)
        self.anomaly_scores = raw_scores
        
        # Calculate anomaly threshold (based on contamination)
        self.anomaly_threshold = np.percentile(raw_scores, (1 - self.contamination) * 100)
        
        # Save the model
        with open(os.path.join(self.model_path, 'one_class_svm_model.pkl'), 'wb') as f:
            pickle.dump(self.one_class_svm, f)
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        return self
    
    def train_vae(self, df, feature_columns=None):
        """
        Train Variational Autoencoder model
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        
        Returns:
        - history: Training history
        """
        # Prepare features
        features = self._prepare_features(df, feature_columns)
        
        # Apply MinMaxScaler for VAE (features should be in [0, 1])
        minmax_scaler = MinMaxScaler()
        features = minmax_scaler.fit_transform(features)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features)
        
        # Create dataset and dataloader
        dataset = MarketAnomalyDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create VAE
        input_size = features.shape[1]
        self.vae = VariationalAutoencoder(input_size, self.latent_size)
        
        # Define optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        # Training loop
        history = {'loss': [], 'reconstruction_loss': [], 'kl_divergence': []}
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            for batch in dataloader:
                # Forward pass
                reconstructed, mu, log_var = self.vae(batch)
                
                # Calculate reconstruction loss
                reconstruction_loss = torch.mean((reconstructed - batch) ** 2)
                
                # Calculate KL divergence
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss
                loss = reconstruction_loss + kl_divergence
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += reconstruction_loss.item()
                epoch_kl_loss += kl_divergence.item()
            
            epoch_loss /= len(dataloader)
            epoch_recon_loss /= len(dataloader)
            epoch_kl_loss /= len(dataloader)
            
            history['loss'].append(epoch_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['kl_divergence'].append(epoch_kl_loss)
            
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.6f}, Recon Loss: {epoch_recon_loss:.6f}, KL Loss: {epoch_kl_loss:.6f}')
        
        # Calculate anomaly scores
        self.vae.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.vae(features_tensor)
            reconstruction_error = torch.mean((reconstructed - features_tensor) ** 2, dim=1).numpy()
        
        self.anomaly_scores = reconstruction_error
        
        # Calculate anomaly threshold (based on contamination)
        self.anomaly_threshold = np.percentile(reconstruction_error, (1 - self.contamination) * 100)
        
        # Save the model
        torch.save({
            'model_state_dict': self.vae.state_dict(),
            'input_size': input_size,
            'latent_size': self.latent_size
        }, os.path.join(self.model_path, 'vae_model.pth'))
        
        # Save the scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Save the minmax scaler
        with open(os.path.join(self.model_path, 'minmax_scaler.pkl'), 'wb') as f:
            pickle.dump(minmax_scaler, f)
        
        return history
    
    def load_isolation_forest(self, model_file=None, scaler_file=None):
        """
        Load a trained Isolation Forest model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'isolation_forest_model.pkl')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.isolation_forest = pickle.load(f)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"Isolation Forest model loaded from {model_file}")
    
    def load_lof(self, model_file=None, scaler_file=None):
        """
        Load a trained Local Outlier Factor model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'lof_model.pkl')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.lof = pickle.load(f)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"Local Outlier Factor model loaded from {model_file}")
    
    def load_one_class_svm(self, model_file=None, scaler_file=None):
        """
        Load a trained One-Class SVM model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'one_class_svm_model.pkl')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        # Load the model
        with open(model_file, 'rb') as f:
            self.one_class_svm = pickle.load(f)
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        print(f"One-Class SVM model loaded from {model_file}")
    
    def load_vae(self, model_file=None, scaler_file=None, minmax_scaler_file=None):
        """
        Load a trained VAE model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        - scaler_file: Path to the scaler file (default: None, uses default path)
        - minmax_scaler_file: Path to the minmax scaler file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'vae_model.pth')
        
        if scaler_file is None:
            scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        
        if minmax_scaler_file is None:
            minmax_scaler_file = os.path.join(self.model_path, 'minmax_scaler.pkl')
        
        # Load the model
        checkpoint = torch.load(model_file)
        
        # Create the VAE
        input_size = checkpoint['input_size']
        self.latent_size = checkpoint['latent_size']
        self.vae = VariationalAutoencoder(input_size, self.latent_size)
        
        # Load the state dictionary
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode
        self.vae.eval()
        
        # Load the scaler
        with open(scaler_file, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        # Load the minmax scaler
        with open(minmax_scaler_file, 'rb') as f:
            self.minmax_scaler = pickle.load(f)
        
        print(f"VAE model loaded from {model_file}")
    
    def detect_anomalies(self, df, feature_columns=None, threshold=None):
        """
        Detect anomalies in market data
        
        Parameters:
        - df: DataFrame with market data
        - feature_columns: List of columns to use as features (default: None, uses all numeric columns)
        - threshold: Anomaly threshold (default: None, uses threshold from training)
        
        Returns:
        - df_anomalies: DataFrame with anomaly scores and labels
        """
        # Create a copy of the DataFrame
        df_anomalies = df.copy()
        
        # If feature columns not specified, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Extract features
        features = df[feature_columns].values
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        # Detect anomalies based on model type
        if self.model_type == 'isolation_forest':
            if self.isolation_forest is None:
                raise ValueError("Isolation Forest model not loaded. Please train or load a model first.")
            
            # Calculate anomaly scores
            anomaly_scores = -self.isolation_forest.decision_function(features)
        
        elif self.model_type == 'lof':
            if self.lof is None:
                raise ValueError("Local Outlier Factor model not loaded. Please train or load a model first.")
            
            # Calculate anomaly scores
            anomaly_scores = -self.lof.decision_function(features)
        
        elif self.model_type == 'one_class_svm':
            if self.one_class_svm is None:
                raise ValueError("One-Class SVM model not loaded. Please train or load a model first.")
            
            # Calculate anomaly scores
            anomaly_scores = -self.one_class_svm.decision_function(features)
        
        elif self.model_type == 'vae':
            if self.vae is None:
                raise ValueError("VAE model not loaded. Please train or load a model first.")
            
            # Apply MinMaxScaler for VAE
            features = self.minmax_scaler.transform(features)
            
            # Convert to PyTorch tensor
            features_tensor = torch.FloatTensor(features)
            
            # Calculate reconstruction error
            with torch.no_grad():
                reconstructed, _, _ = self.vae(features_tensor)
                anomaly_scores = torch.mean((reconstructed - features_tensor) ** 2, dim=1).numpy()
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Add anomaly scores to DataFrame
        df_anomalies['anomaly_score'] = anomaly_scores
        
        # Use threshold from training if not specified
        if threshold is None:
            threshold = self.anomaly_threshold
        
        # Add anomaly labels
        df_anomalies['anomaly'] = (df_anomalies['anomaly_score'] > threshold).astype(int)
        
        return df_anomalies
    
    def plot_anomaly_scores(self, df_anomalies, price_column='close'):
        """
        Plot anomaly scores along with price
        
        Parameters:
        - df_anomalies: DataFrame with anomaly scores and labels
        - price_column: Column to plot (default: 'close')
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot price
        ax1.plot(df_anomalies.index, df_anomalies[price_column], label=price_column.capitalize())
        ax1.set_ylabel(price_column.capitalize())
        ax1.set_title('Price and Anomaly Detection')
        ax1.legend()
        
        # Highlight anomalies
        anomaly_dates = df_anomalies[df_anomalies['anomaly'] == 1].index
        for date in anomaly_dates:
            ax1.axvline(x=date, color='r', linestyle='--', alpha=0.3)
        
        # Plot anomaly scores
        ax2.plot(df_anomalies.index, df_anomalies['anomaly_score'], label='Anomaly Score', color='purple')
        ax2.axhline(y=self.anomaly_threshold, color='r', linestyle='--', label=f'Threshold ({self.anomaly_threshold:.4f})')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_xlabel('Date')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_distribution(self, df_anomalies):
        """
        Plot distribution of anomaly scores
        
        Parameters:
        - df_anomalies: DataFrame with anomaly scores and labels
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of anomaly scores
        plt.hist(df_anomalies['anomaly_score'], bins=50, alpha=0.7)
        
        # Plot threshold line
        plt.axvline(x=self.anomaly_threshold, color='r', linestyle='--', label=f'Threshold ({self.anomaly_threshold:.4f})')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

# Example usage
def example_usage():
    """Example of how to use the MarketAnomalyDetection class"""
    
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
    
    # Add some anomalies
    for i in range(450, 460):
        data['close'][i] = 120 + np.random.normal(0, 2)
        data['high'][i] = data['close'][i] + abs(np.random.normal(0, 1))
        data['low'][i] = data['close'][i] - abs(np.random.normal(0, 1))
        data['volume'][i] = 2000000 + np.random.normal(0, 100000)
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some technical indicators
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = np.random.normal(50, 15, 500)  # Simplified RSI
    
    # Drop NaN values
    df = df.dropna()
    
    # Create a MarketAnomalyDetection instance
    mad = MarketAnomalyDetection(model_type='isolation_forest', contamination=0.05)
    
    # Train the model
    mad.train_isolation_forest(df)
    
    # Detect anomalies
    df_anomalies = mad.detect_anomalies(df)
    
    # Plot anomaly scores
    mad.plot_anomaly_scores(df_anomalies)
    
    # Plot anomaly distribution
    mad.plot_anomaly_distribution(df_anomalies)

if __name__ == '__main__':
    example_usage()
