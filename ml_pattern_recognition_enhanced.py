"""
Enhanced Machine Learning Pattern Recognition Module for Perfect Storm Dashboard

This module implements advanced machine learning models to identify patterns
that precede significant market movements using PyTorch with ONNX Runtime.

Enhancements:
1. Transformer model for sequence modeling
2. Attention mechanism for focusing on important time steps
3. Transfer learning capabilities
4. Ensemble methods for improved prediction accuracy
5. Advanced feature extraction
6. Optimized ONNX Runtime integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import os
import pickle
import json
import time

class MarketPatternDataset(Dataset):
    """Dataset class for market pattern recognition"""
    
    def __init__(self, features, labels=None):
        """
        Initialize the dataset
        
        Parameters:
        - features: Feature tensor
        - labels: Label tensor (optional, for training)
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
        return self.features[idx]

class LSTMModel(nn.Module):
    """LSTM model for pattern recognition"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the LSTM model
        
        Parameters:
        - input_size: Number of input features
        - hidden_size: Number of hidden units
        - num_layers: Number of LSTM layers
        - output_size: Number of output classes
        - dropout: Dropout rate (default: 0.2)
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        
        # Apply sigmoid activation for binary classification
        out = self.sigmoid(out)
        
        return out

class CNNModel(nn.Module):
    """CNN model for pattern recognition"""
    
    def __init__(self, input_size, sequence_length, num_filters, output_size):
        """
        Initialize the CNN model
        
        Parameters:
        - input_size: Number of input features
        - sequence_length: Length of input sequence
        - num_filters: Number of convolutional filters
        - output_size: Number of output classes
        """
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        self.fc_input_size = num_filters*2 * (sequence_length // 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - Output tensor of shape (batch_size, output_size)
        """
        # Transpose input for 1D convolution (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply first convolutional layer
        x = self.relu(self.conv1(x))
        
        # Apply pooling
        x = self.pool(x)
        
        # Apply second convolutional layer
        x = self.relu(self.conv2(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x

class TransformerModel(nn.Module):
    """Transformer model for pattern recognition"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        """
        Initialize the Transformer model
        
        Parameters:
        - input_size: Number of input features
        - d_model: Dimension of the model
        - nhead: Number of heads in multi-head attention
        - num_layers: Number of transformer layers
        - output_size: Number of output classes
        - dropout: Dropout rate (default: 0.1)
        """
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_size)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        - Output tensor of shape (batch_size, output_size)
        """
        # Transpose for transformer (sequence_length, batch_size, input_size)
        x = x.transpose(0, 1)
        
        # Apply input embedding
        x = self.input_embedding(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Get the output from the last time step
        x = x[-1]
        
        # Apply output layer
        x = self.output_layer(x)
        
        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding
        
        Parameters:
        - d_model: Dimension of the model
        - dropout: Dropout rate (default: 0.1)
        - max_len: Maximum sequence length (default: 5000)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        
        # Register buffer (not a parameter, but should be saved and restored in state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor of shape (sequence_length, batch_size, d_model)
        
        Returns:
        - Output tensor of shape (sequence_length, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EnsembleModel(nn.Module):
    """Ensemble model combining multiple pattern recognition models"""
    
    def __init__(self, models, weights=None):
        """
        Initialize the ensemble model
        
        Parameters:
        - models: List of models to ensemble
        - weights: List of weights for each model (default: None, equal weights)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights) / sum(weights)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input tensor
        
        Returns:
        - Output tensor (weighted average of model outputs)
        """
        outputs = [model(x) for model in self.models]
        weighted_outputs = torch.stack([output * weight for output, weight in zip(outputs, self.weights)], dim=0)
        return torch.sum(weighted_outputs, dim=0)

class PatternRecognition:
    """Enhanced class for pattern recognition using machine learning"""
    
    def __init__(self, model_type='transformer', sequence_length=20, hidden_size=128, num_layers=3, 
                 num_filters=64, d_model=128, nhead=4, learning_rate=0.001, batch_size=32, num_epochs=100,
                 model_path='models', use_ensemble=False, use_transfer_learning=False):
        """
        Initialize the PatternRecognition class
        
        Parameters:
        - model_type: Type of model to use ('lstm', 'cnn', 'transformer', default: 'transformer')
        - sequence_length: Length of input sequence (default: 20)
        - hidden_size: Number of hidden units for LSTM (default: 128)
        - num_layers: Number of layers (default: 3)
        - num_filters: Number of convolutional filters for CNN (default: 64)
        - d_model: Dimension of the transformer model (default: 128)
        - nhead: Number of heads in multi-head attention (default: 4)
        - learning_rate: Learning rate for optimization (default: 0.001)
        - batch_size: Batch size for training (default: 32)
        - num_epochs: Number of training epochs (default: 100)
        - model_path: Path to save/load models (default: 'models')
        - use_ensemble: Whether to use ensemble methods (default: False)
        - use_transfer_learning: Whether to use transfer learning (default: False)
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.d_model = d_model
        self.nhead = nhead
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.use_ensemble = use_ensemble
        self.use_transfer_learning = use_transfer_learning
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Initialize model
        self.model = None
        self.input_size = None
        self.output_size = None
        
        # Initialize ONNX session
        self.onnx_session = None
        
        # Initialize ensemble models
        self.ensemble_models = []
        
        # Initialize metrics history
        self.metrics_history = {}
    
    def _prepare_sequence_data(self, df, target_column, threshold=0.02, prediction_horizon=5, add_technical_features=True):
        """
        Prepare sequence data for pattern recognition with enhanced feature extraction
        
        Parameters:
        - df: DataFrame with market data
        - target_column: Column to predict (e.g., 'close')
        - threshold: Threshold for significant movement (default: 0.02, i.e., 2%)
        - prediction_horizon: Number of days to look ahead (default: 5)
        - add_technical_features: Whether to add technical indicators as features (default: True)
        
        Returns:
        - X: Feature sequences
        - y: Labels (1 for significant upward movement, 0 otherwise)
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Add technical features if requested
        if add_technical_features:
            # Add returns
            df_copy['return_1d'] = df_copy[target_column].pct_change()
            df_copy['return_5d'] = df_copy[target_column].pct_change(5)
            df_copy['return_10d'] = df_copy[target_column].pct_change(10)
            
            # Add volatility
            df_copy['volatility_5d'] = df_copy['return_1d'].rolling(5).std()
            df_copy['volatility_10d'] = df_copy['return_1d'].rolling(10).std()
            df_copy['volatility_20d'] = df_copy['return_1d'].rolling(20).std()
            
            # Add moving averages
            df_copy['ma_5d'] = df_copy[target_column].rolling(5).mean()
            df_copy['ma_10d'] = df_copy[target_column].rolling(10).mean()
            df_copy['ma_20d'] = df_copy[target_column].rolling(20).mean()
            
            # Add moving average crossovers
            df_copy['ma_5d_10d_ratio'] = df_copy['ma_5d'] / df_copy['ma_10d']
            df_copy['ma_5d_20d_ratio'] = df_copy['ma_5d'] / df_copy['ma_20d']
            df_copy['ma_10d_20d_ratio'] = df_copy['ma_10d'] / df_copy['ma_20d']
            
            # Add volume features
            if 'volume' in df_copy.columns:
                df_copy['volume_ma_5d'] = df_copy['volume'].rolling(5).mean()
                df_copy['volume_ma_10d'] = df_copy['volume'].rolling(10).mean()
                df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma_5d']
        
        # Calculate future returns
        df_copy['future_return'] = df_copy[target_column].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Create labels based on threshold
        df_copy['label'] = 0
        df_copy.loc[df_copy['future_return'] > threshold, 'label'] = 1  # Significant upward movement
        
        # Drop rows with NaN values
        df_copy = df_copy.dropna()
        
        # Select features (all columns except 'label' and 'future_return')
        feature_columns = [col for col in df_copy.columns if col not in ['label', 'future_return']]
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(df_copy) - self.sequence_length + 1):
            X.append(df_copy[feature_columns].iloc[i:i+self.sequence_length].values)
            y.append(df_copy['label'].iloc[i+self.sequence_length-1])
        
        return np.array(X), np.array(y)
    
    def _create_model(self, input_size, output_size):
        """
        Create a new model
        
        Parameters:
     <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>