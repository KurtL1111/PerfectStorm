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
import plotly.tools as tls
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import os
import pickle
import json
import time
import plotly.graph_objects as go
from dashboard_utils import get_standardized_model_filename, log_with_timestamp
# Additional imports for advanced features and regularization
from technical_indicators import add_technical_indicators  # You must implement this function
from sklearn.utils.class_weight import compute_class_weight

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
        #print(f"Dataset initialized with features shape: {features.shape}, label shape: {labels.shape if labels is not None else 'None'}")

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        if self.labels is not None:
            
            # Get features and labels
            feature = self.features[idx]
            label = self.labels[idx]
            
            # Ensure consistent tensor shapes
            # If feature is 3D [seq_len, input_size] and label is 1D [1]
            # We need to ensure they're compatible for the model
            if len(feature.shape) == 2 and len(label.shape) == 1:
                # Keep feature as is (batch dimension will be added by DataLoader)
                # Reshape label to [1] to ensure it's a 1D tensor with single value
                label = label.view(1)
            
            return feature, label
        else:
            return self.features[idx]

class LSTMModel(nn.Module):
    """LSTM model for pattern recognition"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, layer_norm=True):
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
        self.layer_norm = layer_norm
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Layer normalization for better convergence
        self.ln = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
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
        out = out[:, -1, :]
        out = self.ln(out)
        out = self.fc(out)
        # Apply sigmoid activation for binary classification
        out = self.sigmoid(out)
        # Ensure output is 2D [batch_size, output_size]
        if len(out.shape) > 2:
            out = out.view(out.size(0), -1)
        return out

class CNNModel(nn.Module):
    """CNN model for pattern recognition"""
    
    def __init__(self, input_size, sequence_length, num_filters, output_size, layer_norm=True, device='cpu'):
        """
        Initialize the CNN model
        
        Parameters:
        - input_size: Number of input features
        - sequence_length: Length of input sequence
        - num_filters: Number of convolutional filters
        - output_size: Number of output classes
        - layer_norm: Boolean to enable layer normalization (default: True)
        - device: Device to run the model on (default: 'cpu')
        """
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=3, padding=1)
        # Layer normalization
        self.ln = nn.LayerNorm(num_filters*2 * (sequence_length // 2)) if layer_norm else nn.Identity()
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
        self.dropout = nn.Dropout(0.3)
        self.to(device)    
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
        x = self.ln(x)
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

class TransformerModel(nn.Module):
    """Transformer model for pattern recognition"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1, layer_norm=True):
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
        # Transformer encoder with layer norm
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False, norm_first=layer_norm)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=nn.LayerNorm(d_model) if layer_norm else None)
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
        #print("TransformerModel Output:\n", x)
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
        outputs = []
        output_shapes = []
        
        # Collect outputs from all models
        for i, model in enumerate(self.models):
            try:
                output = model(x)
                #print(f"Model {i} output shape: {output.shape}")
                outputs.append(output)
                output_shapes.append(output.shape)
            except Exception as e:
                print(f"Error in model {i}: {e}")
                continue
        
        if not outputs:
            print("No valid outputs from any models")
            # Return a dummy output of appropriate shape
            return torch.zeros(x.size(0), 1).to(x.device)
        
        # Check if we have different shapes
        if len(set(str(shape) for shape in output_shapes)) > 1:
            print("Different output shapes detected, attempting to reconcile")
            
            # Determine if we have 2D and 3D tensors
            has_2d = any(len(shape) == 2 for shape in output_shapes)
            has_3d = any(len(shape) == 3 for shape in output_shapes)
            
            if has_2d and has_3d:
                # Get the sequence length from the first 3D tensor
                seq_len = next(shape[1] for shape in output_shapes if len(shape) == 3)
                
                # Reshape 2D tensors to match 3D format
                for i, output in enumerate(outputs):
                    if len(output.shape) == 2:
                        # Reshape [batch_size, 1] to [batch_size, seq_len, 1]
                        outputs[i] = output.unsqueeze(1).expand(-1, seq_len, -1)
                        #print(f"Reshaped model {i} output to: {outputs[i].shape}")
        
        try:
            # Stack outputs and apply weights
            #print(f"Attempting to stack outputs with shapes: {[o.shape for o in outputs]}")
            stacked_outputs = torch.stack(outputs, dim=0)
            #print(f"Successfully stacked outputs with shape: {stacked_outputs.shape}")
            #print(f"Stacked outputs shape: {stacked_outputs.shape}")
            
            # Adjust weights view based on output dimension
            if len(stacked_outputs.shape) == 3:  # [num_models, batch_size, output_dim]
                weights_view = self.weights.view(-1, 1, 1)
            elif len(stacked_outputs.shape) == 4:  # [num_models, batch_size, seq_len, output_dim]
                weights_view = self.weights.view(-1, 1, 1, 1)
            else:
                weights_view = self.weights.view(-1, 1)
            
            weighted_outputs = torch.sum(stacked_outputs * weights_view, dim=0)
            #print(f"Weighted outputs shape: {weighted_outputs.shape}")
            return weighted_outputs
        
        except RuntimeError as e:
            print(f"Error in stacking: {e}")
            print(f"Shapes of outputs: {[o.shape for o in outputs]}")
            # Fallback: return the first output
            return outputs[0]

class AttentionLSTM(nn.Module):
    """LSTM model with attention mechanism for pattern recognition"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1, layer_norm=True):
        """
        Initialize the Attention LSTM model
        
        Parameters:
        - input_size: Number of input features
        - hidden_size: Number of hidden units
        - num_layers: Number of LSTM layers
        - output_size: Number of output classes
        - dropout: Dropout rate (default: 0.2)
        """
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
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
        
        # Layer normalization
        out = self.ln(out)
        # Apply attention mechanism
        attention_weights = self.attention(out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(out * attention_weights, dim=1)
        # Apply fully connected layer
        out = self.fc(context_vector)
        # Apply sigmoid activation for binary classification
        out = self.sigmoid(out)
        return out

class MarketPatternRecognition:
    """Enhanced class for market pattern recognition"""
    
    def __init__(self, sequence_length=20, hidden_size=64, num_layers=4, learning_rate=0.001, 
                 batch_size=32, num_epochs=100, model_path='models\\Pattern Recognition Models', model_type='lstm',
                 use_ensemble=True, use_attention=True, use_transfer_learning=True):
        """
        Initialize the MarketPatternRecognition class
        
        Parameters:
        - sequence_length: Length of input sequence (default: 20)
        - hidden_size: Size of hidden layers (default: 64)
        - num_layers: Number of layers in LSTM/Transformer (default: 4)
        - learning_rate: Learning rate for training (default: 0.001)
        - batch_size: Batch size for training (default: 32)
        - num_epochs: Number of training epochs (default: 100)
        - model_path: Path to save/load models (default: 'models')
        - model_type: Type of model to use ('lstm', 'cnn', 'transformer', default: 'lstm')
        - use_ensemble: Whether to use ensemble methods (default: True)
        - use_attention: Whether to use attention mechanism (default: True)
        - use_transfer_learning: Whether to use transfer learning (default: True)
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        self.use_attention = use_attention
        self.use_transfer_learning = use_transfer_learning
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize models
        self.model = None
        self.ensemble_models = []
        
        # Initialize scalers
        self.feature_scaler = None
        
        # Initialize metrics
        self.metrics = {}
        
        # Initialize ONNX models
        self.onnx_model_path = None
        self.onnx_session = None
        
        # Determine device for GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_with_timestamp(f"Using device: {self.device}", log_level="INFO")
    
    def preprocess_data(self, df, feature_cols, target_col=None, train_size=0.8, add_features=True):
        """
        Preprocess data for pattern recognition
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        - target_col: Target column name (optional, for training)
        - train_size: Proportion of data to use for training (default: 0.8)
        
        Returns:
        - processed_data: Dictionary with processed data
        """
        # Advanced feature engineering: add technical indicators, volatility, regime, etc.
        if add_features:
            df = add_technical_indicators(df)
            # Only select numeric columns, exclude target/date/timestamp
            candidate_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in [target_col, 'date', 'timestamp']
            ]
        else:
            # If not adding features, still ensure only numeric columns are used
            candidate_cols = [
                col for col in feature_cols
                if pd.api.types.is_numeric_dtype(df[col]) and col not in [target_col, 'date', 'timestamp']
            ]

        # --- Robust feature column enforcement ---
        if hasattr(self, 'feature_cols_used') and self.feature_cols_used is not None:
            # Inference/prediction: enforce same columns and order as during training
            missing = [col for col in self.feature_cols_used if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in input data required by scaler: {missing}")
            feature_cols_final = self.feature_cols_used.copy()
        else:
            # Training: set feature_cols_used to the columns actually used
            feature_cols_final = candidate_cols.copy()
            self.feature_cols_used = feature_cols_final.copy()

        # Extract features, robust to NaNs
        features = df[feature_cols_final].fillna(0).values
        log_with_timestamp(f"Preprocessing data with features: {feature_cols_final}, shape: {features.shape}", log_level="DEBUG")
        # Initialize feature scaler if not already
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            features = self.feature_scaler.fit_transform(features)
        else:
            features = self.feature_scaler.transform(features)
        
        # Create sequences
        X = []
        for i in range(len(features) - self.sequence_length + 1):
            X.append(features[i:i+self.sequence_length])
        
        X = np.array(X)
        
        # If target column is provided, create labels
        if target_col is not None and target_col in df.columns:
            # Extract target
            target = df[target_col].values
            
            # Create labels (shifted by sequence_length)
            y = target[self.sequence_length-1:]
            
            # Convert to binary classification if needed
            if np.unique(y).size > 2:
                # Assuming positive values are 1, negative are 0
                y = (y > 0).astype(int)
            
            # Split into train and test sets (walk-forward validation recommended for time series)
            split_idx = int(len(X) * train_size)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
            # Create datasets
            train_dataset = MarketPatternDataset(X_train_tensor, y_train_tensor)
            test_dataset = MarketPatternDataset(X_test_tensor, y_test_tensor)
            # Create dataloaders with multi-core processing, adjust workers based on device
            num_workers = min(os.cpu_count(), 8) if os.cpu_count() and self.device.type == "cuda" else min(os.cpu_count(), 4) if os.cpu_count() else 0
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)  # No shuffle for time series
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            # Return processed data
            return {
                'X': X,
                'y': y,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_tensor': X_train_tensor,
                'y_train_tensor': y_train_tensor,
                'X_test_tensor': X_test_tensor,
                'y_test_tensor': y_test_tensor,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'train_loader': train_loader,
                'test_loader': test_loader
            }
        else:
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X)
            
            # Create dataset
            dataset = MarketPatternDataset(X_tensor)
            
            # Create dataloader with multi-core processing, adjust workers based on device
            num_workers = min(os.cpu_count(), 8) if os.cpu_count() and self.device.type == "cuda" else min(os.cpu_count(), 4) if os.cpu_count() else 0
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            
            # Return processed data
            return {
                'X': X,
                'X_tensor': X_tensor,
                'dataset': dataset,
                'dataloader': dataloader
            }
        
    def create_model(self, input_size, output_size=1, device='cpu'):
        """
        Create model for pattern recognition
        
        Parameters:
        - input_size: Number of input features
        - output_size: Number of output classes (default: 1)
        - device: Device to use for training ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - model: Created model
        """
        if self.model_type == 'lstm':
            if self.use_attention:
                model = AttentionLSTM(input_size, self.hidden_size, self.num_layers, output_size, layer_norm=True)
            else:
                model = LSTMModel(input_size, self.hidden_size, self.num_layers, output_size, layer_norm=True)
        elif self.model_type == 'cnn':
            model = CNNModel(input_size, self.sequence_length, self.hidden_size, output_size, layer_norm=True)
        elif self.model_type == 'transformer':
            d_model = 64
            nhead = 8  # More heads for richer attention
            model = TransformerModel(input_size, d_model, nhead, self.num_layers, output_size, layer_norm=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        model.to(device)
        return model
    
    def create_ensemble_models(self, input_size, output_size=1, device='cpu'):
        """
        Create ensemble models for pattern recognition
        
        Parameters:
        - input_size: Number of input features
        - output_size: Number of output classes (default: 1)
        - device: Device to use for training ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - models: List of created models
        """
        models = []
        
        # Create LSTM model
        lstm_model = LSTMModel(input_size, self.hidden_size, self.num_layers, output_size, layer_norm=True)
        models.append(lstm_model)
        # Create CNN model
        cnn_model = CNNModel(input_size, self.sequence_length, self.hidden_size, output_size, layer_norm=True)
        models.append(cnn_model)
        # Create Transformer model
        d_model = 64
        nhead = 8
        transformer_model = TransformerModel(input_size, d_model, nhead, self.num_layers, output_size, layer_norm=True)
        models.append(transformer_model)
        # Create Attention LSTM model
        attention_lstm_model = AttentionLSTM(input_size, self.hidden_size, self.num_layers, output_size, layer_norm=True)
        models.append(attention_lstm_model)
        # Move models to device
        for model in models:
            model.to(device)
        print("Model types in ensemble:", [type(model).__name__ for model in models])
        for i, model in enumerate(models):
            print(f"Model {i} architecture: {model}")
        return models
    
    def train_model(self, train_loader, test_loader=None, device=None, symbol=None, period=None, interval=None):
        """
        Train model for pattern recognition
        
        Parameters:
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for test data (optional)
        - device: Device to use for training ('cpu' or 'cuda', default: None, use self.device)
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - model: Trained model
        """
        # Use class device if not specified
        if device is None:
            device = self.device
        log_with_timestamp(f"Training on device: {device}", log_level="INFO")

        # Get input size from first batch
        for batch in train_loader:
            if isinstance(batch, tuple):
                features, labels = batch
            else:
                features = batch
            # Handle list of tensors with different shapes
            if isinstance(features, list):
                features = features[0]
                input_size = features.shape[2] if len(features.shape) >= 3 else features.shape[1]
                break
            else:
                input_size = features.shape[2] if len(features.shape) >= 3 else features.shape[1]
                break

        if self.use_ensemble:
            self.ensemble_models = self.create_ensemble_models(input_size, device=device)
            for i, model in enumerate(self.ensemble_models):
                log_with_timestamp(f"Training model {i+1}/{len(self.ensemble_models)}...", log_level="INFO")
                # Use the standardized filename format to check if model exists
                model_name = f"{self.model_type}_ensemble_{i}"
                model_filename = get_standardized_model_filename(
                    model_type="pattern_recognition",
                    model_name=model_name,
                    symbol=symbol,
                    period=period,
                    interval=interval,
                    base_path=self.model_path
                ) + ".pth"
                
                if os.path.exists(model_filename):
                    log_with_timestamp(f"Model {i} already exists at {model_filename}, loading instead of training.", log_level="INFO")
                    model.load_state_dict(torch.load(model_filename, map_location=device))
                    model.eval()
                else:
                    self._train_single_model(model, train_loader, test_loader, device, symbol=symbol, period=period, interval=interval, ensemble_idx=i)
            self.model = EnsembleModel(self.ensemble_models)
        else:
            self.model = self.create_model(input_size, device=device)
            # Use the standardized filename format to check if model exists
            model_filename = get_standardized_model_filename(
                model_type="pattern_recognition",
                model_name=self.model_type,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pth"
            
            if os.path.exists(model_filename):
                log_with_timestamp(f"Model already exists at {model_filename}, loading instead of training.", log_level="INFO")
                self.model.load_state_dict(torch.load(model_filename, map_location=device))
                self.model.eval()
            else:
                self._train_single_model(self.model, train_loader, test_loader, device, symbol=symbol, period=period, interval=interval)
        return self.model
    
    def _train_single_model(self, model, train_loader, test_loader=None, device='cpu', symbol=None, period=None, interval=None, ensemble_idx=None):
        """
        Train a single model
        
        Parameters:
        - model: Model to train
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for test data (optional)
        - device: Device to use for training ('cpu' or 'cuda', default: 'cpu')
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        - ensemble_idx: Index if part of ensemble (default: None)
        
        Returns:
        - model: Trained model
        """
        # Get the standardized filename for the model
        from dashboard_utils import get_standardized_model_filename
        
        # Determine the appropriate model name/type
        model_name_for_log = self.model_type
        if ensemble_idx is not None:
            model_name = f"{self.model_type}_ensemble_{ensemble_idx}"
            model_name_for_log = f"{self.model_type}_ensemble_{ensemble_idx} for {symbol} ({period}, {interval})"
        else:
            model_name = self.model_type
            model_name_for_log = f"{self.model_type} for {symbol} ({period}, {interval})"
            
        # Generate standardized filename
        model_pth = get_standardized_model_filename(
            model_type="pattern_recognition",
            model_name=model_name, # Use the specific name for saving
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        ) + ".pth"
            
        # Check if model already exists before training
        if os.path.exists(model_pth):
            log_with_timestamp(f"Model already exists at {model_pth}, loading instead of training.", log_level="INFO")
            model.load_state_dict(torch.load(model_pth, map_location=device))
            model.eval()
            return model
            
        # Move model to device
        model.to(device)
            
        # Compute class weights for imbalanced data
        y_all = []
        for batch in train_loader:
            _, labels = batch
            y_all.extend(labels.cpu().numpy().flatten())
        classes = np.unique(y_all)
        if len(classes) > 1:
            class_weights = compute_class_weight('balanced', classes=classes, y=y_all)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        else:
            class_weights = torch.tensor([1.0], dtype=torch.float32).to(device)
        # Use weighted BCE loss
        def weighted_bce_loss(output, target):
            # Flatten output and target to 1D
            output_flat = output.view(-1)
            target_flat = target.view(-1)
            # Compute weight as

        criterion = weighted_bce_loss
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0
            for features_batch, labels_batch in train_loader:
                features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            test_loss = 0
            if test_loader:
                model.eval()
                with torch.no_grad():
                    for features_batch, labels_batch in test_loader:
                        features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                        outputs = model(features_batch)
                        loss = criterion(outputs, labels_batch)
                        test_loss += loss.item()
                    test_loss /= len(test_loader)

            if (epoch + 1) % max(1, self.num_epochs // 10) == 0 or epoch == self.num_epochs - 1:
                log_with_timestamp(f"Pattern Recognition Model Training progress for {model_name_for_log}: {(epoch + 1) * 100 / self.num_epochs:.0f}% completed. Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}", log_level="INFO")

        # Save the trained model
        torch.save(model.state_dict(), model_pth)
        return model
    
    def evaluate_model(self, model, test_loader, device='cpu'):
        """
        Evaluate model on test data
        
        Parameters:
        - model: Model to evaluate
        - test_loader: DataLoader for test data
        - device: Device to use for evaluation ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - metrics: Dictionary with evaluation metrics
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get batch
                features, labels = batch
                
                # Move batch to device
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Convert outputs to binary predictions
                preds = (outputs > 0.5).float()
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        precision = np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_preds == 1) + 1e-10)
        recall = np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_labels == 1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        return accuracy, precision, recall, f1
    
    def predict(self, X, device):
        """
        Make predictions with the trained model
        
        Parameters:
        - X: Features
        
        Returns:
        - Predictions
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor) and X.dtype != torch.float32:
            X = X.to(torch.float32)
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # If X is a DataLoader, use it directly
        if isinstance(X, DataLoader):
            dataloader = X
        else:
            dataset = MarketPatternDataset(X)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                # If batch is a tuple (features, _), extract features
                if isinstance(batch, tuple):
                    features = batch[0]
                else:
                    features = batch
                    
                # Forward pass
                outputs = self.model(features)
                
                # If outputs is 3D [batch_size, seq_len, 1], take only the last time step
                if len(outputs.shape) == 3:
                    outputs = outputs[:, -1, :]
                    
                # Add predictions to list
                predictions.append(outputs)
        
        # Concatenate predictions
        try:
            predictions = torch.cat(predictions, dim=0)
        except RuntimeError as e:
            #print(f"Error concatenating predictions: {e}")
            #print(f"Prediction shapes: {[p.shape for p in predictions]}")
            # Try to reshape predictions to be consistent
            reshaped_predictions = []
            for p in predictions:
                if len(p.shape) == 3:  # If 3D [batch_size, seq_len, 1]
                    reshaped_predictions.append(p[:, -1, :])
                else:
                    reshaped_predictions.append(p)
            predictions = torch.cat(reshaped_predictions, dim=0)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, dataloader, device):
        """
        Make probability predictions using trained model
        
        Parameters:
        - dataloader: DataLoader for prediction data
        - device: Device to use for prediction ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - probabilities: Model probability predictions
        """
        # Same as predict for binary classification with sigmoid output
        return self.predict(dataloader, device)
    
    def predict_pattern(self, df, feature_cols, threshold=0.5, device='cpu'):
        """
        Predict patterns in market data
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        - threshold: Threshold for binary classification (default: 0.5)
        - device: Device to use for prediction ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - pattern_predictions: DataFrame with pattern predictions
        """
        # Use the feature columns from training if available
        if hasattr(self, 'feature_cols_used') and self.feature_cols_used is not None:
            feature_cols = self.feature_cols_used
        # Preprocess data
        processed_data = self.preprocess_data(df, feature_cols)
        
        # Make predictions
        probabilities = self.predict_proba(processed_data['dataloader'], device)
        
        # Convert to binary predictions
        predictions = (probabilities > threshold).astype(int)
        
        # Create DataFrame with predictions
        pattern_predictions = pd.DataFrame({
            'probability': probabilities.flatten(),
            'prediction': predictions.flatten()
        }, index=df.index[self.sequence_length-1:])
        
        return pattern_predictions
    
    def transfer_learning(self, source_model, train_loader, test_loader=None, device='cpu', freeze_layers=True):
        """
        Apply transfer learning from a source model
        
        Parameters:
        - source_model: Source model to transfer from
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for test data (optional)
        - device: Device to use for training ('cpu' or 'cuda', default: 'cpu')
        - freeze_layers: Whether to freeze source model layers (default: True)
        
        Returns:
        - model: Trained model
        """
        if not self.use_transfer_learning:
            raise ValueError("Transfer learning not enabled. Set use_transfer_learning=True.")
        
        # Get input size from first batch
        for batch in train_loader:
            if isinstance(batch, tuple):
                features, _ = batch
            else:
                features = batch
            if isinstance(features, list):
                features = torch.stack(features)
            input_size = features.shape[2]
            break
        
        # Create new model with same architecture as source model
        if isinstance(source_model, LSTMModel) or isinstance(source_model, AttentionLSTM):
            self.model_type = 'lstm'
            self.use_attention = isinstance(source_model, AttentionLSTM)
        elif isinstance(source_model, CNNModel):
            self.model_type = 'cnn'
        elif isinstance(source_model, TransformerModel):
            self.model_type = 'transformer'
        else:
            raise ValueError(f"Unsupported source model type: {type(source_model)}")
        
        # Create model
        self.model = self.create_model(input_size, device=device)
        
        # Copy weights from source model
        self.model.load_state_dict(source_model.state_dict())
        
        # Freeze layers if requested
        if freeze_layers:
            for name, param in self.model.named_parameters():
                # Freeze all layers except the last one (output layer)
                if 'fc' not in name and 'output_layer' not in name:
                    param.requires_grad = False
        
        # Train model (fine-tuning)
        self._train_single_model(self.model, train_loader, test_loader, device)
        
        return self.model
    
    def export_to_onnx(self, input_size, sequence_length=None, device='cpu'):
        """
        Export model to ONNX format
        
        Parameters:
        - input_size: Number of input features
        - sequence_length: Length of input sequence (default: None, use self.sequence_length)
        - device: Device to use for export ('cpu' or 'cuda', default: 'cpu')
        
        Returns:
        - onnx_model_path: Path to the exported ONNX model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Create dummy input
        dummy_input = torch.randn(1, sequence_length, input_size).to(device)
        
        # Export model
        onnx_model_path = os.path.join(self.model_path, f'pattern_recognition_{self.model_type}.onnx')
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
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
        
        # Load ONNX model with GPU support if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Store ONNX session
        self.onnx_session = session
        
        return session
    
    def predict_onnx(self, dataloader):
        """
        Make predictions using ONNX model
        
        Parameters:
        - dataloader: DataLoader for prediction data
        
        Returns:
        - predictions: Model predictions
        """
        if self.onnx_session is None:
            raise ValueError("ONNX model not loaded. Call load_onnx_model first.")
        
        all_preds = []
        
        for batch in dataloader:
            # Get batch
            if isinstance(batch, tuple):
                features, _ = batch
            else:
                features = batch
            
            # Convert to numpy
            features_np = features.numpy()
            
            # Make predictions batch by batch
            for i in range(len(features_np)):
                # Get single sample
                sample = features_np[i:i+1].astype(np.float32)
                
                # Run inference
                outputs = self.onnx_session.run(
                    None,
                    {'input': sample}
                )
                
                # Store predictions
                all_preds.append(outputs[0])
        
        # Convert to numpy array
        predictions = np.vstack(all_preds)
        
        return predictions
    
    def save_model(self, filename=None, symbol=None, period=None, interval=None):
        """
        Save model to file
        
        Parameters:
        - filename: Filename to save model (default: None, use default filename)
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - filename: Filename of saved model
        """
        from dashboard_utils import get_standardized_model_filename
        
        if filename is None:
            # Use the standardized filename format
            filename = get_standardized_model_filename(
                model_type="pattern_recognition",
                model_name=self.model_type,
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pkl"
        
        # Create dictionary with model parameters
        model_dict = {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'use_ensemble': self.use_ensemble,
            'use_attention': self.use_attention,
            'use_transfer_learning': self.use_transfer_learning,
            'feature_scaler': self.feature_scaler,
            'feature_cols_used': getattr(self, 'feature_cols_used', None),
            'metrics': self.metrics,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save PyTorch model separately only if not in ensemble
        if self.model is not None and not self.use_ensemble:
            model_pth = get_standardized_model_filename(
                model_type="pattern_recognition",
                model_name=f"{self.model_type}_model",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pth"
            torch.save(self.model.state_dict(), model_pth)
            model_dict['model_path'] = model_pth
        
        # Save ensemble models separately with standardized filenames
        if self.ensemble_models:
            ensemble_paths = []
            for i, model in enumerate(self.ensemble_models):
                # Use consistent model naming
                ensemble_model_name = f"{self.model_type}_ensemble_{i}"
                ensemble_path = get_standardized_model_filename(
                    model_type="pattern_recognition",
                    model_name=ensemble_model_name,
                    symbol=symbol,
                    period=period,
                    interval=interval,
                    base_path=self.model_path
                ) + ".pth"
                torch.save(model.state_dict(), ensemble_path)
                ensemble_paths.append(ensemble_path)
            model_dict['ensemble_models_paths'] = ensemble_paths
        
        # Save model dictionary
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)
        
        log_with_timestamp(f"Pattern Recognition Model saved to {filename}", log_level="INFO")
        return filename
    
    def load_model(self, path=None, symbol=None, period=None, interval=None):
        """
        Load model from file
        
        Parameters:
        - path: Path to load model from (default: None, use self.model_path)
        - symbol: Symbol to include in filename search (default: None)
        - period: Time period to include in filename search (default: None)
        - interval: Time interval to include in filename search (default: None)
        
        Returns:
        - success: Whether loading was successful
        """
        from dashboard_utils import get_standardized_model_filename
        
        if path is None:
            path = self.model_path
        
        # Use the standardized filename format
        base_filename = get_standardized_model_filename(
            model_type="pattern_recognition",
            model_name=self.pattern_recognition_method,
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=path
        )
        
        # Load main model
        model_filename = f"{base_filename}.pkl"
        
        try:
            # Load model dict
            with open(model_filename, 'rb') as f:
                model_dict = pickle.load(f)
            # Restore attributes
            self.sequence_length = model_dict.get('sequence_length', self.sequence_length)
            self.hidden_size = model_dict.get('hidden_size', self.hidden_size)
            self.num_layers = model_dict.get('num_layers', self.num_layers)
            self.learning_rate = model_dict.get('learning_rate', self.learning_rate)
            self.batch_size = model_dict.get('batch_size', self.batch_size)
            self.num_epochs = model_dict.get('num_epochs', self.num_epochs)
            self.model_path = model_dict.get('model_path', self.model_path)
            self.model_type = model_dict.get('model_type', self.model_type)
            self.use_ensemble = model_dict.get('use_ensemble', self.use_ensemble)
            self.use_attention = model_dict.get('use_attention', self.use_attention)
            self.use_transfer_learning = model_dict.get('use_transfer_learning', self.use_transfer_learning)
            self.feature_scaler = model_dict.get('feature_scaler', None)
            self.feature_cols_used = model_dict.get('feature_cols_used', None)
            self.metrics = model_dict.get('metrics', {})
            # Restore other files if needed (pattern catalog, stats, etc.)
            # Load pattern catalog if available
            catalog_filename = f"{base_filename}_catalog.pkl"
            if os.path.exists(catalog_filename):
                with open(catalog_filename, 'rb') as f:
                    self.pattern_catalog = pickle.load(f)
            # Load pattern statistics if available
            stats_filename = f"{base_filename}_stats.json"
            if os.path.exists(stats_filename):
                with open(stats_filename, 'r') as f:
                    self.pattern_stats = json.load(f)
            log_with_timestamp(f"Pattern Recognition Model loaded from {model_filename}", log_level="INFO")
            return True
        except Exception as e:
            log_with_timestamp(f"Error loading model: {e}", log_level="ERROR")
            return False
    
    def create_plotly_roc_curve(self, y_true, y_pred):
        """
        Create Plotly ROC curve figure directly
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted probabilities
        
        Returns:
        - fig: Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr, 
                y=tpr,
                mode='lines',
                name=f'ROC curve (area = {roc_auc:.2f})',
                line=dict(color='darkorange', width=2)
            )
        )
        
        # Add diagonal line (random classifier)
        fig.add_trace(
            go.Scatter(
                x=[0, 1], 
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='navy', width=2, dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.05),
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            plot_bgcolor='white'
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_plotly_precision_recall_curve(self, y_true, y_pred):
        """
        Create Plotly precision-recall curve directly
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted probabilities
        
        Returns:
        - fig: Plotly figure
        """
        import plotly.graph_objects as go
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        # Create figure
        fig = go.Figure()
        
        # Add precision-recall curve
        fig.add_trace(
            go.Scatter(
                x=recall, 
                y=precision,
                mode='lines',
                name=f'Precision-Recall (AP = {avg_precision:.2f})',
                line=dict(color='green', width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            legend=dict(x=0.7, y=0.05),
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            plot_bgcolor='white'
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_plotly_confusion_matrix(self, y_true, y_pred):
        """
        Create Plotly confusion matrix directly
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted probabilities (will be thresholded at 0.5)
        
        Returns:
        - fig: Plotly figure
        """
        import plotly.graph_objects as go
        import numpy as np
        from sklearn.metrics import confusion_matrix
        
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Labels for axes
        categories = ['Negative (0)', 'Positive (1)']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=categories,
            y=categories,
            colorscale='Blues',
            showscale=True,
            text=cm.astype(str),
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        # Add annotations
        annotations = []
        for i in range(len(categories)):
            for j in range(len(categories)):
                annotations.append({
                    "x": categories[j],
                    "y": categories[i],
                    "text": str(cm[i, j]),
                    "font": {"color": "white" if cm[i, j] > np.max(cm)/2 else "black"},
                    "showarrow": False
                })
        
        # Calculate metrics
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        if cm.shape[0] == 2 and cm.shape[1] == 2:  # Binary classification
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add metrics as annotation
            fig.add_annotation(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Update layout
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            autosize=True,
            margin=dict(l=50, r=50, b=100, t=50, pad=4),
            annotations=annotations
        )
        
        return fig
    
    def create_plotly_predictions(self, df, predictions, price_col='close'):
        """
        Create Plotly predictions chart directly
        
        Parameters:
        - df: DataFrame with market data
        - predictions: DataFrame with pattern predictions
        - price_col: Column name for price (default: 'close')
        
        Returns:
        - fig: Plotly figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price Chart with Pattern Predictions', 'Pattern Probabilities'),
            row_heights=[0.7, 0.3]
        )
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode='lines',
                name=price_col.capitalize(),
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Add pattern predictions as scatter points
        pattern_indices = predictions.index[predictions['prediction'] == 1]
        if len(pattern_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pattern_indices,
                    y=df.loc[pattern_indices, price_col],
                    mode='markers',
                    name='Pattern Detected',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='red',
                        line=dict(color='red', width=1)
                    )
                ),
                row=1, col=1
            )
        
        # Add probability line
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['probability'],
                mode='lines',
                name='Pattern Probability',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=[0.5] * len(predictions),
                mode='lines',
                name='Threshold (0.5)',
                line=dict(color='red', dash='dash', width=1)
            ),
            row=2, col=1
        )
        
        # Add probabilities for detected patterns
        if len(pattern_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pattern_indices,
                    y=predictions.loc[pattern_indices, 'probability'],
                    mode='markers',
                    name='Detection Points',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='red',
                        line=dict(color='red', width=1)
                    )
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Pattern Recognition Results',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            autosize=True,
            height=800,
            margin=dict(l=50, r=50, b=50, t=70, pad=4),
            plot_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text=price_col.capitalize(), row=1, col=1)
        fig.update_yaxes(title_text='Probability', range=[0, 1], row=2, col=1)
        
        # Add statistics
        num_patterns = len(pattern_indices)
        avg_prob = predictions['probability'].mean()
        max_prob = predictions['probability'].max()
        last_prob = predictions['probability'].iloc[-1] if not predictions.empty else 0
        
        stats_text = (
            f"Patterns detected: {num_patterns}<br>"
            f"Average probability: {avg_prob:.2f}<br>"
            f"Maximum probability: {max_prob:.2f}<br>"
            f"Latest probability: {last_prob:.2f}"
        )
        
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.01,
            y=0.99,
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            align='left',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
    
    def generate_pattern_report(self, df, feature_cols, target_col=None, price_col='close', device='cpu', filename=None, symbol=None, period=None, interval=None):
        """
        Generate comprehensive pattern recognition report
        
        Parameters:
        - df: DataFrame with market data
        - feature_cols: List of feature column names
        - target_col: Column name for target (default: None, will be created based on price_col)
        - price_col: Column name for price (default: 'close')
        - device: Device to use for inference ('cpu' or 'cuda', default: 'cpu')
        - filename: Filename to save the report (default: None, displays the report)
        - symbol: Symbol to include in filenames (default: None)
        - period: Time period to include in filenames (e.g., '1mo', '3mo', '6mo', '1y', '2y', '5y', default: None)
        - interval: Time interval to include in filenames (e.g., '1d', '1h', '30m', '15m', '5m', default: None)
        
        Returns:
        - report: Dictionary with report components
        """
        from dashboard_utils import _create_empty_figure # Ensure this import is present

        # Attempt to load the model first
        model_loaded_successfully = self.load_model(path=self.model_path, symbol=symbol, period=period, interval=interval)

        if not model_loaded_successfully or self.model is None:
            log_with_timestamp(f"Pattern recognition model for {symbol} ({period}, {interval}) not found or failed to load. Skipping prediction. Please train the model first.", "WARNING")
            return {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
                'predictions': pd.DataFrame(),
                'results': pd.DataFrame(), # Ensure this is a DataFrame
                'visualizations': {
                    'roc_curve': _create_empty_figure(f"ROC Curve: Model for {symbol} not trained/loaded."),
                    'precision_recall_curve': _create_empty_figure(f"Precision-Recall: Model for {symbol} not trained/loaded."),
                    'confusion_matrix': _create_empty_figure(f"Confusion Matrix: Model for {symbol} not trained/loaded."),
                    'predictions': _create_empty_figure(f"Predictions Chart: Model for {symbol} not trained/loaded.")
                },
                'status': 'Model not trained or failed to load'
            }

        # Preprocess data - target_col might be None if we are only predicting
        # The preprocess_data method should handle target_col being None
        processed_data = self.preprocess_data(df, feature_cols, target_col, add_features=True) # Assuming add_features=True for consistency

        # If target_col is available (i.e., not just predicting, but also have labels for evaluation)
        if target_col is not None and target_col in df.columns and processed_data.get('test_loader') is not None:
            # Evaluate model
            accuracy, precision, recall, f1 = self.evaluate_model(self.model, processed_data['test_loader'], device)

            # Make predictions
            predictions = self.predict_pattern(df, feature_cols, device=device)

            # Create report components
            report = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions,
                'results': predictions  # Include raw results data for downstream processing
            }

            # Get the test data predictions and labels for visualization
            y_true = processed_data['y_test']
            y_pred = self.predict(processed_data['X_test_tensor'], device).flatten()

            report['visualizations'] = {}  # Initialize
            MIN_TEST_SAMPLES_FOR_EVAL_CHARTS = 10
            valid_for_eval_charts = False
            if hasattr(y_true, 'shape') and hasattr(y_pred, 'shape') and \
               len(y_true) >= MIN_TEST_SAMPLES_FOR_EVAL_CHARTS and len(y_pred) == len(y_true):
                if len(np.unique(y_true)) > 1:
                    valid_for_eval_charts = True
                else:
                    log_with_timestamp("Pattern Rec Report: Only one class present in y_true. ROC, PR, Confusion Matrix are not meaningful.", log_level="WARNING")
            else:
                log_with_timestamp(f"Pattern Rec Report: Insufficient or mismatched test data. y_true len: {len(y_true) if hasattr(y_true, '__len__') else 'N/A'}, y_pred len: {len(y_pred) if hasattr(y_pred, '__len__') else 'N/A'}", log_level="WARNING")

            placeholder_title_suffix = "Error: Insufficient/Invalid Test Data"
            if not valid_for_eval_charts:
                placeholder_title_suffix = "Not enough distinct test samples or classes."

            if valid_for_eval_charts:
                try:
                    report['visualizations']['roc_curve'] = self.create_plotly_roc_curve(y_true, y_pred)
                except Exception as e:
                    log_with_timestamp(f"Error creating ROC curve: {e}", log_level="ERROR")
                    report['visualizations']['roc_curve'] = _create_empty_figure("ROC Curve " + placeholder_title_suffix)
                try:
                    report['visualizations']['precision_recall_curve'] = self.create_plotly_precision_recall_curve(y_true, y_pred)
                except Exception as e:
                    log_with_timestamp(f"Error creating PR curve: {e}", log_level="ERROR")
                    report['visualizations']['precision_recall_curve'] = _create_empty_figure("PR Curve " + placeholder_title_suffix)
                try:
                    report['visualizations']['confusion_matrix'] = self.create_plotly_confusion_matrix(y_true, y_pred)
                except Exception as e:
                    log_with_timestamp(f"Error creating Confusion Matrix: {e}", log_level="ERROR")
                    report['visualizations']['confusion_matrix'] = _create_empty_figure("Confusion Matrix " + placeholder_title_suffix)
            else:
                report['visualizations']['roc_curve'] = _create_empty_figure("ROC Curve: " + placeholder_title_suffix)
                report['visualizations']['precision_recall_curve'] = _create_empty_figure("Precision-Recall Curve: " + placeholder_title_suffix)
                report['visualizations']['confusion_matrix'] = _create_empty_figure("Confusion Matrix: " + placeholder_title_suffix)

            # Predictions chart
            try:
                report['visualizations']['predictions'] = self.create_plotly_predictions(df, predictions, price_col)
            except Exception as e:
                log_with_timestamp(f"Error creating predictions chart: {e}", log_level="ERROR")
                report['visualizations']['predictions'] = _create_empty_figure(f"Predictions Chart Error: {e}")

        else:
            # For prediction only
            # Make predictions
            predictions = self.predict_pattern(df, feature_cols, device=device)

            # Create report components
            report = {
                'predictions': predictions,
                'results': predictions  # Include raw results data for downstream processing
            }

            # Generate visualizations
            report['visualizations'] = {}
            try:
                report['visualizations']['predictions'] = self.create_plotly_predictions(df, predictions, price_col)
            except Exception as e:
                log_with_timestamp(f"Error creating predictions chart: {e}", log_level="ERROR")
                report['visualizations']['predictions'] = _create_empty_figure(f"Predictions Chart Error: {e}")
            # Placeholders for unavailable eval charts
            report['visualizations']['roc_curve'] = _create_empty_figure("ROC Curve Not Available (Prediction Only Mode)", "No training data provided")
            report['visualizations']['precision_recall_curve'] = _create_empty_figure("Precision-Recall Curve Not Available (Prediction Only Mode)", "No training data provided")
            report['visualizations']['confusion_matrix'] = _create_empty_figure("Confusion Matrix Not Available (Prediction Only Mode)", "No training data provided")
        #report['visualizations'] = visualizations # This line was commented out, keeping it as is.
        
        # Save report if filename provided (but remove actual saving part)
        if filename is not None:
            log_with_timestamp(f"Report generated for Pattern Recognition {symbol}. Saving of artifacts from generate_pattern_report is disabled.", "INFO")
            # # Save raw data - This part is removed as per subtask instructions
            # report_data = {k: v for k, v in report.items() if k != 'visualizations'}
            # with open(f"{filename}_data.json", 'w') as f:
            #     # Convert pandas DataFrames to JSON
            #     import json # Already imported at top
            #     json_data = {}
            #     for k, v in report_data.items():
            #         if isinstance(v, pd.DataFrame):
            #             json_data[k] = v.to_json(date_format='iso', orient='split')
            #         else:
            #             json_data[k] = v
            #     json.dump(json_data, f)
        
        return report
