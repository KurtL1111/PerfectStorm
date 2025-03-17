"""
Machine Learning Pattern Recognition Module for Perfect Storm Dashboard

This module implements machine learning models to identify patterns
that precede significant market movements using PyTorch with ONNX Runtime.
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
import os
import pickle

class MarketPatternDataset(Dataset):
    """Dataset class for market pattern recognition"""
    
    def __init__(self, features, labels):
        """
        Initialize the dataset
        
        Parameters:
        - features: Feature tensor
        - labels: Label tensor
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        return self.features[idx], self.labels[idx]

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

class PatternRecognition:
    """Class for pattern recognition using machine learning"""
    
    def __init__(self, model_type='lstm', sequence_length=20, hidden_size=64, num_layers=2, 
                 num_filters=32, learning_rate=0.001, batch_size=32, num_epochs=50,
                 model_path='models'):
        """
        Initialize the PatternRecognition class
        
        Parameters:
        - model_type: Type of model to use ('lstm' or 'cnn', default: 'lstm')
        - sequence_length: Length of input sequence (default: 20)
        - hidden_size: Number of hidden units for LSTM (default: 64)
        - num_layers: Number of LSTM layers (default: 2)
        - num_filters: Number of convolutional filters for CNN (default: 32)
        - learning_rate: Learning rate for optimization (default: 0.001)
        - batch_size: Batch size for training (default: 32)
        - num_epochs: Number of training epochs (default: 50)
        - model_path: Path to save/load models (default: 'models')
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_path = model_path
        
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
    
    def _prepare_sequence_data(self, df, target_column, threshold=0.02, prediction_horizon=5):
        """
        Prepare sequence data for pattern recognition
        
        Parameters:
        - df: DataFrame with market data
        - target_column: Column to predict (e.g., 'close')
        - threshold: Threshold for significant movement (default: 0.02, i.e., 2%)
        - prediction_horizon: Number of days to look ahead (default: 5)
        
        Returns:
        - X: Feature sequences
        - y: Labels (1 for significant upward movement, 0 otherwise)
        """
        # Calculate future returns
        df['future_return'] = df[target_column].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Create labels based on threshold
        df['label'] = 0
        df.loc[df['future_return'] > threshold, 'label'] = 1  # Significant upward movement
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features (all columns except 'label' and 'future_return')
        feature_columns = [col for col in df.columns if col not in ['label', 'future_return']]
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(df) - self.sequence_length + 1):
            X.append(df[feature_columns].iloc[i:i+self.sequence_length].values)
            y.append(df['label'].iloc[i+self.sequence_length-1])
        
        return np.array(X), np.array(y)
    
    def _create_model(self, input_size, output_size):
        """
        Create a new model
        
        Parameters:
        - input_size: Number of input features
        - output_size: Number of output classes
        
        Returns:
        - model: PyTorch model
        """
        if self.model_type == 'lstm':
            model = LSTMModel(input_size, self.hidden_size, self.num_layers, output_size)
        elif self.model_type == 'cnn':
            model = CNNModel(input_size, self.sequence_length, self.num_filters, output_size)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def train(self, df, target_column='close', threshold=0.02, prediction_horizon=5):
        """
        Train the pattern recognition model
        
        Parameters:
        - df: DataFrame with market data
        - target_column: Column to predict (default: 'close')
        - threshold: Threshold for significant movement (default: 0.02, i.e., 2%)
        - prediction_horizon: Number of days to look ahead (default: 5)
        
        Returns:
        - history: Training history
        """
        # Prepare sequence data
        X, y = self._prepare_sequence_data(df, target_column, threshold, prediction_horizon)
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.feature_scaler.fit(X_reshaped)
        X_reshaped = self.feature_scaler.transform(X_reshaped)
        X = X_reshaped.reshape(X.shape)
        
        # Save scaler
        with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create datasets and dataloaders
        train_dataset = MarketPatternDataset(X_train, y_train)
        val_dataset = MarketPatternDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Create model
        self.input_size = X.shape[2]
        self.output_size = 1
        self.model = self._create_model(self.input_size, self.output_size)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted = (outputs >= 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save the model
        self._save_model()
        
        # Export to ONNX
        self._export_to_onnx()
        
        return history
    
    def _save_model(self):
        """Save the PyTorch model"""
        model_file = os.path.join(self.model_path, f'pattern_recognition_{self.model_type}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_filters': self.num_filters,
            'sequence_length': self.sequence_length,
            'model_type': self.model_type
        }, model_file)
        
        print(f"Model saved to {model_file}")
    
    def _export_to_onnx(self):
        """Export the PyTorch model to ONNX format"""
        # Create a dummy input
        dummy_input = torch.randn(1, self.sequence_length, self.input_size)
        
        # Export the model
        onnx_file = os.path.join(self.model_path, f'pattern_recognition_{self.model_type}.onnx')
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"Model exported to ONNX format: {onnx_file}")
    
    def load_model(self, model_file=None):
        """
        Load a trained model
        
        Parameters:
        - model_file: Path to the model file (default: None, uses default path)
        """
        if model_file is None:
            model_file = os.path.join(self.model_path, f'pattern_recognition_{self.model_type}.pth')
        
        # Load the model
        checkpoint = torch.load(model_file)
        
        # Get model parameters
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.num_filters = checkpoint['num_filters']
        self.sequence_length = checkpoint['sequence_length']
        self.model_type = checkpoint['model_type']
        
        # Create the model
        self.model = self._create_model(self.input_size, self.output_size)
        
        # Load the state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Load the feature scaler
        scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        print(f"Model loaded from {model_file}")
    
    def load_onnx_model(self, onnx_file=None):
        """
        Load an ONNX model
        
        Parameters:
        - onnx_file: Path to the ONNX file (default: None, uses default path)
        """
        if onnx_file is None:
            onnx_file = os.path.join(self.model_path, f'pattern_recognition_{self.model_type}.onnx')
        
        # Check if the ONNX file exists
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"ONNX file not found: {onnx_file}")
        
        # Load the ONNX model
        onnx_model = onnx.load(onnx_file)
        
        # Check the model
        onnx.checker.check_model(onnx_model)
        
        # Create an ONNX Runtime session
        self.onnx_session = ort.InferenceSession(onnx_file)
        
        # Load the feature scaler
        scaler_file = os.path.join(self.model_path, 'feature_scaler.pkl')
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        print(f"ONNX model loaded from {onnx_file}")
    
    def predict(self, df, use_onnx=True):
        """
        Predict patterns in market data
        
        Parameters:
        - df: DataFrame with market data
        - use_onnx: Whether to use ONNX Runtime for inference (default: True)
        
        Returns:
        - predictions: Array of pattern probabilities
        """
        # Check if model is loaded
        if self.model is None and self.onnx_session is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Prepare data
        feature_columns = [col for col in df.columns if col not in ['label', 'future_return']]
        
        # Create sequences
        X = []
        for i in range(len(df) - self.sequence_length + 1):
            X.append(df[feature_columns].iloc[i:i+self.sequence_length].values)
        
        X = np.array(X)
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_reshaped = self.feature_scaler.transform(X_reshaped)
        X = X_reshaped.reshape(X.shape)
        
        # Make predictions
        if use_onnx and self.onnx_session is not None:
            # Convert to the format expected by ONNX Runtime
            X_onnx = X.astype(np.float32)
            
            # Run inference
            predictions = []
            for i in range(len(X_onnx)):
                input_name = self.onnx_session.get_inputs()[0].name
                output_name = self.onnx_session.get_outputs()[0].name
                
                # Run inference for a single sample
                pred = self.onnx_session.run([output_name], {input_name: X_onnx[i:i+1]})[0]
                predictions.append(pred[0][0])
            
            predictions = np.array(predictions)
        else:
            # Convert to PyTorch tensor
            X_torch = torch.FloatTensor(X)
            
            # Make predictions
            with torch.no_grad():
                predictions = self.model(X_torch).numpy().flatten()
        
        return predictions
    
    def identify_patterns(self, df, threshold=0.7, use_onnx=True):
        """
        Identify patterns in market data
        
        Parameters:
        - df: DataFrame with market data
        - threshold: Probability threshold for pattern identification (default: 0.7)
        - use_onnx: Whether to use ONNX Runtime for inference (default: True)
        
        Returns:
        - df_patterns: DataFrame with pattern probabilities and signals
        """
        # Make predictions
        predictions = self.predict(df, use_onnx)
        
        # Create a copy of the DataFrame
        df_patterns = df.copy()
        
        # Add pattern probabilities
        pattern_probs = np.zeros(len(df_patterns))
        pattern_probs[-len(predictions):] = predictions
        df_patterns['pattern_probability'] = pattern_probs
        
        # Add pattern signals
        df_patterns['pattern_signal'] = 0
        df_patterns.loc[df_patterns['pattern_probability'] >= threshold, 'pattern_signal'] = 1
        
        return df_patterns
