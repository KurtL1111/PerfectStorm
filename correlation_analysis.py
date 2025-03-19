"""
Correlation Analysis Module for Perfect Storm Dashboard

This module implements correlation analysis for technical indicators:
1. Analyzes correlations between different technical indicators
2. Identifies which indicators provide unique information versus redundant signals
3. Weights signals based on their historical predictive power
4. Provides feature importance and selection capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalysis:
    """Class for correlation analysis of technical indicators"""
    
    def __init__(self, lookback_period=252, correlation_method='pearson', 
                 feature_selection_method='mutual_info', model_path='models'):
        """
        Initialize the CorrelationAnalysis class
        
        Parameters:
        - lookback_period: Period for historical analysis (default: 252, i.e., 1 year)
        - correlation_method: Method for correlation calculation 
                             ('pearson', 'spearman', 'kendall', default: 'pearson')
        - feature_selection_method: Method for feature selection
                                  ('mutual_info', 'random_forest', 'lasso', default: 'mutual_info')
        - model_path: Path to save/load models (default: 'models')
        """
        self.lookback_period = lookback_period
        self.correlation_method = correlation_method
        self.feature_selection_method = feature_selection_method
        self.model_path = model_path
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize correlation history
        self.correlation_history = {}
        
        # Initialize feature importance history
        self.feature_importance_history = {}
        
        # Initialize signal weights
        self.signal_weights = {}
        
        # Initialize redundancy groups
        self.redundancy_groups = []
        
        # Initialize unique indicators
        self.unique_indicators = []
    
    def calculate_correlation_matrix(self, df, indicators, method=None):
        """
        Calculate correlation matrix for indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - method: Correlation method (default: None, use self.correlation_method)
        
        Returns:
        - corr_matrix: Correlation matrix
        """
        if method is None:
            method = self.correlation_method
        
        # Filter DataFrame to include only indicators
        df_indicators = df[indicators].copy()
        
        # Calculate correlation matrix
        corr_matrix = df_indicators.corr(method=method)
        
        # Store in correlation history
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        
        if timestamp not in self.correlation_history:
            self.correlation_history[timestamp] = {}
        
        self.correlation_history[timestamp]['matrix'] = corr_matrix
        self.correlation_history[timestamp]['indicators'] = indicators
        
        return corr_matrix
    
    def calculate_rolling_correlations(self, df, indicators, window=None, method=None):
        """
        Calculate rolling correlations between indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - window: Rolling window size (default: None, use self.lookback_period)
        - method: Correlation method (default: None, use self.correlation_method)
        
        Returns:
        - rolling_corr: Dictionary of rolling correlation DataFrames
        """
        if window is None:
            window = self.lookback_period
        
        if method is None:
            method = self.correlation_method
        
        # Filter DataFrame to include only indicators
        df_indicators = df[indicators].copy()
        
        # Initialize dictionary for rolling correlations
        rolling_corr = {}
        
        # Calculate rolling correlations for each pair of indicators
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                ind1 = indicators[i]
                ind2 = indicators[j]
                
                # Calculate rolling correlation
                rolling_corr[(ind1, ind2)] = df_indicators[ind1].rolling(window=window).corr(df_indicators[ind2], method=method)
        
        return rolling_corr
    
    def identify_redundant_indicators(self, corr_matrix, threshold=0.8):
        """
        Identify redundant indicators based on correlation
        
        Parameters:
        - corr_matrix: Correlation matrix
        - threshold: Correlation threshold for redundancy (default: 0.8)
        
        Returns:
        - redundancy_groups: List of lists, each containing redundant indicators
        """
        # Get indicator names
        indicators = corr_matrix.columns.tolist()
        
        # Initialize redundancy groups
        redundancy_groups = []
        
        # Initialize processed indicators
        processed = set()
        
        # Find redundancy groups
        for i in range(len(indicators)):
            if indicators[i] in processed:
                continue
            
            # Initialize group with current indicator
            group = [indicators[i]]
            processed.add(indicators[i])
            
            # Find highly correlated indicators
            for j in range(len(indicators)):
                if i == j or indicators[j] in processed:
                    continue
                
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    group.append(indicators[j])
                    processed.add(indicators[j])
            
            # Add group if it contains more than one indicator
            if len(group) > 1:
                redundancy_groups.append(group)
        
        # Store redundancy groups
        self.redundancy_groups = redundancy_groups
        
        # Identify unique indicators (not in any redundancy group)
        all_redundant = set()
        for group in redundancy_groups:
            all_redundant.update(group)
        
        self.unique_indicators = [ind for ind in indicators if ind not in all_redundant]
        
        return redundancy_groups
    
    def calculate_mutual_information(self, df, indicators, target_col):
        """
        Calculate mutual information between indicators and target
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (e.g., 'returns')
        
        Returns:
        - mutual_info: Series with mutual information scores
        """
        # Filter DataFrame to include only indicators and target
        df_filtered = df[indicators + [target_col]].dropna().copy()
        
        if len(df_filtered) < 30:
            print("Not enough data to calculate mutual information")
            return pd.Series(index=indicators)
        
        # Split features and target
        X = df_filtered[indicators]
        y = df_filtered[target_col]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Create Series with scores
        mutual_info = pd.Series(mi_scores, index=indicators)
        
        return mutual_info
    
    def calculate_feature_importance(self, df, indicators, target_col, method=None, classification=False):
        """
        Calculate feature importance for indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (e.g., 'returns')
        - method: Feature selection method (default: None, use self.feature_selection_method)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - feature_importance: Series with feature importance scores
        """
        if method is None:
            method = self.feature_selection_method
        
        # Filter DataFrame to include only indicators and target
        df_filtered = df[indicators + [target_col]].dropna().copy()
        
        if len(df_filtered) < 30:
            print("Not enough data to calculate feature importance")
            return pd.Series(index=indicators)
        
        # Split features and target
        X = df_filtered[indicators]
        y = df_filtered[target_col]
        
        # For classification, convert target to binary
        if classification:
            y = (y > 0).astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate feature importance based on method
        if method == 'mutual_info':
            # Use mutual information
            if classification:
                from sklearn.feature_selection import mutual_info_classif
                importance_scores = mutual_info_classif(X_scaled, y)
            else:
                importance_scores = mutual_info_regression(X_scaled, y)
        
        elif method == 'random_forest':
            # Use random forest
            if classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_scaled, y)
            importance_scores = model.feature_importances_
        
        elif method == 'lasso':
            # Use Lasso for regression or Logistic Regression with L1 penalty for classification
            if classification:
                model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            else:
                model = Lasso(alpha=0.01, random_state=42)
            
            model.fit(X_scaled, y)
            importance_scores = np.abs(model.coef_)
            
            # Reshape if needed
            if importance_scores.ndim > 1:
                importance_scores = importance_scores[0]
        
        else:
            # Unknown method, use mutual information
            if classification:
                from sklearn.feature_selection import mutual_info_classif
                importance_scores = mutual_info_classif(X_scaled, y)
            else:
                importance_scores = mutual_info_regression(X_scaled, y)
        
        # Create Series with scores
        feature_importance = pd.Series(importance_scores, index=indicators)
        
        # Store in feature importance history
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        
        if timestamp not in self.feature_importance_history:
            self.feature_importance_history[timestamp] = {}
        
        self.feature_importance_history[timestamp]['importance'] = feature_importance
        self.feature_importance_history[timestamp]['method'] = method
        self.feature_importance_history[timestamp]['indicators'] = indicators
        
        return feature_importance
    
    def select_optimal_features(self, df, indicators, target_col, n_features=None, method=None, classification=False):
        """
        Select optimal features based on importance
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (e.g., 'returns')
        - n_features: Number of features to select (default: None, half of indicators)
        - method: Feature selection method (default: None, use self.feature_selection_method)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - selected_features: List of selected feature names
        """
        if n_features is None:
            n_features = max(1, len(indicators) // 2)
        
        if method is None:
            method = self.feature_selection_method
        
        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(df, indicators, target_col, method, classification)
        
        # Sort features by importance
        sorted_features = feature_importance.sort_values(ascending=False)
        
        # Select top features
        selected_features = sorted_features.index[:n_features].tolist()
        
        return selected_features
    
    def calculate_predictive_power(self, df, indicators, target_col, forward_periods=1, classification=False):
        """
        Calculate predictive power of indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (e.g., 'returns')
        - forward_periods: Number of periods to look forward (default: 1)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - predictive_power: Series with predictive power scores
        """
        # Filter DataFrame to include only indicators and target
        df_filtered = df[indicators + [target_col]].copy()
        
        # Create forward target
        df_filtered[f'forward_{target_col}'] = df_filtered[target_col].shift(-forward_periods)
        
        # Drop NaN values
        df_filtered = df_filtered.dropna()
        
        if len(df_filtered) < 30:
            print("Not enough data to calculate predictive power")
            return pd.Series(index=indicators)
        
        # Initialize predictive power scores
        predictive_power = {}
        
        # Calculate predictive power for each indicator
        for indicator in indicators:
            # Split data
            X = df_filtered[[indicator]].values
            y = df_filtered[f'forward_{target_col}'].values
            
            # For classification, convert target to binary
            if classification:
                y = (y > 0).astype(int)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                if classification:
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate score
                if classification:
                    score = f1_score(y_test, y_pred, average='weighted')
                else:
                    score = r2_score(y_test, y_pred)
    <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>