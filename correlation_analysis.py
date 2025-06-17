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
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add these imports for better visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import display, HTML

class CorrelationAnalysis:
    """Class for correlation analysis of technical indicators"""
    
    def __init__(self, lookback_period=252, correlation_method=['pearson', 'spearman', 'kendall'], 
                 feature_selection_method=['mutual_info', 'random_forest', 'lasso'], model_path='models\\Correlation Analysis Models',
                 data_cache_path='data_cache'):
        """
        Initialize the CorrelationAnalysis class
        
        Parameters:
        - lookback_period: Period for historical analysis (default: 252, i.e., 1 year)
        - correlation_method: Method for correlation calculation 
                             ('pearson', 'spearman', 'kendall', default: 'pearson')
                             Can be a string or a list of methods to use
        - feature_selection_method: Method for feature selection
                                  ('mutual_info', 'random_forest', 'lasso', default: 'mutual_info')
                                  Can be a string or a list of methods to use
        - model_path: Path to save/load models (default: 'models')
        - data_cache_path: Path to save/load processed data (default: 'data_cache')
        """
        self.lookback_period = lookback_period
        
        # Handle single method or list of methods
        self.correlation_method = correlation_method
        self.correlation_methods = [correlation_method] if isinstance(correlation_method, str) else correlation_method
        
        self.feature_selection_method = feature_selection_method
        self.feature_selection_methods = [feature_selection_method] if isinstance(feature_selection_method, str) else feature_selection_method
        
        self.model_path = model_path
        self.data_cache_path = data_cache_path
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Create data cache directory if it doesn't exist
        os.makedirs(data_cache_path, exist_ok=True)
        
        # Initialize correlation history (now keyed by method)
        self.correlation_history = {}
        
        # Initialize feature importance history (now keyed by methods)
        self.feature_importance_history = {}
        
        # Initialize multi-method results
        self.multi_method_results = {}
        
        # Initialize signal weights (now keyed by methods)
        self.signal_weights = {}
        
        # Initialize redundancy groups (now keyed by method)
        self.redundancy_groups = {}
        
        # Initialize unique indicators (now keyed by method)
        self.unique_indicators = {}

    def calculate_correlation_matrix(self, df, indicators=None, method=None):
        """
        Calculate correlation matrix for indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names (default: None, will auto-select all numeric columns except target)
        - method: Correlation method (default: None, use self.correlation_method)
        
        Returns:
        - corr_matrix: Correlation matrix
        """
        if method is None:
            method = self.correlation_method
        # Auto-select all numeric columns except target if indicators not provided
        if indicators is None:
            indicators = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['target', 'returns', 'label']]
        df_indicators = df[indicators].copy()
        corr_matrix = df_indicators.corr(method=method)
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        if timestamp not in self.correlation_history:
            self.correlation_history[timestamp] = {}
        self.correlation_history[timestamp]['matrix'] = corr_matrix
        self.correlation_history[timestamp]['indicators'] = indicators
        return corr_matrix
    
    def calculate_rolling_correlations(self, df, indicators=None, window=None, method=None):
        """
        Calculate rolling correlations between indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names (default: None, will auto-select all numeric columns except target)
        - window: Rolling window size (default: None, use self.lookback_period)
        - method: Correlation method (default: None, use self.correlation_method)
        
        Returns:
        - rolling_corr: Dictionary of rolling correlation DataFrames
        """
        if window is None:
            window = self.lookback_period
        if method is None:
            method = self.correlation_method
        if indicators is None:
            indicators = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['target', 'returns', 'label']]
        df_indicators = df[indicators].copy()
        rolling_corr = {}
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                ind1 = indicators[i]
                ind2 = indicators[j]
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
    
    def calculate_mutual_information(self, df, indicators=None, target_col='returns'):
        """
        Calculate mutual information between indicators and target
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names (default: None, will auto-select all numeric columns except target_col)
        - target_col: Column name of the target (e.g., 'returns')
        
        Returns:
        - mutual_info: Series with mutual information scores
        """
        if indicators is None:
            indicators = [col for col in df.select_dtypes(include=[np.number]).columns if col not in [target_col, 'label']]
        df_filtered = df[indicators + [target_col]].dropna().copy()
        if len(df_filtered) < 30:
            print("Not enough data to calculate mutual information")
            return pd.Series(index=indicators)
        X = df_filtered[indicators]
        y = df_filtered[target_col]
        mi_scores = mutual_info_regression(X, y)
        mutual_info = pd.Series(mi_scores, index=indicators)
        return mutual_info
    
    def calculate_feature_importance(self, df, indicators=None, target_col='returns', method=None, classification=False):
        """
        Calculate feature importance for indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names (default: None, will auto-select all numeric columns except target_col)
        - target_col: Column name of the target (e.g., 'returns')
        - method: Feature selection method (default: None, use self.feature_selection_method)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - feature_importance: Series with feature importance scores
        """
        if method is None:
            method = self.feature_selection_method
        if indicators is None:
            indicators = [col for col in df.select_dtypes(include=[np.number]).columns if col not in [target_col, 'label']]
        df_filtered = df[indicators + [target_col]].dropna().copy()
        if len(df_filtered) < 30:
            print("Not enough data to calculate feature importance")
            return pd.Series(index=indicators)
        X = df_filtered[indicators]
        y = df_filtered[target_col]
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            else:
                raise ValueError(f"Target column '{target_col}' is not 1D. Got shape {y.shape}.")
        if classification:
            y = (y > 0).astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if method == 'mutual_info':
            if classification:
                from sklearn.feature_selection import mutual_info_classif
                importance_scores = mutual_info_classif(X_scaled, y)
            else:
                importance_scores = mutual_info_regression(X_scaled, y)
        elif method == 'random_forest':
            if classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            importance_scores = model.feature_importances_
        elif method == 'lasso':
            if classification:
                model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            else:
                model = Lasso(alpha=0.01, random_state=42)
            model.fit(X_scaled, y)
            importance_scores = np.abs(model.coef_)
            if classification and len(importance_scores.shape) > 1:
                importance_scores = np.mean(importance_scores, axis=0)
        else:
            if classification:
                from sklearn.feature_selection import mutual_info_classif
                importance_scores = mutual_info_classif(X_scaled, y)
            else:
                importance_scores = mutual_info_regression(X_scaled, y)
        feature_importance = pd.Series(importance_scores, index=indicators)
        feature_importance = feature_importance / feature_importance.sum() if feature_importance.sum() > 0 else feature_importance
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        if timestamp not in self.feature_importance_history:
            self.feature_importance_history[timestamp] = {}
        self.feature_importance_history[timestamp]['importance'] = feature_importance
        self.feature_importance_history[timestamp]['indicators'] = indicators
        self.feature_importance_history[timestamp]['method'] = method
        return feature_importance
    
    def select_optimal_indicators(self, df, indicators=None, target_col='returns', n_select=5, method=None, classification=False):
        """
        Select optimal subset of indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names (default: None, will auto-select all numeric columns except target_col)
        - target_col: Column name of the target (e.g., 'returns')
        - n_select: Number of indicators to select (default: 5)
        - method: Feature selection method (default: None, use self.feature_selection_method)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - selected_indicators: List of selected indicator names
        """
        if method is None:
            method = self.feature_selection_method
        if indicators is None:
            indicators = [col for col in df.select_dtypes(include=[np.number]).columns if col not in [target_col, 'label']]
        df_filtered = df[indicators + [target_col]].dropna().copy()
        if len(df_filtered) < 30:
            print("Not enough data to select optimal indicators")
            return indicators[:min(n_select, len(indicators))]
        X = df_filtered[indicators]
        y = df_filtered[target_col]
        if classification:
            y = (y > 0).astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if method == 'mutual_info':
            if classification:
                from sklearn.feature_selection import mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=n_select)
            else:
                selector = SelectKBest(mutual_info_regression, k=n_select)
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
        elif method == 'random_forest':
            if classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:n_select]
            selected_mask = np.zeros(len(indicators), dtype=bool)
            selected_mask[indices] = True
        elif method == 'lasso':
            if classification:
                model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            else:
                model = Lasso(alpha=0.01, random_state=42)
            model.fit(X_scaled, y)
            coefs = model.coef_
            if classification and len(coefs.shape) > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs)
            indices = np.argsort(coefs)[::-1][:n_select]
            selected_mask = np.zeros(len(indicators), dtype=bool)
            selected_mask[indices] = True
        else:
            if classification:
                from sklearn.feature_selection import mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=n_select)
            else:
                selector = SelectKBest(mutual_info_regression, k=n_select)
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
        selected_indicators = [indicators[i] for i in range(len(indicators)) if selected_mask[i]]
        return selected_indicators
    
    def calculate_signal_weights(self, df, indicators=None, target_col='returns', method=None, classification=False, normalize=True):
        """
        Calculate weights for indicator signals based on their predictive power
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names (default: None, will auto-select all numeric columns except target_col)
        - target_col: Column name of the target (e.g., 'returns')
        - method: Feature selection method (default: None, use self.feature_selection_method)
        - classification: Whether to use classification models (default: False)
        - normalize: Whether to normalize weights to sum to 1 (default: True)
        
        Returns:
        - signal_weights: Series with signal weights
        """
        feature_importance = self.calculate_feature_importance(df, indicators, target_col, method, classification)
        signal_weights = feature_importance.copy()
        if normalize and signal_weights.sum() > 0:
            signal_weights = signal_weights / signal_weights.sum()
        self.signal_weights = signal_weights
        return signal_weights
    
    def calculate_weighted_signal(self, df, indicators, signal_cols, weights=None):
        """
        Calculate weighted signal from multiple indicators
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - signal_cols: List of signal column names (corresponding to indicators)
        - weights: Series with signal weights (default: None, use self.signal_weights)
        
        Returns:
        - weighted_signal: Series with weighted signal
        """
        if weights is None:
            weights = self.signal_weights
        
        # Ensure weights are available for all indicators
        if not all(ind in weights.index for ind in indicators):
            print("Weights not available for all indicators, using equal weights")
            weights = pd.Series(1.0 / len(indicators), index=indicators)
        
        # Initialize weighted signal
        weighted_signal = pd.Series(0, index=df.index)
        
        # Calculate weighted signal
        for i, indicator in enumerate(indicators):
            if indicator in df.columns and signal_cols[i] in df.columns:
                weighted_signal += df[signal_cols[i]] * weights[indicator]
        
        return weighted_signal
    
    def evaluate_indicator_performance(self, df, indicators, target_col, window=None, classification=False):
        """
        Evaluate performance of indicators in predicting the target
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (e.g., 'returns')
        - window: Evaluation window size (default: None, use self.lookback_period)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - performance: DataFrame with performance metrics for each indicator
        """
        if window is None:
            window = self.lookback_period
        
        # Filter DataFrame to include only indicators and target
        df_filtered = df[indicators + [target_col]].dropna().copy()
        
        if len(df_filtered) < 30:
            print("Not enough data to evaluate indicator performance")
            return pd.DataFrame(index=indicators)
        
        # Initialize performance DataFrame
        performance = pd.DataFrame(index=indicators)
        
        # For classification, convert target to binary
        if classification:
            y_true = (df_filtered[target_col] > 0).astype(int)
        else:
            y_true = df_filtered[target_col]
        
        # Evaluate each indicator
        for indicator in indicators:
            # Use indicator as predictor
            X = df_filtered[[indicator]]
            
            # Initialize metrics
            metrics = {}
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Initialize arrays for predictions
            y_pred_all = np.array([])
            y_true_all = np.array([])
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
                
                # Train model
                if classification:
                    model = LogisticRegression(random_state=42)
                else:
                    model = LinearRegression()
                
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store predictions and true values
                y_pred_all = np.append(y_pred_all, y_pred)
                y_true_all = np.append(y_true_all, y_test)
            
            # Calculate metrics
            if classification:
                metrics['accuracy'] = accuracy_score(y_true_all, y_pred_all.round())
                metrics['precision'] = precision_score(y_true_all, y_pred_all.round(), zero_division=0)
                metrics['recall'] = recall_score(y_true_all, y_pred_all.round(), zero_division=0)
                metrics['f1'] = f1_score(y_true_all, y_pred_all.round(), zero_division=0)
            else:
                metrics['r2'] = r2_score(y_true_all, y_pred_all)
                metrics['mse'] = mean_squared_error(y_true_all, y_pred_all)
                metrics['mae'] = mean_absolute_error(y_true_all, y_pred_all)
                metrics['correlation'] = np.corrcoef(y_true_all, y_pred_all)[0, 1]
            
            # Add metrics to performance DataFrame
            for metric, value in metrics.items():
                performance.loc[indicator, metric] = value
        
        return performance
    
    def plot_correlation_matrix(self, corr_matrix, figsize=(10, 8), cmap='coolwarm', annot=True):
        """
        Plot correlation matrix as heatmap
        
        Parameters:
        - corr_matrix: Correlation matrix
        - figsize: Figure size (default: (10, 8))
        - cmap: Colormap (default: 'coolwarm')
        - annot: Whether to annotate cells (default: True)
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix, 
            annot=annot, 
            fmt='.2f', 
            cmap=cmap, 
            center=0, 
            linewidths=0.5, 
            ax=ax
        )
        
        # Set title
        ax.set_title('Correlation Matrix')
        fig_plotly = self.convert_mpl_to_plotly(fig)
        plt.close(fig)
        return fig_plotly
    
    def plot_rolling_correlations(self, df, rolling_corr, pairs=None, figsize=(12, 6)):
        """
        Plot rolling correlations
        
        Parameters:
        - df: DataFrame with market data and indicators
        - rolling_corr: Dictionary of rolling correlation DataFrames
        - pairs: List of indicator pairs to plot (default: None, plot all)
        - figsize: Figure size (default: (12, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        # If pairs not specified, use all pairs
        if pairs is None:
            pairs = list(rolling_corr.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling correlations
        for pair in pairs:
            ax.plot(df.index, rolling_corr[pair], label=f'{pair[0]} vs {pair[1]}')
        
        # Add horizontal lines at 0, 0.5, and -0.5
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
        ax.set_title('Rolling Correlations')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)

        fig_plotly = self.convert_mpl_to_plotly(fig)
        plt.close(fig)
        return fig_plotly
    
    def plot_feature_importance(self, feature_importance, figsize=(10, 6), color='blue', alpha=0.7):
        """
        Plot feature importance
        
        Parameters:
        - feature_importance: Series with feature importance scores
        - figsize: Figure size (default: (10, 6))
        - color: Bar color (default: 'blue')
        - alpha: Bar transparency (default: 0.7)
        
        Returns:
        - fig: Matplotlib figure
        """
        # Sort feature importance
        feature_importance = feature_importance.sort_values(ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot feature importance
        ax.bar(feature_importance.index, feature_importance.values, color=color, alpha=alpha)
        
        # Set labels and title
        ax.set_xlabel('Indicator')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        fig_plotly = self.convert_mpl_to_plotly(fig)
        plt.close(fig)
        return fig_plotly
    
    def plot_redundancy_groups(self, redundancy_groups, figsize=(10, 6)):
        """
        Plot redundancy groups as network graph
        
        Parameters:
        - redundancy_groups: List of lists, each containing redundant indicators
        - figsize: Figure size (default: (10, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        try:
            import networkx as nx
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes and edges
            for group in redundancy_groups:
                for i in range(len(group)):
                    G.add_node(group[i])
                    for j in range(i+1, len(group)):
                        G.add_edge(group[i], group[j])
            
            # Add unique indicators as isolated nodes
            for indicator in self.unique_indicators:
                G.add_node(indicator)
            
            # Set node colors
            node_colors = []
            for node in G.nodes():
                if node in self.unique_indicators:
                    node_colors.append('green')  # Unique indicators
                else:
                    node_colors.append('red')    # Redundant indicators
            
            # Draw graph
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=500)
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Set title
            ax.set_title('Indicator Redundancy Groups')
            
            # Remove axis
            ax.axis('off')
            
            fig_plotly = self.convert_mpl_to_plotly(fig)
            plt.close(fig)
            return fig_plotly
        
        except ImportError:
            print("NetworkX not available, using alternative visualization")
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot redundancy groups
            for i, group in enumerate(redundancy_groups):
                ax.text(0.1, 0.9 - i*0.1, f"Group {i+1}: {', '.join(group)}", fontsize=12)
            
            # Plot unique indicators
            ax.text(0.1, 0.9 - len(redundancy_groups)*0.1, f"Unique: {', '.join(self.unique_indicators)}", fontsize=12, color='green')
            
            # Set title
            ax.set_title('Indicator Redundancy Groups')
            
            # Remove axis
            ax.axis('off')
            
            fig_plotly = self.convert_mpl_to_plotly(fig)
            return fig_plotly
    
    def plot_indicator_performance(self, performance, metric=None, figsize=(10, 6), color='purple', alpha=0.7):
        """
        Plot indicator performance
        
        Parameters:
        - performance: DataFrame with performance metrics for each indicator
        - metric: Metric to plot (default: None, use first available metric)
        - figsize: Figure size (default: (10, 6))
        - color: Bar color (default: 'purple')
        - alpha: Bar transparency (default: 0.7)
        
        Returns:
        - fig: Matplotlib figure
        """
        # If metric not specified, use first available
        if metric is None and not performance.empty:
            metric = performance.columns[0]
        
        # Sort performance by metric
        if metric in performance.columns:
            performance_sorted = performance.sort_values(metric, ascending=False)
        else:
            performance_sorted = performance
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot performance
        if metric in performance.columns:
            ax.bar(performance_sorted.index, performance_sorted[metric], color=color, alpha=alpha)
            
            # Set labels and title
            ax.set_xlabel('Indicator')
            ax.set_ylabel(metric)
            ax.set_title(f'Indicator Performance ({metric})')
        else:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center')
            ax.set_title('Indicator Performance')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        fig_plotly = self.convert_mpl_to_plotly(fig)
        return fig_plotly
    
    def save_correlation_analysis(self, filename=None, symbol=None, period=None, interval=None, method_combination=None):
        """
        Save correlation analysis results to file
        
        Parameters:
        - filename: Filename to save results (default: None, will create a name based on parameters)
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        - method_combination: Specific method combination to save (default: None, use self.correlation_method and self.feature_selection_method)
        
        Returns:
        - filename: Path to the saved file
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Determine which methods to use in the filename
        if method_combination:
            # Split the combination into correlation and feature selection methods
            combo_parts = method_combination.split('_')
            feat_method = combo_parts[-1]
            corr_method = '_'.join(combo_parts[:-1])
        else:
            # Use default methods from class attributes
            corr_method = self.correlation_method
            feat_method = self.feature_selection_method
        
        # Create filename if not provided
        if filename is None:
            filename = get_standardized_model_filename(
                model_type="correlation_analysis",
                model_name=f"{corr_method}_{feat_method}",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pkl"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get the results to save - either for specific method combination or default
        if method_combination and method_combination in self.multi_method_results:
            results = self.multi_method_results[method_combination]
            correlation_history = {max(self.correlation_history.keys()): {'matrix': results['correlation_matrix']}}
            redundancy_groups = results['redundancy_groups']
            unique_indicators = results['unique_indicators']
            
            # Get feature importance if available
            if 'feature_importance' in results:
                feature_importance_history = {
                    max(self.correlation_history.keys()): {
                        'importance': results['feature_importance'],
                        'method': feat_method
                    }
                }
                signal_weights = results.get('signal_weights', None)
            else:
                feature_importance_history = {}
                signal_weights = None
        else:
            # Use the default class attributes
            correlation_history = self.correlation_history
            feature_importance_history = self.feature_importance_history
            signal_weights = self.signal_weights
            redundancy_groups = self.redundancy_groups
            unique_indicators = self.unique_indicators
        
        # Create dictionary with results to save
        results_dict = {
            'correlation_history': correlation_history,
            'feature_importance_history': feature_importance_history,
            'signal_weights': signal_weights,
            'redundancy_groups': redundancy_groups,
            'unique_indicators': unique_indicators,
            'parameters': {
                'lookback_period': self.lookback_period,
                'correlation_method': corr_method,
                'feature_selection_method': feat_method,
                'model_path': self.model_path,
                'symbol': symbol,
                'period': period,
                'interval': interval
            }
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(results_dict, f)
        
        print(f"Saved correlation analysis model to {filename}")    
        return filename
    
    def load_correlation_analysis(self, filename=None, symbol=None, period=None, interval=None):
        """
        Load correlation analysis results from file
        
        Parameters:
        - filename: Filename to load results from (default: None, will create a name based on parameters)
        - symbol: Symbol to include in filename search (default: None)
        - period: Time period to include in filename search (default: None)
        - interval: Time interval to include in filename search (default: None)
        
        Returns:
        - success: Whether loading was successful
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Create filename if not provided
        if filename is None:
            # Determine current method combo for filename
            corr_method = self.correlation_methods[0] if isinstance(self.correlation_method, list) else self.correlation_method
            feat_method = self.feature_selection_methods[0] if isinstance(self.feature_selection_method, list) else self.feature_selection_method
            combo_name = f"{corr_method}_{feat_method}"
            filename = get_standardized_model_filename(
                model_type="correlation_analysis",
                model_name=combo_name, # Use determined combo name
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pkl"

        if not os.path.exists(filename):
            # print(f"Correlation model file not found: {filename}")
             return False # Return False if file doesn't exist
            
        try:
            # Load from file
            with open(filename, 'rb') as f:
                results_dict = pickle.load(f)

            # --- Store loaded parameters ---
            self.loaded_parameters = results_dict.get('parameters', {})
            print(f"Stored loaded parameters: {self.loaded_parameters}")
            # -------------------------------

            # Set attributes
            self.correlation_history = results_dict.get('correlation_history', {})
            self.feature_importance_history = results_dict.get('feature_importance_history', {})
            self.signal_weights = results_dict.get('signal_weights')
            self.redundancy_groups = results_dict.get('redundancy_groups', [])
            self.unique_indicators = results_dict.get('unique_indicators', [])

            # Set parameters from the loaded file if they exist
            parameters = self.loaded_parameters
            if parameters:
                self.lookback_period = parameters.get('lookback_period', self.lookback_period)
                 # Careful: Only update methods if they are single strings in the loaded config,
                 # otherwise keep the potentially list-based methods from current init
                loaded_corr_method = parameters.get('correlation_method')
                if isinstance(loaded_corr_method, str):
                    self.correlation_method = loaded_corr_method
                    self.correlation_methods = [loaded_corr_method]
                loaded_feat_method = parameters.get('feature_selection_method')
                if isinstance(loaded_feat_method, str):
                     self.feature_selection_method = loaded_feat_method
                     self.feature_selection_methods = [loaded_feat_method]
                # self.model_path = parameters.get('model_path', self.model_path) # Keep existing model path usually


            print(f"Successfully loaded correlation analysis from {filename}")
            return True

        except Exception as e:
            print(f"Error loading correlation analysis from {filename}: {e}")
            self.loaded_parameters = {} # Clear loaded params on error
            return False

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
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(family="Arial, sans-serif", size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Close matplotlib figure to prevent display
            plt.close(fig_mpl)
            
            return fig_plotly
        except ImportError:
            print("Plotly not available, returning Matplotlib figure")
            return fig_mpl

    def create_correlation_dashboard(self, report):
        """
        Create an interactive dashboard to display correlation analysis report
        
        Parameters:
        - report: Dictionary with report components from generate_correlation_report
        
        Returns:
        - dashboard: HTML representation of the dashboard
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            
            # Check if visualizations are available
            if 'visualizations' not in report:
                print("No visualizations available in the report")
                return None
            
            # Create a dashboard with tabs
            dashboard_html = """
            <html>
            <head>
                <title>Correlation Analysis Dashboard</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    .dashboard-container {
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }
                    .dashboard-header {
                        background-color: #2c3e50;
                        color: white;
                        padding: 15px 20px;
                        font-size: 24px;
                    }
                    .tabs {
                        display: flex;
                        background-color: #34495e;
                    }
                    .tab {
                        padding: 12px 20px;
                        cursor: pointer;
                        color: white;
                        border: none;
                        background: none;
                        font-size: 16px;
                        outline: none;
                    }
                    .tab.active {
                        background-color: #2c3e50;
                        border-bottom: 3px solid #3498db;
                    }
                    .tab-content {
                        display: none;
                        padding: 20px;
                    }
                    .tab-content.active {
                        display: block;
                    }
                    .summary-section {
                        margin-bottom: 20px;
                        padding: 15px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                    }
                    .summary-title {
                        font-size: 18px;
                        margin-bottom: 10px;
                        color: #2c3e50;
                    }
                    .summary-item {
                        margin: 5px 0;
                    }
                    .plotly-graph {
                        width: 100%;
                        height: 500px;
                        border: none;
                    }
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                    }
                    th, td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                    tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                </style>
                <script>
                    function openTab(evt, tabName) {
                        var i, tabcontent, tablinks;
                        tabcontent = document.getElementsByClassName("tab-content");
                        for (i = 0; i < tabcontent.length; i++) {
                            tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                        }
                        tablinks = document.getElementsByClassName("tab");
                        for (i = 0; i < tablinks.length; i++) {
                            tablinks[i].className = tablinks[i].className.replace(" active", "");
                        }
                        document.getElementById(tabName).className += " active";
                        evt.currentTarget.className += " active";
                    }
                </script>
            </head>
            <body>
                <div class="dashboard-container">
                    <div class="dashboard-header">Correlation Analysis Dashboard</div>
                    <div class="tabs">
            """
            
            # Add tabs for each visualization
            for i, viz_name in enumerate(report['visualizations'].keys()):
                active = " active" if i == 0 else ""
                # Format the tab name for display
                display_name = viz_name.replace('_', ' ').title()
                dashboard_html += f'<button class="tab{active}" onclick="openTab(event, \'{viz_name}\')">{display_name}</button>\n'
            
            # Add summary tab
            dashboard_html += '<button class="tab" onclick="openTab(event, \'summary\')">Summary</button>\n'
            
            dashboard_html += """
                    </div>
            """
            
            # Add content for each visualization
            for i, (viz_name, fig) in enumerate(report['visualizations'].items()):
                active = " active" if i == 0 else ""
                
                dashboard_html += f'<div id="{viz_name}" class="tab-content{active}">\n'
                
                # Convert Plotly figure to HTML
                if hasattr(fig, 'to_html'):
                    # If it's already a Plotly figure
                    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                elif isinstance(fig, plt.Figure):
                    # If it's a Matplotlib figure
                    try:
                        plotly_fig = self.convert_mpl_to_plotly(fig)
                        fig_html = plotly_fig.to_html(full_html=False, include_plotlyjs='cdn')
                    except Exception as e:
                        fig_html = f"<p>Error converting figure: {str(e)}</p>"
                else:
                    fig_html = "<p>Unsupported figure type</p>"
                
                dashboard_html += fig_html
                dashboard_html += '</div>\n'
            
            # Add summary tab content
            dashboard_html += '<div id="summary" class="tab-content">\n'
            dashboard_html += '<div class="summary-section">\n'
            dashboard_html += '<div class="summary-title">Redundancy Groups</div>\n'
            
            # Add redundancy groups
            if 'redundancy_groups' in report and report['redundancy_groups']:
                dashboard_html += '<table>\n'
                dashboard_html += '<tr><th>Group</th><th>Indicators</th></tr>\n'
                for i, group in enumerate(report['redundancy_groups']):
                    dashboard_html += f'<tr><td>Group {i+1}</td><td>{", ".join(group)}</td></tr>\n'
                dashboard_html += '</table>\n'
            else:
                dashboard_html += '<p>No redundancy groups identified</p>\n'
            
            dashboard_html += '</div>\n'
            
            # Add unique indicators
            dashboard_html += '<div class="summary-section">\n'
            dashboard_html += '<div class="summary-title">Unique Indicators</div>\n'
            if 'unique_indicators' in report and report['unique_indicators']:
                dashboard_html += '<ul>\n'
                for indicator in report['unique_indicators']:
                    dashboard_html += f'<li>{indicator}</li>\n'
                dashboard_html += '</ul>\n'
            else:
                dashboard_html += '<p>No unique indicators identified</p>\n'
            dashboard_html += '</div>\n'
            
            # Add feature importance
            if 'feature_importance' in report and report['feature_importance'] is not None:
                dashboard_html += '<div class="summary-section">\n'
                dashboard_html += '<div class="summary-title">Feature Importance</div>\n'
                dashboard_html += '<table>\n'
                dashboard_html += '<tr><th>Indicator</th><th>Importance</th></tr>\n'
                
                # Sort by importance
                sorted_importance = report['feature_importance'].sort_values(ascending=False)
                
                for indicator, importance in sorted_importance.items():
                    dashboard_html += f'<tr><td>{indicator}</td><td>{importance:.4f}</td></tr>\n'
                
                dashboard_html += '</table>\n'
                dashboard_html += '</div>\n'
            
            # Add optimal indicators
            if 'optimal_indicators' in report and report['optimal_indicators']:
                dashboard_html += '<div class="summary-section">\n'
                dashboard_html += '<div class="summary-title">Optimal Indicators</div>\n'
                dashboard_html += '<ul>\n'
                for indicator in report['optimal_indicators']:
                    dashboard_html += f'<li>{indicator}</li>\n'
                dashboard_html += '</ul>\n'
                dashboard_html += '</div>\n'
            
            dashboard_html += '</div>\n'  # Close summary tab
            
            # Close dashboard container
            dashboard_html += """
                </div>
            </body>
            </html>
            """
            
            return HTML(dashboard_html)
        
        except ImportError as e:
            print(f"Error creating dashboard. Required modules not available: {e}")
            print("Displaying raw visualizations instead...")
            
            # Display raw visualizations
            for viz_name, fig in report['visualizations'].items():
                print(f"\n--- {viz_name.replace('_', ' ').title()} ---")
                display(fig)
            
            return None
        
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            return None

    def display_correlation_report(self, report, save_html=False, filename=None):
        """
        Display correlation analysis report in an interactive dashboard
        
        Parameters:
        - report: Dictionary with report components from generate_correlation_report
        - save_html: Whether to save the dashboard as HTML (default: False)
        - filename: Filename to save the dashboard (default: None)
        
        Returns:
        - dashboard: HTML representation of the dashboard if in notebook environment
        """
        # Create dashboard
        dashboard = self.create_correlation_dashboard(report)
        
        # If in Jupyter notebook, display dashboard
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                display(dashboard)
        except:
            print("Not in a notebook environment, dashboard cannot be displayed directly")
        
        # Save dashboard as HTML if requested
        if save_html:
            if filename is None:
                filename = os.path.join(self.model_path, 'correlation_dashboard.html')
            
            if dashboard is not None:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(dashboard.data)
                print(f"Dashboard saved to {filename}")
        
        return dashboard

    def generate_correlation_report(self, df, indicators, target_col=None, filename=None, symbol=None, period=None, interval=None, display_dashboard=True, save_dashboard=False, dashboard_filename=None):
        """
        Generate comprehensive correlation analysis report
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (default: None)
        - filename: Filename to save the report (default: None, displays the report)
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        - display_dashboard: Whether to display the dashboard (default: True)
        - save_dashboard: Whether to save the dashboard as HTML (default: False)
        - dashboard_filename: Filename to save the dashboard (default: None)
        
        Returns:
        - report: Dictionary with report components
        """
        # First try to load all available models for this symbol/period/interval
        if symbol and period and interval:
            # Check if we should try to load multiple models
            if len(self.correlation_methods) > 1 or len(self.feature_selection_methods) > 1:
                loaded_models = self.load_all_available_models(symbol, period, interval)
                if loaded_models:
                    print(f"Loaded {len(loaded_models)} existing correlation models")
                    
                    # If we have multiple method results loaded, generate comparison
                    if len(loaded_models) > 1:
                        # Generate method comparison visualizations
                        feature_importance_comparison_fig = self.plot_method_comparison(
                            key_metric='feature_importance',
                            top_n=10
                        )
                        
                        redundancy_groups_comparison_fig = self.plot_method_comparison(
                            key_metric='redundancy_groups'
                        )
                        
                        # Use the default method combination for standard report
                        default_combo = f"{self.correlation_method}_{self.feature_selection_method}"
                        if default_combo in self.multi_method_results:
                            results = self.multi_method_results[default_combo]
                        else:
                            # Just use the first available combination
                            combo = list(self.multi_method_results.keys())[0]
                            results = self.multi_method_results[combo]
                            
                        # Create report with standard components for default/first method
                        report = {
                            'correlation_matrix': results['correlation_matrix'],
                            'redundancy_groups': results['redundancy_groups'],
                            'unique_indicators': results['unique_indicators'],
                            'symbol': symbol,
                            'period': period,
                            'interval': interval
                        }
                        
                        # Add feature importance if available
                        if 'feature_importance' in results:
                            report['feature_importance'] = results['feature_importance']
                        
                        # Add signal weights if available
                        if 'signal_weights' in results:
                            report['signal_weights'] = results['signal_weights']
                        
                        # Add optimal indicators if available
                        if 'optimal_indicators' in results:
                            report['optimal_indicators'] = results['optimal_indicators']
                        
                        # Create visualizations
                        figs = {}
                        figs['correlation_matrix'] = self.plot_correlation_matrix(results['correlation_matrix'])
                        figs['redundancy_groups'] = self.plot_redundancy_groups(results['redundancy_groups'])
                        
                        if 'feature_importance' in results:
                            figs['feature_importance'] = self.plot_feature_importance(results['feature_importance'])
                        
                        # Add method comparison visualizations
                        figs['feature_importance_comparison'] = feature_importance_comparison_fig
                        figs['redundancy_groups_comparison'] = redundancy_groups_comparison_fig
                        
                        # Add visualizations to report
                        report['visualizations'] = figs
                        
                        # Display dashboard if requested
                        if display_dashboard:
                            self.display_correlation_report(report, save_dashboard, dashboard_filename)
                        
                        return report

        # If we get here, we need to generate new analysis
        # Check if we have cached processed data
        cached_data = self._load_processed_data(symbol, period, interval)
        
        if cached_data is not None:
            print(f"Using cached processed data for {symbol} ({period}, {interval})")
            report = cached_data
        else:
            # No cached data found, generate new correlation analysis
            print("No cached data found, generating new correlation analysis...")
            
            # First, check if we should run multi-method analysis
            if len(self.correlation_methods) > 1 or len(self.feature_selection_methods) > 1:
                print(f"Running multi-method analysis with {len(self.correlation_methods)} correlation methods and {len(self.feature_selection_methods)} feature selection methods")
                
                # Run the multi-method analysis
                self.run_multi_method_analysis(
                    df, 
                    indicators, 
                    target_col=target_col,
                    threshold=0.8,
                    n_select=5,
                    classification=False
                )
                
                # Generate method comparison visualizations
                feature_importance_comparison_fig = self.plot_method_comparison(
                    key_metric='feature_importance',
                    top_n=10
                )
                
                redundancy_groups_comparison_fig = self.plot_method_comparison(
                    key_metric='redundancy_groups'
                )
                
                # Save all method combinations
                saved_files = self.save_all_method_combinations(symbol, period, interval)
                print(f"Saved {len(saved_files)} method combinations")
                
                # Use the default method combination for standard report
                default_combo = f"{self.correlation_method}_{self.feature_selection_method}"
                if default_combo in self.multi_method_results:
                    results = self.multi_method_results[default_combo]
                else:
                    # Just use the first available combination
                    combo = list(self.multi_method_results.keys())[0]
                    results = self.multi_method_results[combo]
                
                # Create report with standard components for default/first method
                report = {
                    'correlation_matrix': results['correlation_matrix'],
                    'redundancy_groups': results['redundancy_groups'],
                    'unique_indicators': results['unique_indicators'],
                    'symbol': symbol,
                    'period': period,
                    'interval': interval
                }
                
                # Add feature importance if available
                if 'feature_importance' in results:
                    report['feature_importance'] = results['feature_importance']
                
                # Add signal weights if available
                if 'signal_weights' in results:
                    report['signal_weights'] = results['signal_weights']
                
                # Add optimal indicators if available
                if 'optimal_indicators' in results:
                    report['optimal_indicators'] = results['optimal_indicators']
                
                # Create visualizations
                figs = {}
                figs['correlation_matrix'] = self.plot_correlation_matrix(results['correlation_matrix'])
                figs['redundancy_groups'] = self.plot_redundancy_groups(results['redundancy_groups'])
                
                if 'feature_importance' in results:
                    figs['feature_importance'] = self.plot_feature_importance(results['feature_importance'])
                
                # Add method comparison visualizations
                figs['feature_importance_comparison'] = feature_importance_comparison_fig
                figs['redundancy_groups_comparison'] = redundancy_groups_comparison_fig
                
                # Add visualizations to report
                report['visualizations'] = figs
            else:
                # Get report from existing method for single-method analysis
                report = self._generate_correlation_report(df, indicators, target_col, filename, symbol, period, interval)
                
            # Cache the processed data for future use
            self._save_processed_data(report, symbol, period, interval)
        
        # Display dashboard if requested
        if display_dashboard:
            self.display_correlation_report(report, save_dashboard, dashboard_filename)
        
        return report

    def _generate_correlation_report(self, df, indicators, target_col=None, filename=None, symbol=None, period=None, interval=None):
        """
        Internal method to generate the correlation report without dashboard display
        This contains the original generate_correlation_report functionality
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (default: None)
        - filename: Filename to save the report (default: None, displays the report)
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - report: Dictionary with report components
        """
        # Check if model already exists
        model_exists, model_filename = self.check_existing_model(symbol, period, interval)
        
        if model_exists:
            print(f"Found existing correlation model at {model_filename}, loading...")
            if self.load_correlation_analysis(model_filename):
                # Get the latest correlation matrix from history
                if self.correlation_history:
                    latest_timestamp = max(self.correlation_history.keys())
                    corr_matrix = self.correlation_history[latest_timestamp]['matrix']
                    
                    # Create report components from loaded data
                    report = {
                        'correlation_matrix': corr_matrix,
                        'redundancy_groups': self.redundancy_groups,
                        'unique_indicators': self.unique_indicators,
                        'symbol': symbol,
                        'period': period,
                        'interval': interval
                    }
                    
                    # Add feature importance if available
                    if self.feature_importance_history and target_col is not None:
                        latest_fi_timestamp = max(self.feature_importance_history.keys())
                        feature_importance = self.feature_importance_history[latest_fi_timestamp]['importance']
                        report['feature_importance'] = feature_importance
                        report['signal_weights'] = self.signal_weights
                        
                        # Generate visualizations
                        figs = {}
                        figs['correlation_matrix'] = self.plot_correlation_matrix(corr_matrix)
                        figs['redundancy_groups'] = self.plot_redundancy_groups(self.redundancy_groups)
                        if feature_importance is not None:
                            figs['feature_importance'] = self.plot_feature_importance(feature_importance)
                        
                        # Add visualizations to report
                        report['visualizations'] = figs
                        
                        return report
        
        # If we get here, either no existing model was found or loading failed
        print("Generating new correlation analysis...")
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(df, indicators)
        
        # Identify redundant indicators
        redundancy_groups = self.identify_redundant_indicators(corr_matrix)
        
        # Calculate feature importance if target column provided
        feature_importance = None
        signal_weights = None
        optimal_indicators = None
        
        if target_col is not None:
            feature_importance = self.calculate_feature_importance(df, indicators, target_col)
            
            # Calculate signal weights
            signal_weights = self.calculate_signal_weights(df, indicators, target_col)
            
            # Select optimal indicators
            optimal_indicators = self.select_optimal_indicators(df, indicators, target_col)
        print("feature_importance",feature_importance)
        # Create report components
        report = {
            'correlation_matrix': corr_matrix,
            'redundancy_groups': redundancy_groups,
            'unique_indicators': self.unique_indicators,
            'symbol': symbol,
            'period': period,
            'interval': interval
        }
        
        if target_col is not None:
            report['feature_importance'] = feature_importance
            report['signal_weights'] = signal_weights
            report['optimal_indicators'] = optimal_indicators
        
        # Generate visualizations
        figs = {}
        
        # Correlation matrix heatmap
        figs['correlation_matrix'] = self.plot_correlation_matrix(corr_matrix)
        
        # Redundancy groups
        figs['redundancy_groups'] = self.plot_redundancy_groups(redundancy_groups)
        
        # Feature importance if available
        if feature_importance is not None:
            figs['feature_importance'] = self.plot_feature_importance(feature_importance)
        
        # Add visualizations to report
        report['visualizations'] = figs
        
        # Save report if filename provided or generate default filename
        self.save_correlation_analysis(filename, symbol, period, interval)
        
        return report

    def check_existing_model(self, symbol=None, period=None, interval=None):
        """
        Check if a correlation analysis model already exists for the given parameters
        
        Parameters:
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - exists: Boolean indicating if the model exists
        - filename: Path to the existing file if it exists, None otherwise
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Create the expected filename using the same function used by save_correlation_analysis
        filename = get_standardized_model_filename(
            model_type="correlation_analysis",
            model_name=f"{self.correlation_method}_{self.feature_selection_method}",
            symbol=symbol,
            period=period,
            interval=interval,
            base_path=self.model_path
        ) + ".pkl"
        
        # Check if the file exists
        if os.path.exists(filename):
            return True, filename
        
        return False, None

    def _save_processed_data(self, data_dict, symbol=None, period=None, interval=None):
        """
        Save processed data to cache
        
        Parameters:
        - data_dict: Dictionary with processed data components
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - filename: Path to the saved cache file
        """
        # Create cache filename
        if symbol is None:
            symbol = "generic"
        
        cache_filename = os.path.join(
            self.data_cache_path, 
            f"correlation_data_{symbol}_{period}_{interval}.pkl"
        )
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            
            # Save to file
            with open(cache_filename, 'wb') as f:
                pickle.dump(data_dict, f)
            
            print(f"Cached correlation processed data to {cache_filename}")
            return cache_filename
        except Exception as e:
            print(f"Error caching correlation data: {e}")
            return None
            
    def _load_processed_data(self, symbol=None, period=None, interval=None):
        """
        Load processed data from cache
        
        Parameters:
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - data_dict: Dictionary with processed data components or None if not found
        """
        # Create cache filename
        if symbol is None:
            symbol = "generic"
        
        cache_filename = os.path.join(
            self.data_cache_path, 
            f"correlation_data_{symbol}_{period}_{interval}.pkl"
        )
        
        # Check if cache file exists
        if not os.path.exists(cache_filename):
            print(f"No cached data found at {cache_filename}")
            return None
        
        try:
            # Load from file
            with open(cache_filename, 'rb') as f:
                data_dict = pickle.load(f)
            
            print(f"Loaded cached correlation data from {cache_filename}")
            return data_dict
        except Exception as e:
            print(f"Error loading cached correlation data: {e}")
            return None

    def run_multi_method_analysis(self, df, indicators, target_col=None, 
                         correlation_methods=None, feature_selection_methods=None,
                         threshold=0.8, n_select=5, classification=False):
        """
        Run correlation analysis with multiple combinations of correlation and feature selection methods
        
        Parameters:
        - df: DataFrame with market data and indicators
        - indicators: List of indicator column names
        - target_col: Column name of the target (default: None)
        - correlation_methods: List of correlation methods to use (default: None, use self.correlation_methods)
        - feature_selection_methods: List of feature selection methods to use (default: None, use self.feature_selection_methods)
        - threshold: Correlation threshold for redundancy (default: 0.8)
        - n_select: Number of indicators to select (default: 5)
        - classification: Whether to use classification models (default: False)
        
        Returns:
        - results: Dictionary with results for each method combination
        """
        # Use class attributes if parameters not provided
        if correlation_methods is None:
            correlation_methods = self.correlation_methods
        
        if feature_selection_methods is None:
            feature_selection_methods = self.feature_selection_methods
        
        # Initialize results dictionary
        results = {}
        
        # Iterate over all combinations of methods
        for corr_method in correlation_methods:
            for feat_method in feature_selection_methods:
                # Create key for this combination
                combo_key = f"{corr_method}_{feat_method}"
                print(f"Running analysis with correlation method: {corr_method}, feature selection method: {feat_method}")
                
                # Initialize results for this combination
                results[combo_key] = {}
                
                # Calculate correlation matrix
                corr_matrix = self.calculate_correlation_matrix(df, indicators, method=corr_method)
                results[combo_key]['correlation_matrix'] = corr_matrix
                
                # Identify redundant indicators
                redundancy_groups = self.identify_redundant_indicators(corr_matrix, threshold=threshold)
                results[combo_key]['redundancy_groups'] = redundancy_groups
                results[combo_key]['unique_indicators'] = self.unique_indicators
                
                # Calculate feature importance if target column provided
                if target_col is not None:
                    # Calculate feature importance
                    feature_importance = self.calculate_feature_importance(
                        df, indicators, target_col, method=feat_method, classification=classification
                    )
                    results[combo_key]['feature_importance'] = feature_importance
                    
                    # Calculate signal weights
                    signal_weights = self.calculate_signal_weights(
                        df, indicators, target_col, method=feat_method, classification=classification
                    )
                    results[combo_key]['signal_weights'] = signal_weights
                    
                    # Select optimal indicators
                    optimal_indicators = self.select_optimal_indicators(
                        df, indicators, target_col, n_select=n_select, method=feat_method, classification=classification
                    )
                    results[combo_key]['optimal_indicators'] = optimal_indicators
        
        # Store all results in class attribute
        self.multi_method_results = results
        
        return results
    
    def compare_method_combinations(self, key_metric='unique_indicators', sort_by=None):
        """
        Compare results across different method combinations
        
        Parameters:
        - key_metric: The metric to compare (default: 'unique_indicators')
                     Options: 'unique_indicators', 'redundancy_groups', 'feature_importance', 'optimal_indicators'
        - sort_by: Column to sort by (for feature importance comparison)
        
        Returns:
        - comparison: DataFrame with comparison results
        """
        if not self.multi_method_results:
            print("No multi-method results available. Run run_multi_method_analysis first.")
            return None
        
        # Initialize comparison DataFrame
        if key_metric == 'unique_indicators':
            # Compare number of unique indicators
            data = []
            for combo, results in self.multi_method_results.items():
                if 'unique_indicators' in results:
                    # Safely extract methods from the combo key, handling cases with underscores in method names
                    combo_parts = combo.split('_')
                    # Last part is always the feature selection method
                    feat_method = combo_parts[-1]
                    # All other parts (joined back with underscore) form the correlation method
                    corr_method = '_'.join(combo_parts[:-1])
                    
                    data.append({
                        'Correlation Method': corr_method,
                        'Feature Selection Method': feat_method,
                        'Number of Unique Indicators': len(results['unique_indicators']),
                        'Unique Indicators': ', '.join(results['unique_indicators'])
                    })
            
            comparison = pd.DataFrame(data)
            
            # Sort by number of unique indicators
            if comparison.empty:
                return comparison
            comparison = comparison.sort_values('Number of Unique Indicators', ascending=False)
        
        elif key_metric == 'redundancy_groups':
            # Compare redundancy groups
            data = []
            for combo, results in self.multi_method_results.items():
                if 'redundancy_groups' in results:
                    # Safely extract methods from the combo key, handling cases with underscores in method names
                    combo_parts = combo.split('_')
                    # Last part is always the feature selection method
                    feat_method = combo_parts[-1]
                    # All other parts (joined back with underscore) form the correlation method
                    corr_method = '_'.join(combo_parts[:-1])
                    
                    data.append({
                        'Correlation Method': corr_method,
                        'Feature Selection Method': feat_method,
                        'Number of Redundancy Groups': len(results['redundancy_groups']),
                        'Total Redundant Indicators': sum(len(group) for group in results['redundancy_groups'])
                    })
            
            comparison = pd.DataFrame(data)
            
            # Sort by number of redundancy groups
            if comparison.empty:
                return comparison
            comparison = comparison.sort_values('Number of Redundancy Groups', ascending=False)
        
        elif key_metric == 'feature_importance':
            # Compare feature importance
            all_indicators = set()
            
            # Collect all indicators across all method combinations
            for results in self.multi_method_results.values():
                if 'feature_importance' in results:
                    all_indicators.update(results['feature_importance'].index)
            
            # Create DataFrame with feature importance from each method
            data = {}
            for combo, results in self.multi_method_results.items():
                if 'feature_importance' in results:
                    data[combo] = pd.Series(0, index=all_indicators)
                    for ind in results['feature_importance'].index:
                        data[combo][ind] = results['feature_importance'][ind]
            
            comparison = pd.DataFrame(data)
            
            # Sort by a specific column if requested
            if sort_by is not None and sort_by in comparison.columns:
                comparison = comparison.sort_values(sort_by, ascending=False)
        
        elif key_metric == 'optimal_indicators':
            # Compare optimal indicators
            data = []
            for combo, results in self.multi_method_results.items():
                if 'optimal_indicators' in results:
                    # Safely extract methods from the combo key, handling cases with underscores in method names
                    combo_parts = combo.split('_')
                    # Last part is always the feature selection method
                    feat_method = combo_parts[-1]
                    # All other parts (joined back with underscore) form the correlation method
                    corr_method = '_'.join(combo_parts[:-1])
                    
                    data.append({
                        'Correlation Method': corr_method,
                        'Feature Selection Method': feat_method,
                        'Number of Optimal Indicators': len(results['optimal_indicators']),
                        'Optimal Indicators': ', '.join(results['optimal_indicators'])
                    })
            
            comparison = pd.DataFrame(data)
            
            # Sort by number of optimal indicators
            if comparison.empty:
                return comparison
            comparison = comparison.sort_values('Number of Optimal Indicators', ascending=False)
        
        else:
            print(f"Invalid key_metric: {key_metric}")
            print("Valid options are: 'unique_indicators', 'redundancy_groups', 'feature_importance', 'optimal_indicators'")
            return None
        
        return comparison

    def plot_method_comparison(self, key_metric='feature_importance', top_n=10, figsize=(12, 8)):
        """
        Plot comparison of results across different method combinations
        
        Parameters:
        - key_metric: The metric to compare (default: 'feature_importance')
                     Options: 'unique_indicators', 'redundancy_groups', 'feature_importance', 'optimal_indicators'
        - top_n: Number of top indicators to include (for feature importance comparison)
        - figsize: Figure size (default: (12, 8))
        
        Returns:
        - fig: Plotly figure
        """
        if not self.multi_method_results:
            print("No multi-method results available. Run run_multi_method_analysis first.")
            return None
        
        # Get comparison data
        comparison = self.compare_method_combinations(key_metric=key_metric)
        
        if comparison is None or comparison.empty:
            print(f"No comparison data available for {key_metric}")
            return None
        
        # Create appropriate visualization based on metric
        if key_metric == 'unique_indicators' or key_metric == 'redundancy_groups':
            # Bar chart for counting metrics
            try:
                import plotly.express as px
                
                # Determine which column to plot
                if key_metric == 'unique_indicators':
                    y_col = 'Number of Unique Indicators'
                    title = 'Number of Unique Indicators by Method Combination'
                else:
                    y_col = 'Number of Redundancy Groups'
                    title = 'Number of Redundancy Groups by Method Combination'
                
                # Create labels for x-axis combining both methods
                comparison['Method Combination'] = comparison['Correlation Method'] + ' + ' + comparison['Feature Selection Method']
                
                # Create bar chart
                fig = px.bar(
                    comparison,
                    x='Method Combination',
                    y=y_col,
                    color='Correlation Method',
                    barmode='group',
                    title=title,
                    labels={y_col: y_col, 'Method Combination': 'Method Combination'},
                    height=figsize[1] * 70,
                    width=figsize[0] * 70
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Method Combination',
                    yaxis_title=y_col,
                    legend_title='Correlation Method',
                    font=dict(family="Arial, sans-serif", size=12)
                )
                
                return fig
            
            except ImportError:
                print("Plotly not available, using Matplotlib instead")
                
                # Fallback to Matplotlib
                fig, ax = plt.subplots(figsize=figsize)
                
                # Determine which column to plot
                if key_metric == 'unique_indicators':
                    y_col = 'Number of Unique Indicators'
                    title = 'Number of Unique Indicators by Method Combination'
                else:
                    y_col = 'Number of Redundancy Groups'
                    title = 'Number of Redundancy Groups by Method Combination'
                
                # Create labels for x-axis combining both methods
                comparison['Method Combination'] = comparison['Correlation Method'] + ' + ' + comparison['Feature Selection Method']
                
                # Create bar chart
                comparison.plot.bar(x='Method Combination', y=y_col, ax=ax, legend=False)
                
                # Set labels and title
                ax.set_xlabel('Method Combination')
                ax.set_ylabel(y_col)
                ax.set_title(title)
                
                # Rotate x-axis labels
                plt.xticks(rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, axis='y', alpha=0.3)
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
        
        elif key_metric == 'feature_importance':
            # Heatmap for feature importance comparison
            try:
                import plotly.express as px
                
                # Get top N indicators by average importance
                avg_importance = comparison.mean(axis=1).sort_values(ascending=False)
                top_indicators = avg_importance.head(top_n).index.tolist()
                
                # Filter DataFrame to include only top indicators
                df_top = comparison.loc[top_indicators]
                
                # Melt DataFrame for heatmap
                df_melted = df_top.reset_index().melt(
                    id_vars='index',
                    var_name='Method Combination',
                    value_name='Importance'
                )
                df_melted.rename(columns={'index': 'Indicator'}, inplace=True)
                
                # Create heatmap
                fig = px.density_heatmap(
                    df_melted,
                    x='Method Combination',
                    y='Indicator',
                    z='Importance',
                    title=f'Feature Importance Comparison (Top {top_n} Indicators)',
                    height=figsize[1] * 70,
                    width=figsize[0] * 70
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Method Combination',
                    yaxis_title='Indicator',
                    font=dict(family="Arial, sans-serif", size=12)
                )
                
                return fig
            
            except ImportError:
                print("Plotly not available, using Matplotlib instead")
                
                # Fallback to Matplotlib
                fig, ax = plt.subplots(figsize=figsize)
                
                # Get top N indicators by average importance
                avg_importance = comparison.mean(axis=1).sort_values(ascending=False)
                top_indicators = avg_importance.head(top_n).index.tolist()
                
                # Filter DataFrame to include only top indicators
                df_top = comparison.loc[top_indicators]
                
                # Create heatmap
                sns.heatmap(
                    df_top,
                    cmap='viridis',
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5,
                    ax=ax
                )
                
                # Set title
                ax.set_title(f'Feature Importance Comparison (Top {top_n} Indicators)')
                
                # Rotate x-axis labels
                plt.xticks(rotation=45, ha='right')
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
        
        elif key_metric == 'optimal_indicators':
            # Visualization for optimal indicators comparison
            try:
                import plotly.express as px
                
                # Create bar chart for number of optimal indicators
                fig = px.bar(
                    comparison,
                    x='Feature Selection Method',
                    y='Number of Optimal Indicators',
                    color='Correlation Method',
                    barmode='group',
                    title='Number of Optimal Indicators by Method Combination',
                    labels={'Number of Optimal Indicators': 'Number of Optimal Indicators'},
                    height=figsize[1] * 70,
                    width=figsize[0] * 70
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Feature Selection Method',
                    yaxis_title='Number of Optimal Indicators',
                    legend_title='Correlation Method',
                    font=dict(family="Arial, sans-serif", size=12)
                )
                
                return fig
            
            except ImportError:
                print("Plotly not available, using Matplotlib instead")
                
                # Fallback to Matplotlib
                fig, ax = plt.subplots(figsize=figsize)
                
                # Pivot data for grouped bar chart
                pivot_data = comparison.pivot(
                    index='Feature Selection Method',
                    columns='Correlation Method',
                    values='Number of Optimal Indicators'
                )
                
                # Create grouped bar chart
                pivot_data.plot.bar(ax=ax)
                
                # Set labels and title
                ax.set_xlabel('Feature Selection Method')
                ax.set_ylabel('Number of Optimal Indicators')
                ax.set_title('Number of Optimal Indicators by Method Combination')
                
                # Add grid
                ax.grid(True, axis='y', alpha=0.3)
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
        
        else:
            print(f"Invalid key_metric: {key_metric}")
            print("Valid options are: 'unique_indicators', 'redundancy_groups', 'feature_importance', 'optimal_indicators'")
            return None

    def load_multi_method_model(self, filename=None, symbol=None, period=None, interval=None, method_combination=None):
        """
        Load a specific correlation model for a given method combination
        
        Parameters:
        - filename: Filename to load results from (default: None, will create a name based on parameters)
        - symbol: Symbol to include in filename search (default: None)
        - period: Time period to include in filename search (default: None)
        - interval: Time interval to include in filename search (default: None)
        - method_combination: Specific method combination to load (default: None, use self.correlation_method and self.feature_selection_method)
        
        Returns:
        - success: Whether loading was successful
        - results_dict: The loaded model data if successful, None otherwise
        """
        from dashboard_utils import get_standardized_model_filename
        
        # Determine which methods to use in the filename
        if method_combination:
            # Split the combination into correlation and feature selection methods
            combo_parts = method_combination.split('_')
            feat_method = combo_parts[-1]
            corr_method = '_'.join(combo_parts[:-1])
        else:
            # Use default methods from class attributes
            corr_method = self.correlation_method
            feat_method = self.feature_selection_method
        
        # Create filename if not provided
        if filename is None:
            filename = get_standardized_model_filename(
                model_type="correlation_analysis",
                model_name=f"{corr_method}_{feat_method}",
                symbol=symbol,
                period=period,
                interval=interval,
                base_path=self.model_path
            ) + ".pkl"
            
        try:
            # Load from file
            with open(filename, 'rb') as f:
                results_dict = pickle.load(f)
            
            # Store results in multi_method_results if it's a different combination than default
            if method_combination and method_combination != f"{self.correlation_method}_{self.feature_selection_method}":
                # Create a multi-method result format for this combination
                self.multi_method_results[method_combination] = {
                    'correlation_matrix': results_dict['correlation_history'][max(results_dict['correlation_history'].keys())]['matrix'],
                    'redundancy_groups': results_dict['redundancy_groups'],
                    'unique_indicators': results_dict['unique_indicators']
                }
                
                # Add feature importance if available
                if 'feature_importance_history' in results_dict and results_dict['feature_importance_history']:
                    latest_timestamp = max(results_dict['feature_importance_history'].keys())
                    self.multi_method_results[method_combination]['feature_importance'] = results_dict['feature_importance_history'][latest_timestamp]['importance']
                    self.multi_method_results[method_combination]['signal_weights'] = results_dict.get('signal_weights', None)
            else:
                # Set default class attributes
                self.correlation_history = results_dict['correlation_history']
                self.feature_importance_history = results_dict['feature_importance_history']
                self.signal_weights = results_dict['signal_weights']
                self.redundancy_groups = results_dict['redundancy_groups']
                self.unique_indicators = results_dict['unique_indicators']
            
            print(f"Successfully loaded correlation analysis for {method_combination} from {filename}")
            return True, results_dict
        
        except Exception as e:
            print(f"Error loading correlation analysis for {method_combination} from {filename}: {e}")
            return False, None

    def load_all_available_models(self, symbol=None, period=None, interval=None, models_dir=None):
        """
        Load all available correlation models for different method combinations
        
        Parameters:
        - symbol: Symbol to include in filename search (default: None)
        - period: Time period to include in filename search (default: None)
        - interval: Time interval to include in filename search (default: None)
        - models_dir: Directory to search for models (default: self.model_path)
        
        Returns:
        - loaded_models: Dictionary with loaded model data for each method combination
        """
        import os
        import pickle
        if models_dir is None:
            models_dir = self.model_path
        loaded_models = {}
        # Print debug info
        print(f"Searching for models in: {models_dir}")
        print(f"Looking for symbol: {symbol}, period: {period}, interval: {interval}")
        files = os.listdir(models_dir)
        print("All files in models directory:", files)
        for fname in files:
            # Accept files ending with .pkl or .pkl.pkl
            if not (fname.endswith(".pkl") or fname.endswith(".pkl.pkl")):
                continue
            # Print file name and filter check values
            if symbol in fname and period in fname and interval in fname:
                print(f"File passed filter: {fname}")
                try:
                    method_combo = fname.split("correlation_analysis_")[1].split(f"_{symbol}")[0]
                except Exception:
                    method_combo = fname
                full_path = os.path.join(models_dir, fname)
                try:
                    with open(full_path, "rb") as f:
                        report = pickle.load(f)
                    # Regenerate visualizations if missing or empty
                    if not report.get('visualizations') or not isinstance(report['visualizations'], dict) or not report['visualizations']:
                        figs = {}
                        if report.get('correlation_matrix') is not None:
                            figs['correlation_matrix'] = self.plot_correlation_matrix(report['correlation_matrix'])
                        if report.get('redundancy_groups') is not None:
                            figs['redundancy_groups'] = self.plot_redundancy_groups(report['redundancy_groups'])
                        if report.get('feature_importance') is not None:
                            figs['feature_importance'] = self.plot_feature_importance(report['feature_importance'])
                        report['visualizations'] = figs
                    loaded_models[method_combo] = report
                    print(f"Successfully loaded correlation analysis for {method_combo} from {full_path}")
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
        print(f"Successfully loaded {len(loaded_models)} correlation models")
        return loaded_models

    def save_all_method_combinations(self, symbol=None, period=None, interval=None):
        """
        Save all method combinations in multi_method_results
        
        Parameters:
        - symbol: Symbol to include in filename (default: None)
        - period: Time period to include in filename (default: None)
        - interval: Time interval to include in filename (default: None)
        
        Returns:
        - saved_files: List of saved filenames
        """
        if not self.multi_method_results:
            print("No multi-method results available. Run run_multi_method_analysis first.")
            return []
        
        # Initialize list to store saved filenames
        saved_files = []
        
        # Iterate through all method combinations in multi_method_results
        for method_combination in self.multi_method_results.keys():
            # Save this method combination
            filename = self.save_correlation_analysis(
                symbol=symbol,
                period=period,
                interval=interval,
                method_combination=method_combination
            )
            
            saved_files.append(filename)
        
        print(f"Saved {len(saved_files)} method combinations")
        return saved_files
    
    def get_latest_report(self):
        """
        Retrieves the data components for the most recently calculated or loaded correlation report.
        Does NOT recalculate or regenerate visualizations.

        Returns:
            dict: A dictionary containing the latest available report components,
                  similar to the structure generated by generate_correlation_report,
                  but without regenerating visualizations. Returns None for components
                  that are not available.
        """
        report = {
            'correlation_matrix': None,
            'redundancy_groups': [], # Default to empty list
            'unique_indicators': [],  # Default to empty list
            'feature_importance': None,
            'signal_weights': None,
            'optimal_indicators': None,
            # Metadata (attempt to retrieve if model was loaded)
            'symbol': None,
            'period': None,
            'interval': None,
            # Placeholder for visualizations - these should be regenerated by the caller
            'visualizations': {}
        }

        # Determine the key for the currently configured method combination
        current_combo_key = None
        if hasattr(self, 'correlation_method') and hasattr(self, 'feature_selection_method'):
             # Handle cases where methods might be lists (take the first element for default report)
             corr_method = self.correlation_methods[0] if isinstance(self.correlation_method, list) else self.correlation_method
             feat_method = self.feature_selection_methods[0] if isinstance(self.feature_selection_method, list) else self.feature_selection_method
             current_combo_key = f"{corr_method}_{feat_method}"

        # 1. Prioritize retrieving from multi_method_results if it exists and has the current key
        retrieved_from_multi = False
        if hasattr(self, 'multi_method_results') and self.multi_method_results and current_combo_key in self.multi_method_results:
            results = self.multi_method_results[current_combo_key]
            print(f"Retrieving latest report data from multi_method_results for key: {current_combo_key}")
            report['correlation_matrix'] = results.get('correlation_matrix')
            report['redundancy_groups'] = results.get('redundancy_groups', [])
            report['unique_indicators'] = results.get('unique_indicators', [])
            report['feature_importance'] = results.get('feature_importance')
            report['signal_weights'] = results.get('signal_weights')
            report['optimal_indicators'] = results.get('optimal_indicators')
             # Try to get metadata stored with multi-method results if structure allows
            report['symbol'] = results.get('symbol', report['symbol'])
            report['period'] = results.get('period', report['period'])
            report['interval'] = results.get('interval', report['interval'])
            retrieved_from_multi = True


        # 2. If not retrieved from multi-method results, fallback to history/attributes
        if not retrieved_from_multi:
             print("Retrieving latest report data from history/attributes.")
             # Get latest correlation matrix from history
             if hasattr(self, 'correlation_history') and self.correlation_history:
                try:
                    # History keys might be timestamps or other objects, find the max
                    latest_ts = max(self.correlation_history.keys())
                    report['correlation_matrix'] = self.correlation_history[latest_ts].get('matrix')
                except ValueError: # Handle empty history
                     print("Warning: Correlation history is empty.")
                     pass

             # Get latest feature importance from history
             if hasattr(self, 'feature_importance_history') and self.feature_importance_history:
                  try:
                      latest_fi_ts = max(self.feature_importance_history.keys())
                      report['feature_importance'] = self.feature_importance_history[latest_fi_ts].get('importance')
                  except ValueError:
                      print("Warning: Feature importance history is empty.")
                      pass

             # Get directly from attributes (assume they hold the latest state if history not used)
             report['redundancy_groups'] = getattr(self, 'redundancy_groups', [])
             report['unique_indicators'] = getattr(self, 'unique_indicators', [])
             report['signal_weights'] = getattr(self, 'signal_weights', None)
             # Note: optimal_indicators might not be stored as a direct attribute,
             # depends on where it's calculated and stored. Adjust if necessary.
             # report['optimal_indicators'] = getattr(self, 'optimal_indicators', None) # Example


        # 3. Retrieve Metadata (If stored during loading)
        if hasattr(self, 'loaded_parameters') and self.loaded_parameters:
             print("Updating metadata from loaded parameters.")
             report['symbol'] = report['symbol'] or self.loaded_parameters.get('symbol')
             report['period'] = report['period'] or self.loaded_parameters.get('period')
             report['interval'] = report['interval'] or self.loaded_parameters.get('interval')
        elif not retrieved_from_multi: # Only print if not retrieved elsewhere
             print("Warning: Could not retrieve metadata (symbol, period, interval). Not loaded?")


        # Regenerate visualizations if missing or empty
        if not report.get('visualizations') or not isinstance(report['visualizations'], dict) or not report['visualizations']:
            figs = {}
            # Only generate if data is present
            if report.get('correlation_matrix') is not None:
                figs['correlation_matrix'] = self.plot_correlation_matrix(report['correlation_matrix'])
            if report.get('redundancy_groups') is not None:
                figs['redundancy_groups'] = self.plot_redundancy_groups(report['redundancy_groups'])
            if report.get('feature_importance') is not None:
                figs['feature_importance'] = self.plot_feature_importance(report['feature_importance'])
            report['visualizations'] = figs

        return report
