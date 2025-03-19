"""
Portfolio Optimization Module for Perfect Storm Dashboard

This module extends beyond single-asset analysis to:
1. Provide portfolio-level recommendations
2. Suggest position sizing based on signal strength
3. Implement risk management rules
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.optimize as sco
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Class for portfolio optimization"""
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the PortfolioOptimizer class
        
        Parameters:
        - risk_free_rate: Annual risk-free rate (default: 0.02 or 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Initialize portfolio history
        self.portfolio_history = {}
    
    def calculate_portfolio_metrics(self, returns_df):
        """
        Calculate portfolio metrics
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        
        Returns:
        - metrics: Dictionary with portfolio metrics
        """
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Calculate annualized metrics
        ann_mean_returns = mean_returns * 252
        ann_cov_matrix = cov_matrix * 252
        
        # Calculate asset volatilities
        asset_volatilities = np.sqrt(np.diag(ann_cov_matrix))
        
        # Create metrics dictionary
        metrics = {
            'mean_returns': mean_returns,
            'ann_mean_returns': ann_mean_returns,
            'cov_matrix': cov_matrix,
            'ann_cov_matrix': ann_cov_matrix,
            'corr_matrix': corr_matrix,
            'asset_volatilities': asset_volatilities
        }
        
        return metrics
    
    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        - weights: Array of asset weights
        - mean_returns: Series of mean returns
        - cov_matrix: Covariance matrix of returns
        
        Returns:
        - returns: Portfolio returns
        - volatility: Portfolio volatility
        - sharpe_ratio: Portfolio Sharpe ratio
        """
        # Calculate portfolio returns
        returns = np.sum(mean_returns * weights) * 252
        
        # Calculate portfolio volatility
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (returns - self.risk_free_rate) / volatility
        
        return returns, volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights, mean_returns, cov_matrix):
        """
        Calculate negative Sharpe ratio for optimization
        
        Parameters:
        - weights: Array of asset weights
        - mean_returns: Series of mean returns
        - cov_matrix: Covariance matrix of returns
        
        Returns:
        - negative_sharpe: Negative Sharpe ratio
        """
        returns, volatility, sharpe = self.portfolio_performance(weights, mean_returns, cov_matrix)
        return -sharpe
    
    def portfolio_volatility(self, weights, mean_returns, cov_matrix):
        """
        Calculate portfolio volatility
        
        Parameters:
        - weights: Array of asset weights
        - mean_returns: Series of mean returns
        - cov_matrix: Covariance matrix of returns
        
        Returns:
        - volatility: Portfolio volatility
        """
        return self.portfolio_performance(weights, mean_returns, cov_matrix)[1]
    
    def optimize_portfolio(self, returns_df, target_return=None, min_weight=0.0, max_weight=1.0):
        """
        Optimize portfolio weights
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - target_return: Target portfolio return (default: None, maximize Sharpe ratio)
        - min_weight: Minimum weight for each asset (default: 0.0)
        - max_weight: Maximum weight for each asset (default: 1.0)
        
        Returns:
        - optimal_weights: Series of optimal asset weights
        - performance: Dictionary with portfolio performance metrics
        """
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(returns_df)
        mean_returns = metrics['mean_returns']
        cov_matrix = metrics['cov_matrix']
        
        # Get number of assets
        num_assets = len(mean_returns)
        
        # Set initial weights
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Set bounds for weights
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # Set constraint that weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        if target_return is None:
            # Maximize Sharpe ratio
            result = sco.minimize(self.negative_sharpe_ratio, init_weights, 
                                 args=(mean_returns, cov_matrix), method='SLSQP', 
                                 bounds=bounds, constraints=constraints)
            optimal_weights = result['x']
        else:
            # Minimize volatility for target return
            def target_return_constraint(weights):
                return np.sum(mean_returns * weights) * 252 - target_return
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                          {'type': 'eq', 'fun': target_return_constraint})
            
            result = sco.minimize(self.portfolio_volatility, init_weights, 
                                 args=(mean_returns, cov_matrix), method='SLSQP', 
                                 bounds=bounds, constraints=constraints)
            optimal_weights = result['x']
        
        # Calculate portfolio performance
        returns, volatility, sharpe = self.portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        
        # Create performance dictionary
        performance = {
            'returns': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
        
        # Create Series of optimal weights
        optimal_weights_series = pd.Series(optimal_weights, index=returns_df.columns)
        
        return optimal_weights_series, performance
    
    def efficient_frontier(self, returns_df, num_portfolios=100, min_weight=0.0, max_weight=1.0):
        """
        Calculate efficient frontier
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - num_portfolios: Number of portfolios to generate (default: 100)
        - min_weight: Minimum weight for each asset (default: 0.0)
        - max_weight: Maximum weight for each asset (default: 1.0)
        
        Returns:
        - efficient_frontier: DataFrame with efficient frontier data
        """
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(returns_df)
        mean_returns = metrics['mean_returns']
        cov_matrix = metrics['cov_matrix']
        
        # Get number of assets
        num_assets = len(mean_returns)
        
        # Calculate minimum volatility portfolio
        min_vol_weights, min_vol_performance = self.optimize_portfolio(returns_df, min_weight=min_weight, max_weight=max_weight)
        min_vol_return = min_vol_performance['returns']
        min_vol_volatility = min_vol_performance['volatility']
        
        # Calculate maximum Sharpe ratio portfolio
        max_sharpe_weights, max_sharpe_performance = self.optimize_portfolio(returns_df, min_weight=min_weight, max_weight=max_weight)
        max_sharpe_return = max_sharpe_performance['returns']
        max_sharpe_volatility = max_sharpe_performance['volatility']
        
        # Calculate maximum return portfolio
        max_return_idx = mean_returns.idxmax()
        max_return = mean_returns[max_return_idx] * 252
        
        # Generate efficient frontier
        target_returns = np.linspace(min_vol_return, max_return, num_portfolios)
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                weights, performance = self.optimize_portfolio(returns_df, target_return=target_return, 
                                                             min_weight=min_weight, max_weight=max_weight)
                efficient_portfolios.append({
                    'return': performance['returns'],
                    'volatility': performance['volatility'],
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'weights': weights
                })
            except:
                # Skip if optimization fails
                continue
        
        # Create DataFrame
        efficient_frontier = pd.DataFrame(efficient_portfolios)
        
        # Add minimum volatility and maximum Sharpe ratio portfolios
        efficient_frontier = efficient_frontier.append({
            'return': min_vol_return,
            'volatility': min_vol_volatility,
            'sharpe_ratio': min_vol_performance['sharpe_ratio'],
            'weights': min_vol_weights,
            'portfolio_type': 'Minimum Volatility'
        }, ignore_index=True)
        
        efficient_frontier = efficient_frontier.append({
            'return': max_sharpe_return,
            'volatility': max_sharpe_volatility,
            'sharpe_ratio': max_sharpe_performance['sharpe_ratio'],
            'weights': max_sharpe_weights,
            'portfolio_type': 'Maximum Sharpe Ratio'
        }, ignore_index=True)
        
        return efficient_frontier
    
    def plot_efficient_frontier(self, efficient_frontier, figsize=(10, 6)):
        """
        Plot efficient frontier
        
        Parameters:
        - efficient_frontier: DataFrame with efficient frontier data
        - figsize: Figure size (default: (10, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        ax.scatter(efficient_frontier['volatility'], efficient_frontier['return'], 
                  c=efficient_frontier['sharpe_ratio'], cmap='viridis', 
                  marker='o', s=10, alpha=0.7)
        
        # Highlight minimum volatility and maximum Sharpe ratio portfolios
        min_vol_idx = efficient_frontier[efficient_frontier['portfolio_type'] == 'Minimum Volatility'].index
        max_sharpe_idx = efficient_frontier[efficient_frontier['portfolio_type'] == 'Maximum Sharpe Ratio'].index
        
        if not min_vol_idx.empty:
            min_vol = efficient_frontier.loc[min_vol_idx[0]]
            ax.scatter(min_vol['volatility'], min_vol['return'], 
                      marker='*', color='r', s=200, label='Minimum Volatility')
        
        if not max_sharpe_idx.empty:
            max_sharpe = efficient_frontier.loc[max_sharpe_idx[0]]
            ax.scatter(max_sharpe['volatility'], max_sharpe['return'], 
                      marker='*', color='g', s=200, label='Maximum Sharpe Ratio')
        
        # Plot capital market line
        x_min, x_max = ax.get_xlim()
        y_min = self.risk_free_rate
        
        if not max_sharpe_idx.empty:
            max_sharpe = efficient_frontier.loc[max_sharpe_idx[0]]
            slope = (max_sharpe['return'] - self.risk_free_rate) / max_sharpe['volatility']
            y_max = self.risk_free_rate + slope * x_max
            ax.plot([0, x_max], [self.risk_free_rate, y_max], 'k--', label='Capital Market Line')
        
        # Set title and labels
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Sharpe Ratio')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_portfolio_weights(self, weights, figsize=(10, 6)):
        """
        Plot portfolio weights
        
        Parameters:
        - weights: Series of asset weights
        - figsize: Figure size (default: (10, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort weights
        sorted_weights = weights.sort_values(ascending=False)
        
        # Plot weights
        sorted_weights.plot(kind='bar', ax=ax)
        
        # Set title and labels
        ax.set_title('Portfolio Weights')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight')
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


class PositionSizer:
    """Class for position sizing based on signal strength"""
    
    def __init__(self, base_position_size=0.1, max_position_size=0.25):
        """
        Initialize the PositionSizer class
        
        Parameters:
        - base_position_size: Base position size as fraction of portfolio (default: 0.1 or 10%)
        - max_position_size: Maximum position size as fraction of portfolio (default: 0.25 or 25%)
        """
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, signal_strength, volatility=None, risk_tolerance=1.0):
        """
        Calculate position size based on signal strength
        
        Parameters:
        - signal_strength: Signal strength (-1 to 1)
        - volatility: Asset volatility (default: None)
        - risk_tolerance: Risk tolerance factor (default: 1.0)
        
        Returns:
        - position_size: Position size as fraction of portfolio
        """
        # Calculate base position size adjusted for risk tolerance
        adjusted_base_size = self.base_position_size * risk_tolerance
        
        # Calculate maximum position size adjusted for risk tolerance
        adjusted_max_size = self.max_position_size * risk_tolerance
        
        # Calculate position size based on signal strength
        abs_signal = abs(signal_strength)
        position_size = adjusted_base_size + (adjusted_max_size - adjusted_base_size) * abs_signal
        
        # Adjust for volatility if provided
        if volatility is not None:
            # Normalize volatility (assuming average volatility is around 0.2 or 20%)
            normalized_volatility = volatility / 0.2
            
            # Adjust position size inversely with volatility
            position_size = position_size / normalized_volatility
        
        # Ensure position size is within limits
        position_size = min(position_size, adjusted_max_size)
        
        # Determine direction based on signal
        if signal_strength < 0:
            position_size = -position_size
        
        return position_size
    
    def calculate_portfolio_positions(self, signals_df, volatilities=None, risk_tolerance=1.0):
        """
        Calculate position sizes for multiple assets
        
        Parameters:
        - signals_df: DataFrame with signal strengths for each as<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>