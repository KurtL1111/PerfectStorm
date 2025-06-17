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
        efficient_frontier = pd.concat([efficient_frontier, pd.DataFrame([{
            'return': min_vol_return,
            'volatility': min_vol_volatility,
            'sharpe_ratio': min_vol_performance['sharpe_ratio'],
            'weights': min_vol_weights,
            'portfolio_type': 'Minimum Volatility'
        }])], ignore_index=True)
        
        efficient_frontier = pd.concat([efficient_frontier, pd.DataFrame([{
            'return': max_sharpe_return,
            'volatility': max_sharpe_volatility,
            'sharpe_ratio': max_sharpe_performance['sharpe_ratio'],
            'weights': max_sharpe_weights,
            'portfolio_type': 'Maximum Sharpe Ratio'
        }])], ignore_index=True)
        
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
        scatter = ax.scatter(efficient_frontier['volatility'], efficient_frontier['return'], 
                  c=efficient_frontier['sharpe_ratio'], cmap='viridis', 
                  marker='o', s=10, alpha=0.7)
        
        # Highlight minimum volatility and maximum Sharpe ratio portfolios
        min_vol_portfolio = efficient_frontier[efficient_frontier['portfolio_type'] == 'Minimum Volatility']
        max_sharpe_portfolio = efficient_frontier[efficient_frontier['portfolio_type'] == 'Maximum Sharpe Ratio']
        
        ax.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['return'], 
                  color='r', marker='*', s=200, label='Minimum Volatility')
        
        ax.scatter(max_sharpe_portfolio['volatility'], max_sharpe_portfolio['return'], 
                  color='g', marker='*', s=200, label='Maximum Sharpe Ratio')
        
        # Add colorbar using the scatter plot handle
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
        
        # Set labels and title
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.set_title('Efficient Frontier')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.close(fig)
        return fig
    
    def plot_asset_weights(self, weights, figsize=(10, 6)):
        """
        Plot asset weights
        
        Parameters:
        - weights: Series of asset weights
        - figsize: Figure size (default: (10, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot weights
        weights.plot(kind='bar', ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Weights')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def calculate_position_sizes(self, weights, total_capital, signal_strengths=None):
        """
        Calculate position sizes based on weights and signal strengths
        
        Parameters:
        - weights: Series of asset weights
        - total_capital: Total capital to allocate
        - signal_strengths: Series of signal strengths (default: None, equal strengths)
        
        Returns:
        - position_sizes: Series of position sizes
        """
        # If signal strengths not provided, use equal strengths
        if signal_strengths is None:
            signal_strengths = pd.Series(1.0, index=weights.index)
        
        # Normalize signal strengths to sum to 1
        normalized_signals = signal_strengths / signal_strengths.sum()
        
        # Adjust weights based on signal strengths
        adjusted_weights = weights * normalized_signals
        
        # Normalize adjusted weights to sum to 1
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Calculate position sizes
        position_sizes = adjusted_weights * total_capital
        
        return position_sizes
    
    def calculate_risk_adjusted_position_sizes(self, weights, total_capital, volatilities, 
                                              target_portfolio_volatility=0.15, max_position_size=None):
        """
        Calculate risk-adjusted position sizes
        
        Parameters:
        - weights: Series of asset weights
        - total_capital: Total capital to allocate
        - volatilities: Series of asset volatilities
        - target_portfolio_volatility: Target portfolio volatility (default: 0.15 or 15%)
        - max_position_size: Maximum position size as percentage of total capital (default: None)
        
        Returns:
        - position_sizes: Series of position sizes
        """
        # Calculate risk contribution of each asset
        risk_contributions = weights * volatilities
        
        # Calculate total portfolio risk
        portfolio_risk = risk_contributions.sum()
        
        # Calculate risk adjustment factor
        risk_adjustment = target_portfolio_volatility / portfolio_risk
        
        # Adjust weights based on risk
        risk_adjusted_weights = weights * risk_adjustment
        
        # Calculate position sizes
        position_sizes = risk_adjusted_weights * total_capital
        
        # Apply maximum position size if specified
        if max_position_size is not None:
            max_size = total_capital * max_position_size
            position_sizes = position_sizes.clip(upper=max_size)
        
        return position_sizes
    
    def calculate_equal_risk_contribution(self, returns_df, total_capital, max_position_size=None):
        """
        Calculate position sizes with equal risk contribution
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - total_capital: Total capital to allocate
        - max_position_size: Maximum position size as percentage of total capital (default: None)
        
        Returns:
        - position_sizes: Series of position sizes
        """
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(returns_df)
        cov_matrix = metrics['cov_matrix']
        
        # Get number of assets
        num_assets = len(returns_df.columns)
        
        # Define objective function for equal risk contribution
        def objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contribution = weights * np.dot(cov_matrix, weights) / portfolio_vol
            return np.sum((risk_contribution - portfolio_vol / num_assets) ** 2)
        
        # Set initial weights
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Set bounds for weights
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        # Set constraint that weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Optimize weights
        result = sco.minimize(objective, init_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
        
        # Get optimal weights
        optimal_weights = pd.Series(result['x'], index=returns_df.columns)
        
        # Calculate position sizes
        position_sizes = optimal_weights * total_capital
        
        # Apply maximum position size if specified
        if max_position_size is not None:
            max_size = total_capital * max_position_size
            position_sizes = position_sizes.clip(upper=max_size)
        
        return position_sizes
    
    def implement_risk_management_rules(self, position_sizes, total_capital, max_drawdown=0.2, 
                                       stop_loss_pct=0.05, take_profit_pct=0.1):
        """
        Implement risk management rules
        
        Parameters:
        - position_sizes: Series of position sizes
        - total_capital: Total capital to allocate
        - max_drawdown: Maximum allowed drawdown (default: 0.2 or 20%)
        - stop_loss_pct: Stop loss percentage (default: 0.05 or 5%)
        - take_profit_pct: Take profit percentage (default: 0.1 or 10%)
        
        Returns:
        - risk_management: Dictionary with risk management parameters
        """
        # Calculate maximum capital at risk
        max_capital_at_risk = total_capital * max_drawdown
        
        # Calculate stop loss amounts
        stop_loss_amounts = position_sizes * stop_loss_pct
        
        # Calculate take profit amounts
        take_profit_amounts = position_sizes * take_profit_pct
        
        # Calculate total capital at risk
        total_capital_at_risk = stop_loss_amounts.sum()
        
        # Adjust position sizes if total capital at risk exceeds maximum
        if total_capital_at_risk > max_capital_at_risk:
            adjustment_factor = max_capital_at_risk / total_capital_at_risk
            adjusted_position_sizes = position_sizes * adjustment_factor
            adjusted_stop_loss_amounts = stop_loss_amounts * adjustment_factor
            adjusted_take_profit_amounts = take_profit_amounts * adjustment_factor
        else:
            adjusted_position_sizes = position_sizes
            adjusted_stop_loss_amounts = stop_loss_amounts
            adjusted_take_profit_amounts = take_profit_amounts
        
        # Create risk management dictionary
        risk_management = {
            'position_sizes': adjusted_position_sizes,
            'stop_loss_amounts': adjusted_stop_loss_amounts,
            'take_profit_amounts': adjusted_take_profit_amounts,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_drawdown': max_drawdown,
            'max_capital_at_risk': max_capital_at_risk,
            'total_capital_at_risk': adjusted_stop_loss_amounts.sum()
        }
        
        return risk_management
    
    def calculate_portfolio_var(self, returns_df, weights, confidence_level=0.95, time_horizon=1):
        """
        Calculate portfolio Value at Risk (VaR)
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - weights: Series of asset weights
        - confidence_level: Confidence level for VaR (default: 0.95 or 95%)
        - time_horizon: Time horizon in days (default: 1)
        
        Returns:
        - var: Value at Risk
        """
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate VaR using historical method
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * np.sqrt(time_horizon)
        
        return var
    
    def calculate_portfolio_cvar(self, returns_df, weights, confidence_level=0.95, time_horizon=1):
        """
        Calculate portfolio Conditional Value at Risk (CVaR)
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - weights: Series of asset weights
        - confidence_level: Confidence level for CVaR (default: 0.95 or 95%)
        - time_horizon: Time horizon in days (default: 1)
        
        Returns:
        - cvar: Conditional Value at Risk
        """
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * np.sqrt(time_horizon)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = -portfolio_returns[portfolio_returns < -var / np.sqrt(time_horizon)].mean() * np.sqrt(time_horizon)
        
        return cvar
    
    def optimize_portfolio_with_risk_constraints(self, returns_df, max_var=0.05, max_cvar=0.07, 
                                               min_weight=0.0, max_weight=1.0):
        """
        Optimize portfolio with risk constraints
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - max_var: Maximum allowed VaR (default: 0.05 or 5%)
        - max_cvar: Maximum allowed CVaR (default: 0.07 or 7%)
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
        
        # Define VaR constraint
        def var_constraint(weights):
            return max_var - self.calculate_portfolio_var(returns_df, pd.Series(weights, index=returns_df.columns))
        
        # Define CVaR constraint
        def cvar_constraint(weights):
            return max_cvar - self.calculate_portfolio_cvar(returns_df, pd.Series(weights, index=returns_df.columns))
        
        # Set constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': var_constraint},         # VaR constraint
            {'type': 'ineq', 'fun': cvar_constraint}         # CVaR constraint
        ]
        
        # Maximize Sharpe ratio
        result = sco.minimize(self.negative_sharpe_ratio, init_weights, 
                             args=(mean_returns, cov_matrix), method='SLSQP', 
                             bounds=bounds, constraints=constraints)
        
        # Get optimal weights
        optimal_weights = pd.Series(result['x'], index=returns_df.columns)
        
        # Calculate portfolio performance
        returns, volatility, sharpe = self.portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        
        # Calculate VaR and CVaR
        var = self.calculate_portfolio_var(returns_df, optimal_weights)
        cvar = self.calculate_portfolio_cvar(returns_df, optimal_weights)
        
        # Create performance dictionary
        performance = {
            'returns': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'var': var,
            'cvar': cvar
        }
        
        return optimal_weights, performance
    
    def rebalance_portfolio(self, current_weights, target_weights, threshold=0.05):
        """
        Determine if portfolio rebalancing is needed
        
        Parameters:
        - current_weights: Series of current asset weights
        - target_weights: Series of target asset weights
        - threshold: Rebalancing threshold (default: 0.05 or 5%)
        
        Returns:
        - rebalance_needed: Whether rebalancing is needed
        - rebalance_weights: Series of weights to rebalance to
        """
        # Calculate weight differences
        weight_diff = (current_weights - target_weights).abs()
        
        # Check if any weight difference exceeds threshold
        rebalance_needed = (weight_diff > threshold).any()
        
        # Return rebalance information
        return rebalance_needed, target_weights
    
    def generate_portfolio_recommendations(self, returns_df, total_capital, signal_strengths=None, 
                                          risk_profile='moderate', current_weights=None):
        """
        Generate comprehensive portfolio recommendations
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - total_capital: Total capital to allocate
        - signal_strengths: Series of signal strengths (default: None, equal strengths)
        - risk_profile: Risk profile ('conservative', 'moderate', 'aggressive', default: 'moderate')
        - current_weights: Series of current asset weights (default: None)
        
        Returns:
        - recommendations: Dictionary with portfolio recommendations
        """
        # Set risk parameters based on risk profile
        if risk_profile == 'conservative':
            target_volatility = 0.10  # 10%
            max_drawdown = 0.15      # 15%
            stop_loss_pct = 0.03     # 3%
            take_profit_pct = 0.07   # 7%
            max_position_size = 0.15 # 15%
        elif risk_profile == 'moderate':
            target_volatility = 0.15  # 15%
            max_drawdown = 0.20      # 20%
            stop_loss_pct = 0.05     # 5%
            take_profit_pct = 0.10   # 10%
            max_position_size = 0.20 # 20%
        elif risk_profile == 'aggressive':
            target_volatility = 0.25  # 25%
            max_drawdown = 0.30      # 30%
            stop_loss_pct = 0.07     # 7%
            take_profit_pct = 0.15   # 15%
            max_position_size = 0.30 # 30%
        else:
            raise ValueError(f"Unknown risk profile: {risk_profile}")
        
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(returns_df)
        
        # Optimize portfolio
        optimal_weights, performance = self.optimize_portfolio(returns_df)
        
        # Calculate position sizes
        if signal_strengths is not None:
            position_sizes = self.calculate_position_sizes(optimal_weights, total_capital, signal_strengths)
        else:
            position_sizes = optimal_weights * total_capital
        
        # Calculate risk-adjusted position sizes
        risk_adjusted_position_sizes = self.calculate_risk_adjusted_position_sizes(
            optimal_weights, total_capital, metrics['asset_volatilities'], 
            target_volatility, max_position_size
        )
        
        # Implement risk management rules
        risk_management = self.implement_risk_management_rules(
            risk_adjusted_position_sizes, total_capital, 
            max_drawdown, stop_loss_pct, take_profit_pct
        )
        
        # Calculate VaR and CVaR
        var = self.calculate_portfolio_var(returns_df, optimal_weights)
        cvar = self.calculate_portfolio_cvar(returns_df, optimal_weights)
        
        # Check if rebalancing is needed
        rebalance_needed = False
        if current_weights is not None:
            rebalance_needed, _ = self.rebalance_portfolio(current_weights, optimal_weights)
        
        # Create recommendations dictionary
        recommendations = {
            'optimal_weights': optimal_weights,
            'performance': performance,
            'position_sizes': position_sizes,
            'risk_adjusted_position_sizes': risk_adjusted_position_sizes,
            'risk_management': risk_management,
            'var': var,
            'cvar': cvar,
            'rebalance_needed': rebalance_needed,
            'risk_profile': risk_profile
        }
        
        # Store in portfolio history
        timestamp = pd.Timestamp.now()
        self.portfolio_history[timestamp] = recommendations
        
        return recommendations
    
    def plot_portfolio_allocation(self, weights, figsize=(10, 6), kind='pie'):
        """
        Plot portfolio allocation
        
        Parameters:
        - weights: Series of asset weights
        - figsize: Figure size (default: (10, 6))
        - kind: Plot type ('pie' or 'bar', default: 'pie')
        
        Returns:
        - fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if kind == 'pie':
            # Plot pie chart
            weights.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            
            # Set title
            ax.set_title('Portfolio Allocation')
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_aspect('equal')
        
        elif kind == 'bar':
            # Plot bar chart
            weights.plot(kind='bar', ax=ax)
            
            # Set labels and title
            ax.set_xlabel('Asset')
            ax.set_ylabel('Weight')
            ax.set_title('Portfolio Allocation')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
        
        else:
            raise ValueError(f"Unknown plot kind: {kind}")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_risk_contribution(self, weights, returns_df, figsize=(10, 6)):
        """
        Plot risk contribution of each asset
        
        Parameters:
        - weights: Series of asset weights
        - returns_df: DataFrame with asset returns (each column is an asset)
        - figsize: Figure size (default: (10, 6))
        
        Returns:
        - fig: Matplotlib figure
        """
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(returns_df)
        cov_matrix = metrics['cov_matrix']
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Calculate risk contribution
        risk_contribution = weights * np.dot(cov_matrix, weights) * 252 / portfolio_vol
        
        # Normalize to percentage
        risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot risk contribution
        risk_contribution_pct.plot(kind='bar', ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Asset')
        ax.set_ylabel('Risk Contribution (%)')
        ax.set_title('Portfolio Risk Contribution')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_portfolio_report(self, returns_df, total_capital, signal_strengths=None, 
                                 risk_profile='moderate', current_weights=None, filename=None):
        """
        Generate comprehensive portfolio report
        
        Parameters:
        - returns_df: DataFrame with asset returns (each column is an asset)
        - total_capital: Total capital to allocate
        - signal_strengths: Series of signal strengths (default: None, equal strengths)
        - risk_profile: Risk profile ('conservative', 'moderate', 'aggressive', default: 'moderate')
        - current_weights: Series of current asset weights (default: None)
        - filename: Filename to save the report (default: None, displays the report)
        
        Returns:
        - report: Dictionary with report components
        """
        # Generate portfolio recommendations
        recommendations = self.generate_portfolio_recommendations(
            returns_df, total_capital, signal_strengths, risk_profile, current_weights
        )
        
        # Calculate efficient frontier
        efficient_frontier = self.efficient_frontier(returns_df)
        
        # Create report components
        report = {
            'recommendations': recommendations,
            'efficient_frontier': efficient_frontier,
            'returns_df': returns_df
        }
        
        # Generate visualizations
        figs = {}
        
        # Efficient frontier
        figs['efficient_frontier'] = self.plot_efficient_frontier(efficient_frontier)
        
        # Portfolio allocation
        figs['portfolio_allocation_pie'] = self.plot_portfolio_allocation(
            recommendations['optimal_weights'], kind='pie'
        )
        
        figs['portfolio_allocation_bar'] = self.plot_portfolio_allocation(
            recommendations['optimal_weights'], kind='bar'
        )
        
        # Risk contribution
        figs['risk_contribution'] = self.plot_risk_contribution(
            recommendations['optimal_weights'], returns_df
        )
        
        # Add visualizations to report
        report['visualizations'] = figs
        
        # Save report if filename provided
        if filename is not None:
            # Save visualizations
            for name, fig in figs.items():
                fig.savefig(f"{filename}_{name}.png", dpi=300, bbox_inches='tight')
            
            # Save report data
            import pickle
            with open(f"{filename}_data.pkl", 'wb') as f:
                pickle.dump(report, f)
        
        return report
