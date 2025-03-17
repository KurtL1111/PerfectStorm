"""
Backtesting Framework for Perfect Storm Dashboard

This module implements a comprehensive backtesting system that allows users to:
1. Test the Perfect Storm strategy against historical data
2. Calculate key performance metrics (Sharpe ratio, drawdown, win rate)
3. Compare performance against benchmark indices
4. Optimize indicator parameters for specific assets or market conditions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid
import itertools
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class BacktestingEngine:
    """Class for backtesting trading strategies"""
    
    def __init__(self, initial_capital=100000, commission=0.001, risk_free_rate=0.02):
        """
        Initialize the BacktestingEngine class
        
        Parameters:
        - initial_capital: Initial capital for backtesting (default: 100000)
        - commission: Commission rate for trades (default: 0.001, i.e., 0.1%)
        - risk_free_rate: Annual risk-free rate for performance metrics (default: 0.02, i.e., 2%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        
        # Initialize results
        self.results = None
        self.trades = None
        self.metrics = None
        self.benchmark_results = None
        self.optimization_results = None
    
    def run_backtest(self, df, strategy_func, strategy_params=None, benchmark_col=None):
        """
        Run a backtest on historical data
        
        Parameters:
        - df: DataFrame with market data
        - strategy_func: Function that generates buy/sell signals
        - strategy_params: Dictionary of parameters for the strategy function (default: None)
        - benchmark_col: Column to use as benchmark (default: None)
        
        Returns:
        - results: DataFrame with backtest results
        """
        # Make a copy of the DataFrame
        df_backtest = df.copy()
        
        # Generate signals
        if strategy_params is None:
            strategy_params = {}
        
        df_backtest = strategy_func(df_backtest, **strategy_params)
        
        # Ensure 'buy_signal' and 'sell_signal' columns exist
        if 'buy_signal' not in df_backtest.columns or 'sell_signal' not in df_backtest.columns:
            raise ValueError("Strategy function must add 'buy_signal' and 'sell_signal' columns to the DataFrame")
        
        # Initialize results DataFrame
        results = pd.DataFrame(index=df_backtest.index)
        results['close'] = df_backtest['close']
        results['buy_signal'] = df_backtest['buy_signal']
        results['sell_signal'] = df_backtest['sell_signal']
        
        # Initialize position and portfolio value
        results['position'] = 0
        results['cash'] = self.initial_capital
        results['holdings'] = 0
        results['portfolio_value'] = self.initial_capital
        
        # Initialize trade log
        trades = []
        
        # Simulate trading
        for i in range(1, len(results)):
            # Default: carry forward previous position and portfolio value
            results.loc[results.index[i], 'position'] = results.loc[results.index[i-1], 'position']
            results.loc[results.index[i], 'cash'] = results.loc[results.index[i-1], 'cash']
            results.loc[results.index[i], 'holdings'] = results.loc[results.index[i-1], 'position'] * results.loc[results.index[i], 'close']
            
            # Check for buy signal
            if results.loc[results.index[i], 'buy_signal'] == 1 and results.loc[results.index[i-1], 'position'] == 0:
                # Calculate number of shares to buy
                price = results.loc[results.index[i], 'close']
                available_cash = results.loc[results.index[i], 'cash']
                commission_cost = available_cash * self.commission
                shares = (available_cash - commission_cost) / price
                
                # Update position and cash
                results.loc[results.index[i], 'position'] = shares
                results.loc[results.index[i], 'cash'] = 0
                results.loc[results.index[i], 'holdings'] = shares * price
                
                # Log trade
                trades.append({
                    'date': results.index[i],
                    'type': 'buy',
                    'price': price,
                    'shares': shares,
                    'value': shares * price,
                    'commission': commission_cost
                })
            
            # Check for sell signal
            elif results.loc[results.index[i], 'sell_signal'] == 1 and results.loc[results.index[i-1], 'position'] > 0:
                # Calculate value of shares to sell
                price = results.loc[results.index[i], 'close']
                shares = results.loc[results.index[i-1], 'position']
                value = shares * price
                commission_cost = value * self.commission
                
                # Update position and cash
                results.loc[results.index[i], 'position'] = 0
                results.loc[results.index[i], 'cash'] = value - commission_cost
                results.loc[results.index[i], 'holdings'] = 0
                
                # Log trade
                trades.append({
                    'date': results.index[i],
                    'type': 'sell',
                    'price': price,
                    'shares': shares,
                    'value': value,
                    'commission': commission_cost
                })
            
            # Update portfolio value
            results.loc[results.index[i], 'portfolio_value'] = results.loc[results.index[i], 'cash'] + results.loc[results.index[i], 'holdings']
        
        # Calculate daily returns
        results['daily_return'] = results['portfolio_value'].pct_change()
        
        # Calculate benchmark returns if provided
        if benchmark_col is not None and benchmark_col in df_backtest.columns:
            results['benchmark'] = df_backtest[benchmark_col]
            results['benchmark_return'] = results['benchmark'].pct_change()
        
        # Store results
        self.results = results
        self.trades = pd.DataFrame(trades)
        
        # Calculate performance metrics
        self.calculate_metrics()
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
        - metrics: Dictionary of performance metrics
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Get daily returns
        daily_returns = self.results['daily_return'].dropna()
        
        # Total return
        metrics['total_return'] = (self.results['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = (self.results.index[-1] - self.results.index[0]).days
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (365 / days) - 1
        
        # Volatility (annualized)
        metrics['volatility'] = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = daily_returns - daily_risk_free
        metrics['sharpe_ratio'] = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Sortino ratio (downside risk only)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = (daily_returns.mean() - daily_risk_free) * np.sqrt(252) / downside_deviation
        else:
            metrics['sortino_ratio'] = np.inf
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        metrics['max_drawdown'] = drawdown.min()
        
        # Calmar ratio
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.inf
        
        # Win rate
        if len(self.trades) > 0:
            # Calculate profit/loss for each trade
            buy_trades = self.trades[self.trades['type'] == 'buy'].copy()
            sell_trades = self.trades[self.trades['type'] == 'sell'].copy()
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                buy_trades['trade_id'] = range(len(buy_trades))
                sell_trades['trade_id'] = range(len(sell_trades))
                
                trades_pl = pd.DataFrame()
                trades_pl['buy_date'] = buy_trades['date'].values
                trades_pl['buy_price'] = buy_trades['price'].values
                trades_pl['sell_date'] = sell_trades['date'].values
                trades_pl['sell_price'] = sell_trades['price'].values
                trades_pl['shares'] = buy_trades['shares'].values
                trades_pl['buy_commission'] = buy_trades['commission'].values
                trades_pl['sell_commission'] = sell_trades['commission'].values
                
                trades_pl['profit_loss'] = (trades_pl['sell_price'] - trades_pl['buy_price']) * trades_pl['shares'] - trades_pl['buy_commission'] - trades_pl['sell_commission']
                trades_pl['return'] = trades_pl['profit_loss'] / (trades_pl['buy_price'] * trades_pl['shares'])
                
                # Calculate win rate
                metrics['win_rate'] = len(trades_pl[trades_pl['profit_loss'] > 0]) / len(trades_pl)
                
                # Calculate average win and loss
                winning_trades = trades_pl[trades_pl['profit_loss'] > 0]
                losing_trades = trades_pl[trades_pl['profit_loss'] <= 0]
                
                if len(winning_trades) > 0:
                    metrics['avg_win'] = winning_trades['return'].mean()
                else:
                    metrics['avg_win'] = 0
                
                if len(losing_trades) > 0:
                    metrics['avg_loss'] = losing_trades['return'].mean()
                else:
                    metrics['avg_loss'] = 0
                
                # Calculate profit factor
                if len(losing_trades) > 0 and abs(losing_trades['profit_loss'].sum()) > 0:
                    metrics['profit_factor'] = winning_trades['profit_loss'].sum() / abs(losing_trades['profit_loss'].sum())
                else:
                    metrics['profit_factor'] = np.inf
            else:
                metrics['win_rate'] = 0
                metrics['avg_win'] = 0
                metrics['avg_loss'] = 0
                metrics['profit_factor'] = 0
        else:
            metrics['win_rate'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0
            metrics['profit_factor'] = 0
        
        # Calculate benchmark metrics if available
        if 'benchmark_return' in self.results.columns:
            benchmark_returns = self.results['benchmark_return'].dropna()
            
            # Benchmark total return
            metrics['benchmark_total_return'] = (self.results['benchmark'].iloc[-1] / self.results['benchmark'].iloc[0]) - 1
            
            # Benchmark annualized return
            metrics['benchmark_annualized_return'] = (1 + metrics['benchmark_total_return']) ** (365 / days) - 1
            
            # Benchmark volatility
            metrics['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(252)
            
            # Benchmark Sharpe ratio
            benchmark_excess_returns = benchmark_returns - daily_risk_free
            metrics['benchmark_sharpe_ratio'] = benchmark_excess_returns.mean() / benchmark_returns.std() * np.sqrt(252)
            
            # Benchmark maximum drawdown
            benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
            benchmark_running_max = benchmark_cumulative_returns.cummax()
            benchmark_drawdown = (benchmark_cumulative_returns / benchmark_running_max) - 1
            metrics['benchmark_max_drawdown'] = benchmark_drawdown.min()
            
            # Alpha and Beta
            covariance = np.cov(daily_returns.values, benchmark_returns.values)[0, 1]
            variance = np.var(benchmark_returns.values)
            metrics['beta'] = covariance / variance
            metrics['alpha'] = metrics['annualized_return'] - (self.risk_free_rate + metrics['beta'] * (metrics['benchmark_annualized_return'] - self.risk_free_rate))
            
            # Information ratio
            tracking_error = (daily_returns - benchmark_returns).std() * np.sqrt(252)
            metrics['information_ratio'] = (metrics['annualized_return'] - metrics['benchmark_annualized_return']) / tracking_error
        
        # Store metrics
        self.metrics = metrics
        
        return metrics
    
    def plot_results(self, benchmark=True, figsize=(12, 8)):
        """
        Plot backtest results
        
        Parameters:
        - benchmark: Whether to include benchmark in the plot (default: True)
        - figsize: Figure size (default: (12, 8))
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot portfolio value
        portfolio_values = self.results['portfolio_value']
        axes[0].plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
        
        # Plot benchmark if available
        if benchmark and 'benchmark' in self.results.columns:
            # Normalize benchmark to start at initial capital
            benchmark_values = self.results['benchmark'] / self.results['benchmark'].iloc[0] * self.initial_capital
            axes[0].plot(benchmark_values.index, benchmark_values, label='Benchmark', alpha=0.7)
        
        # Add buy/sell markers
        buy_signals = self.results[self.results['buy_signal'] == 1]
        sell_signals = self.results[self.results['sell_signal'] == 1]
        
        if len(buy_signals) > 0:
            axes[0].scatter(buy_signals.index, buy_signals['portfolio_value'], marker='^', color='green', label='Buy Signal', alpha=0.7)
        
        if len(sell_signals) > 0:
            axes[0].scatter(sell_signals.index, sell_signals['portfolio_value'], marker='v', color='red', label='Sell Signal', alpha=0.7)
        
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title('Backtest Results')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis as currency
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot drawdown
        daily_returns = self.results['daily_return'].dropna()
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        
        axes[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis as percentage
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Plot position
        axes[2].fill_between(self.results.index, self.results['position'], 0, color='blue', alpha=0.3)
        axes[2].set_ylabel('Position (Shares)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, benchmark=True, figsize=(12, 6)):
        """
        Plot distribution of returns
        
        Parameters:
        - benchmark: Whether to include benchmark in the plot (default: True)
        - figsize: Figure size (default: (12, 6))
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Get daily returns
        daily_returns = self.results['daily_return'].dropna()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot strategy returns distribution
        sns.histplot(daily_returns, kde=True, stat='density', label='Strategy Returns')
        
        # Plot benchmark returns distribution if available
        if benchmark and 'benchmark_return' in self.results.columns:
            benchmark_returns = self.results['benchmark_return'].dropna()
            sns.histplot(benchmark_returns, kde=True, stat='density', label='Benchmark Returns', alpha=0.7)
        
        # Plot normal distribution for comparison
        x = np.linspace(daily_returns.min(), daily_returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, daily_returns.mean(), daily_returns.std()), 'r-', label='Normal Distribution')
        
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.title('Distribution of Daily Returns')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, figsize=(12, 8)):
        """
        Plot monthly returns heatmap
        
        Parameters:
        - figsize: Figure size (default: (12, 8))
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Get daily returns
        daily_returns = self.results['daily_return'].dropna()
        
        # Calculate monthly returns
        monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.unstack()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', center=0, linewidths=1, cbar=True)
        
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        plt.tight_layout()
        plt.show()
    
    def print_metrics(self):
        """Print performance metrics"""
        if self.metrics is None:
            raise ValueError("No metrics available. Run a backtest first.")
        
        print("=== Performance Metrics ===")
        print(f"Total Return: {self.metrics['total_return']:.2%}")
        print(f"Annualized Return: {self.metrics['annualized_return']:.2%}")
        print(f"Volatility: {self.metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {self.metrics['win_rate']:.2%}")
        print(f"Average Win: {self.metrics['avg_win']:.2%}")
        print(f"Average Loss: {self.metrics['avg_loss']:.2%}")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        
        if 'benchmark_total_return' in self.metrics:
            print("\n=== Benchmark Comparison ===")
            print(f"Benchmark Total Return: {self.metrics['benchmark_total_return']:.2%}")
            print(f"Benchmark Annualized Return: {self.metrics['benchmark_annualized_return']:.2%}")
            print(f"Benchmark Volatility: {self.metrics['benchmark_volatility']:.2%}")
            print(f"Benchmark Sharpe Ratio: {self.metrics['benchmark_sharpe_ratio']:.2f}")
            print(f"Benchmark Maximum Drawdown: {self.metrics['benchmark_max_drawdown']:.2%}")
            print(f"Alpha: {self.metrics['alpha']:.2%}")
            print(f"Beta: {self.metrics['beta']:.2f}")
            print(f"Information Ratio: {self.metrics['information_ratio']:.2f}")
    
    def optimize_parameters(self, df, strategy_func, param_grid, metric='sharpe_ratio', maximize=True):
        """
        Optimize strategy parameters
        
        Parameters:
        - df: DataFrame with market data
        - strategy_func: Function that generates buy/sell signals
        - param_grid: Dictionary of parameter grids for the strategy function
        - metric: Metric to optimize (default: 'sharpe_ratio')
        - maximize: Whether to maximize or minimize the metric (default: True)
        
        Returns:
        - best_params: Dictionary of best parameters
        - best_metrics: Dictionary of metrics for the best parameters
        """
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Initialize results
        optimization_results = []
        
        # Run backtest for each parameter combination
        for params in param_combinations:
            # Run backtest
            self.run_backtest(df, strategy_func, strategy_params=params)
            
            # Store results
            result = {
                'params': params,
                'metrics': self.metrics.copy()
            }
            optimization_results.append(result)
        
        # Find best parameters
        if maximize:
            best_result = max(optimization_results, key=lambda x: x['metrics'][metric])
        else:
            best_result = min(optimization_results, key=lambda x: x['metrics'][metric])
        
        # Store optimization results
        self.optimization_results = optimization_results
        
        return best_result['params'], best_result['metrics']
    
    def plot_optimization_results(self, param_name, metric='sharpe_ratio', figsize=(12, 6)):
        """
        Plot optimization results for a specific parameter
        
        Parameters:
        - param_name: Name of the parameter to plot
        - metric: Metric to plot (default: 'sharpe_ratio')
        - figsize: Figure size (default: (12, 6))
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available. Run parameter optimization first.")
        
        # Extract parameter values and metrics
        param_values = []
        metric_values = []
        
        for result in self.optimization_results:
            if param_name in result['params']:
                param_values.append(result['params'][param_name])
                metric_values.append(result['metrics'][metric])
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot parameter vs. metric
        plt.scatter(param_values, metric_values, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(param_values, metric_values, 1)
        p = np.poly1d(z)
        plt.plot(param_values, p(param_values), 'r--', alpha=0.7)
        
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.title(f'{metric} vs. {param_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def walk_forward_optimization(self, df, strategy_func, param_grid, window_size=252, step_size=63, metric='sharpe_ratio', maximize=True):
        """
        Perform walk-forward optimization
        
        Parameters:
        - df: DataFrame with market data
        - strategy_func: Function that generates buy/sell signals
        - param_grid: Dictionary of parameter grids for the strategy function
        - window_size: Size of the training window in days (default: 252, i.e., 1 year)
        - step_size: Size of the step in days (default: 63, i.e., 3 months)
        - metric: Metric to optimize (default: 'sharpe_ratio')
        - maximize: Whether to maximize or minimize the metric (default: True)
        
        Returns:
        - walk_forward_results: Dictionary of walk-forward optimization results
        """
        # Initialize results
        walk_forward_results = {
            'train_periods': [],
            'test_periods': [],
            'best_params': [],
            'test_metrics': []
        }
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Iterate through time periods
        for i in range(0, len(df) - window_size - step_size, step_size):
            # Define train and test periods
            train_start = i
            train_end = i + window_size
            test_start = train_end
            test_end = test_start + step_size
            
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            # Optimize parameters on train data
            best_params = None
            best_metric_value = -np.inf if maximize else np.inf
            
            for params in param_combinations:
                # Run backtest on train data
                self.run_backtest(train_df, strategy_func, strategy_params=params)
                
                # Check if this is the best result
                if maximize:
                    if self.metrics[metric] > best_metric_value:
                        best_metric_value = self.metrics[metric]
                        best_params = params
                else:
                    if self.metrics[metric] < best_metric_value:
                        best_metric_value = self.metrics[metric]
                        best_params = params
            
            # Run backtest on test data with best parameters
            self.run_backtest(test_df, strategy_func, strategy_params=best_params)
            
            # Store results
            walk_forward_results['train_periods'].append((train_df.index[0], train_df.index[-1]))
            walk_forward_results['test_periods'].append((test_df.index[0], test_df.index[-1]))
            walk_forward_results['best_params'].append(best_params)
            walk_forward_results['test_metrics'].append(self.metrics.copy())
        
        return walk_forward_results
    
    def monte_carlo_simulation(self, num_simulations=1000, confidence_level=0.95):
        """
        Perform Monte Carlo simulation to estimate the distribution of future returns
        
        Parameters:
        - num_simulations: Number of simulations to run (default: 1000)
        - confidence_level: Confidence level for the prediction interval (default: 0.95)
        
        Returns:
        - simulation_results: Dictionary of simulation results
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Get daily returns
        daily_returns = self.results['daily_return'].dropna()
        
        # Initialize simulation results
        simulation_results = {
            'simulations': [],
            'final_values': [],
            'confidence_interval': None
        }
        
        # Run simulations
        for _ in range(num_simulations):
            # Sample returns with replacement
            sampled_returns = np.random.choice(daily_returns, size=252, replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + sampled_returns).cumprod()
            
            # Calculate portfolio value
            portfolio_value = self.results['portfolio_value'].iloc[-1] * cumulative_returns
            
            # Store results
            simulation_results['simulations'].append(portfolio_value)
            simulation_results['final_values'].append(portfolio_value[-1])
        
        # Calculate confidence interval
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bound = np.percentile(simulation_results['final_values'], lower_percentile)
        upper_bound = np.percentile(simulation_results['final_values'], upper_percentile)
        
        simulation_results['confidence_interval'] = (lower_bound, upper_bound)
        
        return simulation_results
    
    def plot_monte_carlo_simulation(self, simulation_results, figsize=(12, 8)):
        """
        Plot Monte Carlo simulation results
        
        Parameters:
        - simulation_results: Dictionary of simulation results from monte_carlo_simulation
        - figsize: Figure size (default: (12, 8))
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot simulations
        for i, simulation in enumerate(simulation_results['simulations']):
            if i == 0:
                ax1.plot(range(len(simulation)), simulation, color='blue', alpha=0.1, label='Simulations')
            else:
                ax1.plot(range(len(simulation)), simulation, color='blue', alpha=0.1)
        
        # Plot confidence interval
        lower_bound, upper_bound = simulation_results['confidence_interval']
        ax1.axhline(y=lower_bound, color='red', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
        ax1.axhline(y=upper_bound, color='green', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
        
        # Plot current portfolio value
        current_value = self.results['portfolio_value'].iloc[-1]
        ax1.axhline(y=current_value, color='black', linestyle='-', label=f'Current Value ({current_value:.2f})')
        
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Monte Carlo Simulation')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot distribution of final values
        sns.histplot(simulation_results['final_values'], kde=True, ax=ax2)
        
        # Add vertical lines for confidence interval
        ax2.axvline(x=lower_bound, color='red', linestyle='--')
        ax2.axvline(x=upper_bound, color='green', linestyle='--')
        
        # Add vertical line for current value
        ax2.axvline(x=current_value, color='black', linestyle='-')
        
        ax2.set_xlabel('Final Portfolio Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis as currency
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.show()

class PerfectStormStrategy:
    """Class for Perfect Storm trading strategy"""
    
    @staticmethod
    def generate_signals(df, rsi_threshold_low=40, rsi_threshold_high=65, 
                         macd_threshold=0, stoch_threshold_low=20, stoch_threshold_high=80,
                         cci_threshold_low=-100, cci_threshold_high=100,
                         bb_threshold=0, min_signals_buy=3, min_signals_sell=3):
        """
        Generate buy/sell signals based on Perfect Storm strategy
        
        Parameters:
        - df: DataFrame with market data and indicators
        - rsi_threshold_low: RSI threshold for buy signal (default: 40)
        - rsi_threshold_high: RSI threshold for sell signal (default: 65)
        - macd_threshold: MACD threshold (default: 0)
        - stoch_threshold_low: Stochastic threshold for buy signal (default: 20)
        - stoch_threshold_high: Stochastic threshold for sell signal (default: 80)
        - cci_threshold_low: CCI threshold for buy signal (default: -100)
        - cci_threshold_high: CCI threshold for sell signal (default: 100)
        - bb_threshold: Bollinger Bands threshold (default: 0)
        - min_signals_buy: Minimum number of buy signals required (default: 3)
        - min_signals_sell: Minimum number of sell signals required (default: 3)
        
        Returns:
        - df: DataFrame with buy/sell signals
        """
        # Make a copy of the DataFrame
        df_signals = df.copy()
        
        # Initialize buy/sell signals
        df_signals['buy_signal'] = 0
        df_signals['sell_signal'] = 0
        
        # Check if all required indicators are available
        required_indicators = ['rsi', 'macd_line', 'macd_signal', 'stoch_k', 'cci', 'bb_upper', 'bb_lower']
        missing_indicators = [ind for ind in required_indicators if ind not in df_signals.columns]
        
        if missing_indicators:
            raise ValueError(f"Missing indicators: {', '.join(missing_indicators)}")
        
        # Generate individual signals
        df_signals['rsi_buy'] = df_signals['rsi'] < rsi_threshold_low
        df_signals['rsi_sell'] = df_signals['rsi'] > rsi_threshold_high
        
        df_signals['macd_buy'] = df_signals['macd_line'] > df_signals['macd_signal']
        df_signals['macd_sell'] = df_signals['macd_line'] < df_signals['macd_signal']
        
        df_signals['stoch_buy'] = df_signals['stoch_k'] < stoch_threshold_low
        df_signals['stoch_sell'] = df_signals['stoch_k'] > stoch_threshold_high
        
        df_signals['cci_buy'] = df_signals['cci'] < cci_threshold_low
        df_signals['cci_sell'] = df_signals['cci'] > cci_threshold_high
        
        df_signals['bb_buy'] = df_signals['close'] < df_signals['bb_lower'] - bb_threshold
        df_signals['bb_sell'] = df_signals['close'] > df_signals['bb_upper'] + bb_threshold
        
        # Count buy and sell signals
        buy_columns = ['rsi_buy', 'macd_buy', 'stoch_buy', 'cci_buy', 'bb_buy']
        sell_columns = ['rsi_sell', 'macd_sell', 'stoch_sell', 'cci_sell', 'bb_sell']
        
        df_signals['buy_count'] = df_signals[buy_columns].sum(axis=1)
        df_signals['sell_count'] = df_signals[sell_columns].sum(axis=1)
        
        # Generate final signals
        df_signals.loc[df_signals['buy_count'] >= min_signals_buy, 'buy_signal'] = 1
        df_signals.loc[df_signals['sell_count'] >= min_signals_sell, 'sell_signal'] = 1
        
        # Ensure we don't have buy and sell signals on the same day
        df_signals.loc[df_signals['buy_signal'] & df_signals['sell_signal'], 'sell_signal'] = 0
        
        # Ensure we don't have consecutive signals of the same type
        for i in range(1, len(df_signals)):
            if df_signals['buy_signal'].iloc[i] == 1 and df_signals['buy_signal'].iloc[i-1] == 1:
                df_signals.loc[df_signals.index[i], 'buy_signal'] = 0
            if df_signals['sell_signal'].iloc[i] == 1 and df_signals['sell_signal'].iloc[i-1] == 1:
                df_signals.loc[df_signals.index[i], 'sell_signal'] = 0
        
        return df_signals

# Example usage
def example_usage():
    """Example of how to use the BacktestingEngine class"""
    
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
    df['macd_line'] = np.random.normal(0, 1, 500)  # Simplified MACD
    df['macd_signal'] = np.random.normal(0, 1, 500)  # Simplified MACD signal
    df['stoch_k'] = np.random.normal(50, 20, 500)  # Simplified Stochastic
    df['cci'] = np.random.normal(0, 100, 500)  # Simplified CCI
    df['bb_upper'] = df['close'] + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['close'] - 2 * df['close'].rolling(window=20).std()
    
    # Drop NaN values
    df = df.dropna()
    
    # Create a BacktestingEngine instance
    engine = BacktestingEngine(initial_capital=100000, commission=0.001)
    
    # Run backtest
    results = engine.run_backtest(df, PerfectStormStrategy.generate_signals)
    
    # Print metrics
    engine.print_metrics()
    
    # Plot results
    engine.plot_results()
    
    # Optimize parameters
    param_grid = {
        'rsi_threshold_low': [30, 35, 40, 45],
        'rsi_threshold_high': [60, 65, 70, 75],
        'min_signals_buy': [2, 3, 4],
        'min_signals_sell': [2, 3, 4]
    }
    
    best_params, best_metrics = engine.optimize_parameters(df, PerfectStormStrategy.generate_signals, param_grid)
    
    print("\n=== Best Parameters ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print("\n=== Best Metrics ===")
    for metric, value in best_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

if __name__ == '__main__':
    example_usage()
