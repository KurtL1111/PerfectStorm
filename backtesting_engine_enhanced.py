"""
Enhanced Backtesting Framework for Perfect Storm Dashboard

This module implements a comprehensive backtesting system that allows users to:
1. Test the Perfect Storm strategy against historical data
2. Calculate key performance metrics (Sharpe ratio, drawdown, win rate)
3. Compare performance against benchmark indices
4. Optimize indicator parameters for specific assets or market conditions
5. Implement walk-forward testing and Monte Carlo simulations
6. Analyze strategy robustness across different market regimes
7. Visualize performance with advanced charts and analytics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.cluster import KMeans
import itertools
import os
import pickle
import warnings
import joblib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import empyrical
warnings.filterwarnings('ignore')

class BacktestingEngine:
    """Enhanced class for backtesting trading strategies"""
    
    def __init__(self, initial_capital=100000, commission=0.001, risk_free_rate=0.02, 
                 slippage_model='fixed', slippage_value=0.0005, position_sizing='fixed',
                 max_position_size=0.2, stop_loss=None, take_profit=None):
        """
        Initialize the BacktestingEngine class
        
        Parameters:
        - initial_capital: Initial capital for backtesting (default: 100000)
        - commission: Commission rate for trades (default: 0.001, i.e., 0.1%)
        - risk_free_rate: Annual risk-free rate for performance metrics (default: 0.02, i.e., 2%)
        - slippage_model: Model for slippage ('fixed', 'percentage', 'volatility', default: 'fixed')
        - slippage_value: Value for slippage model (default: 0.0005, i.e., 0.05%)
        - position_sizing: Method for position sizing ('fixed', 'percent', 'kelly', 'volatility', default: 'fixed')
        - max_position_size: Maximum position size as fraction of portfolio (default: 0.2, i.e., 20%)
        - stop_loss: Stop loss as fraction of entry price (default: None)
        - take_profit: Take profit as fraction of entry price (default: None)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Initialize results
        self.results = None
        self.trades = None
        self.metrics = None
        self.benchmark_results = None
        self.optimization_results = None
        self.monte_carlo_results = None
        self.regime_analysis = None
        self.walk_forward_results = None
    
    def run_backtest(self, df, strategy_func, strategy_params=None, benchmark_col=None,
                    regime_labels=None, rebalance_freq=None, risk_manager=None):
        """
        Run a backtest on historical data
        
        Parameters:
        - df: DataFrame with market data
        - strategy_func: Function that generates buy/sell signals
        - strategy_params: Dictionary of parameters for the strategy function (default: None)
        - benchmark_col: Column to use as benchmark (default: None)
        - regime_labels: Column with market regime labels (default: None)
        - rebalance_freq: Frequency for portfolio rebalancing ('daily', 'weekly', 'monthly', default: None)
        - risk_manager: Function for dynamic risk management (default: None)
        
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
        
        # Add signal strength if available
        if 'signal_strength' in df_backtest.columns:
            results['signal_strength'] = df_backtest['signal_strength']
        else:
            results['signal_strength'] = 1.0  # Default signal strength
        
        # Add market regime if available
        if regime_labels is not None and regime_labels in df_backtest.columns:
            results['market_regime'] = df_backtest[regime_labels]
        
        # Initialize position and portfolio value
        results['position'] = 0
        results['cash'] = self.initial_capital
        results['holdings'] = 0
        results['portfolio_value'] = self.initial_capital
        
        # Initialize trade log
        trades = []
        
        # Initialize stop loss and take profit tracking
        active_trades = {}  # Dictionary to track active trades for stop loss/take profit
        
        # Determine rebalance dates if rebalance frequency is specified
        rebalance_dates = None
        if rebalance_freq is not None:
            if rebalance_freq == 'daily':
                rebalance_dates = results.index
            elif rebalance_freq == 'weekly':
                rebalance_dates = results.index[results.index.weekday == 4]  # Friday
            elif rebalance_freq == 'monthly':
                rebalance_dates = results.index[results.index.day == 1]  # First day of month
        
        # Simulate trading
        for i in range(1, len(results)):
            current_date = results.index[i]
            prev_date = results.index[i-1]
            
            # Default: carry forward previous position and portfolio value
            results.loc[current_date, 'position'] = results.loc[prev_date, 'position']
            results.loc[current_date, 'cash'] = results.loc[prev_date, 'cash']
            
            # Update holdings value based on current price
            current_price = results.loc[current_date, 'close']
            results.loc[current_date, 'holdings'] = results.loc[current_date, 'position'] * current_price
            
            # Check for stop loss or take profit if active trades exist
            if self.stop_loss is not None or self.take_profit is not None:
                for trade_id, trade_info in list(active_trades.items()):
                    entry_price = trade_info['entry_price']
                    position = trade_info['position']
                    
                    # Check stop loss
                    if self.stop_loss is not None:
                        stop_price = entry_price * (1 - self.stop_loss)
                        if current_price <= stop_price:
                            # Execute stop loss
                            value = position * current_price
                            commission_cost = value * self.commission
                            
                            # Update position and cash
                            results.loc[current_date, 'position'] -= position
                            results.loc[current_date, 'cash'] += value - commission_cost
                            results.loc[current_date, 'holdings'] = results.loc[current_date, 'position'] * current_price
                            
                            # Log trade
                            trades.append({
                                'date': current_date,
                                'type': 'stop_loss',
                                'price': current_price,
                                'shares': position,
                                'value': value,
                                'commission': commission_cost,
                                'pnl': (current_price - entry_price) * position - commission_cost
                            })
                            
                            # Remove from active trades
                            del active_trades[trade_id]
                            continue
                    
                    # Check take profit
                    if self.take_profit is not None:
                        take_profit_price = entry_price * (1 + self.take_profit)
                        if current_price >= take_profit_price:
                            # Execute take profit
                            value = position * current_price
                            commission_cost = value * self.commission
                            
                            # Update position and cash
                            results.loc[current_date, 'position'] -= position
                            results.loc[current_date, 'cash'] += value - commission_cost
                            results.loc[current_date, 'holdings'] = results.loc[current_date, 'position'] * current_price
                            
                            # Log trade
                            trades.append({
                                'date': current_date,
                                'type': 'take_profit',
                                'price': current_price,
                                'shares': position,
                                'value': value,
                                'commission': commission_cost,
                                'pnl': (current_price - entry_price) * position - commission_cost
                            })
                            
                            # Remove from active trades
                            del active_trades[trade_id]
                            continue
            
            # Apply dynamic risk management if provided
            if risk_manager is not None:
                risk_action = risk_manager(df_backtest.iloc[:i+1], results.iloc[:i+1])
                if risk_action == 'exit_all' and results.loc[prev_date, 'position'] > 0:
                    # Force exit all positions
                    price = results.loc[current_date, 'close']
                    shares = results.loc[prev_date, 'position']
                    value = shares * price
                    commission_cost = value * self.commission
                    
                    # Update position and cash
                    results.loc[current_date, 'position'] = 0
                    results.loc[current_date, 'cash'] = results.loc[prev_date, 'cash'] + value - commission_cost
                    results.loc[current_date, 'holdings'] = 0
                    
                    # Log trade
                    trades.append({
                        'date': current_date,
                        'type': 'risk_exit',
                        'price': price,
                        'shares': shares,
                        'value': value,
                        'commission': commission_cost
                    })
                    
                    # Skip to next iteration
                    results.loc[current_date, 'portfolio_value'] = results.loc[current_date, 'cash'] + results.loc[current_date, 'holdings']
                    continue
            
            # Check for rebalancing
            if rebalance_dates is not None and current_date in rebalance_dates:
                # Sell all positions
                if results.loc[prev_date, 'position'] > 0:
                    price = results.loc[current_date, 'close']
                    shares = results.loc[prev_date, 'position']
                    value = shares * price
                    commission_cost = value * self.commission
                    
                    # Update position and cash
                    results.loc[current_date, 'position'] = 0
                    results.loc[current_date, 'cash'] = results.loc[prev_date, 'cash'] + value - commission_cost
                    results.loc[current_date, 'holdings'] = 0
                    
                    # Log trade
                    trades.append({
                        'date': current_date,
                        'type': 'rebalance_sell',
                        'price': price,
                        'shares': shares,
                        'value': value,
                        'commission': commission_cost
                    })
                
                # Re-evaluate buy signal
                if results.loc[current_date, 'buy_signal'] == 1:
                    # Calculate position size
                    price = results.loc[current_date, 'close']
                    available_cash = results.loc[current_date, 'cash']
                    position_size = self._calculate_position_size(
                        available_cash, price, results.loc[current_date, 'signal_strength'],
                        df_backtest.iloc[:i+1]
                    )
                    
                    # Apply slippage
                    execution_price = self._apply_slippage(price, 'buy', df_backtest.iloc[:i+1])
                    
                    # Calculate shares and commission
                    shares = position_size / execution_price
                    value = shares * execution_price
                    commission_cost = value * self.commission
                    
                    # Update position and cash
                    results.loc[current_date, 'position'] = shares
                    results.loc[current_date, 'cash'] -= value + commission_cost
                    results.loc[current_date, 'holdings'] = shares * price
                    
                    # Log trade
                    trade_id = len(trades)
                    trades.append({
                        'date': current_date,
                        'type': 'rebalance_buy',
                        'price': execution_price,
                        'shares': shares,
                        'value': value,
                        'commission': commission_cost
                    })
                    
                    # Add to active trades for stop loss/take profit tracking
                    if self.stop_loss is not None or self.take_profit is not None:
                        active_trades[trade_id] = {
                            'entry_price': execution_price,
                            'position': shares
                        }
            
            # Check for buy signal
            elif results.loc[current_date, 'buy_signal'] == 1 and results.loc[prev_date, 'position'] == 0:
                # Calculate position size
                price = results.loc[current_date, 'close']
                available_cash = results.loc[current_date, 'cash']
                position_size = self._calculate_position_size(
                    available_cash, price, results.loc[current_date, 'signal_strength'],
                    df_backtest.iloc[:i+1]
                )
                
                # Apply slippage
                execution_price = self._apply_slippage(price, 'buy', df_backtest.iloc[:i+1])
                
                # Calculate shares and commission
                shares = position_size / execution_price
                value = shares * execution_price
                commission_cost = value * self.commission
                
                # Update position and cash
                results.loc[current_d<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>