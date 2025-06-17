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
# import empyrical - Consider using this for more metrics if installed

class BacktestingEngine:
    # ... (constructor remains the same) ...
    def __init__(self, initial_capital=50000, commission=0.001, risk_free_rate=0.02,
                 slippage_model='fixed', slippage_value=0.0005, position_sizing='fixed',
                 max_position_size=0.5, stop_loss=None, take_profit=None):
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.position_sizing_method = position_sizing # Renamed attribute
        self.max_position_size = max_position_size
        self.base_stop_loss = stop_loss # Store base value
        self.base_take_profit = take_profit # Store base value
        self.current_stop_loss = stop_loss
        self.current_take_profit = take_profit

        # Initialize results
        self.results = None
        self.trades = None
        self.metrics = None
        self.benchmark_results = None
        self.optimization_results = None
        self.monte_carlo_results = None
        self.regime_analysis = None
        self.walk_forward_results = None

    # --- Add method to update risk parameters dynamically ---
    def update_risk_parameters(self, stop_loss=None, take_profit=None):
        """
        Updates current stop-loss and take-profit levels.
        
        Args:
            stop_loss (float, optional): New stop-loss value.
            take_profit (float, optional): New take-profit value.
        """
        if stop_loss is not None:
            self.current_stop_loss = stop_loss
        if take_profit is not None:
            self.current_take_profit = take_profit
        #print(f"Risk parameters updated: SL={self.current_stop_loss}, TP={self.current_take_profit}")

    # --- Modify run_backtest ---
    def run_backtest(self, df, strategy_func=None, strategy_params=None, benchmark_col=None, rebalance_freq=None, risk_manager=None, regime_col='market_regime', **kwargs):
        """
        Run a backtest on historical data using pre-calculated signals.

        Args:
            df (pd.DataFrame): DataFrame WITH 'buy_signal' and 'sell_signal' columns. Optionally includes 'signal_strength' and regime_col.
            strategy_func (callable, optional): DEPRECATED - signals should be pre-calculated. Kept for backward compat.
            strategy_params (dict, optional): DEPRECATED.
            benchmark_col (str, optional): Column for benchmark comparison.
            rebalance_freq (str, optional): Rebalancing frequency ('daily', 'weekly', 'monthly').
            risk_manager (callable, optional): Custom risk management function.
            regime_col (str, optional): Column name containing market regime information.
            **kwargs: Additional arguments for compatibility.

        Returns:
            pd.DataFrame: DataFrame with backtest results.
        """
        # If a strategy_func is provided, apply it to generate signals (for backward compatibility)
        df_backtest = df.copy()
        if strategy_func is not None and (('buy_signal' not in df_backtest.columns) or ('sell_signal' not in df_backtest.columns)):
            try:
                df_backtest = strategy_func(df_backtest, strategy_params)
            except Exception:
                pass

        # Ensure required signal columns exist
        if 'buy_signal' not in df_backtest.columns or 'sell_signal' not in df_backtest.columns:
            raise ValueError("Input DataFrame must contain 'buy_signal' and 'sell_signal' columns.")

        # Robustly coerce columns to numeric types where needed
        results = pd.DataFrame(index=df_backtest.index)
        results['close'] = pd.to_numeric(df_backtest['close'], errors='coerce')
        results['buy_signal'] = pd.to_numeric(df_backtest['buy_signal'], errors='coerce').fillna(0).astype(int)
        results['sell_signal'] = pd.to_numeric(df_backtest['sell_signal'], errors='coerce').fillna(0).astype(int)
        # Robustly handle signal_strength: if column missing, use 1.0 for all rows
        if 'signal_strength' in df_backtest.columns:
            results['signal_strength'] = pd.to_numeric(df_backtest['signal_strength'], errors='coerce').fillna(1.0)
        else:
            results['signal_strength'] = 1.0  # Will be broadcast to all rows
        results['market_regime'] = df_backtest.get(regime_col, 'unknown')

        results['position'] = 0.0
        results['cash'] = float(self.initial_capital)
        results['holdings'] = 0.0
        results['portfolio_value'] = float(self.initial_capital)

        trades = []
        active_trades = {}
        current_position = 0
        trade_id_counter = 0

        rebalance_dates = None
        if rebalance_freq is not None:
            if rebalance_freq == 'daily':
                rebalance_dates = results.index
            elif rebalance_freq == 'weekly':
                dt_index = pd.to_datetime(results.index)
                rebalance_dates = results.index[pd.DatetimeIndex(dt_index).dayofweek == 4]
            elif rebalance_freq == 'monthly':
                dt_index = pd.to_datetime(results.index)
                rebalance_dates = results.index[pd.DatetimeIndex(dt_index).day == 1]

        for i in range(1, len(results)):
            current_date = results.index[i]
            prev_date = results.index[i-1]
            current_price = results.loc[current_date, 'close']
            current_low = df_backtest['low'].iloc[i] if 'low' in df_backtest.columns else current_price
            current_high = df_backtest['high'].iloc[i] if 'high' in df_backtest.columns else current_price

            # --- NEW: Adjust Risk Parameters Based on Regime ---
            current_regime = results.loc[current_date, 'market_regime']
            if 'trending' in str(current_regime):
                self.update_risk_parameters(stop_loss=self.base_stop_loss * 1.5 if self.base_stop_loss else None,
                                           take_profit=self.base_take_profit * 1.5 if self.base_take_profit else None)
            elif 'ranging' in str(current_regime):
                self.update_risk_parameters(stop_loss=self.base_stop_loss * 0.7 if self.base_stop_loss else None,
                                            take_profit=self.base_take_profit * 0.7 if self.base_take_profit else None)
            elif 'volatile' in str(current_regime):
                self.update_risk_parameters(stop_loss=self.base_stop_loss * 2.0 if self.base_stop_loss else None,
                                           take_profit=self.base_take_profit * 2.0 if self.base_take_profit else None)
            else:
                self.update_risk_parameters(stop_loss=self.base_stop_loss, take_profit=self.base_take_profit)

            # Carry forward values robustly
            prev_position = results.loc[prev_date, 'position']
            prev_cash = results.loc[prev_date, 'cash']
            try:
                prev_position_val = float(np.asarray(prev_position)) if pd.notna(prev_position) else 0.0
            except Exception:
                prev_position_val = 0.0
            try:
                prev_cash_val = float(np.asarray(prev_cash)) if pd.notna(prev_cash) else 0.0
            except Exception:
                prev_cash_val = 0.0
            results.loc[current_date, 'position'] = prev_position_val
            results.loc[current_date, 'cash'] = prev_cash_val
            try:
                position_val = float(np.asarray(results.loc[current_date, 'position']))
            except Exception:
                position_val = 0.0
            try:
                price_val = float(np.asarray(current_price)) if pd.notna(current_price) else 0.0
            except Exception:
                price_val = 0.0
            results.loc[current_date, 'holdings'] = position_val * price_val

            # Check for stop loss/take profit
            if current_position > 0 and active_trades:
                exited_via_sl_tp = False
                trade_keys = list(active_trades.keys())
                for trade_id in trade_keys:
                    if trade_id not in active_trades:
                        continue
                    trade_info = active_trades[trade_id]
                    entry_price = trade_info['entry_price']
                    position_size = trade_info['position']
                    exit_price = None
                    exit_type = None
                    if self.current_stop_loss is not None:
                        stop_price = entry_price * (1 - self.current_stop_loss)
                        if current_low <= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop_loss'
                    if self.current_take_profit is not None:
                        take_profit_price = entry_price * (1 + self.current_take_profit)
                        if current_high >= take_profit_price:
                            exit_price = take_profit_price
                            exit_type = 'take_profit'
                    if exit_type:
                        try:
                            position_size_val = float(np.asarray(position_size))
                        except Exception:
                            position_size_val = 0.0
                        try:
                            exit_price_val = float(np.asarray(exit_price))
                        except Exception:
                            exit_price_val = 0.0
                        value = position_size_val * exit_price_val
                        commission_cost = value * self.commission
                        pnl = (exit_price_val - entry_price) * position_size_val - commission_cost
                        results.loc[current_date, 'position'] = float(np.asarray(results.loc[current_date, 'position'])) - position_size_val
                        results.loc[current_date, 'cash'] = float(np.asarray(results.loc[current_date, 'cash'])) + value - commission_cost
                        current_position = 0
                        trades.append({
                            'date': current_date, 'type': exit_type, 'price': exit_price_val,
                            'shares': -position_size_val, 'value': value, 'commission': commission_cost, 'pnl': pnl,
                            'trade_id': trade_id
                        })
                        del active_trades[trade_id]
                        exited_via_sl_tp = True
                if exited_via_sl_tp:
                    try:
                        position_val = float(np.asarray(results.loc[current_date, 'position']))
                    except Exception:
                        position_val = 0.0
                    try:
                        price_val = float(np.asarray(current_price)) if pd.notna(current_price) else 0.0
                    except Exception:
                        price_val = 0.0
                    results.loc[current_date, 'holdings'] = position_val * price_val
                    try:
                        cash_val = float(np.asarray(results.loc[current_date, 'cash']))
                    except Exception:
                        cash_val = 0.0
                    try:
                        holdings_val = float(np.asarray(results.loc[current_date, 'holdings']))
                    except Exception:
                        holdings_val = 0.0
                    results.loc[current_date, 'portfolio_value'] = cash_val + holdings_val
                    continue

            # Apply dynamic risk management
            if risk_manager is not None:
                risk_action = risk_manager(df_backtest.iloc[:i+1], results.iloc[:i+1])
                try:
                    prev_position_val = float(np.asarray(results.loc[prev_date, 'position']))
                except Exception:
                    prev_position_val = 0.0
                if risk_action == 'exit_all' and prev_position_val > 0:
                    try:
                        price = float(np.asarray(results.loc[current_date, 'close']))
                    except Exception:
                        price = 0.0
                    try:
                        shares = float(np.asarray(results.loc[prev_date, 'position']))
                    except Exception:
                        shares = 0.0
                    value = shares * price
                    commission_cost = value * self.commission
                    results.loc[current_date, 'position'] = 0.0
                    try:
                        prev_cash_val = float(np.asarray(results.loc[prev_date, 'cash']))
                    except Exception:
                        prev_cash_val = 0.0
                    results.loc[current_date, 'cash'] = prev_cash_val + value - commission_cost
                    results.loc[current_date, 'holdings'] = 0.0
                    try:
                        prev_close_val = float(np.asarray(results.loc[prev_date, 'close']))
                    except Exception:
                        prev_close_val = 0.0
                    trades.append({
                        'date': current_date,
                        'type': 'risk_exit',
                        'price': price,
                        'shares': shares,
                        'value': value,
                        'commission': commission_cost,
                        'pnl': value - shares * prev_close_val - commission_cost
                    })
                    active_trades = {}
                    continue

            # Check for rebalancing
            if rebalance_dates is not None and current_date in rebalance_dates:
                try:
                    prev_position_val = float(np.asarray(results.loc[prev_date, 'position']))
                except Exception:
                    prev_position_val = 0.0
                if prev_position_val > 0:
                    try:
                        price = float(np.asarray(results.loc[current_date, 'close']))
                    except Exception:
                        price = 0.0
                    try:
                        shares = float(np.asarray(results.loc[prev_date, 'position']))
                    except Exception:
                        shares = 0.0
                    value = shares * price
                    commission_cost = value * self.commission
                    results.loc[current_date, 'position'] = 0.0
                    try:
                        prev_cash_val = float(np.asarray(results.loc[prev_date, 'cash']))
                    except Exception:
                        prev_cash_val = 0.0
                    results.loc[current_date, 'cash'] = prev_cash_val + value - commission_cost
                    results.loc[current_date, 'holdings'] = 0.0
                    try:
                        prev_close_val = float(np.asarray(results.loc[prev_date, 'close']))
                    except Exception:
                        prev_close_val = 0.0
                    trades.append({
                        'date': current_date,
                        'type': 'rebalance_exit',
                        'price': price,
                        'shares': shares,
                        'value': value,
                        'commission': commission_cost,
                        'pnl': value - shares * prev_close_val - commission_cost
                    })
                    active_trades = {}

            is_buy_signal = int(np.asarray(results.loc[current_date, 'buy_signal'])) == 1
            is_sell_signal = int(np.asarray(results.loc[current_date, 'sell_signal'])) == 1

            if is_buy_signal and current_position == 0:
                try:
                    portfolio_value = float(np.asarray(results.loc[prev_date, 'portfolio_value']))
                except Exception:
                    portfolio_value = float(self.initial_capital)
                try:
                    cash_available = float(np.asarray(results.loc[prev_date, 'cash']))
                except Exception:
                    cash_available = float(self.initial_capital)
                if self.position_sizing_method == 'fixed':
                    investment = portfolio_value * self.max_position_size
                elif getattr(self, 'position_sizing', None) == 'percent':
                    position_value = portfolio_value * self.max_position_size
                    investment = position_value
                elif getattr(self, 'position_sizing', None) == 'kelly':
                    win_rate = 0.5
                    win_loss_ratio = 1.5
                    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
                    kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
                    investment = portfolio_value * kelly_fraction
                elif getattr(self, 'position_sizing', None) == 'volatility':
                    if 'volatility' in df_backtest.columns:
                        volatility = df_backtest.loc[current_date, 'volatility']
                        try:
                            volatility = float(np.asarray(volatility))
                        except Exception:
                            volatility = 0.01
                        vol_factor = 0.1 / max(0.01, volatility)
                        investment = portfolio_value * min(self.max_position_size, vol_factor)
                    else:
                        investment = portfolio_value * self.max_position_size
                else:
                    investment = portfolio_value * self.max_position_size
                try:
                    investment = min(float(investment), float(cash_available))
                except Exception:
                    investment = float(self.initial_capital)
                try:
                    signal_strength = float(np.asarray(results.loc[current_date, 'signal_strength']))
                except Exception:
                    signal_strength = 1.0
                investment *= signal_strength
                if self.slippage_model == 'fixed':
                    entry_price_slippage = float(np.asarray(current_price)) * (1 + self.slippage_value)
                elif self.slippage_model == 'percentage':
                    entry_price_slippage = float(np.asarray(current_price)) * (1 + self.slippage_value)
                elif self.slippage_model == 'volatility':
                    if 'volatility' in df_backtest.columns:
                        volatility = df_backtest.loc[current_date, 'volatility']
                        try:
                            volatility = float(np.asarray(volatility))
                        except Exception:
                            volatility = 0.01
                        entry_price_slippage = float(np.asarray(current_price)) * (1 + self.slippage_value * volatility)
                    else:
                        entry_price_slippage = float(np.asarray(current_price)) * (1 + self.slippage_value)
                else:
                    entry_price_slippage = float(np.asarray(current_price)) * (1 + self.slippage_value)
                if entry_price_slippage > 0:
                    try:
                        shares = int(investment / entry_price_slippage)
                    except Exception:
                        shares = 0
                    if shares > 0:
                        value = shares * entry_price_slippage
                        commission_cost = value * self.commission
                        results.loc[current_date, 'position'] = float(shares)
                        results.loc[current_date, 'cash'] = float(np.asarray(results.loc[current_date, 'cash'])) - (value + commission_cost)
                        results.loc[current_date, 'holdings'] = float(shares) * float(np.asarray(current_price))
                        current_position = 1
                        trade_id_counter += 1
                        trades.append({
                             'date': current_date, 'type': 'buy', 'price': entry_price_slippage,
                             'shares': shares, 'value': value, 'commission': commission_cost, 'pnl': -commission_cost,
                             'trade_id': trade_id_counter
                         })
                        active_trades[trade_id_counter] = {'entry_price': entry_price_slippage, 'position': shares}

            elif is_sell_signal and current_position > 0:
                try:
                    shares = float(np.asarray(results.loc[prev_date, 'position']))
                except Exception:
                    shares = 0.0
                if self.slippage_model == 'fixed':
                    exit_price_slippage = float(np.asarray(current_price)) * (1 - self.slippage_value)
                elif self.slippage_model == 'percentage':
                    exit_price_slippage = float(np.asarray(current_price)) * (1 - self.slippage_value)
                elif self.slippage_model == 'volatility':
                    if 'volatility' in df_backtest.columns:
                        volatility = df_backtest.loc[current_date, 'volatility']
                        try:
                            volatility = float(np.asarray(volatility))
                        except Exception:
                            volatility = 0.01
                        exit_price_slippage = float(np.asarray(current_price)) * (1 - self.slippage_value * volatility)
                    else:
                        exit_price_slippage = float(np.asarray(current_price)) * (1 - self.slippage_value)
                else:
                    exit_price_slippage = float(np.asarray(current_price)) * (1 - self.slippage_value)
                value = shares * exit_price_slippage
                commission_cost = value * self.commission
                results.loc[current_date, 'position'] = 0.0
                results.loc[current_date, 'cash'] = float(np.asarray(results.loc[current_date, 'cash'])) + (value - commission_cost)
                results.loc[current_date, 'holdings'] = 0.0
                current_position = 0
                pnl_total = -commission_cost
                entry_total_value = 0.0
                trade_ids_closed = list(active_trades.keys())
                for trade_id, trade_info in active_trades.items():
                    try:
                        entry_price_val = float(np.asarray(trade_info['entry_price']))
                    except Exception:
                        entry_price_val = 0.0
                    try:
                        position_val = float(np.asarray(trade_info['position']))
                    except Exception:
                        position_val = 0.0
                    pnl_trade = (exit_price_slippage - entry_price_val) * position_val
                    pnl_total += pnl_trade
                    entry_total_value += entry_price_val * position_val
                trades.append({
                    'date': current_date, 'type': 'sell', 'price': exit_price_slippage,
                    'shares': -shares, 'value': value, 'commission': commission_cost, 'pnl': pnl_total,
                    'trade_id': trade_ids_closed
                 })
                active_trades = {}

            try:
                position_val = float(np.asarray(results.loc[current_date, 'position']))
            except Exception:
                position_val = 0.0
            try:
                price_val = float(np.asarray(current_price)) if pd.notna(current_price) else 0.0
            except Exception:
                price_val = 0.0
            results.loc[current_date, 'holdings'] = position_val * price_val
            try:
                cash_val = float(np.asarray(results.loc[current_date, 'cash']))
            except Exception:
                cash_val = 0.0
            try:
                holdings_val = float(np.asarray(results.loc[current_date, 'holdings']))
            except Exception:
                holdings_val = 0.0
            results.loc[current_date, 'portfolio_value'] = cash_val + holdings_val

        # Compute daily returns for performance metrics
        results['daily_return'] = results['portfolio_value'].pct_change().fillna(0)
        self.results = results
        self.trades = pd.DataFrame(trades)
        self.calculate_performance_metrics()
        return self.results

    # --- (calculate_performance_metrics remains the same) ---
    # --- (optimize_parameters remains the same, but use conviction thresholds/weights as params) ---
    # --- (walk_forward_optimization remains the same structure) ---
    # --- (run_monte_carlo_simulation remains the same structure) ---
    # --- (analyze_market_regimes remains the same structure) ---
    # --- (Plotting functions remain the same structure) ---
    # --- (save/load results remain the same structure) ---
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the backtest results and trades.

        Returns:
            dict: Dictionary of calculated performance metrics.
        """
        # Ensure self.results and self.trades are populated before calculation
        if self.results is None or self.trades is None:
            print("Warning: Cannot calculate metrics. Run backtest first.")
            self.metrics = {}
            return self.metrics

        returns = self.results['daily_return'].fillna(0) # Fill NaNs in returns
        if len(returns) < 2:
            self.metrics = {} # Not enough data
            return self.metrics

        trading_days_per_year = 252
        total_days = len(returns)
        years = total_days / trading_days_per_year

        total_return = (self.results['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(trading_days_per_year)

        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / trading_days_per_year)
        sharpe_ratio = (excess_returns.mean() * trading_days_per_year) / volatility if volatility > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(trading_days_per_year)
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Trade Stats
        num_trades = len(self.trades) // 2 # Assuming each trade has buy/sell or entry/exit
        win_rate = 0
        profit_factor = 0
        if num_trades > 0 and 'pnl' in self.trades:
            winning_trades = self.trades[self.trades['pnl'] > 0]
            losing_trades = self.trades[self.trades['pnl'] <= 0] # Include commission-only losses
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            gross_profit = winning_trades['pnl'].sum()
            gross_loss = abs(losing_trades['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        # Calculate benchmark metrics if available
        benchmark_metrics = {}
        if 'benchmark_return' in self.results.columns:
            benchmark_returns = self.results['benchmark_return'].dropna()
            
            # Benchmark total return
            benchmark_total_return = self.results['benchmark_cumulative_return'].iloc[-1]
            
            # Benchmark annualized return
            benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / years) - 1
            
            # Benchmark volatility
            benchmark_volatility = benchmark_returns.std() * np.sqrt(trading_days_per_year)
            
            # Benchmark Sharpe ratio
            benchmark_excess_returns = benchmark_returns - self.risk_free_rate / trading_days_per_year
            benchmark_sharpe_ratio = benchmark_excess_returns.mean() / benchmark_returns.std() * np.sqrt(trading_days_per_year)
            
            # Benchmark maximum drawdown
            benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
            benchmark_running_max = benchmark_cumulative_returns.cummax()
            benchmark_drawdown = (benchmark_cumulative_returns / benchmark_running_max - 1)
            benchmark_max_drawdown = benchmark_drawdown.min()
            
            # Alpha and beta
            covariance = np.cov(returns.values, benchmark_returns.values)[0, 1]
            variance = np.var(benchmark_returns.values)
            beta = covariance / variance if variance > 0 else 0
            alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_annualized_return - self.risk_free_rate))
            
            # Information ratio
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(trading_days_per_year)
            information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error if tracking_error > 0 else 0
            
            # Store benchmark metrics
            benchmark_metrics = {
                'benchmark_total_return': benchmark_total_return,
                'benchmark_annualized_return': benchmark_annualized_return,
                'benchmark_volatility': benchmark_volatility,
                'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
                'benchmark_max_drawdown': benchmark_max_drawdown,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            }
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            # Add more metrics from empyrical if desired
        }
        # Add benchmark metrics if available
        metrics.update(benchmark_metrics)
        
        # Store metrics
        self.metrics = metrics

        return metrics
    def optimize_parameters(self, df, strategy_func, param_grid, metric='sharpe_ratio',
                           n_jobs=None, verbose=True):
        """
        Optimize strategy parameters using grid search.
        """
        # Create parameter grid
        param_combinations = list(ParameterGrid(param_grid))
        n_combinations = len(param_combinations)
        
        if verbose:
            print(f"Optimizing parameters with {n_combinations} combinations")
        
        # Set number of parallel jobs
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)

        # Run optimization
        results = []
        if n_jobs > 1:
            # Prepare arguments for static method
            engine_state = self.get_engine_state()
            args_list = [(engine_state, df, strategy_func, params, metric) for params in param_combinations]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                if verbose:
                    results = list(tqdm(executor.map(BacktestingEngine._evaluate_params_static, args_list), total=n_combinations))
                else:
                    results = list(executor.map(BacktestingEngine._evaluate_params_static, args_list))
        else:
            # Sequential processing (use self)
            for params in tqdm(param_combinations) if verbose else param_combinations:
                try:
                    self.run_backtest(df, strategy_func, **params)
                    if metric in self.metrics:
                        score = self.metrics[metric]
                    else:
                        score = float('-inf')
                    if 'drawdown' in metric:
                        score = -score
                    results.append({**params, 'score': score, 'metric': metric})
                except Exception as e:
                    print(f"Error evaluating parameters {params}: {e}")
                    results.append({**params, 'score': float('-inf'), 'metric': metric})
        
        # Convert results to DataFrame
        optimization_results = pd.DataFrame(results)
        
        # Sort by score
        optimization_results = optimization_results.sort_values('score', ascending=False)
        
        # Get best parameters
        best_params = optimization_results.iloc[0].drop(['score', 'metric']).to_dict()
        
        # Store optimization results
        self.optimization_results = optimization_results
        
        if verbose:
            print(f"Best parameters: {best_params}")
            print(f"Best {metric}: {optimization_results.iloc[0]['score']}")
        
        return best_params, optimization_results
        
        # Static method and get_engine_state are now at the class level
    @staticmethod
    def _evaluate_params_static(args):
        """
        Static helper for multiprocessing parameter evaluation.
        """
        engine_state, df, strategy_func, params, metric = args
        engine = BacktestingEngine(**engine_state)
        try:
            engine.run_backtest(df, strategy_func, **params)
            if metric in engine.metrics:
                score = engine.metrics[metric]
            else:
                score = float('-inf')
            if 'drawdown' in metric:
                score = -score
            return {**params, 'score': score, 'metric': metric}
        except Exception as e:
            print(f"Error evaluating parameters {params}: {e}")
            return {**params, 'score': float('-inf'), 'metric': metric}

    def get_engine_state(self):
        """
        Helper to serialize engine state for multiprocessing.
        """
        return dict(
            initial_capital=self.initial_capital,
            commission=self.commission,
            risk_free_rate=self.risk_free_rate,
            slippage_model=self.slippage_model,
            slippage_value=self.slippage_value,
            position_sizing=self.position_sizing_method,
            max_position_size=self.max_position_size,
            stop_loss=self.base_stop_loss,
            take_profit=self.base_take_profit
        )
    
    def walk_forward_optimization(self, df, strategy_func, param_grid, window_size=252,
                                 step_size=63, metric='sharpe_ratio', n_jobs=None, verbose=True):
        """
        Perform walk-forward optimization.
        """
        # Calculate number of windows
        total_days = len(df)
        n_windows = max(1, (total_days - window_size) // step_size + 1)
        
        if verbose:
            print(f"Performing walk-forward optimization with {n_windows} windows")
        
        # Initialize results
        walk_forward_results = []
        
        # Iterate through windows
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > total_days:
                end_idx = total_days
            
            # Get in-sample and out-of-sample data
            in_sample = df.iloc[start_idx:end_idx]
            
            # Skip window if not enough data
            if len(in_sample) < window_size // 2:
                continue
            
            # Get out-of-sample data (next step_size days or remaining days)
            out_sample_end = min(end_idx + step_size, total_days)
            out_sample = df.iloc[end_idx:out_sample_end]
            
            # Skip window if no out-of-sample data
            if len(out_sample) == 0:
                continue
            
            if verbose:
                print(f"Window {i+1}/{n_windows}: In-sample {in_sample.index[0]} to {in_sample.index[-1]}, "
                     f"Out-of-sample {out_sample.index[0]} to {out_sample.index[-1]}")
            
            # Optimize parameters on in-sample data
            best_params, _ = self.optimize_parameters(in_sample, strategy_func, param_grid, metric, n_jobs, verbose=False)
            
            # Test best parameters on out-of-sample data
            self.run_backtest(out_sample, strategy_func, strategy_params=best_params)
            
            # Store results
            window_results = {
                'window': i + 1,
                'in_sample_start': in_sample.index[0],
                'in_sample_end': in_sample.index[-1],
                'out_sample_start': out_sample.index[0],
                'out_sample_end': out_sample.index[-1],
                'best_params': best_params,
                'in_sample_metric': self.metrics[metric],
                'out_sample_metric': self.metrics[metric]
            }
            
            walk_forward_results.append(window_results)
        
        # Convert results to DataFrame
        walk_forward_df = pd.DataFrame(walk_forward_results)
        
        # Store walk-forward results
        self.walk_forward_results = walk_forward_df
        
        return walk_forward_df
    
    def run_monte_carlo_simulation(self, df, strategy_func, strategy_params=None, 
                                  n_simulations=1000, confidence_level=0.95, seed=None):
        """
        Run Monte Carlo simulation to assess strategy robustness.
        """
        # Run backtest to get base results
        self.run_backtest(df, strategy_func, strategy_params)
        
        # Extract daily returns
        returns = self.results['daily_return'].dropna()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Run Monte Carlo simulations
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Resample daily returns with replacement
            sim_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative returns
            sim_cumulative_returns = (1 + sim_returns).cumprod()
            
            # Store simulated returns
            simulated_returns.append(sim_cumulative_returns)
        
        # Convert to DataFrame
        simulated_returns_df = pd.DataFrame(simulated_returns).T
        
        # Calculate final equity values
        final_equity = simulated_returns_df.iloc[-1] * self.initial_capital
        
        # Calculate drawdowns for each simulation
        drawdowns = []
        for i in range(n_simulations):
            sim_returns = simulated_returns_df.iloc[:, i]
            running_max = sim_returns.cummax()
            drawdown = (sim_returns / running_max - 1)
            drawdowns.append(drawdown.min())
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(final_equity, 5)  # 95% VaR
        var_99 = np.percentile(final_equity, 1)  # 99% VaR
        
        cvar_95 = final_equity[final_equity <= var_95].mean()  # 95% CVaR
        cvar_99 = final_equity[final_equity <= var_99].mean()  # 99% CVaR
        
        # Calculate confidence intervals for final equity
        ci_lower = np.percentile(final_equity, (1 - confidence_level) / 2 * 100)
        ci_upper = np.percentile(final_equity, (1 + confidence_level) / 2 * 100)
        
        # Calculate confidence intervals for max drawdown
        drawdown_ci_lower = np.percentile(drawdowns, (1 - confidence_level) / 2 * 100)
        drawdown_ci_upper = np.percentile(drawdowns, (1 + confidence_level) / 2 * 100)
        
        # Store Monte Carlo results
        monte_carlo_results = {
            'simulated_returns': simulated_returns_df,
            'final_equity': final_equity,
            'drawdowns': drawdowns,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'final_equity_mean': final_equity.mean(),
            'final_equity_std': final_equity.std(),
            'final_equity_ci_lower': ci_lower,
            'final_equity_ci_upper': ci_upper,
            'max_drawdown_mean': np.mean(drawdowns),
            'max_drawdown_std': np.std(drawdowns),
            'max_drawdown_ci_lower': drawdown_ci_lower,
            'max_drawdown_ci_upper': drawdown_ci_upper,
            'n_simulations': n_simulations,
            'confidence_level': confidence_level
        }
        
        # Store Monte Carlo results
        self.monte_carlo_results = monte_carlo_results
        
        return monte_carlo_results
    
    def analyze_market_regimes(self, df, strategy_func, strategy_params=None, 
                              regime_col=None, n_regimes=3):
        """
        Analyze strategy performance across different market regimes.
        """
        # Make a copy of the DataFrame
        df_regime = df.copy()
        
        # Detect market regimes if not provided
        if regime_col is None or regime_col not in df_regime.columns:
            # Calculate features for regime detection
            df_regime['returns'] = df_regime['close'].pct_change()
            df_regime['volatility'] = df_regime['returns'].rolling(window=20).std() * np.sqrt(252)
            df_regime['trend'] = df_regime['close'].rolling(window=50).mean().pct_change(20)
            
            # Drop NaN values
            df_features = df_regime.dropna()[['volatility', 'trend']]
            
            # Standardize features
            features_std = (df_features - df_features.mean()) / df_features.std()
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            df_regime.loc[df_features.index, 'regime_cluster'] = kmeans.fit_predict(features_std)
            
            # Determine regime characteristics
            regime_stats = df_regime.groupby('regime_cluster')[['volatility', 'trend']].mean()
            
            # Label regimes
            regime_labels = [''] * n_regimes
            
            # Identify trending regime (highest absolute trend)
            trending_idx = abs(regime_stats['trend']).idxmax()
            if regime_stats.loc[trending_idx, 'trend'] > 0:
                regime_labels[trending_idx] = 'trending_up'
            else:
                regime_labels[trending_idx] = 'trending_down'
            
            # Identify volatile regime (highest volatility)
            volatile_idx = regime_stats['volatility'].idxmax()
            if volatile_idx != trending_idx:
                regime_labels[volatile_idx] = 'volatile'
            
            # Identify ranging regime (lowest volatility and trend)
            remaining_idx = [i for i in range(n_regimes) if i != trending_idx and i != volatile_idx]
            if remaining_idx:
                regime_labels[remaining_idx[0]] = 'ranging'
            
            # Map cluster numbers to regime labels
            cluster_to_regime = {i: label for i, label in enumerate(regime_labels)}
            df_regime['market_regime'] = df_regime['regime_cluster'].map(cluster_to_regime)
            
            # Forward fill regime for NaN values
            df_regime['market_regime'] = df_regime['market_regime'].ffill()
            
            # Use the new regime column
            regime_col = 'market_regime'
        
        # Run backtest with regime labels
        self.run_backtest(df_regime, strategy_func, strategy_params, regime_labels=regime_col)
        
        # Analyze performance by regime
        regime_performance = {}
        
        for regime in df_regime[regime_col].unique():
            if pd.isna(regime) or regime == '':
                continue
                
            # Get data for this regime
            regime_data = self.results[self.results[regime_col] == regime]
            
            if len(regime_data) < 5:  # Skip regimes with too few data points
                continue
            
            # Calculate regime-specific metrics
            regime_returns = regime_data['daily_return'].dropna()
            
            # Total return
            regime_total_return = (1 + regime_returns).prod() - 1
            
            # Annualized return
            trading_days_per_year = 252
            regime_days = len(regime_returns)
            regime_years = regime_days / trading_days_per_year
            regime_annualized_return = (1 + regime_total_return) ** (1 / regime_years) - 1 if regime_years > 0 else 0
            
            # Volatility
            regime_volatility = regime_returns.std() * np.sqrt(trading_days_per_year)
            
            # Sharpe ratio
            regime_excess_returns = regime_returns - self.risk_free_rate / trading_days_per_year
            regime_sharpe_ratio = regime_excess_returns.mean() / regime_returns.std() * np.sqrt(trading_days_per_year) if regime_returns.std() > 0 else 0
            
            # Maximum drawdown
            regime_cumulative_returns = (1 + regime_returns).cumprod()
            regime_running_max = regime_cumulative_returns.cummax()
            regime_drawdown = (regime_cumulative_returns / regime_running_max - 1)
            regime_max_drawdown = regime_drawdown.min()
            
            # Win rate
            regime_trades = self.trades[self.trades['date'].isin(regime_data.index)]
            regime_win_rate = len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades) if len(regime_trades) > 0 else 0
            
            # Store regime metrics
            regime_performance[regime] = {
                'total_return': regime_total_return,
                'annualized_return': regime_annualized_return,
                'volatility': regime_volatility,
                'sharpe_ratio': regime_sharpe_ratio,
                'max_drawdown': regime_max_drawdown,
                'win_rate': regime_win_rate,
                'days': regime_days,
                'years': regime_years,
                'trades': len(regime_trades)
            }
        
        # Calculate regime distribution
        regime_counts = df_regime[regime_col].value_counts()
        regime_distribution = regime_counts / regime_counts.sum()
        
        # Calculate regime transitions
        regime_transitions = pd.crosstab(
            df_regime[regime_col].shift(), 
            df_regime[regime_col], 
            normalize='index'
        )
        
        # Store regime analysis
        regime_analysis = {
            'regime_performance': regime_performance,
            'regime_distribution': regime_distribution,
            'regime_transitions': regime_transitions
        }
        
        # Store regime analysis
        self.regime_analysis = regime_analysis
        
        return regime_analysis
    
    def plot_equity_curve(self, benchmark=True, log_scale=False, figsize=(12, 6)):
        """
        Plot equity curve.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        ax.plot(self.results.index, self.results['portfolio_value'], label='Strategy')
        
        # Plot benchmark if available
        if benchmark and 'benchmark' in self.results.columns:
            # Normalize benchmark to initial capital
            benchmark_values = self.results['benchmark'] / self.results['benchmark'].iloc[0] * self.initial_capital
            ax.plot(self.results.index, benchmark_values, label='Benchmark', alpha=0.7)
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Equity Curve')
        
        # Add grid and legend
        ax.grid(True)
        ax.legend()
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        plt.close(fig)
        return fig
    
    def plot_drawdown(self, figsize=(12, 6)):
        """
        Plot drawdown.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Calculate drawdown
        equity_curve = self.results['portfolio_value']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100  # Convert to percentage
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='red', label='Drawdown')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Portfolio Drawdown')
        
        # Add grid and legend
        ax.grid(True)
        ax.legend()
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Invert y-axis (drawdowns are negative)
        ax.invert_yaxis()
        
        plt.close(fig)
        return fig
    
    def plot_monthly_returns(self, figsize=(12, 8)):
        """
        Plot monthly returns heatmap.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Calculate monthly returns
        monthly_returns = self.results['daily_return'].groupby([
            self.results.index.year, 
            self.results.index.month
        ]).apply(lambda x: (1 + x).prod() - 1) * 100  # Convert to percentage
        
        # Reshape to matrix
        monthly_returns = monthly_returns.unstack()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            monthly_returns, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn', 
            center=0, 
            linewidths=1, 
            ax=ax
        )
        
        # Add labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        ax.set_title('Monthly Returns (%)')
        
        # Set month names
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        return fig
    
    def plot_trade_analysis(self, figsize=(12, 10)):
        """
        Plot trade analysis.
        """
        if self.trades is None or len(self.trades) == 0:
            raise ValueError("No trades available. Run a backtest first.")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot trade P&L
        axes[0, 0].bar(
            range(len(self.trades)), 
            self.trades['pnl'], 
            color=['green' if pnl > 0 else 'red' for pnl in self.trades['pnl']]
        )
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('P&L')
        axes[0, 0].set_title('Trade P&L')
        axes[0, 0].grid(True)
        
        # Plot trade P&L distribution
        axes[0, 1].hist(self.trades['pnl'], bins=20, color='blue', alpha=0.7)
        axes[0, 1].axvline(0, color='black', linestyle='--')
        axes[0, 1].set_xlabel('P&L')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('P&L Distribution')
        axes[0, 1].grid(True)
        
        # Plot trade duration
        if 'date' in self.trades.columns:
            # Calculate trade durations
            trade_dates = pd.to_datetime(self.trades['date'])
            entry_dates = []
            exit_dates = []
            
            for i, row in self.trades.iterrows():
                if row['type'] in ['buy', 'entry']:
                    entry_dates.append(row['date'])
                elif row['type'] in ['sell', 'exit', 'stop_loss', 'take_profit', 'risk_exit']:
                    if entry_dates:
                        exit_dates.append((entry_dates.pop(), row['date']))
            
            durations = [(exit_date - entry_date).days for entry_date, exit_date in exit_dates]
            
            axes[1, 0].hist(durations, bins=20, color='green', alpha=0.7)
            axes[1, 0].set_xlabel('Duration (days)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Trade Duration')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Trade dates not available', ha='center', va='center')
            axes[1, 0].set_title('Trade Duration')
        
        # Plot cumulative P&L
        cumulative_pnl = self.trades['pnl'].cumsum()
        axes[1, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, color='blue')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative P&L')
        axes[1, 1].set_title('Cumulative P&L')
        axes[1, 1].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_monte_carlo_simulation(self, figsize=(12, 10)):
        """
        Plot Monte Carlo simulation results.
        """
        if self.monte_carlo_results is None:
            raise ValueError("No Monte Carlo simulation results available. Run a Monte Carlo simulation first.")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot equity curves
        simulated_returns = self.monte_carlo_results['simulated_returns']
        
        # Plot a subset of simulations for clarity
        n_to_plot = min(100, simulated_returns.shape[1])
        for i in range(n_to_plot):
            axes[0, 0].plot(simulated_returns.index, simulated_returns.iloc[:, i], color='blue', alpha=0.1)
        
        # Plot mean and confidence intervals
        mean_returns = simulated_returns.mean(axis=1)
        ci_lower = simulated_returns.quantile(0.05, axis=1)
        ci_upper = simulated_returns.quantile(0.95, axis=1)
        
        axes[0, 0].plot(simulated_returns.index, mean_returns, color='black', label='Mean')
        axes[0, 0].fill_between(simulated_returns.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='90% CI')
        
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].set_title('Monte Carlo Simulations')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot final equity distribution
        final_equity = self.monte_carlo_results['final_equity']
        axes[0, 1].hist(final_equity, bins=50, color='green', alpha=0.7)
        
        # Add VaR lines
        var_95 = self.monte_carlo_results['var_95']
        var_99 = self.monte_carlo_results['var_99']
        
        axes[0, 1].axvline(var_95, color='red', linestyle='--', label='95% VaR')
        axes[0, 1].axvline(var_99, color='darkred', linestyle='--', label='99% VaR')
        
        axes[0, 1].set_xlabel('Final Equity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Final Equity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot drawdown distribution
        drawdowns = self.monte_carlo_results['drawdowns']
        axes[1, 0].hist(drawdowns, bins=50, color='red', alpha=0.7)
        
        # Add mean and confidence intervals
        drawdown_mean = np.mean(drawdowns)
        drawdown_ci_lower = self.monte_carlo_results['max_drawdown_ci_lower']
        drawdown_ci_upper = self.monte_carlo_results['max_drawdown_ci_upper']
        
        axes[1, 0].axvline(drawdown_mean, color='black', linestyle='-', label='Mean')
        axes[1, 0].axvline(drawdown_ci_lower, color='red', linestyle='--', label='95% CI Lower')
        axes[1, 0].axvline(drawdown_ci_upper, color='red', linestyle='--', label='95% CI Upper')
        
        axes[1, 0].set_xlabel('Maximum Drawdown')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Maximum Drawdown Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot summary statistics
        axes[1, 1].axis('off')
        
        summary_text = (
            f"Monte Carlo Simulation Summary\n"
            f"-----------------------------\n"
            f"Number of simulations: {self.monte_carlo_results['n_simulations']}\n"
            f"Confidence level: {self.monte_carlo_results['confidence_level']}\n\n"
            f"Final Equity:\n"
            f"  Mean: ${self.monte_carlo_results['final_equity_mean']:.2f}\n"
            f"  Std Dev: ${self.monte_carlo_results['final_equity_std']:.2f}\n"
            f"  95% CI: (${self.monte_carlo_results['final_equity_ci_lower']:.2f}, "
            f"${self.monte_carlo_results['final_equity_ci_upper']:.2f})\n\n"
            f"Value at Risk (VaR):\n"
            f"  95% VaR: ${self.monte_carlo_results['var_95']:.2f}\n"
            f"  99% VaR: ${self.monte_carlo_results['var_99']:.2f}\n\n"
            f"Maximum Drawdown:\n"
            f"  Mean: {self.monte_carlo_results['max_drawdown_mean']:.2%}\n"
            f"  Std Dev: {self.monte_carlo_results['max_drawdown_std']:.2%}\n"
            f"  95% CI: ({self.monte_carlo_results['max_drawdown_ci_lower']:.2%}, "
            f"{self.monte_carlo_results['max_drawdown_ci_upper']:.2%})"
        )
        
        axes[1, 1].text(0, 1, summary_text, va='top', ha='left', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_regime_analysis(self, figsize=(12, 10)):
        """
        Plot regime analysis results.
        """
        if self.regime_analysis is None:
            raise ValueError("No regime analysis results available. Run regime analysis first.")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot regime distribution
        regime_distribution = self.regime_analysis['regime_distribution']
        axes[0, 0].bar(regime_distribution.index, regime_distribution.values, color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Market Regime')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Regime Distribution')
        axes[0, 0].grid(True)
        
        # Plot regime performance
        regime_performance = self.regime_analysis['regime_performance']
        regimes = list(regime_performance.keys())
        returns = [regime_performance[regime]['annualized_return'] * 100 for regime in regimes]
        sharpe_ratios = [regime_performance[regime]['sharpe_ratio'] for regime in regimes]
        
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        ax1.bar(regimes, returns, color='green', alpha=0.7, label='Return')
        ax2.plot(regimes, sharpe_ratios, 'ro-', label='Sharpe')
        
        ax1.set_xlabel('Market Regime')
        ax1.set_ylabel('Annualized Return (%)')
        ax2.set_ylabel('Sharpe Ratio')
        ax1.set_title('Performance by Regime')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.grid(True)
        
        # Plot regime transitions
        regime_transitions = self.regime_analysis['regime_transitions']
        
        sns.heatmap(
            regime_transitions, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues', 
            linewidths=1, 
            ax=axes[1, 0]
        )
        
        axes[1, 0].set_xlabel('To Regime')
        axes[1, 0].set_ylabel('From Regime')
        axes[1, 0].set_title('Regime Transition Probabilities')
        
        # Plot drawdown by regime
        drawdowns = [regime_performance[regime]['max_drawdown'] * 100 for regime in regimes]
        win_rates = [regime_performance[regime]['win_rate'] * 100 for regime in regimes]
        
        ax3 = axes[1, 1]
        ax4 = ax3.twinx()
        
        ax3.bar(regimes, drawdowns, color='red', alpha=0.7, label='Drawdown')
        ax4.plot(regimes, win_rates, 'bo-', label='Win Rate')
        
        ax3.set_xlabel('Market Regime')
        ax3.set_ylabel('Maximum Drawdown (%)')
        ax4.set_ylabel('Win Rate (%)')
        ax3.set_title('Risk Metrics by Regime')
        
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
        
        ax3.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_performance_report(self, filename=None):
        """
        Generate a comprehensive performance report.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Plot equity curve
        ax_equity = axes[0, 0]
        ax_equity.plot(self.results.index, self.results['portfolio_value'], label='Strategy')
        
        # Plot benchmark if available
        if 'benchmark' in self.results.columns:
            # Normalize benchmark to initial capital
            benchmark_values = self.results['benchmark'] / self.results['benchmark'].iloc[0] * self.initial_capital
            ax_equity.plot(self.results.index, benchmark_values, label='Benchmark', alpha=0.7)
        
        ax_equity.set_xlabel('Date')
        ax_equity.set_ylabel('Portfolio Value')
        ax_equity.set_title('Equity Curve')
        ax_equity.grid(True)
        ax_equity.legend()
        
        # Plot drawdown
        ax_drawdown = axes[0, 1]
        
        equity_curve = self.results['portfolio_value']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100  # Convert to percentage
        
        ax_drawdown.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax_drawdown.plot(drawdown.index, drawdown, color='red', label='Drawdown')
        
        ax_drawdown.set_xlabel('Date')
        ax_drawdown.set_ylabel('Drawdown (%)')
        ax_drawdown.set_title('Portfolio Drawdown')
        ax_drawdown.grid(True)
        ax_drawdown.legend()
        ax_drawdown.invert_yaxis()
        
        # Plot monthly returns heatmap
        ax_monthly = axes[1, 0]
        
        # Calculate monthly returns
        monthly_returns = self.results['daily_return'].groupby([
            self.results.index.year, 
            self.results.index.month
        ]).apply(lambda x: (1 + x).prod() - 1) * 100  # Convert to percentage
        
        # Reshape to matrix
        monthly_returns = monthly_returns.unstack()
        
        # Plot heatmap
        sns.heatmap(
            monthly_returns, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn', 
            center=0, 
            linewidths=1, 
            ax=ax_monthly
        )
        
        ax_monthly.set_xlabel('Month')
        ax_monthly.set_ylabel('Year')
        ax_monthly.set_title('Monthly Returns (%)')
        
        # Set month names
        ax_monthly.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Plot trade analysis
        ax_trades = axes[1, 1]
        
        if self.trades is not None and len(self.trades) > 0:
            # Plot trade P&L
            ax_trades.bar(
                range(len(self.trades)), 
                self.trades['pnl'], 
                color=['green' if pnl > 0 else 'red' for pnl in self.trades['pnl']]
            )
            ax_trades.set_xlabel('Trade Number')
            ax_trades.set_ylabel('P&L')
            ax_trades.set_title('Trade P&L')
            ax_trades.grid(True)
        else:
            ax_trades.text(0.5, 0.5, 'No trades available', ha='center', va='center')
            ax_trades.set_title('Trade P&L')
        
        # Plot performance metrics
        ax_metrics = axes[2, 0]
        ax_metrics.axis('off')
        
        if self.metrics is not None:
            metrics_text = (
                f"Performance Metrics\n"
                f"------------------\n"
                f"Total Return: {self.metrics['total_return']:.2%}\n"
                f"Annualized Return: {self.metrics['annualized_return']:.2%}\n"
                f"Volatility: {self.metrics['volatility']:.2%}\n"
                f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
                f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}\n"
                f"Maximum Drawdown: {self.metrics['max_drawdown']:.2%}\n"
                f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}\n"
                f"Win Rate: {self.metrics['win_rate']:.2%}\n"
                f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
                f"Average Trade: {self.metrics['avg_trade']:.2f}\n"
                f"Total Trades: {self.metrics['total_trades']}\n"
                f"Trading Days: {self.metrics['trading_days']}\n"
            )
            
            # Add benchmark metrics if available
            if 'benchmark_total_return' in self.metrics:
                metrics_text += (
                    f"\nBenchmark Metrics\n"
                    f"----------------\n"
                    f"Benchmark Return: {self.metrics['benchmark_total_return']:.2%}\n"
                    f"Benchmark Ann. Return: {self.metrics['benchmark_annualized_return']:.2%}\n"
                    f"Benchmark Volatility: {self.metrics['benchmark_volatility']:.2%}\n"
                    f"Benchmark Sharpe: {self.metrics['benchmark_sharpe_ratio']:.2f}\n"
                    f"Benchmark Max DD: {self.metrics['benchmark_max_drawdown']:.2%}\n"
                    f"Alpha: {self.metrics['alpha']:.2%}\n"
                    f"Beta: {self.metrics['beta']:.2f}\n"
                    f"Information Ratio: {self.metrics['information_ratio']:.2f}\n"
                )
            
            ax_metrics.text(0, 1, metrics_text, va='top', ha='left', fontsize=12)
        else:
            ax_metrics.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
        
        # Plot distribution of returns
        ax_dist = axes[2, 1]
        
        returns = self.results['daily_return'].dropna() * 100  # Convert to percentage
        
        ax_dist.hist(returns, bins=50, color='blue', alpha=0.7)
        ax_dist.axvline(0, color='black', linestyle='--')
        
        # Add normal distribution fit
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std()) * len(returns) * (returns.max() - returns.min()) / 50
        ax_dist.plot(x, y, 'r-', linewidth=2)
        
        ax_dist.set_xlabel('Daily Return (%)')
        ax_dist.set_ylabel('Frequency')
        ax_dist.set_title('Distribution of Daily Returns')
        ax_dist.grid(True)
        
        # Format x-axis dates
        for ax in [ax_equity, ax_drawdown]:
            fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if filename is provided
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, filename='backtest_results.pkl'):
        """
        Save backtest results to file.
        """
        # Create dictionary with all results
        results_dict = {
            'results': self.results,
            'trades': self.trades,
            'metrics': self.metrics,
            'optimization_results': self.optimization_results,
            'monte_carlo_results': self.monte_carlo_results,
            'regime_analysis': self.regime_analysis,
            'walk_forward_results': self.walk_forward_results,
            'parameters': {
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'risk_free_rate': self.risk_free_rate,
                'slippage_model': self.slippage_model,
                'slippage_value': self.slippage_value,
                'position_sizing': self.position_sizing,
                'max_position_size': self.max_position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(results_dict, f)
    
    def load_results(self, filename='backtest_results.pkl'):
        """
        Load backtest results from file.
        """
        try:
            # Load from file
            with open(filename, 'rb') as f:
                results_dict = pickle.load(f)
            
            # Set attributes
            self.results = results_dict['results']
            self.trades = results_dict['trades']
            self.metrics = results_dict['metrics']
            self.optimization_results = results_dict['optimization_results']
            self.monte_carlo_results = results_dict['monte_carlo_results']
            self.regime_analysis = results_dict['regime_analysis']
            self.walk_forward_results = results_dict['walk_forward_results']
            
            # Set parameters
            parameters = results_dict['parameters']
            self.initial_capital = parameters['initial_capital']
            self.commission = parameters['commission']
            self.risk_free_rate = parameters['risk_free_rate']
            self.slippage_model = parameters['slippage_model']
            self.slippage_value = parameters['slippage_value']
            self.position_sizing = parameters['position_sizing']
            self.max_position_size = parameters['max_position_size']
            self.stop_loss = parameters['stop_loss']
            self.take_profit = parameters['take_profit']
            
            return True
        
        except Exception as e:
            print(f"Error loading results: {e}")
            return False