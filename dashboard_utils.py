from dash import dcc
from dash import html
def create_correlation_multi_method_charts(figures):
    """
    Create a Dash component (Tabs) to display all correlation analysis charts for each method combination.
    Expects figures dict with keys like 'correlation_matrix__pearson_rfecv'.
    Returns a Div with dcc.Tabs for each chart type, each containing a tab per method combination.
    """
    chart_types = ['correlation_matrix', 'redundancy_groups', 'feature_importance']
    # Find all method combinations for each chart type
    method_combos = {ct: [] for ct in chart_types}
    for key in figures:
        for ct in chart_types:
            if key.startswith(ct + '__'):
                method_combos[ct].append(key.split('__', 1)[1])

    # For each chart type, create a dcc.Tabs with a tab for each method (only if the value is a Plotly figure)
    chart_tabs = []
    import plotly.graph_objs as go
    for ct in chart_types:
        tabs = []
        for method in sorted(method_combos[ct]):
            fig_key = f"{ct}__{method}"
            fig = figures.get(fig_key)
            # Only add tab if fig is a Plotly Figure (strict type check)
            if isinstance(fig, go.Figure):
                tabs.append(dcc.Tab(label=method, value=fig_key, children=[
                    dcc.Graph(figure=fig, id=f"{ct}-chart-{method}")
                ]))
        if tabs:
            chart_tabs.append(html.Div([
                html.H4(ct.replace('_', ' ').title()),
                dcc.Tabs(tabs, id=f"{ct}-multi-method-tabs", value=tabs[0].value)
            ], style={'marginBottom': '30px'}))

    # If there are no valid chart tabs, return a plain message (not a Div as a figure!)
    if not chart_tabs:
        return html.Div("No multi-method correlation charts available.")
    return html.Div(chart_tabs)
# --- Advanced Backtesting Visualizations ---
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_walk_forward_optimization_chart(wfo_results):
    """
    Visualize walk-forward optimization results as a bar/line chart of out-of-sample metrics per window.
    """
    if wfo_results is None or len(wfo_results) == 0:
        return go.Figure().update_layout(title="Walk-Forward Optimization: No Results")
    if not isinstance(wfo_results, pd.DataFrame):
        wfo_results = pd.DataFrame(wfo_results)
    metric_col = 'out_sample_metric' if 'out_sample_metric' in wfo_results.columns else wfo_results.columns[-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=wfo_results['window'],
        y=wfo_results[metric_col],
        name='Out-of-Sample Metric',
        marker_color='royalblue',
        hovertext=[f"Params: {p}" for p in wfo_results['best_params']]
    ))
    fig.update_layout(
        title="Walk-Forward Optimization (Out-of-Sample Metric per Window)",
        xaxis_title="Window",
        yaxis_title=metric_col.replace('_', ' ').title(),
        height=400
    )
    return fig

def create_monte_carlo_simulation_chart(mc_results):
    """
    Visualize Monte Carlo simulation results: equity curves and final equity distribution.
    """
    if mc_results is None or 'simulated_returns' not in mc_results:
        return go.Figure().update_layout(title="Monte Carlo Simulation: No Results")
    sim_returns = mc_results['simulated_returns']
    n_to_plot = min(50, sim_returns.shape[1])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Simulated Equity Curves", "Final Equity Distribution"))
    # Plot equity curves
    for i in range(n_to_plot):
        fig.add_trace(go.Scatter(x=sim_returns.index, y=sim_returns.iloc[:, i],
                                 mode='lines', line=dict(width=1),
                                 marker_color='rgba(0,0,255,0.15)', showlegend=False),
                      row=1, col=1)
    # Plot mean equity curve
    mean_curve = sim_returns.mean(axis=1)
    fig.add_trace(go.Scatter(x=sim_returns.index, y=mean_curve, mode='lines',
                             line=dict(color='black', width=2), name='Mean'), row=1, col=1)
    # Final equity histogram
    final_equity = mc_results['final_equity']
    fig.add_trace(go.Histogram(x=final_equity, nbinsx=30, marker_color='green', name='Final Equity'), row=1, col=2)
    # VaR lines
    if 'var_95' in mc_results:
        fig.add_vline(x=mc_results['var_95'], line_dash='dash', line_color='red', row=1, col=2, annotation_text='95% VaR', annotation_position='top')
    if 'var_99' in mc_results:
        fig.add_vline(x=mc_results['var_99'], line_dash='dot', line_color='darkred', row=1, col=2, annotation_text='99% VaR', annotation_position='top')
    fig.update_layout(
        title="Monte Carlo Simulation Results",
        height=400,
        showlegend=False
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_xaxes(title_text="Final Equity", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    return fig

def create_regime_analysis_chart(regime_analysis):
    """
    Visualize regime analysis: performance by regime, regime distribution, and transition matrix.
    """
    if regime_analysis is None or 'regime_performance' not in regime_analysis:
        return go.Figure().update_layout(title="Regime Analysis: No Results")
    perf = regime_analysis['regime_performance']
    regimes = list(perf.keys())
    returns = [perf[r]['annualized_return']*100 for r in regimes]
    sharpe = [perf[r]['sharpe_ratio'] for r in regimes]
    drawdown = [perf[r]['max_drawdown']*100 for r in regimes]
    # Bar chart: returns, sharpe, drawdown
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Performance by Regime", "Regime Transition Matrix"))
    fig.add_trace(go.Bar(x=regimes, y=returns, name='Ann. Return (%)', marker_color='green'), row=1, col=1)
    fig.add_trace(go.Bar(x=regimes, y=drawdown, name='Max Drawdown (%)', marker_color='red'), row=1, col=1)
    fig.add_trace(go.Scatter(x=regimes, y=sharpe, name='Sharpe Ratio', mode='lines+markers', marker_color='blue'), row=1, col=1)
    fig.update_yaxes(title_text="Metric Value", row=1, col=1)
    # Regime transition matrix heatmap
    if 'regime_transitions' in regime_analysis:
        trans = regime_analysis['regime_transitions']
        if hasattr(trans, 'values'):
            fig.add_trace(go.Heatmap(z=trans.values, x=trans.columns, y=trans.index, colorscale='Blues',
                                     colorbar=dict(title='Prob')), row=1, col=2)
    fig.update_layout(title="Market Regime Analysis", height=400, barmode='group')
    return fig

# --- Imports ---
import os
import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np
import math # For math.ceil

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html
import dash_bootstrap_components as dbc # For styled HTML components
from adaptive_thresholds_enhanced import EnhancedAdaptiveThresholds
from portfolio_optimization import PortfolioOptimizer

# --- Centralized File Naming ---
def get_standardized_model_filename(model_type, model_name, symbol=None, period=None, interval=None, base_path=None, include_timestamp=True, extension="pkl"):
    """
    Create a standardized filename for model saving/loading.
    Handles potential list types for model_name if it's constructed from multiple parts.

    Parameters:
    - model_type: Type of model (e.g., 'anomaly_detection')
    - model_name: Specific name/algorithm of the model. Can be a list to be joined by '_'.
    - symbol, period, interval: Optional metadata
    - base_path: Base directory (defaults to 'models/{Model Type Name}')
    - include_timestamp: Whether to include a YYYYMMDD timestamp (default: True)
    - extension: File extension (default: 'pkl')

    Returns:
    - Full path to the model file
    """
    timestamp_str = f"_{datetime.now().strftime('%Y%m%d')}" if include_timestamp else ""

    if isinstance(model_name, list):
        model_name_str = "_".join(filter(None, model_name))
    else:
        model_name_str = model_name if model_name else "model"

    symbol_str = f"_{symbol}" if symbol else "_generic" # Use _generic if no symbol
    period_str = f"_{period}" if period else ""
    interval_str = f"_{interval}" if interval else ""

    # Standardize and clean parts to avoid issues from various inputs
    filename_parts = [
        str(model_type).replace(" ", "_").lower(),
        str(model_name_str).replace(" ", "_").lower(),
        str(symbol_str).replace(" ", "_"),
        str(period_str),
        str(interval_str),
        timestamp_str
    ]
    filename_base = "_".join(filter(None, filename_parts))
    filename_base = filename_base.replace("__", "_").replace("..", "_").replace("/", "_").replace("\\", "_") # General cleaning
    filename_base = "".join(c for c in filename_base if c.isalnum() or c in ['_', '-']) # Ensure valid chars


    if base_path is None:
        dir_name_parts = [word.capitalize() for word in model_type.split('_')]
        dir_name = " ".join(dir_name_parts) + " Models"
        base_path = os.path.join("models", dir_name)

    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, f"{filename_base}.{extension}")

# --- Common Visualization Utilities ---

def _create_empty_figure(title="Data Unavailable or Error"):
    fig = go.Figure()
    fig.update_layout(
        title={'text': title, 'x':0.5, 'xanchor': 'center'},
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{
            'text': title,
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': 0.5, 'showarrow': False,
            'font': {'size': 16}
        }],
        plot_bgcolor='rgba(240,240,240,0.5)', # Light background for placeholder
        paper_bgcolor='rgba(255,255,255,1)'
    )
    return fig



def create_market_data_info(df, symbol, market_breadth_data, sentiment_data):
    if df is None or df.empty:
        return dbc.Alert("Market data not available for info display.", color="warning")

    if len(df) < 2:
        latest_data = df.iloc[-1]
        price_change = 0.0
        price_change_pct = 0.0
    else:
        latest_data = df.iloc[-1]
        prev_data = df.iloc[-2]
        price_change = latest_data['close'] - prev_data['close']
        price_change_pct = (price_change / prev_data['close']) * 100 if prev_data['close'] != 0 else 0

    change_color = "text-success" if price_change >= 0 else "text-danger"
    arrow = "▲" if price_change >= 0 else "▼"

    mb_html = []
    if market_breadth_data and isinstance(market_breadth_data, dict):
        mb_html.extend([
            html.P(f"Adv. Issues: {market_breadth_data.get('advancing_issues', 'N/A')}", className="mb-1"),
            html.P(f"Decl. Issues: {market_breadth_data.get('declining_issues', 'N/A')}", className="mb-1"),
            html.P(f"ARMS Index: {latest_data.get('arms_index', 'N/A'):.2f}" if isinstance(latest_data.get('arms_index'), float) else "N/A", className="mb-1"),
        ])
    else:
        mb_html.append(html.P("Market breadth N/A", className="text-muted mb-1"))

    sent_html = []
    if sentiment_data and isinstance(sentiment_data, dict):
        def get_sentiment_val(key):
            val = sentiment_data.get(key, 0)
            if isinstance(val, list):
                val = val[-1] if val else 0
            try:
                return float(val) * 100
            except (ValueError, TypeError):
                return "N/A"

        bullish_pct = get_sentiment_val('bullish')
        bearish_pct = get_sentiment_val('bearish')
        neutral_pct = get_sentiment_val('neutral')

        sent_html.extend([
            html.P(f"AAII Bullish: {f'{bullish_pct:.1f}%' if isinstance(bullish_pct, float) else 'N/A'}", className="mb-1"),
            html.P(f"AAII Bearish: {f'{bearish_pct:.1f}%' if isinstance(bearish_pct, float) else 'N/A'}", className="mb-1"),
            html.P(f"AAII Neutral: {f'{neutral_pct:.1f}%' if isinstance(neutral_pct, float) else 'N/A'}", className="mb-1"),
        ])
    else:
        sent_html.append(html.P("Sentiment data N/A", className="text-muted mb-1"))

    date_str = latest_data.name.strftime('%Y-%m-%d %H:%M') if isinstance(latest_data.name, pd.Timestamp) else str(latest_data.name)

    return dbc.Row([
        dbc.Col([
            html.H5(f"{symbol} - {date_str}", className="mb-1"),
            html.H6(f"${latest_data['close']:.2f} ", className=f"d-inline {change_color}"),
            html.Span(f"{arrow} {price_change:+.2f} ({price_change_pct:+.2f}%)", className=change_color),
        ], md=3, className="border-end"),
        dbc.Col([html.Strong("Market Breadth", className="d-block mb-2"), *mb_html], md=3, className="border-end"),
        dbc.Col([html.Strong("AAII Sentiment", className="d-block mb-2"), *sent_html], md=3, className="border-end"),
        dbc.Col([
            html.Strong("Key Indicators", className="d-block mb-2"),
            html.P(f"RSI (14): {latest_data.get('rsi', 'N/A'):.2f}", className="mb-1"),
            html.P(f"MACD Line: {latest_data.get('macd_line', 'N/A'):.2f}", className="mb-1"),
            html.P(f"Stoch %K: {latest_data.get('stoch_k', 'N/A'):.2f}", className="mb-1"),
        ], md=3),
    ], className="p-2")

def create_main_chart(df, symbol):
    """
    Create main chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    - symbol: Stock symbol
    
    Returns:
    - Main chart figure
    """
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{symbol} Price", "Volume"))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['bb_upper'],
            mode='lines',
            line=dict(width=1, color='rgba(173, 204, 255, 0.7)'),
            name="Upper BB"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['bb_middle'],
            mode='lines',
            line=dict(width=1, color='rgba(173, 204, 255, 0.7)'),
            name="Middle BB"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['bb_lower'],
            mode='lines',
            line=dict(width=1, color='rgba(173, 204, 255, 0.7)'),
            name="Lower BB",
            fill='tonexty',
            fillcolor='rgba(173, 204, 255, 0.2)'
        ),
        row=1, col=1
    )
    
    # Add buy/sell signals if available
    if 'buy_signal' in df.columns and 'sell_signal' in df.columns:
        # Filter for buy signals
        buy_signals = df[df['buy_signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['low'] * 0.99,  # Slightly below the low price
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name="Buy Signal"
                ),
                row=1, col=1
            )
        
        # Filter for sell signals
        sell_signals = df[df['sell_signal'] == 1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['high'] * 1.01,  # Slightly above the high price
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name="Sell Signal"
                ),
                row=1, col=1
            )
    
    # Add volume chart
    colors = ['red' if row['open'] > row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_indicators_chart(df):
    """
    Create indicators chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    
    Returns:
    - Indicators chart figure
    """
    # Create figure with 2 rows for MACD and RSI
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.5, 0.5],
                        subplot_titles=("MACD", "RSI"))
    # Add MACD traces
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_line'],
            mode='lines',
            name="MACD Line"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            mode='lines',
            name="Signal Line"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['macd_hist'],
            name="MACD Histogram",
            marker_color=['red' if val < 0 else 'green' for val in df['macd_hist']]
        ),
        row=1, col=1
    )
    # Add RSI trace
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            mode='lines',
            name="RSI"
        ),
        row=2, col=1
    )
    # Get adaptive RSI thresholds using get_threshold_recommendations
    thresholds_model = EnhancedAdaptiveThresholds()
    rsi_recommendation = thresholds_model.get_threshold_recommendations(df, indicator_col='rsi')
    rsi_overbought = rsi_recommendation['upper_threshold']
    rsi_oversold = rsi_recommendation['lower_threshold']
    # Add adaptive overbought line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[rsi_overbought] * len(df),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name=f"Overbought (Adaptive: {rsi_overbought:.2f})"
        ),
        row=2, col=1
    )
    # Add adaptive oversold line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[rsi_oversold] * len(df),
            mode='lines',
            line=dict(dash='dash', color='green'),
            name=f"Oversold (Adaptive: {rsi_oversold:.2f})"
        ),
        row=2, col=1
    )
    # Update layout
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    # Update y-axis labels
    fig.update_yaxes(title_text="MACD", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    return fig

def create_moving_averages_chart(df):
    """
    Create moving averages chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    
    Returns:
    - Moving averages chart figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name="Close Price"
        )
    )
    
    # Add moving averages
    ma_periods = [5, 9, 20, 50, 100, 200]
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, period in enumerate(ma_periods):
        if f'ma_{period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'ma_{period}'],
                    mode='lines',
                    name=f"{period}-day MA",
                    line=dict(color=colors[i % len(colors)])
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Moving Averages",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis label
    fig.update_yaxes(title_text="Price ($)")
    
    return fig

def create_volume_chart(df):
    """
    Create volume chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    
    Returns:
    - Volume chart figure
    """
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.5, 0.5],
                        subplot_titles=("Volume", "Chaikin Money Flow"))
    
    # Add volume
    colors = ['red' if row['open'] > row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=1, col=1
    )
    
    # Add volume moving average
    if 'volume_ma' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['volume_ma'],
                mode='lines',
                name="Volume MA",
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    # Add Chaikin Money Flow
    if 'cmf' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['cmf'],
                mode='lines',
                name="CMF",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[0] * len(df),
                mode='lines',
                line=dict(dash='dash', color='black'),
                name="Zero Line"
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="CMF", row=2, col=1)
    
    return fig

def create_oscillators_chart(df):
    """
    Create oscillators chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    
    Returns:
    - Oscillators chart figure
    """
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.5, 0.5],
                        subplot_titles=("Stochastic Oscillator", "CCI"))
    
    # Add Stochastic Oscillator
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['stoch_k'],
                mode='lines',
                name="%K"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['stoch_d'],
                mode='lines',
                name="%D"
            ),
            row=1, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[80] * len(df),
                mode='lines',
                line=dict(dash='dash', color='red'),
                name="Overbought (80)"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[20] * len(df),
                mode='lines',
                line=dict(dash='dash', color='green'),
                name="Oversold (20)"
            ),
            row=1, col=1
        )
    
    # Add CCI
    if 'cci' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['cci'],
                mode='lines',
                name="CCI"
            ),
            row=2, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[100] * len(df),
                mode='lines',
                line=dict(dash='dash', color='red'),
                name="Overbought (100)"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[-100] * len(df),
                mode='lines',
                line=dict(dash='dash', color='green'),
                name="Oversold (-100)"
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Stochastic", row=1, col=1)
    fig.update_yaxes(title_text="CCI", row=2, col=1)
    
    return fig

def create_sentiment_chart(df, sentiment_data):
    """
    Create sentiment chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    - sentiment_data: Dictionary with sentiment data
    
    Returns:
    - Sentiment chart figure
    """
    # Create figure
    fig = make_subplots(rows=1, cols=1)
    
    # Check if sentiment data is available
    if sentiment_data is None:
        fig.add_annotation(
            text="Sentiment data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create sentiment chart
    fig.add_trace(
        go.Bar(
            x=['Bullish', 'Bearish', 'Neutral'],
            y=[sentiment_data['bullish'], sentiment_data['bearish'], sentiment_data['neutral']],
            marker_color=['green', 'red', 'blue']
        )
    )
    
    # Add bulls vs bears ratio if available
    if 'bulls_bears_ratio' in df.columns:
        # Create a secondary y-axis for the ratio
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        
        # Add sentiment bars
        fig.add_trace(
            go.Bar(
                x=['Bullish', 'Bearish', 'Neutral'],
                y=[sentiment_data['bullish'], sentiment_data['bearish'], sentiment_data['neutral']],
                marker_color=['green', 'red', 'blue'],
                name="Sentiment"
            ),
            secondary_y=False
        )
        
        # Add bulls vs bears ratio line
        latest_ratio = df['bulls_bears_ratio'].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=['Bullish', 'Bearish', 'Neutral'],
                y=[latest_ratio, latest_ratio, latest_ratio],
                mode='lines',
                name=f"Bulls/Bears Ratio: {latest_ratio:.2f}",
                line=dict(color='black', dash='dash')
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title="AAII Investor Sentiment Survey",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Percentage (%)", secondary_y=False)
    if 'bulls_bears_ratio' in df.columns:
        fig.update_yaxes(title_text="Bulls/Bears Ratio", secondary_y=True)
    
    return fig

def create_backtesting_results(backtest_results):
    """
    Create a formatted display for backtesting results.
    
    Parameters:
    - backtest_results: Dictionary with backtesting performance metrics.
    
    Returns:
    - HTML Div displaying backtesting results.
    """
    def safe_format(val, fmt):
        if isinstance(val, (float, int)) and not (isinstance(val, float) and (val != val)):
            try:
                return format(val, fmt)
            except Exception:
                return str(val)
        return str(val)

    return html.Div([
        html.H4("Performance Metrics"),
        html.P(f"Sharpe Ratio: {safe_format(backtest_results.get('sharpe_ratio', 'N/A'), '.2f')}"),
        html.P(f"Maximum Drawdown: {safe_format(backtest_results.get('max_drawdown', 'N/A'), '.2%')}"),
        html.P(f"Win Rate: {safe_format(backtest_results.get('win_rate', 'N/A'), '.2%')}"),
        html.P(f"Total Return: {safe_format(backtest_results.get('total_return', 'N/A'), '.2%')}"),
        html.P(f"Annualized Volatility: {safe_format(backtest_results.get('volatility', 'N/A'), '.2%')}"),
        html.P(f"Profit Factor: {safe_format(backtest_results.get('profit_factor', 'N/A'), '.2f')}")
    ])

def create_perfect_storm_analysis(df):
    """
    Create Perfect Storm analysis
    
    Parameters:
    - df: DataFrame with stock and indicator data
    
    Returns:
    - Perfect Storm analysis
    """
    # Get the latest data
    latest_data = df.iloc[-1]
    
    # Check if we have all the required indicators
    required_indicators = ['rsi', 'macd_line', 'macd_signal', 'stoch_k', 'stoch_d', 'cci', 'bb_upper', 'bb_lower']
    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
    
    if missing_indicators:
        return html.Div([
            html.P(f"Missing indicators: {', '.join(missing_indicators)}"),
            html.P("Cannot perform Perfect Storm analysis without all required indicators.")
        ])
    
    # Create analysis
    analysis = []
    
    # Check for buy signals
    buy_signals = []
    if latest_data.get('rsi', 0) < 40:
        buy_signals.append("RSI is below 40")
    if latest_data.get('macd_line', 0) > latest_data.get('macd_signal', 0):
        buy_signals.append("MACD line is above signal line")
    if latest_data.get('stoch_k', 0) < 20:
        buy_signals.append("Stochastic %K is below 20 (oversold)")
    if latest_data.get('cci', 0) < -100:
        buy_signals.append("CCI is below -100 (oversold)")
    if latest_data.get('close', 0) < latest_data.get('bb_lower', 0):
        buy_signals.append("Price is below lower Bollinger Band")
    
    # Check for sell signals
    sell_signals = []
    if latest_data.get('rsi', 0) > 65:
        sell_signals.append("RSI is above 65")
    if latest_data.get('macd_line', 0) < latest_data.get('macd_signal', 0):
        sell_signals.append("MACD line is below signal line")
    if latest_data.get('stoch_k', 0) > 80:
        sell_signals.append("Stochastic %K is above 80 (overbought)")
    if latest_data.get('cci', 0) > 100:
        sell_signals.append("CCI is above 100 (overbought)")
    if latest_data.get('close', 0) > latest_data.get('bb_upper', 0):
        sell_signals.append("Price is above upper Bollinger Band")
    
    # Add buy signals
    if buy_signals:
        analysis.append(html.Div([
            html.H4("Buy Signals", style={'color': 'green'}),
            html.Ul([html.Li(signal) for signal in buy_signals])
        ]))
    
    # Add sell signals
    if sell_signals:
        analysis.append(html.Div([
            html.H4("Sell Signals", style={'color': 'red'}),
            html.Ul([html.Li(signal) for signal in sell_signals])
        ]))
    
    # Add overall recommendation
    if len(buy_signals) > len(sell_signals) and len(buy_signals) >= 3:
        analysis.append(html.Div([
            html.H4("Overall Recommendation", style={'fontWeight': 'bold'}),
            html.P("BUY", style={'color': 'green', 'fontSize': '24px', 'fontWeight': 'bold'})
        ]))
    elif len(sell_signals) > len(buy_signals) and len(sell_signals) >= 3:
        analysis.append(html.Div([
            html.H4("Overall Recommendation", style={'fontWeight': 'bold'}),
            html.P("SELL", style={'color': 'red', 'fontSize': '24px', 'fontWeight': 'bold'})
        ]))
    else:
        analysis.append(html.Div([
            html.H4("Overall Recommendation", style={'fontWeight': 'bold'}),
            html.P("HOLD", style={'color': 'blue', 'fontSize': '24px', 'fontWeight': 'bold'})
        ]))
    
    return analysis

def create_backtesting_chart(results, trades, metrics):
    """
    Creates a comprehensive backtesting chart including price, signals, and portfolio value.

    Args:
        results (pd.DataFrame): DataFrame from BacktestingEngine containing portfolio value, close prices, etc.
                                Expected columns: 'close', 'portfolio_value'. Index should be datetime.
        trades (pd.DataFrame): DataFrame from BacktestingEngine listing individual trades.
                               Expected columns: 'date', 'type' ('buy'/'sell'), 'price'.
        metrics (dict): Dictionary containing performance metrics from BacktestingEngine.
                        Expected keys: 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'.

    Returns:
        go.Figure: A Plotly figure object visualizing the backtest results.
                   Returns an empty figure with an error message if essential data is missing.
    """
    # Basic data validation
    if results is None or results.empty:
        return go.Figure().update_layout(title="Backtesting Error: Results data is missing.")
    if trades is None: # Allow trades to be empty if no trades occurred
        trades = pd.DataFrame(columns=['date', 'type', 'price'])
        print("Warning: No trades were provided for backtesting chart.")
    if metrics is None:
        metrics = {} # Allow empty metrics, annotations will show N/A
        print("Warning: Performance metrics missing for backtesting chart annotations.")

    # --- Create Subplots ---
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05, # Reduce spacing slightly
        row_heights=[0.6, 0.4],
        subplot_titles=("Price & Signals", "Portfolio Performance")
    )

    # --- Top Subplot: Price and Signals ---
    # Plot Price (Using closing price from results)
    if 'close' in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['close'],
                mode='lines',
                name="Price",
                line=dict(color='black', width=1)
            ),
            row=1, col=1
        )
    else:
         # Handle case where close price isn't in results - less likely but possible
        fig.add_trace(go.Scatter(x=results.index, y=[None]*len(results.index), name="Price Data Missing"), row=1, col=1)


    # Plot Buy Signals from trades DataFrame
    if not trades.empty and 'type' in trades.columns and 'date' in trades.columns and 'price' in trades.columns:
        buy_trades = trades[trades['type'] == 'buy'].sort_values('date')
        if not buy_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_trades['date'],
                    y=buy_trades['price'], # Use actual trade price for signal marker
                    mode='markers',
                    name="Buy",
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )

        # Plot Sell Signals from trades DataFrame
        sell_trades = trades[trades['type'] == 'sell'].sort_values('date')
        if not sell_trades.empty:
             # Check if sell trades occurred when a position was held
             # Note: Simple check; robust check needs position tracking state during backtest.
             valid_sell_markers = sell_trades[sell_trades['price'].notna()] # Filter out any invalid price points

             fig.add_trace(
                 go.Scatter(
                    x=valid_sell_markers['date'],
                    y=valid_sell_markers['price'], # Use actual trade price
                    mode='markers',
                    name="Sell",
                    marker=dict(symbol='triangle-down', size=10, color='red')
                 ),
                 row=1, col=1
            )


    # --- Bottom Subplot: Portfolio Performance ---
    # Plot Portfolio Value
    initial_capital = metrics.get('initial_capital', results['portfolio_value'].iloc[0] if not results.empty else 100000) # Get initial capital if possible

    buy_hold_value = None  # Ensure buy_hold_value is always defined

    if 'portfolio_value' in results.columns:
        fig.add_trace(
            go.Scatter(
                x=results.index,
                y=results['portfolio_value'],
                mode='lines',
                name="Strategy Value",
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
    else:
        fig.add_trace(go.Scatter(x=results.index, y=[None]*len(results.index), name="Portfolio Value Missing"), row=2, col=1)

    # Plot Buy & Hold Benchmark
    if 'close' in results.columns and not results['close'].empty:
         # Use the actual start price and initial capital
         start_price = results['close'].iloc[0]
         if start_price > 0:
              buy_hold_value = (results['close'] / start_price) * initial_capital
              fig.add_trace(
                  go.Scatter(
                      x=results.index,
                      y=buy_hold_value,
                      mode='lines',
                      name="Buy & Hold",
                      line=dict(color='grey', dash='dash', width=1)
                  ),
                  row=2, col=1
              )

    # Add Initial Capital Line
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=[initial_capital] * len(results),
            mode='lines',
            name="Initial Capital",
            line=dict(color='red', dash='dot', width=1)
        ),
        row=2, col=1
    )

    # --- Add Performance Metrics Annotation ---
    if metrics:
        total_return = metrics.get('total_return', float('nan'))
        sharpe_ratio = metrics.get('sharpe_ratio', float('nan'))
        max_drawdown = metrics.get('max_drawdown', float('nan'))
        win_rate = metrics.get('win_rate', float('nan'))
        total_trades = metrics.get('total_trades', 'N/A')
        # Calculate Buy & Hold Return for Alpha comparison if possible
        if buy_hold_value is not None:
            bh_return = (buy_hold_value.iloc[-1] / initial_capital - 1)
        else:
            bh_return = float('nan')
        alpha = total_return - bh_return # Simple alpha calculation

        stats_text = (
            f"<b>Strategy Metrics:</b><br>"
            f"  Total Return: {total_return:.2%}<br>"
            f"  Sharpe Ratio: {sharpe_ratio:.2f}<br>"
            f"  Max Drawdown: {max_drawdown:.2%}<br>"
            f"  Win Rate: {win_rate:.2%}<br>"
            f"  Trades: {total_trades}<br>"
             f"  Buy & Hold Return: {bh_return:.2%}<br>"
            f"  Alpha (vs B&H): {alpha:.2%}"
        )

        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.38, # Positioned near bottom left of lower plot
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            align="left"
        )

    # --- Final Layout Updates ---
    fig.update_layout(
        height=700,
        title_text="Backtesting Results",
        xaxis_rangeslider_visible=False, # No range slider on price chart
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Update axes titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1) # Only label bottom x-axis

    return fig

def create_correlation_report_charts(correlation_report):
    """
    Create plotly figures from correlation report
    
    Parameters:
    - correlation_report: Dictionary with report components from CorrelationAnalysis.generate_correlation_report
    
    Returns:
    - Dictionary of plotly figures for correlation report visualizations
    """
    figures = {}

    # Helper to style figures
    def style_figure(fig, name):
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12),
            title=dict(
                text=name.replace('_', ' ').title(),
                font=dict(size=20)
            )
        )
        return fig

    # If multi-method results are present, extract all charts for each method combination
    if correlation_report is not None and 'multi_method_results' in correlation_report and correlation_report['multi_method_results']:
        multi_results = correlation_report['multi_method_results']
        for method_name, result in multi_results.items():
            # Each result should have its own visualizations dict
            visualizations = result.get('visualizations', {})
            # For each expected chart type, add to figures with method suffix
            for chart_type in ['correlation_matrix', 'redundancy_groups', 'feature_importance']:
                key = f"{chart_type}__{method_name}"
                if chart_type in visualizations:
                    figures[key] = style_figure(visualizations[chart_type], key)
                else:
                    empty_fig = go.Figure().update_layout(title=f"{chart_type.replace('_', ' ').title()} Not Available ({method_name})")
                    figures[key] = style_figure(empty_fig, key)
        # If no results found, fall back to default
        if not figures:
            for chart_type in ['correlation_matrix', 'redundancy_groups', 'feature_importance']:
                empty_fig = go.Figure().update_layout(title=f"{chart_type.replace('_', ' ').title()} Not Available")
                figures[chart_type] = style_figure(empty_fig, chart_type)
        return figures

    # Otherwise, use the default single-method logic
    # Check if visualizations are available in the report
    if correlation_report is None or 'visualizations' not in correlation_report:
        # Create empty placeholder figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Correlation Data Available",
            annotations=[{
                'text': 'Correlation analysis data not available',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        return {
            'correlation_matrix': empty_fig,
            'redundancy_groups': empty_fig,
            'feature_importance': empty_fig
        }

    visualizations = correlation_report['visualizations']
    for chart_type in ['correlation_matrix', 'redundancy_groups', 'feature_importance']:
        if chart_type in visualizations:
            figures[chart_type] = style_figure(visualizations[chart_type], chart_type)
        else:
            empty_fig = go.Figure().update_layout(title=f"{chart_type.replace('_', ' ').title()} Not Available")
            figures[chart_type] = style_figure(empty_fig, chart_type)

    return figures

def create_correlation_dashboard_component(correlation_report):
    """
    Create a Dash HTML component displaying a summary of the correlation analysis
    
    Parameters:
    - correlation_report: Dictionary with report components from CorrelationAnalysis.generate_correlation_report
    
    Returns:
    - Dash HTML component with correlation summary
    """
    if correlation_report is None or not all(key in correlation_report for key in ['redundancy_groups', 'unique_indicators']):
        return html.Div([
            html.P("Correlation analysis data not available or incomplete.")
        ])
    
    # Create summary of redundancy groups
    redundancy_groups = html.Div([
        html.H4("Redundancy Groups", style={'color': '#2c3e50'}),
        html.P("These groups of indicators provide similar information and can be reduced to a single representative:"),
        html.Table(
            # Header row
            [html.Tr([html.Th("Group"), html.Th("Indicators")])] +
            # Data rows
            [html.Tr([
                html.Td(f"Group {i+1}"), 
                html.Td(", ".join(group))
            ]) for i, group in enumerate(correlation_report['redundancy_groups'])],
            style={
                'width': '100%',
                'border-collapse': 'collapse',
                'margin-bottom': '20px'
            }
        ) if correlation_report['redundancy_groups'] else html.P("No redundancy groups identified.")
    ])
    
    # Create summary of unique indicators
    unique_indicators = html.Div([
        html.H4("Unique Indicators", style={'color': '#2c3e50'}),
        html.P("These indicators provide unique information and should be prioritized:"),
        html.Ul([
            html.Li(indicator) for indicator in correlation_report['unique_indicators']
        ]) if correlation_report['unique_indicators'] else html.P("No unique indicators identified.")
    ])
    
    # Create summary of feature importance if available
    feature_importance = html.Div()
    if 'feature_importance' in correlation_report and correlation_report['feature_importance'] is not None:
        # Sort by importance
        sorted_importance = correlation_report['feature_importance'].sort_values(ascending=False)
        # Take top 10 for display
        top_indicators = sorted_importance.head(10)
        
        feature_importance = html.Div([
            html.H4("Top 10 Most Important Indicators", style={'color': '#2c3e50'}),
            html.P("These indicators have the highest predictive power:"),
            html.Table(
                # Header row
                [html.Tr([html.Th("Indicator"), html.Th("Importance")])] +
                # Data rows
                [html.Tr([
                    html.Td(indicator), 
                    html.Td(f"{importance:.4f}")
                ]) for indicator, importance in top_indicators.items()],
                style={
                    'width': '100%',
                    'border-collapse': 'collapse',
                    'margin-bottom': '20px'
                }
            )
        ])
    
    # Create recommendations
    optimal_indicators = html.Div()
    if 'optimal_indicators' in correlation_report and correlation_report['optimal_indicators']:
        optimal_indicators = html.Div([
            html.H4("Recommended Optimal Indicator Set", style={'color': '#2c3e50'}),
            html.P("For the best trading performance, consider focusing on this subset of indicators:"),
            html.Ul([
                html.Li(indicator) for indicator in correlation_report['optimal_indicators']
            ])
        ])
    
    # Multi-method analysis results (new section)
    multi_method_section = html.Div()
    if 'multi_method_results' in correlation_report and correlation_report['multi_method_results']:
        # Extract results from multiple method combinations
        multi_results = correlation_report['multi_method_results']
        
        # Create a section for consistent indicators across methods
        consistent_indicators_div = html.Div()
        if 'consistent_indicators' in correlation_report:
            consistent_indicators = correlation_report.get('consistent_indicators', [])
            consistent_indicators_div = html.Div([
                html.H4("Consistent Indicators Across Methods", style={'color': '#2c3e50'}),
                html.P("These indicators show consistent importance across different correlation and feature selection methods:"),
                html.Ul([
                    html.Li(indicator) for indicator in consistent_indicators
                ]) if consistent_indicators else html.P("No consistently important indicators identified across methods.")
            ])
        
        # Create a comparison of results across different methods if available
        method_comparison_table = html.Div()
        if len(multi_results) > 1:
            # Create a table comparing feature importance across methods
            # Show top 5 indicators from each method
            top_indicators_by_method = {}
            all_top_indicators = set()
            
            for method_name, result in multi_results.items():
                if 'feature_importance' in result:
                    # Get top 5 indicators
                    top_5 = result['feature_importance'].nlargest(5).index.tolist()
                    top_indicators_by_method[method_name] = top_5
                    all_top_indicators.update(top_5)
            
            if top_indicators_by_method:
                # Create table header with method names
                method_names = list(top_indicators_by_method.keys())
                
                method_comparison_table = html.Div([
                    html.H4("Top 5 Indicators by Method", style={'color': '#2c3e50'}),
                    html.P("Comparison of top indicators across different analysis methods:"),
                    html.Table(
                        # Header row
                        [html.Tr([html.Th("Method")] + [html.Th(f"#{i+1}") for i in range(5)])] +
                        # Data rows for each method
                        [html.Tr([
                            html.Td(method_name),
                            *[html.Td(top_indicators_by_method[method_name][i] if i < len(top_indicators_by_method[method_name]) else "-") 
                              for i in range(5)]
                        ]) for method_name in method_names],
                        style={
                            'width': '100%',
                            'border-collapse': 'collapse',
                            'margin-bottom': '20px'
                        }
                    )
                ])
        
        # Create a summary of redundancy groups that are consistent across methods
        consistent_redundancy_div = html.Div()
        if 'consistent_redundant_pairs' in correlation_report:
            consistent_pairs = correlation_report.get('consistent_redundant_pairs', [])
            consistent_redundancy_div = html.Div([
                html.H4("Consistently Redundant Indicators", style={'color': '#2c3e50'}),
                html.P("These indicator pairs show consistent redundancy across different correlation methods:"),
                html.Ul([
                    html.Li(f"{pair[0]} and {pair[1]}") for pair in consistent_pairs
                ]) if consistent_pairs else html.P("No consistently redundant indicator pairs identified across methods.")
            ])
        
        # Combine multi-method sections
        multi_method_section = html.Div([
            html.H3("Multi-Method Analysis Results", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '20px'}),
            html.P(f"Analysis performed using {len(multi_results)} different combinations of correlation and feature selection methods."),
            consistent_indicators_div,
            consistent_redundancy_div,
            method_comparison_table
        ], style={
            'backgroundColor': '#f0f8ff',
            'padding': '15px',
            'borderRadius': '5px',
            'border': '1px solid #b0c4de',
            'marginTop': '20px',
            'marginBottom': '20px'
        }) if multi_results else html.Div()
    
    # Combine all elements
    return html.Div([
        html.H3("Correlation Analysis Summary", style={'textAlign': 'center', 'marginBottom': '20px'}),
        redundancy_groups,
        unique_indicators,
        feature_importance,
        optimal_indicators,
        multi_method_section,  # Add the multi-method section
        html.Hr(),
        html.P("This analysis helps identify which indicators provide unique information and which are redundant, allowing for a more focused trading strategy.", style={'fontStyle': 'italic'})
    ], style={
        'backgroundColor': '#f9f9f9',
        'padding': '20px',
        'borderRadius': '5px',
        'border': '1px solid #ddd',
        'marginTop': '20px',
        'marginBottom': '20px'
    })

def create_portfolio_report_charts(portfolio_report, symbol='Portfolio'):
    """
    Create Plotly figures from portfolio report visualizations
    
    Parameters:
    - portfolio_report: Dictionary with report components from PortfolioOptimizer.generate_portfolio_report
    - symbol: Stock symbol or portfolio name
    
    Returns:
    - Dictionary of plotly figures for portfolio report visualizations
    """
    figures = {}
    
    # Check if report is available
    if portfolio_report is None or 'visualizations' not in portfolio_report:
        # Create empty placeholder figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Portfolio Report Data Available",
            annotations=[{
                'text': 'Portfolio report data not available',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        return {
            'efficient_frontier': empty_fig,
            'portfolio_allocation_pie': empty_fig,
            'portfolio_allocation_bar': empty_fig,
            'risk_contribution': empty_fig,
            'portfolio_performance': empty_fig
        }
    
    # Create efficient frontier figure
    if 'efficient_frontier' in portfolio_report:
        efficient_frontier = portfolio_report['efficient_frontier']
        
        # Create figure
        fig_ef = go.Figure()
        
        # Add scatter points for efficient frontier
        # Extract the relevant columns from the DataFrame
        fig_ef.add_trace(
            go.Scatter(
                x=efficient_frontier['volatility'],
                y=efficient_frontier['return'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=efficient_frontier['sharpe_ratio'],
                    colorscale='Viridis',
                    colorbar=dict(title='Sharpe Ratio'),
                    showscale=True
                ),
                text=[f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.2f}" 
                     for r, v, s in zip(efficient_frontier['return'], 
                                        efficient_frontier['volatility'], 
                                        efficient_frontier['sharpe_ratio'])],
                hoverinfo='text',
                name='Portfolios'
            )
        )
        
        # Highlight minimum volatility and maximum Sharpe ratio portfolios
        min_vol_portfolio = efficient_frontier[efficient_frontier['portfolio_type'] == 'Minimum Volatility']
        max_sharpe_portfolio = efficient_frontier[efficient_frontier['portfolio_type'] == 'Maximum Sharpe Ratio']
        
        if not min_vol_portfolio.empty:
            fig_ef.add_trace(
                go.Scatter(
                    x=min_vol_portfolio['volatility'],
                    y=min_vol_portfolio['return'],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='red',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name='Minimum Volatility'
                )
            )
        
        if not max_sharpe_portfolio.empty:
            fig_ef.add_trace(
                go.Scatter(
                    x=max_sharpe_portfolio['volatility'],
                    y=max_sharpe_portfolio['return'],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='green',
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name='Maximum Sharpe Ratio'
                )
            )
        
        # Update layout
        fig_ef.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Annualized)',
            yaxis_title='Return (Annualized)',
            xaxis=dict(tickformat='.0%'),
            yaxis=dict(tickformat='.0%'),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template='plotly_white',
            height=600
        )
        
        figures['efficient_frontier'] = fig_ef
    
    # Extract weights from recommendations
    if 'recommendations' in portfolio_report and 'optimal_weights' in portfolio_report['recommendations']:
        weights = portfolio_report['recommendations']['optimal_weights']
        
        # Create pie chart for portfolio allocation
        fig_pie = px.pie(
            data_frame=pd.DataFrame({
                'Asset': weights.index,
                'Weight': weights.values,
                'Percentage': [round(w * 100, 2) for w in weights.values]
            }),
            values='Weight',
            names='Asset',
            title=f"{symbol} Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.G10,
            hole=0.3,
            hover_data=['Percentage']
        )
        
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='%{label}: %{value:.2%}'
        )
        
        figures['portfolio_allocation_pie'] = fig_pie
        
        # Create bar chart for portfolio allocation
        fig_bar = px.bar(
            x=weights.index,
            y=weights.values,
            title=f"{symbol} Portfolio Weights",
            color=weights.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Asset', 'y': 'Weight'},
            text=[f"{w:.2%}" for w in weights.values]
        )
        
        fig_bar.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='%{x}: %{y:.2%}'
        )
        
        fig_bar.update_layout(
            height=500,
            yaxis=dict(tickformat='.0%'),
            xaxis={'categoryorder':'total descending'}
        )
        
        figures['portfolio_allocation_bar'] = fig_bar
        
        # Create risk contribution chart if returns_df is available
        if 'returns_df' in portfolio_report and portfolio_report['returns_df'] is not None:
            returns_df = portfolio_report['returns_df']
            
            # Create a PortfolioOptimizer instance to calculate risk contribution
            optimizer = PortfolioOptimizer()
            metrics = optimizer.calculate_portfolio_metrics(returns_df)
            cov_matrix = metrics['cov_matrix']
            
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)
            
            # Calculate risk contribution
            risk_contribution = weights * np.dot(cov_matrix, weights) * 252 / portfolio_vol
            
            # Normalize to percentage
            risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100
            
            # Create bar chart for risk contribution
            fig_risk = px.bar(
                x=risk_contribution_pct.index,
                y=risk_contribution_pct.values,
                title=f"{symbol} Risk Contribution",
                color=risk_contribution_pct.values,
                color_continuous_scale='Viridis',
                labels={'x': 'Asset', 'y': 'Risk Contribution (%)'},
                text=[f"{r:.2f}%" for r in risk_contribution_pct.values]
            )
            
            fig_risk.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                hovertemplate='%{x}: %{y:.2f}%'
            )
            
            fig_risk.update_layout(
                height=500,
                xaxis={'categoryorder':'total descending'}
            )
            
            figures['risk_contribution'] = fig_risk
        
        # Create performance metrics visualization
        if 'performance' in portfolio_report['recommendations']:
            performance = portfolio_report['recommendations']['performance']
            
            # Get additional risk metrics
            var = portfolio_report['recommendations'].get('var', 0)
            cvar = portfolio_report['recommendations'].get('cvar', 0)
            
            # Create gauge charts for key metrics
            fig_perf = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=("Expected Return", "Volatility", "Sharpe Ratio", "VaR (95%)"),
                vertical_spacing=0.1
            )
            
            # Expected Return gauge
            fig_perf.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=performance.get('returns', 0) * 100,
                    title={'text': "Expected Return (%)"},
                    number={'suffix': "%", 'valueformat': '.2f'},
                    gauge={
                        'axis': {'range': [None, max(15, math.ceil(performance.get('returns', 0) * 100))]},
                        'bar': {'color': "green" if performance.get('returns', 0) > 0 else "red"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgray"},
                            {'range': [5, 10], 'color': "gray"},
                            {'range': [10, 15], 'color': "darkgray"}
                        ]
                    }
                ),
                row=1, col=1
            )
            
            # Volatility gauge
            fig_perf.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=performance.get('volatility', 0) * 100,
                    title={'text': "Volatility (%)"},
                    number={'suffix': "%", 'valueformat': '.2f'},
                    gauge={
                        'axis': {'range': [0, max(30, math.ceil(performance.get('volatility', 0) * 100))]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "yellow"},
                            {'range': [20, 30], 'color': "lightcoral"}
                        ]
                    }
                ),
                row=1, col=2
            )
            
            # Sharpe Ratio gauge
            sharpe = performance.get('sharpe_ratio', 0)
            fig_perf.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sharpe,
                    title={'text': "Sharpe Ratio"},
                    number={'valueformat': '.2f'},
                    gauge={
                        'axis': {'range': [0, max(3, math.ceil(sharpe))]},
                        'bar': {'color': "blue"},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "lightblue"},
                            {'range': [2, 3], 'color': "royalblue"}
                        ]
                    }
                ),
                row=2, col=1
            )
            
            # VaR gauge
            fig_perf.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=var * 100 if var else 0,
                    title={'text': "Value at Risk (95%)"},
                    number={'suffix': "%", 'valueformat': '.2f'},
                    gauge={
                        'axis': {'range': [0, max(10, math.ceil(var * 100) if var else 5)]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 3], 'color': "lightgreen"},
                            {'range': [3, 6], 'color': "yellow"},
                            {'range': [6, 10], 'color': "lightcoral"}
                        ]
                    }
                ),
                row=2, col=2
            )
            
            fig_perf.update_layout(
                title=f"{symbol} Portfolio Performance Metrics",
                height=700,
                grid={'rows': 2, 'columns': 2, 'pattern': "independent"}
            )
            
            figures['portfolio_performance'] = fig_perf
    
    # Return all figures
    return figures

def create_portfolio_report_component(portfolio_report, symbol='Portfolio'):
    """
    Create a Dash HTML component displaying a summary of the portfolio report
    
    Parameters:
    - portfolio_report: Dictionary with report components from PortfolioOptimizer.generate_portfolio_report
    - symbol: Stock symbol or portfolio name
    
    Returns:
    - Dash HTML component with portfolio report summary
    """
    if portfolio_report is None or 'recommendations' not in portfolio_report:
        return html.Div([
            html.P("Portfolio report data not available or incomplete.")
        ])
    
    # Extract recommendations
    recommendations = portfolio_report['recommendations']
    
    # Check if we need to rebalance
    rebalance_needed = recommendations.get('rebalance_needed', False)
    rebalance_style = {'color': 'red', 'fontWeight': 'bold'} if rebalance_needed else {'color': 'green'}
    
    # Get risk profile
    risk_profile = recommendations.get('risk_profile', 'moderate')
    
    # Create performance section
    performance = recommendations.get('performance', {})
    performance_section = html.Div([
        html.H4("Portfolio Performance", style={'color': '#2c3e50'}),
        html.Table(
            # Header row
            [html.Tr([html.Th("Metric"), html.Th("Value")])],
            style={
                'width': '100%',
                'border-collapse': 'collapse',
                'margin-bottom': '20px'
            }
        ) if not performance else html.Table(
            # Header row
            [html.Tr([html.Th("Metric"), html.Th("Value")])] +
            # Data rows
            [
                html.Tr([html.Td("Expected Return"), html.Td(f"{performance.get('returns', 0):.2%}")]),
                html.Tr([html.Td("Volatility"), html.Td(f"{performance.get('volatility', 0):.2%}")]),
                html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{performance.get('sharpe_ratio', 0):.2f}")]),
                html.Tr([html.Td("Value at Risk (95%)"), html.Td(f"{recommendations.get('var', 0):.2%}")]),
                html.Tr([html.Td("Conditional VaR (95%)"), html.Td(f"{recommendations.get('cvar', 0):.2%}")])
            ],
            style={
                'width': '100%',
                'border-collapse': 'collapse',
                'margin-bottom': '20px'
            }
        )
    ])
    
    # Create position sizes section
    if 'risk_adjusted_position_sizes' in recommendations:
        position_sizes = recommendations['risk_adjusted_position_sizes']
        position_section = html.Div([
            html.H4("Recommended Positions (Risk-Adjusted)", style={'color': '#2c3e50'}),
            html.Table(
                # Header row
                [html.Tr([html.Th("Asset"), html.Th("Allocation"), html.Th("Position Size")])] +
                # Data rows
                [html.Tr([
                    html.Td(asset), 
                    html.Td(f"{weight:.2%}"),
                    html.Td(f"${pos_size:,.2f}")
                ]) for asset, (weight, pos_size) in zip(
                    position_sizes.index, 
                    zip(recommendations['optimal_weights'], position_sizes)
                )],
                style={
                    'width': '100%',
                    'border-collapse': 'collapse',
                    'margin-bottom': '20px'
                }
            )
        ])
    else:
        position_section = html.Div()
    
    # Create risk management section
    if 'risk_management' in recommendations:
        risk_mgmt = recommendations['risk_management']
        risk_section = html.Div([
            html.H4("Risk Management Parameters", style={'color': '#2c3e50'}),
            html.Table(
                # Header row
                [html.Tr([html.Th("Parameter"), html.Th("Value")])] +
                # Data rows
                [
                    html.Tr([html.Td("Maximum Drawdown"), html.Td(f"{risk_mgmt.get('max_drawdown', 0):.2%}")]),
                    html.Tr([html.Td("Stop Loss"), html.Td(f"{risk_mgmt.get('stop_loss_pct', 0):.2%}")]),
                    html.Tr([html.Td("Take Profit"), html.Td(f"{risk_mgmt.get('take_profit_pct', 0):.2%}")]),
                    html.Tr([html.Td("Total Capital at Risk"), html.Td(f"${risk_mgmt.get('total_capital_at_risk', 0):,.2f}")])
                ],
                style={
                    'width': '100%',
                    'border-collapse': 'collapse',
                    'margin-bottom': '20px'
                }
            )
        ])
    else:
        risk_section = html.Div()
    
    # Combine all sections
    return html.Div([
        html.H3(f"{symbol} Portfolio Optimization Report", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P(f"Risk Profile: {risk_profile.capitalize()}", style={'fontWeight': 'bold'}),

        html.P(f"Rebalancing: {'Needed' if rebalance_needed else 'Not Needed'}", style=rebalance_style),
        html.Hr(),
        performance_section,
        position_section,
        risk_section,
        html.Hr(),
        html.P("This portfolio optimization is based on historical returns and volatility. Future performance may vary.", 
              style={'fontStyle': 'italic', 'marginTop': '20px'})
    ], style={
        'backgroundColor': '#f9f9f9',
        'padding': '20px',
        'borderRadius': '5px',
        'border': '1px solid #ddd',
        'marginTop': '20px',
        'marginBottom': '20px'
    })