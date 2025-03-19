"""
Main application for Perfect Storm Dashboard

This dashboard implements the "Perfect Storm" investment strategy
developed by John R. Connelley in his book "Tech Smart".
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px    # Added import for Plotly Express
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_data_retrieval import MarketDataRetriever
from technical_indicators import TechnicalIndicators
from backtesting_engine_enhanced import BacktestingEngine
from ml_pattern_recognition_enhanced import PatternRecognition
from ml_clustering_enhanced import PerfectStormClustering
from ml_anomaly_detection_enhanced import MarketAnomalyDetection
from adaptive_thresholds_enhanced import EnhancedAdaptiveThresholds
from market_regime_detection import MarketRegimeDetection
from correlation_analysis import CorrelationAnalysis

# Initialize the app
app = dash.Dash(__name__, title="Perfect Storm Dashboard")
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Perfect Storm Investment Strategy Dashboard", style={'textAlign': 'center'}),
    html.P("Based on John R. Connelley's strategy from 'Tech Smart'", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Stock Symbol:"),
            dcc.Input(id='symbol-input', type='text', value='AAPL'),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Time Period:"),
            dcc.Dropdown(
                id='period-dropdown',
                options=[
                    {'label': '1 Month', 'value': '1mo'},
                    {'label': '3 Months', 'value': '3mo'},
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '2 Years', 'value': '2y'},
                    {'label': '5 Years', 'value': '5y'},
                ],
                value='1y'
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Button('Update Dashboard', id='update-button', n_clicks=0),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px', 'textAlign': 'center'}),
    ]),
    
    html.Div([
        dcc.Loading(
            id="loading-1",
            type="circle",
            children=[
                html.Div(id='market-data-info', style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd', 'borderRadius': '5px', 'margin': '10px 0'}),
                dcc.Graph(id='main-chart', style={'height': '600px'}),
                html.Div([
                    html.Div([
                        html.H3("Technical Indicators"),
                        dcc.Graph(id='indicators-chart', style={'height': '400px'}),
                    ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),
                ]),
                html.Div([
                    html.Div([
                        html.H3("Moving Averages"),
                        dcc.Graph(id='moving-averages-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                    html.Div([
                        html.H3("Volume Analysis"),
                        dcc.Graph(id='volume-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                ]),
                html.Div([
                    html.Div([
                        html.H3("Oscillators"),
                        dcc.Graph(id='oscillators-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                    html.Div([
                        html.H3("Market Sentiment"),
                        dcc.Graph(id='sentiment-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                ]),
                html.Div([
                    html.H3("Perfect Storm Analysis"),
                    html.Div(id='perfect-storm-analysis', style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
                ]),
                html.Div([
                    html.Div([
                        html.H3("Pattern Recognition"),
                        dcc.Graph(id='patterns-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                    html.Div([
                        html.H3("Market Clustering"),
                        dcc.Graph(id='clusters-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                ]),
                html.Div([
                    html.Div([
                        html.H3("Anomaly Detection"),
                        dcc.Graph(id='anomalies-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                    html.Div([
                        html.H3("Market Regime Detection"),
                        dcc.Graph(id='market-regime-chart', style={'height': '400px'}),
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                ]),
                html.Div([
                    html.H3("Backtesting Results"),
                    dcc.Graph(id='backtesting-results-chart', style={'height': '400px'}),
                ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),
            ]
        )
    ]),
    
    html.Div(id='alerts-div', style={'padding': '10px', 'backgroundColor': '#ffe6e6', 'border': '1px solid red', 'margin': '10px'}),
    dcc.Interval(id='real-time-alerts', interval=60000, n_intervals=0),
    
    html.Div([
        html.Hr(),
        html.P("Data sources: Yahoo Finance, MarketWatch, AAII Investor Sentiment Survey", style={'textAlign': 'center'}),
        html.P("© 2025 Perfect Storm Dashboard", style={'textAlign': 'center'}),
    ]),
])

@callback(
    [Output('market-data-info', 'children'),
     Output('main-chart', 'figure'),
     Output('indicators-chart', 'figure'),
     Output('moving-averages-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('oscillators-chart', 'figure'),
     Output('sentiment-chart', 'figure'),
     Output('patterns-chart', 'figure'),
     Output('clusters-chart', 'figure'),
     Output('anomalies-chart', 'figure'),
     Output('market-regime-chart', 'figure'),
     Output('backtesting-results-chart', 'figure'),   # Added output for backtesting results
     Output('perfect-storm-analysis', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('period-dropdown', 'value')]
)
def update_dashboard(n_clicks, symbol, period):
    """
    Update the dashboard with the latest data.
    """
    import os
    # Retrieve AlphaVantage API key from environment variables (or use "demo")
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "25WNVRI1YIXCDIH1")
    # Initialize data retriever with the API key
    data_retriever = MarketDataRetriever(api_key=api_key)
    
    # Get stock data
    stock_data = data_retriever.get_stock_history(symbol, interval='1d', period=period)
    
    # Check if stock data retrieval failed
    if stock_data is None:
        error_message = html.Div([
            html.H3("Data Retrieval Error"),
            html.P(f"Failed to retrieve stock data for {symbol}. Please check the symbol and try again."),
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No Data Available")
        # Return 13 outputs including empty figure for backtesting results
        return (error_message, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, "No data available for analysis")
    
    # Get market breadth data
    market_breadth_data = data_retriever.get_market_breadth_data()
    
    # Get sentiment data
    sentiment_data = data_retriever.get_sentiment_data()
    
    # Calculate technical indicators
    df = TechnicalIndicators.calculate_all_indicators(stock_data, market_breadth_data, sentiment_data)
    
    # Create market data info
    market_data_info = create_market_data_info(df, symbol, market_breadth_data, sentiment_data)
    
    # Create main chart
    main_chart = create_main_chart(df, symbol)
    
    # Create indicators chart
    indicators_chart = create_indicators_chart(df)
    
    # Create moving averages chart
    moving_averages_chart = create_moving_averages_chart(df)
    
    # Create volume chart
    volume_chart = create_volume_chart(df)
    
    # Create oscillators chart
    oscillators_chart = create_oscillators_chart(df)
    
    # Create sentiment chart
    sentiment_chart = create_sentiment_chart(df, sentiment_data)

    # ML Analysis
    pattern_model = PatternRecognition()
    # Optionally try loading ONNX model for inference
    try:
        pattern_model.load_onnx_model()
        # Run ONNX inference as a test; ignore output for now
        _ = pattern_model.predict(df, use_onnx=True)
    except Exception as e:
        print("ONNX runtime for pattern recognition not used, fallback to PyTorch:", e)
    patterns = pattern_model.detect_patterns(df)
    
    clustering_model = PerfectStormClustering()
    clusters = clustering_model.cluster_data(df)
    
    anomaly_model = MarketAnomalyDetection()
    anomalies = anomaly_model.detect_anomalies(df)
    
    # Adaptive thresholds integration
    thresholds_model = EnhancedAdaptiveThresholds()
    dynamic_thresholds = thresholds_model.compute_thresholds(df)
    
    # Market Regime Analysis and Backtesting – existing integration
    regime_analyzer = MarketRegimeDetection(df)
    market_regimes = regime_analyzer.detect_market_regimes()
    backtester = BacktestingEngine()
    backtest_results = backtester.run_backtest(df, market_regimes)  # Using market regime output as signals
    
    # Create Perfect Storm analysis and append adaptive thresholds info
    perfect_storm_analysis = create_perfect_storm_analysis(df)
    perfect_storm_analysis = html.Div([
        perfect_storm_analysis,
        html.H4("Adaptive Thresholds"),
        html.Pre(str(dynamic_thresholds))
    ])
    
    patterns_chart = px.line(patterns, title="Pattern Recognition")
    clusters_chart = px.scatter(clusters, title="Market Clustering")
    anomalies_chart = px.scatter(anomalies, title="Anomaly Detection")

    regime_chart = go.Figure()
    regime_chart.add_trace(go.Scatter(y=market_regimes, mode='lines+markers', name='Market Regimes'))
    regime_chart.update_layout(title="Market Regime Detection", xaxis_title="Time", yaxis_title="Regime")

    backtesting_results = create_backtesting_results(backtest_results)
    
    return (market_data_info, main_chart, indicators_chart, moving_averages_chart,
            volume_chart, oscillators_chart, sentiment_chart, patterns_chart,
            clusters_chart, anomalies_chart, regime_chart, backtesting_results,
            perfect_storm_analysis)

# New callback for real-time alerts
@callback(
    Output('alerts-div', 'children'),
    Input('real-time-alerts', 'n_intervals')
)
def update_alerts(n):
    """
    Check conditions and update alerts in real time.
    For example, if adaptive thresholds or ML modules trigger a "perfect storm".
    """
    # Dummy implementation – replace with real alert checks (e.g., using AdaptiveThresholds or pattern models)
    import random
    alerts = []
    if random.random() > 0.8:
        alerts.append("ALERT: Significant market signal detected!")
    if not alerts:
        alerts.append("No alerts at this time.")
    return html.Ul([html.Li(alert) for alert in alerts])

def create_market_data_info(df, symbol, market_breadth_data, sentiment_data):
    """
    Create market data information
    
    Parameters:
    - df: DataFrame with stock and indicator data
    - symbol: Stock symbol
    - market_breadth_data: Dictionary with market breadth data
    - sentiment_data: Dictionary with sentiment data
    
    Returns:
    - Market data information
    """
    # Get the latest data
    latest_data = df.iloc[-1]
    
    # Calculate price change and percentage change
    price_change = latest_data['close'] - df.iloc[-2]['close']
    price_change_pct = (price_change / df.iloc[-2]['close']) * 100
    
    # Check if market breadth data is available
    market_breadth_section = html.Div([
        html.H4("Market Breadth"),
        html.P("Market breadth data unavailable"),
    ], style={'width': '25%', 'display': 'inline-block'})
    
    if market_breadth_data is not None:
        market_breadth_section = html.Div([
            html.H4("Market Breadth"),
            html.P(f"Advancing Issues: {market_breadth_data['advancing_issues']}"),
            html.P(f"Declining Issues: {market_breadth_data['declining_issues']}"),
            html.P(f"ARMS Index: {latest_data.get('arms_index', 'N/A')}"),
        ], style={'width': '25%', 'display': 'inline-block'})
    
    # Check if sentiment data is available
    sentiment_section = html.Div([
        html.H4("Sentiment"),
        html.P("Sentiment data unavailable"),
    ], style={'width': '25%', 'display': 'inline-block'})
    
    if sentiment_data is not None:
        sentiment_section = html.Div([
            html.H4("Sentiment"),
            html.P(f"Bullish: {sentiment_data['bullish']:.1f}%"),
            html.P(f"Bearish: {sentiment_data['bearish']:.1f}%"),
            html.P(f"Neutral: {sentiment_data['neutral']:.1f}%"),
        ], style={'width': '25%', 'display': 'inline-block'})
    
    # Create market data info
    market_data_info = html.Div([
        html.Div([
            html.H3(f"{symbol} - {latest_data.name.strftime('%Y-%m-%d')}"),
            html.P(f"Close: ${latest_data['close']:.2f} ({price_change_pct:.2f}%)"),
        ], style={'width': '25%', 'display': 'inline-block'}),
        
        market_breadth_section,
        sentiment_section,
        
        html.Div([
            html.H4("Key Indicators"),
            html.P(f"RSI: {latest_data.get('rsi', 'N/A')}"),
            html.P(f"MACD: {latest_data.get('macd_line', 'N/A')}"),
            html.P(f"CCI: {latest_data.get('cci', 'N/A')}"),
        ], style={'width': '25%', 'display': 'inline-block'}),
    ])
    
    return market_data_info

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
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.5, 0.5],
                        subplot_titles=("MACD", "RSI"))
    
    # Add MACD
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
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            mode='lines',
            name="RSI"
        ),
        row=2, col=1
    )
    
    # Add RSI overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[70] * len(df),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name="Overbought (70)"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[30] * len(df),
            mode='lines',
            line=dict(dash='dash', color='green'),
            name="Oversold (30)"
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

def create_patterns_chart(df):
    """
    Create patterns chart
    
    Parameters:
    - df: DataFrame with stock and indicator data
    
    Returns:
    - Patterns chart figure
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
    
    # Add patterns if available
    if 'patterns' in df.columns:
        for pattern in df['patterns'].unique():
            pattern_data = df[df['patterns'] == pattern]
            fig.add_trace(
                go.Scatter(
                    x=pattern_data.index,
                    y=pattern_data['close'],
                    mode='markers',
                    name=f"Pattern: {pattern}",
                    marker=dict(size=10)
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Detected Patterns",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis label
    fig.update_yaxes(title_text="Price ($)")
    
    return fig
def create_clusters_chart(df):
    """
    Create clusters chart
    Parameters:
    - df: DataFrame with stock and indicator data
    Returns:
    - Clusters chart figure
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
    
    # Add clusters if available
    if 'clusters' in df.columns:
        for cluster in df['clusters'].unique():
            cluster_data = df[df['clusters'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data.index,
                    y=cluster_data['close'],
                    mode='markers',
                    name=f"Cluster: {cluster}",
                    marker=dict(size=10)
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Market Clusters",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis label
    fig.update_yaxes(title_text="Price ($)")
    
    return fig
def create_anomalies_chart(df):
    """
    Create anomalies chart
    Parameters:
    - df: DataFrame with stock and indicator data
    Returns:
    - Anomalies chart figure
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
    
    # Add anomalies if available
    if 'anomalies' in df.columns:
        for anomaly in df['anomalies'].unique():
            anomaly_data = df[df['anomalies'] == anomaly]
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data.index,
                    y=anomaly_data['close'],
                    mode='markers',
                    name=f"Anomaly: {anomaly}",
                    marker=dict(size=10)
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Detected Anomalies",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis label
    fig.update_yaxes(title_text="Price ($)")
    
    return fig
def create_market_regime_chart(df):
    """
    Create market regime chart
    Parameters:
    - df: DataFrame with stock and indicator data
    Returns:
    - Market regime chart figure
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
    
    # Add market regimes if available
    if 'market_regime' in df.columns:
        for regime in df['market_regime'].unique():
            regime_data = df[df['market_regime'] == regime]
            fig.add_trace(
                go.Scatter(
                    x=regime_data.index,
                    y=regime_data['close'],
                    mode='markers',
                    name=f"Market Regime: {regime}",
                    marker=dict(size=10)
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Market Regimes",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update y-axis label
    fig.update_yaxes(title_text="Price ($)")
    
    return fig

def create_backtesting_results(backtest_results):
    """
    Create a formatted display for backtesting results.
    
    Parameters:
    - backtest_results: Dictionary with backtesting performance metrics.
    
    Returns:
    - HTML Div displaying backtesting results.
    """
    return html.Div([
        html.H4("Performance Metrics"),
        html.P(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 'N/A'):.2f}"),
        html.P(f"Maximum Drawdown: {backtest_results.get('max_drawdown', 'N/A'):.2%}"),
        html.P(f"Win Rate: {backtest_results.get('win_rate', 'N/A'):.2%}"),
        html.P(f"Total Return: {backtest_results.get('total_return', 'N/A'):.2%}"),
        html.P(f"Annualized Volatility: {backtest_results.get('volatility', 'N/A'):.2%}"),
        html.P(f"Profit Factor: {backtest_results.get('profit_factor', 'N/A'):.2f}")
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
    
    return html.Div(analysis)

if __name__ == '__main__':
    app.run_server(debug=True, host='localhost')
