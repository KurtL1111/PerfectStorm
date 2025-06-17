import os
import sys
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_data_retrieval import MarketDataRetriever
from technical_indicators import TechnicalIndicators

# Initialize the app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))

# Define the layout
app.layout = html.Div([
    html.H1("Perfect Storm Dashboard - Demo Scenarios", 
            style={'textAlign': 'center', 'marginBottom': '30px', 'marginTop': '20px'}),
    
    dbc.Tabs([
        dbc.Tab(label="Scenario 1: Bullish Perfect Storm", children=[
            html.Div([
                html.H3("Bullish Perfect Storm Scenario", 
                        style={'marginTop': '20px', 'marginBottom': '10px'}),
                html.P("This scenario demonstrates a bullish Perfect Storm setup where multiple technical indicators align to signal a strong buying opportunity."),
                
                dbc.Card([
                    dbc.CardHeader("Scenario Details"),
                    dbc.CardBody([
                        html.P("Stock: AAPL"),
                        html.P("Date: March 15, 2023"),
                        html.P("Time Frame: Daily"),
                        html.P("Market Regime: Bullish Trending"),
                        html.P("Perfect Storm Confidence: 92%")
                    ])
                ], style={'marginBottom': '20px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Price Action", style={'textAlign': 'center'}),
                        dcc.Graph(id='bullish-price-chart')
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Technical Indicators", style={'textAlign': 'center'}),
                        dcc.Graph(id='bullish-indicators-chart')
                    ], width=12, lg=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Perfect Storm Analysis", style={'textAlign': 'center'}),
                        html.Div([
                            html.H5("Indicator Alignment"),
                            
                            html.Div([
                                html.Div("RSI (65): Bullish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '65%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("MACD: Bullish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '80%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Stochastic (82): Bullish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '82%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Bollinger Bands: Bullish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '75%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Moving Averages: Bullish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '90%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Volume: Bullish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '70%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.H5("Action Items", style={'marginTop': '20px'}),
                            dbc.Alert([
                                html.Strong("BUY Signal Detected"),
                                html.P("Confidence: 92%"),
                                html.P("Suggested Entry: $150.25"),
                                html.P("Stop Loss: $145.50"),
                                html.P("Take Profit: $165.75")
                            ], color="success")
                        ], className="analysis-panel")
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Pattern Recognition", style={'textAlign': 'center'}),
                        dcc.Graph(id='bullish-pattern-chart'),
                        html.Div([
                            html.H5("Detected Patterns"),
                            html.Ul([
                                html.Li("Cup and Handle (Confidence: 87%)"),
                                html.Li("Golden Cross (Confidence: 95%)"),
                                html.Li("Bullish Engulfing (Confidence: 82%)")
                            ])
                        ], className="analysis-panel")
                    ], width=12, lg=6)
                ])
            ])
        ]),
        
        dbc.Tab(label="Scenario 2: Bearish Perfect Storm", children=[
            html.Div([
                html.H3("Bearish Perfect Storm Scenario", 
                        style={'marginTop': '20px', 'marginBottom': '10px'}),
                html.P("This scenario demonstrates a bearish Perfect Storm setup where multiple technical indicators align to signal a strong selling opportunity."),
                
                dbc.Card([
                    dbc.CardHeader("Scenario Details"),
                    dbc.CardBody([
                        html.P("Stock: MSFT"),
                        html.P("Date: September 22, 2023"),
                        html.P("Time Frame: Daily"),
                        html.P("Market Regime: Bearish Trending"),
                        html.P("Perfect Storm Confidence: 88%")
                    ])
                ], style={'marginBottom': '20px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Price Action", style={'textAlign': 'center'}),
                        dcc.Graph(id='bearish-price-chart')
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Technical Indicators", style={'textAlign': 'center'}),
                        dcc.Graph(id='bearish-indicators-chart')
                    ], width=12, lg=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Perfect Storm Analysis", style={'textAlign': 'center'}),
                        html.Div([
                            html.H5("Indicator Alignment"),
                            
                            html.Div([
                                html.Div("RSI (28): Bearish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '72%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("MACD: Bearish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '85%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Stochastic (15): Bearish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '85%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Bollinger Bands: Bearish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '80%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Moving Averages: Bearish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '88%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Volume: Bearish", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '75%'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.H5("Action Items", style={'marginTop': '20px'}),
                            dbc.Alert([
                                html.Strong("SELL Signal Detected"),
                                html.P("Confidence: 88%"),
                                html.P("Suggested Entry: $325.75"),
                                html.P("Stop Loss: $335.25"),
                                html.P("Take Profit: $300.50")
                            ], color="danger")
                        ], className="analysis-panel")
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Pattern Recognition", style={'textAlign': 'center'}),
                        dcc.Graph(id='bearish-pattern-chart'),
                        html.Div([
                            html.H5("Detected Patterns"),
                            html.Ul([
                                html.Li("Head and Shoulders (Confidence: 92%)"),
                                html.Li("Death Cross (Confidence: 90%)"),
                                html.Li("Bearish Engulfing (Confidence: 85%)")
                            ])
                        ], className="analysis-panel")
                    ], width=12, lg=6)
                ])
            ])
        ]),
        
        dbc.Tab(label="Scenario 3: Market Regime Change", children=[
            html.Div([
                html.H3("Market Regime Change Scenario", 
                        style={'marginTop': '20px', 'marginBottom': '10px'}),
                html.P("This scenario demonstrates how the Perfect Storm Dashboard detects and adapts to changes in market regimes."),
                
                dbc.Card([
                    dbc.CardHeader("Scenario Details"),
                    dbc.CardBody([
                        html.P("Stock: GOOGL"),
                        html.P("Date Range: July 1 - August 15, 2023"),
                        html.P("Time Frame: Daily"),
                        html.P("Initial Regime: Ranging"),
                        html.P("New Regime: Bullish Trending"),
                        html.P("Transition Confidence: 85%")
                    ])
                ], style={'marginBottom': '20px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Market Regime Detection", style={'textAlign': 'center'}),
                        dcc.Graph(id='regime-change-chart')
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Regime Characteristics", style={'textAlign': 'center'}),
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Ranging Regime (Before)"),
                                        dbc.CardBody([
                                            html.P("Average Daily Return: 0.12%"),
                                            html.P("Volatility: Low (0.8%)"),
                                            html.P("Average Volume: 1.2M"),
                                            html.P("Average RSI: 52"),
                                            html.P("Optimal Strategy: Mean Reversion")
                                        ])
                                    ])
                                ], width=12, lg=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Bullish Trending Regime (After)"),
                                        dbc.CardBody([
                                            html.P("Average Daily Return: 0.45%"),
                                            html.P("Volatility: Medium (1.5%)"),
                                            html.P("Average Volume: 2.1M"),
                                            html.P("Average RSI: 68"),
                                            html.P("Optimal Strategy: Trend Following")
                                        ])
                                    ])
                                ], width=12, lg=6)
                            ])
                        ], className="analysis-panel")
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Adaptive Parameters", style={'textAlign': 'center'}),
                        html.Div([
                            html.H5("Parameter Adjustments"),
                            
                            html.Div([
                                html.Div("RSI Thresholds", className="indicator-label"),
                                html.Div([
                                    html.Span("Before: 30/70", style={'marginRight': '10px'}),
                                    html.Span("→"),
                                    html.Span("After: 40/80", style={'marginLeft': '10px'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Stop Loss", className="indicator-label"),
                                html.Div([
                                    html.Span("Before: 2%", style={'marginRight': '10px'}),
                                    html.Span("→"),
                                    html.Span("After: 3%", style={'marginLeft': '10px'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Take Profit", className="indicator-label"),
                                html.Div([
                                    html.Span("Before: 3%", style={'marginRight': '10px'}),
                                    html.Span("→"),
                                    html.Span("After: 6%", style={'marginLeft': '10px'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("Position Size", className="indicator-label"),
                                html.Div([
                                    html.Span("Before: 5%", style={'marginRight': '10px'}),
                                    html.Span("→"),
                                    html.Span("After: 8%", style={'marginLeft': '10px'})
                                ])
                            ], className="perfect-storm-indicator"),
                            
                            html.H5("Strategy Adjustment", style={'marginTop': '20px'}),
                            dbc.Alert([
                                html.Strong("Strategy Change Detected"),
                                html.P("Previous: Mean Reversion"),
                                html.P("New: Trend Following"),
                                html.P("Confidence: 85%"),
                                html.P("Recommended Action: Increase position size and extend profit targets")
                            ], color="info")
                        ], className="analysis-panel")
                    ], width=12, lg=6)
                ])
            ])
        ]),
        
        dbc.Tab(label="Scenario 4: Portfolio Optimization", children=[
            html.Div([
                html.H3("Portfolio Optimization Scenario", 
                        style={'marginTop': '20px', 'marginBottom': '10px'}),
                html.P("This scenario demonstrates how the Perfect Storm Dashboard optimizes a portfolio based on signal strength and risk management."),
                
                dbc.Card([
                    dbc.CardHeader("Scenario Details"),
                    dbc.CardBody([
                        html.P("Portfolio: AAPL, MSFT, GOOGL, AMZN, TSLA"),
                        html.P("Date: October 5, 2023"),
                        html.P("Initial Capital: $100,000"),
                        html.P("Risk Tolerance: Moderate"),
                        html.P("Investment Horizon: 6 months")
                    ])
                ], style={'marginBottom': '20px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Signal Strength Analysis", style={'textAlign': 'center'}),
                        dcc.Graph(id='portfolio-signals-chart')
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Optimal Allocation", style={'textAlign': 'center'}),
                        dcc.Graph(id='portfolio-allocation-chart')
                    ], width=12, lg=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Portfolio Recommendations", style={'textAlign': 'center'}),
                        html.Div([
                            html.H5("Asset Allocation"),
                            
                            html.Div([
                                html.Div("AAPL (Strong Buy)", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '35%'})
                                ]),
                                html.Div("35%", className="indicator-value")
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("MSFT (Buy)", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-positive", 
                                             style={'width': '25%'})
                                ]),
                                html.Div("25%", className="indicator-value")
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("GOOGL (Neutral)", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-neutral", 
                                             style={'width': '15%'})
                                ]),
                                html.Div("15%", className="indicator-value")
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("AMZN (Neutral)", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-neutral", 
                                             style={'width': '15%'})
                                ]),
                                html.Div("15%", className="indicator-value")
                            ], className="perfect-storm-indicator"),
                            
                            html.Div([
                                html.Div("TSLA (Sell)", className="indicator-label"),
                                html.Div(className="indicator-bar-container", children=[
                                    html.Div(className="indicator-bar indicator-bar-negative", 
                                             style={'width': '10%'})
                                ]),
                                html.Div("10%", className="indicator-value")
                            ], className="perfect-storm-indicator"),
                            
                            html.H5("Risk Management", style={'marginTop': '20px'}),
                            dbc.Alert([
                                html.Strong("Portfolio Risk Analysis"),
                                html.P("Expected Return: 12.5% (6 months)"),
                                html.P("Expected Volatility: 15.2%"),
                                html.P("Sharpe Ratio: 1.65"),
                                html.P("Maximum Drawdown: 8.3%"),
                                html.P("Diversification Score: 82/100")
                            ], color="info")
                        ], className="analysis-panel")
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Correlation Analysis", style={'textAlign': 'center'}),
                        dcc.Graph(id='portfolio-correlation-chart'),
                        html.Div([
                            html.H5("Correlation Insights"),
                            html.Ul([
                                html.Li("AAPL and MSFT show moderate correlation (0.65)"),
                                html.Li("TSLA has lowest correlation with other assets (0.35 avg)"),
                                html.Li("GOOGL and AMZN show high correlation (0.82)"),
                                html.Li("Overall portfolio correlation: Medium")
                            ])
                        ], className="analysis-panel")
                    ], width=12, lg=6)
                ])
            ])
        ]),
        
        dbc.Tab(label="Scenario 5: Backtesting Results", children=[
            html.Div([
                html.H3("Backtesting Results Scenario", 
                        style={'marginTop': '20px', 'marginBottom': '10px'}),
                html.P("This scenario demonstrates the backtesting capabilities of the Perfect Storm Dashboard, showing historical performance of the strategy."),
                
                dbc.Card([
                    dbc.CardHeader("Scenario Details"),
                    dbc.CardBody([
                        html.P("Stock: SPY (S&P 500 ETF)"),
                        html.P("Date Range: January 2020 - December 2023"),
                        html.P("Initial Capital: $100,000"),
                        html.P("Strategy: Perfect Storm with default parameters"),
                        html.P("Benchmark: Buy and Hold SPY")
                    ])
                ], style={'marginBottom': '20px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Equity Curve", style={'textAlign': 'center'}),
                        dcc.Graph(id='backtest-equity-chart')
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Performance Metrics", style={'textAlign': 'center'}),
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Perfect Storm Strategy"),
                                        dbc.CardBody([
                                            html.P("Total Return: 87.5%"),
                                            html.P("Annualized Return: 21.2%"),
                                            html.P("Sharpe Ratio: 1.85"),
                                            html.P("Maximum Drawdown: 15.3%"),
                                            html.P("Win Rate: 68.5%"),
                                            html.P("Profit Factor: 2.35"),
                                            html.P("Recovery Factor: 5.72")
                                        ])
                                    ])
                                ], width=12, lg=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Buy and Hold Benchmark"),
                                        dbc.CardBody([
                                            html.P("Total Return: 58.2%"),
                                            html.P("Annualized Return: 14.8%"),
                                            html.P("Sharpe Ratio: 1.12"),
                                            html.P("Maximum Drawdown: 33.9%"),
                                            html.P("Win Rate: N/A"),
                                            html.P("Profit Factor: N/A"),
                                            html.P("Recovery Factor: 1.72")
                                        ])
                                    ])
                                ], width=12, lg=6)
                            ])
                        ], className="analysis-panel")
                    ], width=12, lg=6),
                    dbc.Col([
                        html.H4("Trade Analysis", style={'textAlign': 'center'}),
                        dcc.Graph(id='backtest-trades-chart'),
                        html.Div([
                            html.H5("Trade Statistics"),
                            html.Ul([
                                html.Li("Total Trades: 42"),
                                html.Li("Winning Trades: 29 (68.5%)"),
                                html.Li("Losing Trades: 13 (31.5%)"),
                                html.Li("Average Winner: 8.2%"),
                                html.Li("Average Loser: -3.5%"),
                                html.Li("Largest Winner: 18.7%"),
                                html.Li("Largest Loser: -7.2%"),
                                html.Li("Average Holding Period: 24 days")
                            ])
                        ], className="analysis-panel")
                    ], width=12, lg=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Parameter Optimization", style={'textAlign': 'center'}),
                        dcc.Graph(id='backtest-optimization-chart')
                    ], width=12)
                ])
            ])
        ])
    ])
])

# Callback for bullish price chart
@app.callback(
    Output('bullish-price-chart', 'figure'),
    Input('bullish-price-chart', 'id')
)
def update_bullish_price_chart(_):
    # Create sample data for a bullish scenario
    dates = pd.date_range(start='2023-02-15', end='2023-03-15')
    n = len(dates)
    
    # Create price data with an uptrend
    close = np.linspace(130, 140, n//2).tolist() + np.linspace(140, 155, n-n//2).tolist()
    open_prices = [c - np.random.uniform(0.5, 1.5) for c in close]
    high = [max(o, c) + np.random.uniform(0.5, 2.0) for o, c in zip(open_prices, close)]
    low = [min(o, c) - np.random.uniform(0.5, 2.0) for o, c in zip(open_prices, close)]
    
    # Create volume data
    volume = [np.random.uniform(0.8, 1.2) * 1000000 for _ in range(n)]
    # Add volume spike at the breakout
    volume[n//2] = 2500000
    volume[n//2+1] = 2200000
    volume[n//2+2] = 1800000
    
    # Create moving averages
    ma20 = pd.Series(close).rolling(window=5).mean()
    ma50 = pd.Series(close).rolling(window=10).mean()
    ma200 = pd.Series(close).rolling(window=20).mean()
    
    # Create Bollinger Bands
    ma20_series = pd.Series(close).rolling(window=5).mean()
    std20 = pd.Series(close).rolling(window=5).std()
    upper_band = ma20_series + 2 * std20
    lower_band = ma20_series - 2 * std20
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=('Price', 'Volume'))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=open_prices,
            high=high,
            low=low,
            close=close,
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma20,
            name='5-day MA',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma50,
            name='10-day MA',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma200,
            name='20-day MA',
            line=dict(color='purple', width=1)
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_band,
            name='Upper Band',
            line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lower_band,
            name='Lower Band',
            line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add buy signal
    fig.add_trace(
        go.Scatter(
            x=[dates[n//2]],
            y=[close[n//2] - 5],
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name='Buy Signal'
        ),
        row=1, col=1
    )
    
    # Add volume
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volume,
            name='Volume',
            marker=dict(color='rgba(0, 0, 255, 0.5)')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='AAPL Price Action - Bullish Perfect Storm',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig

# Callback for bullish indicators chart
@app.callback(
    Output('bullish-indicators-chart', 'figure'),
    Input('bullish-indicators-chart', 'id')
)
def update_bullish_indicators_chart(_):
    # Create sample data for a bullish scenario
    dates = pd.date_range(start='2023-02-15', end='2023-03-15')
    n = len(dates)
    
    # Create RSI data (rising from oversold to bullish)
    rsi = np.linspace(35, 65, n)
    
    # Create MACD data (crossing above signal line)
    macd_line = np.linspace(-2, 2, n)
    signal_line = np.linspace(0, 0, n)
    histogram = macd_line - signal_line
    
    # Create Stochastic data (rising from oversold)
    stoch_k = np.linspace(20, 80, n)
    stoch_d = np.linspace(25, 75, n)
    
    # Create figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.33, 0.33, 0.33],
                        subplot_titles=('RSI', 'MACD', 'Stochastic'))
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rsi,
            name='RSI',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add RSI overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[70, 70],
            name='Overbought',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[30, 30],
            name='Oversold',
            line=dict(color='green', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add MACD line and signal
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=macd_line,
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=signal_line,
            name='Signal',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # Add MACD histogram
    fig.add_trace(
        go.Bar(
            x=dates,
            y=histogram,
            name='Histogram',
            marker=dict(color=['red' if h < 0 else 'green' for h in histogram])
        ),
        row=2, col=1
    )
    
    # Add Stochastic
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stoch_k,
            name='%K',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stoch_d,
            name='%D',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # Add Stochastic overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[80, 80],
            name='Overbought',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[20, 20],
            name='Oversold',
            line=dict(color='green', width=1, dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Technical Indicators - Bullish Alignment',
        xaxis_title='Date',
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly_white'
    )
    
    # Update y-axis ranges
    fig.update_yaxes(range=[0, 100], title_text='RSI', row=1, col=1)
    fig.update_yaxes(title_text='MACD', row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text='Stochastic', row=3, col=1)
    
    return fig

# Callback for bullish pattern chart
@app.callback(
    Output('bullish-pattern-chart', 'figure'),
    Input('bullish-pattern-chart', 'id')
)
def update_bullish_pattern_chart(_):
    # Create sample data for a cup and handle pattern
    dates = pd.date_range(start='2023-01-15', end='2023-03-15')
    n = len(dates)
    
    # Create cup and handle pattern
    first_part = np.linspace(150, 130, n//5)
    cup_bottom = np.linspace(130, 130, n//5)
    second_part = np.linspace(130, 150, n//5)
    handle = np.linspace(150, 145, n//5)
    breakout = np.linspace(145, 165, n//5)
    
    close = np.concatenate([first_part, cup_bottom, second_part, handle, breakout])
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=close,
            name='Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add pattern annotations
    fig.add_shape(
        type="path",
        path="M 0,0 C 0.5,0.5 0.5,-0.5 1,0",
        xref="x domain",
        yref="y domain",
        x0=0.1,
        y0=0.4,
        x1=0.9,
        y1=0.6,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Add text annotations
    fig.add_annotation(
        x=dates[n//5],
        y=close[n//5] + 5,
        text="Cup Formation Begins",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//5*3],
        y=close[n//5*3] + 5,
        text="Cup Formation Complete",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//5*4],
        y=close[n//5*4] - 5,
        text="Handle Formation",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//5*4 + 5],
        y=close[n//5*4 + 5] + 5,
        text="Breakout",
        showarrow=True,
        arrowhead=1
    )
    
    # Update layout
    fig.update_layout(
        title='Cup and Handle Pattern Detected',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_white'
    )
    
    return fig

# Callback for bearish price chart
@app.callback(
    Output('bearish-price-chart', 'figure'),
    Input('bearish-price-chart', 'id')
)
def update_bearish_price_chart(_):
    # Create sample data for a bearish scenario
    dates = pd.date_range(start='2023-08-22', end='2023-09-22')
    n = len(dates)
    
    # Create price data with a downtrend
    close = np.linspace(350, 340, n//2).tolist() + np.linspace(340, 320, n-n//2).tolist()
    open_prices = [c + np.random.uniform(0.5, 1.5) for c in close]
    high = [max(o, c) + np.random.uniform(0.5, 2.0) for o, c in zip(open_prices, close)]
    low = [min(o, c) - np.random.uniform(0.5, 2.0) for o, c in zip(open_prices, close)]
    
    # Create volume data
    volume = [np.random.uniform(0.8, 1.2) * 1000000 for _ in range(n)]
    # Add volume spike at the breakdown
    volume[n//2] = 2500000
    volume[n//2+1] = 2200000
    volume[n//2+2] = 1800000
    
    # Create moving averages
    ma20 = pd.Series(close).rolling(window=5).mean()
    ma50 = pd.Series(close).rolling(window=10).mean()
    ma200 = pd.Series(close).rolling(window=20).mean()
    
    # Create Bollinger Bands
    ma20_series = pd.Series(close).rolling(window=5).mean()
    std20 = pd.Series(close).rolling(window=5).std()
    upper_band = ma20_series + 2 * std20
    lower_band = ma20_series - 2 * std20
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=('Price', 'Volume'))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=open_prices,
            high=high,
            low=low,
            close=close,
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma20,
            name='5-day MA',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma50,
            name='10-day MA',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ma200,
            name='20-day MA',
            line=dict(color='purple', width=1)
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_band,
            name='Upper Band',
            line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lower_band,
            name='Lower Band',
            line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add sell signal
    fig.add_trace(
        go.Scatter(
            x=[dates[n//2]],
            y=[close[n//2] + 5],
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Sell Signal'
        ),
        row=1, col=1
    )
    
    # Add volume
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volume,
            name='Volume',
            marker=dict(color='rgba(255, 0, 0, 0.5)')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='MSFT Price Action - Bearish Perfect Storm',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig

# Callback for bearish indicators chart
@app.callback(
    Output('bearish-indicators-chart', 'figure'),
    Input('bearish-indicators-chart', 'id')
)
def update_bearish_indicators_chart(_):
    # Create sample data for a bearish scenario
    dates = pd.date_range(start='2023-08-22', end='2023-09-22')
    n = len(dates)
    
    # Create RSI data (falling from overbought to bearish)
    rsi = np.linspace(75, 28, n)
    
    # Create MACD data (crossing below signal line)
    macd_line = np.linspace(2, -2, n)
    signal_line = np.linspace(0, 0, n)
    histogram = macd_line - signal_line
    
    # Create Stochastic data (falling from overbought)
    stoch_k = np.linspace(80, 15, n)
    stoch_d = np.linspace(75, 20, n)
    
    # Create figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.33, 0.33, 0.33],
                        subplot_titles=('RSI', 'MACD', 'Stochastic'))
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rsi,
            name='RSI',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add RSI overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[70, 70],
            name='Overbought',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[30, 30],
            name='Oversold',
            line=dict(color='green', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add MACD line and signal
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=macd_line,
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=signal_line,
            name='Signal',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # Add MACD histogram
    fig.add_trace(
        go.Bar(
            x=dates,
            y=histogram,
            name='Histogram',
            marker=dict(color=['red' if h < 0 else 'green' for h in histogram])
        ),
        row=2, col=1
    )
    
    # Add Stochastic
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stoch_k,
            name='%K',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=stoch_d,
            name='%D',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # Add Stochastic overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[80, 80],
            name='Overbought',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[dates[0], dates[-1]],
            y=[20, 20],
            name='Oversold',
            line=dict(color='green', width=1, dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Technical Indicators - Bearish Alignment',
        xaxis_title='Date',
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly_white'
    )
    
    # Update y-axis ranges
    fig.update_yaxes(range=[0, 100], title_text='RSI', row=1, col=1)
    fig.update_yaxes(title_text='MACD', row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text='Stochastic', row=3, col=1)
    
    return fig

# Callback for bearish pattern chart
@app.callback(
    Output('bearish-pattern-chart', 'figure'),
    Input('bearish-pattern-chart', 'id')
)
def update_bearish_pattern_chart(_):
    # Create sample data for a head and shoulders pattern
    dates = pd.date_range(start='2023-07-22', end='2023-09-22')
    n = len(dates)
    
    # Create head and shoulders pattern
    left_shoulder = np.linspace(320, 340, n//7) + np.linspace(340, 330, n//7)
    head_up = np.linspace(330, 350, n//7)
    head_down = np.linspace(350, 325, n//7)
    right_shoulder = np.linspace(325, 335, n//7) + np.linspace(335, 325, n//7)
    breakdown = np.linspace(325, 300, n - n//7*6)
    
    close = np.concatenate([left_shoulder, head_up, head_down, right_shoulder, breakdown])
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=close,
            name='Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add neckline
    fig.add_shape(
        type="line",
        x0=dates[0],
        y0=325,
        x1=dates[n//7*6],
        y1=325,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add text annotations
    fig.add_annotation(
        x=dates[n//7],
        y=close[n//7] + 5,
        text="Left Shoulder",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//7*3],
        y=close[n//7*3] + 5,
        text="Head",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//7*5],
        y=close[n//7*5] + 5,
        text="Right Shoulder",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//7*6 + 5],
        y=close[n//7*6 + 5] - 5,
        text="Breakdown",
        showarrow=True,
        arrowhead=1
    )
    
    fig.add_annotation(
        x=dates[n//7*3],
        y=325 - 5,
        text="Neckline",
        showarrow=False,
        font=dict(color="red")
    )
    
    # Update layout
    fig.update_layout(
        title='Head and Shoulders Pattern Detected',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_white'
    )
    
    return fig

# Callback for regime change chart
@app.callback(
    Output('regime-change-chart', 'figure'),
    Input('regime-change-chart', 'id')
)
def update_regime_change_chart(_):
    # Create sample data for a regime change scenario
    dates = pd.date_range(start='2023-07-01', end='2023-08-15')
    n = len(dates)
    
    # Create price data with a regime change from ranging to trending
    ranging_part = np.random.normal(1200, 20, n//2)
    trending_part = np.linspace(1200, 1350, n - n//2)
    
    close = np.concatenate([ranging_part, trending_part])
    
    # Create volume data
    volume_ranging = np.random.uniform(0.8, 1.2, n//2) * 1000000
    volume_trending = np.random.uniform(1.5, 2.5, n - n//2) * 1000000
    volume = np.concatenate([volume_ranging, volume_trending])
    
    # Create volatility data
    volatility_ranging = np.random.uniform(0.5, 1.0, n//2)
    volatility_trending = np.random.uniform(1.0, 2.0, n - n//2)
    volatility = np.concatenate([volatility_ranging, volatility_trending])
    
    # Create regime labels
    regime = ['Ranging'] * (n//2) + ['Bullish Trending'] * (n - n//2)
    
    # Create figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=('Price', 'Volume', 'Volatility'))
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=close,
            name='Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add regime background colors
    fig.add_vrect(
        x0=dates[0],
        x1=dates[n//2-1],
        fillcolor="yellow",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    
    fig.add_vrect(
        x0=dates[n//2],
        x1=dates[-1],
        fillcolor="green",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    
    # Add regime transition line
    fig.add_vline(
        x=dates[n//2], 
        line_width=2, 
        line_dash="dash", 
        line_color="black"
    )
    
    # Add regime labels
    fig.add_annotation(
        x=dates[n//4],
        y=close.max() + 30,
        text="Ranging Regime",
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.add_annotation(
        x=dates[n//2 + n//4],
        y=close.max() + 30,
        text="Bullish Trending Regime",
        showarrow=False,
        font=dict(size=14)
    )
    
    # Add volume
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volume,
            name='Volume',
            marker=dict(color=['rgba(255, 255, 0, 0.5)' if r == 'Ranging' else 'rgba(0, 255, 0, 0.5)' for r in regime])
        ),
        row=2, col=1
    )
    
    # Add volatility
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=volatility,
            name='Volatility',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Market Regime Change Detection',
        xaxis_title='Date',
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='Volatility (%)', row=3, col=1)
    
    return fig

# Callback for portfolio signals chart
@app.callback(
    Output('portfolio-signals-chart', 'figure'),
    Input('portfolio-signals-chart', 'id')
)
def update_portfolio_signals_chart(_):
    # Create sample data for portfolio signals
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Signal strength for each indicator (0-100)
    rsi = [65, 58, 52, 48, 35]
    macd = [75, 60, 50, 45, 30]
    stochastic = [70, 62, 55, 50, 25]
    bollinger = [68, 55, 48, 52, 32]
    moving_avg = [80, 65, 52, 48, 25]
    volume = [72, 60, 50, 45, 35]
    
    # Create figure
    fig = go.Figure()
    
    # Add radar chart for each asset
    for i, asset in enumerate(assets):
        fig.add_trace(
            go.Scatterpolar(
                r=[rsi[i], macd[i], stochastic[i], bollinger[i], moving_avg[i], volume[i], rsi[i]],
                theta=['RSI', 'MACD', 'Stochastic', 'Bollinger', 'Moving Avg', 'Volume', 'RSI'],
                fill='toself',
                name=asset
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Signal Strength Analysis by Asset',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        height=500,
        template='plotly_white'
    )
    
    return fig

# Callback for portfolio allocation chart
@app.callback(
    Output('portfolio-allocation-chart', 'figure'),
    Input('portfolio-allocation-chart', 'id')
)
def update_portfolio_allocation_chart(_):
    # Create sample data for portfolio allocation
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    allocations = [35, 25, 15, 15, 10]
    colors = ['green', 'green', 'yellow', 'yellow', 'red']
    
    # Create figure
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=assets,
            values=allocations,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hole=0.4
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Optimal Portfolio Allocation',
        height=500,
        template='plotly_white',
        annotations=[dict(text='100K', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

# Callback for portfolio correlation chart
@app.callback(
    Output('portfolio-correlation-chart', 'figure'),
    Input('portfolio-correlation-chart', 'id')
)
def update_portfolio_correlation_chart(_):
    # Create sample data for portfolio correlation
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Correlation matrix
    correlation = [
        [1.00, 0.65, 0.55, 0.48, 0.35],
        [0.65, 1.00, 0.60, 0.52, 0.40],
        [0.55, 0.60, 1.00, 0.82, 0.38],
        [0.48, 0.52, 0.82, 1.00, 0.32],
        [0.35, 0.40, 0.38, 0.32, 1.00]
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=correlation,
            x=assets,
            y=assets,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            text=[[f'{val:.2f}' for val in row] for row in correlation],
            texttemplate='%{text}',
            textfont={"size":12}
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Asset Correlation Matrix',
        height=500,
        template='plotly_white'
    )
    
    return fig

# Callback for backtest equity chart
@app.callback(
    Output('backtest-equity-chart', 'figure'),
    Input('backtest-equity-chart', 'id')
)
def update_backtest_equity_chart(_):
    # Create sample data for backtest equity curve
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='W')
    n = len(dates)
    
    # Create equity curves
    strategy_equity = [100000]
    benchmark_equity = [100000]
    
    # Generate equity curves with realistic performance
    for i in range(1, n):
        # Strategy returns (with higher returns and lower drawdowns)
        if i < n//4:  # 2020 (COVID crash and recovery)
            strategy_return = np.random.normal(0.005, 0.02)
            benchmark_return = np.random.normal(0.003, 0.03)
        elif i < n//2:  # 2021 (bull market)
            strategy_return = np.random.normal(0.008, 0.015)
            benchmark_return = np.random.normal(0.006, 0.02)
        elif i < 3*n//4:  # 2022 (bear market)
            strategy_return = np.random.normal(0.001, 0.02)
            benchmark_return = np.random.normal(-0.002, 0.025)
        else:  # 2023 (recovery)
            strategy_return = np.random.normal(0.006, 0.015)
            benchmark_return = np.random.normal(0.005, 0.02)
        
        strategy_equity.append(strategy_equity[-1] * (1 + strategy_return))
        benchmark_equity.append(benchmark_equity[-1] * (1 + benchmark_return))
    
    # Create drawdowns
    strategy_drawdown = []
    benchmark_drawdown = []
    
    strategy_peak = strategy_equity[0]
    benchmark_peak = benchmark_equity[0]
    
    for i in range(n):
        strategy_peak = max(strategy_peak, strategy_equity[i])
        benchmark_peak = max(benchmark_peak, benchmark_equity[i])
        
        strategy_drawdown.append((strategy_equity[i] / strategy_peak - 1) * 100)
        benchmark_drawdown.append((benchmark_equity[i] / benchmark_peak - 1) * 100)
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=('Equity Curve', 'Drawdown'))
    
    # Add equity curves
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strategy_equity,
            name='Perfect Storm Strategy',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=benchmark_equity,
            name='Buy and Hold',
            line=dict(color='gray', width=2)
        ),
        row=1, col=1
    )
    
    # Add drawdowns
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strategy_drawdown,
            name='Strategy Drawdown',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=benchmark_drawdown,
            name='Benchmark Drawdown',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Backtest Results: Perfect Storm Strategy vs. Buy and Hold',
        xaxis_title='Date',
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
    
    return fig

# Callback for backtest trades chart
@app.callback(
    Output('backtest-trades-chart', 'figure'),
    Input('backtest-trades-chart', 'id')
)
def update_backtest_trades_chart(_):
    # Create sample data for backtest trades
    trade_numbers = list(range(1, 43))
    trade_returns = []
    
    # Generate trade returns with more winners than losers
    for i in range(42):
        if i % 3 == 0:  # Every third trade is a loser
            trade_returns.append(np.random.uniform(-7, -1))
        else:
            trade_returns.append(np.random.uniform(1, 18))
    
    # Create figure
    fig = go.Figure()
    
    # Add trade returns
    fig.add_trace(
        go.Bar(
            x=trade_numbers,
            y=trade_returns,
            marker=dict(color=['green' if r > 0 else 'red' for r in trade_returns])
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=43,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title='Trade-by-Trade Returns',
        xaxis_title='Trade Number',
        yaxis_title='Return (%)',
        height=400,
        template='plotly_white'
    )
    
    return fig

# Callback for backtest optimization chart
@app.callback(
    Output('backtest-optimization-chart', 'figure'),
    Input('backtest-optimization-chart', 'id')
)
def update_backtest_optimization_chart(_):
    # Create sample data for parameter optimization
    param1 = list(range(5, 30, 5))  # Short window
    param2 = list(range(20, 80, 10))  # Long window
    
    # Create grid of parameters
    param_grid = []
    sharpe_ratios = []
    returns = []
    drawdowns = []
    
    for p1 in param1:
        for p2 in param2:
            if p1 < p2:  # Short window must be less than long window
                param_grid.append((p1, p2))
                # Generate random performance metrics
                sharpe_ratios.append(np.random.uniform(0.5, 2.0))
                returns.append(np.random.uniform(10, 30))
                drawdowns.append(np.random.uniform(5, 25))
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in param_grid],
            y=[p[1] for p in param_grid],
            mode='markers',
            marker=dict(
                size=15,
                color=sharpe_ratios,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio'),
                sizemode='area',
                sizeref=2.*max(returns)/(40.**2),
                sizemin=4,
                line=dict(width=1, color='black')
            ),
            text=[f'Short Window: {p[0]}<br>Long Window: {p[1]}<br>Sharpe: {s:.2f}<br>Return: {r:.2f}%<br>Drawdown: {d:.2f}%' 
                  for p, s, r, d in zip(param_grid, sharpe_ratios, returns, drawdowns)],
            hoverinfo='text'
        )
    )
    
    # Highlight optimal parameters
    optimal_idx = sharpe_ratios.index(max(sharpe_ratios))
    optimal_params = param_grid[optimal_idx]
    
    fig.add_trace(
        go.Scatter(
            x=[optimal_params[0]],
            y=[optimal_params[1]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name='Optimal Parameters'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Parameter Optimization (Optimal: Short={optimal_params[0]}, Long={optimal_params[1]})',
        xaxis_title='Short Window',
        yaxis_title='Long Window',
        height=500,
        template='plotly_white'
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='localhost')
