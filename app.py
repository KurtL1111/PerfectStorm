
# --- Enhanced Imports and Setup ---
import os
import pickle
import json # Added for json load/dump in clustering and pattern_recognition save/load
from datetime import datetime # Added for clustering and pattern_recognition save/load

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
# Plotly imports are used in dashboard_utils and individual modules
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
# from datetime import datetime, timedelta # already imported datetime

from market_data_retrieval import MarketDataRetriever # Corresponds to input_file_8.py
from technical_indicators import TechnicalIndicators # Corresponds to input_file_11.py
from backtesting_engine import BacktestingEngine # Corresponds to input_file_4.py

from ml_pattern_recognition_enhanced import MarketPatternRecognition # Corresponds to input_file_10.py
from ml_clustering_enhanced_completed import PerfectStormClustering # Corresponds to input_file_6.py
from ml_anomaly_detection_enhanced import MarketAnomalyDetection # Corresponds to input_file_7.py
from adaptive_thresholds_enhanced import EnhancedAdaptiveThresholds # Corresponds to input_file_5.py
from market_regime_detection_completed import MarketRegimeDetection # Corresponds to input_file_1.py
from correlation_analysis import CorrelationAnalysis # Corresponds to input_file_13.py
# strategy_logic is imported by quantitative_strategy
# quantitative_strategy is imported by callbacks
# portfolio_optimization is imported by callbacks

# import traceback # Available in callbacks.py
import logging # Available in callbacks.py

# dashboard_utils is imported by callbacks.py and other modules
# Ensure this is the correct path/module if dashboard_utils.py is in the same directory
# For the tool environment, if it's directly provided as a file, direct import is fine.
# from dashboard_utils import (
#     # ... functions listed ...
# )
from callbacks import register_callbacks # Corresponds to input_file_12.py

# Set up logging (this is often done in app.py or main execution script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s') # This is in callbacks.py and app.py, once is enough

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Perfect Storm Dashboard") # Added bootstrap theme
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Perfect Storm Investment Strategy Dashboard", className="text-center text-primary my-4"))),
    dbc.Row(dbc.Col(html.P("Based on John R. Connelley's strategy from 'Tech Smart'", className="text-center text-muted mb-4"))),

    dbc.Card([ # Input Controls Card
        dbc.CardHeader("Dashboard Controls"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Stock Symbol:"),
                    dbc.Input(id='symbol-input', type='text', value='BTC/USD', placeholder="e.g., TSLA, AAPL, BTC/USD"),
                ], md=3), # Use md for medium screen column width

                dbc.Col([
                    dbc.Label("Time Period:"),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '2 Years', 'value': '2y'},
                            {'label': '5 Years', 'value': '5y'},
                            {'label': '10 Years', 'value': '10y'},
                            {'label': 'All-Time', 'value': 'max'},
                        ], value='5y'
                    ),  # Added 10y and All-Time (max) options
                ], md=3),

                dbc.Col([
                    dbc.Label("Interval:"),
                    dcc.Dropdown(
                        id='interval-dropdown',
                        options=[
                            {'label': '1 Day', 'value': '1d'}, {'label': '1 Hour', 'value': '1h'},
                            {'label': '30 Minutes', 'value': '30m'}, {'label': '15 Minutes', 'value': '15m'},
                            {'label': '5 Minutes', 'value': '5m'}, {'label': '1 Minute', 'value': '1m'},
                        ], value='1m'
                    ),
                ], md=3),

                dbc.Col([
                    dbc.Button('Update Dashboard', id='update-button', n_clicks=0, color="primary", className="w-100 mt-4") # w-100 for full width, mt-4 for margin
                ], md=3, className="d-flex align-items-end"), # Align button to bottom
            ]),
        ]),
    ], className="mb-4"),

    dbc.Card([ # Manual Market Breadth Input Card
        dbc.CardHeader("Manual Market Breadth Input"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Input(id='adv-issues-input', type='number', placeholder='Advancing Issues'), md=3),
                dbc.Col(dbc.Input(id='dec-issues-input', type='number', placeholder='Declining Issues'), md=3),
                dbc.Col(dbc.Input(id='adv-volume-input', type='number', placeholder='Advancing Volume'), md=3),
                dbc.Col(dbc.Input(id='dec-volume-input', type='number', placeholder='Declining Volume'), md=3),
            ], className="mb-2"),
            dbc.Button("Save Market Breadth Data", id="save-breadth-button", n_clicks=0, color="secondary"),
            html.Div(id='save-breadth-output', className="mt-2 text-muted small")
        ]),
    ], className="mb-4"),

    dbc.Card([ # Portfolio Optimization Card
        dbc.CardHeader("Portfolio Optimization"),
        dbc.CardBody([
            html.P("Create an optimized portfolio based on historical returns.", className="card-text"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Stock Symbols (comma-separated):"),
                    dbc.Input(id='portfolio-symbols-input', type='text',
                              value='AAPL,MSFT,GOOGL,AMZN',
                              placeholder='e.g., AAPL,MSFT,GOOGL'),
                ], md=6),
                dbc.Col([
                    dbc.Label("Look-back Period:"),
                    dcc.Dropdown(
                        id='portfolio-period-dropdown',
                        options=[
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '2 Years', 'value': '2y'},
                            {'label': '5 Years', 'value': '5y'},
                            {'label': '10 Years', 'value': '10y'},
                            {'label': 'All-Time', 'value': 'max'},
                        ], value='1y'
                    ),  # Added 10y and All-Time (max) options
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Total Capital ($):"),
                    dbc.Input(id='portfolio-capital-input', type='number', value=100000, min=1000),
                ], md=4),
                dbc.Col([
                    dbc.Label("Risk Profile:"),
                    dcc.Dropdown(
                        id='portfolio-risk-profile-dropdown',
                        options=[
                            {'label': 'Conservative', 'value': 'conservative'},
                            {'label': 'Moderate', 'value': 'moderate'},
                            {'label': 'Aggressive', 'value': 'aggressive'},
                        ], value='moderate'
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Button("Generate Portfolio Report", id="generate-portfolio-button",
                               color="success", className="w-100 mt-4"), # Primary color used above, maybe success or info here
                ], md=4, className="d-flex align-items-end"),
            ]),
            html.Div(id='portfolio-report-status', className="mt-2 text-muted small"),
        ]),
    ], className="mb-4"),

    # Initially hidden Portfolio Report Container
    dbc.Collapse(
        html.Div(
            id="portfolio-report-container",
            children=[
                dbc.Card(
                    dbc.CardBody(
                        html.Div([
                            html.H3("Portfolio Optimization Report", className="text-center mb-3"),
                            dbc.Row(
                                dbc.Col(html.Div(id='portfolio-report-summary'), width=12),
                                className="mb-3"
                            ),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='portfolio-efficient-frontier'), md=12, className="mb-3"),
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='portfolio-allocation-pie'), md=6),
                                dbc.Col(dcc.Graph(id='portfolio-allocation-bar'), md=6),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='portfolio-risk-contribution'), md=6),
                                dbc.Col(dcc.Graph(id='portfolio-performance-metrics'), md=6),
                            ]),
                        ])
                    )
                )
            ]
        ),
        id="portfolio-report-collapse-container",
        is_open=False
    ),
    # Main Dashboard Content Area
    dcc.Loading(
        id="loading-main-content",
        type="default", # "circle", "cube", "dot", "default"
        children=[
            dbc.Card(dbc.CardBody(html.Div(id='market-data-info')), className="mb-3"), # Wrap in Card
            dbc.Row([
                 dbc.Col(dcc.Graph(id='main-chart'), md=12) # Use md=12 for full width on medium screens and up
            ], className="mb-3"),

            # Tabs for different chart groups
            dbc.Tabs([
                dbc.Tab(label="Core Indicators", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='indicators-chart'), md=12), # MACD & RSI
                    ]),
                     dbc.Row([
                        dbc.Col(dcc.Graph(id='moving-averages-chart'), md=6),
                        dbc.Col(dcc.Graph(id='volume-chart'), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='oscillators-chart'), md=6),
                        dbc.Col(dcc.Graph(id='sentiment-chart'), md=6), # Could be with core or its own tab
                    ]),
                ]),
                dbc.Tab(label="Pattern Recognition", children=[
                    dbc.Row(dbc.Col(dcc.Graph(id='prediction-patterns-chart'), md=12)),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='roc-curve-chart'), md=4),
                        dbc.Col(dcc.Graph(id='prec-recall-chart'), md=4),
                        dbc.Col(dcc.Graph(id='confusion-matrix-chart'), md=4),
                    ]),
                ]),
                dbc.Tab(label="Clustering Analysis", children=[
                     dbc.Row([
                        dbc.Col(dcc.Graph(id='clusters-scatter-chart'), md=6),
                        dbc.Col(dcc.Graph(id='cluster-anomalies-chart'), md=6), # Changed from cluster_tsne, tsne/umap can be optional internal to scatter func
                    ]),
                    dbc.Row([
                         dbc.Col(dcc.Graph(id='clusters-tsne-chart'), md=6), # if tsne specific vis is desired
                         dbc.Col(dcc.Graph(id='clusters-umap-chart'), md=6) # if umap specific vis is desired
                    ]),
                    dbc.Row(dbc.Col(dcc.Graph(id='clusters-over-time-chart'), md=12)),
                ]),
                dbc.Tab(label="Anomaly Detection", children=[
                    dbc.Row(dbc.Col(dcc.Graph(id='anomaly-scores-chart'), md=12)), # Changed to full width
                    dbc.Row(dbc.Col(dcc.Graph(id='price-anomalies-chart'), md=12)),
                ]),
                dbc.Tab(label="Market Regime", children=[
                     dbc.Row(dbc.Col(dcc.Graph(id='market-regime-chart'), md=12)),
                     dbc.Row([
                         dbc.Col(dcc.Graph(id='regime-transition-matrix'), md=6),
                         dbc.Col(dcc.Graph(id='regime-stats-chart'), md=6),
                     ]),
                     dbc.Row(dbc.Col(dcc.Graph(id='returns-distribution-chart'), md=12)),
                ]),
                 dbc.Tab(label="Correlation Analysis", children=[
                    dbc.Row([
                        dbc.Col(html.Div(id='correlation-multi-method-charts'), md=12)
                    ]),
                ]),
                dbc.Tab(label="Backtesting & Strategy", children=[
                    dbc.Card(dbc.CardBody(html.Div(id='perfect-storm-analysis')), className="mb-3"), # Strategy output
                    dbc.Row(dbc.Col(dcc.Graph(id='backtesting-results-chart'), md=12)), # Backtest equity
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='walk-forward-optimization-chart'), md=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='monte-carlo-simulation-chart'), md=12),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='regime-analysis-summary-chart'), md=12),
                    ], className="mb-3"),
                ]),
            ]),
        ],
        # fullscreen=True, # Optional: if you want loading overlay for entire content area
        className="mt-4"
    ), # End of dcc.Loading

    dbc.Alert(id='alerts-div', children="No critical alerts.", color="info", className="mt-4"), # Using dbc.Alert
    dcc.Interval(id='real-time-alerts', interval=60*1000, n_intervals=0), # 60 seconds

    html.Hr(className="my-4"),
    dbc.Row(dbc.Col(html.P("Data sources: Yahoo Finance, AlphaVantage, MarketWatch (manual), AAII Investor Sentiment Survey (local CSV)", className="text-center text-muted small"))),
    dbc.Row(dbc.Col(html.P("© 2025 Perfect Storm Dashboard", className="text-center text-muted small"))),

], fluid=True, className="dbc") # Using Bootstrap CSS, dbc for Bootstrap components styling

# Register callbacks (from callbacks.py)
register_callbacks(app)

# Add a callback to toggle the portfolio report visibility
from dash.dependencies import Output, Input, State
@app.callback(
    Output("portfolio-report-collapse-container", "is_open"),
    [Input("generate-portfolio-button", "n_clicks")],
    [State("portfolio-report-collapse-container", "is_open")],
    prevent_initial_call=True
)
def toggle_portfolio_report(n_clicks, is_open):
    if n_clicks and n_clicks > 0: # Check if n_clicks is not None and greater than 0
        return True # Always open when button is clicked
    # If you want the button to also close it, you could use:
    # if n_clicks:
    #     return not is_open
    return is_open # Keep current state if button not reason for trigger


if __name__ == '__main__':
    # Ensure the models directories exist
    base_model_dir = "models"
    model_types = ["Anomaly Detection Models", "Clustering Models", "Correlation Analysis Models",
                   "Market Regime Models", "Pattern Recognition Models", "Adaptive Threshold Models"]
    for model_type_dir_namepart in model_types:
        # Construct path similar to how get_standardized_model_filename might
        # e.g., "Anomaly Detection Models"
        os.makedirs(os.path.join(base_model_dir, model_type_dir_namepart), exist_ok=True)
    os.makedirs("data_cache", exist_ok=True)

    # Configure logging for the application (if not done in callbacks)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    app.run(debug=True, host='localhost') # Use app.run for newer Dash versions
