import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from indicator_calculations import PerfectStormIndicators
from backtesting_engine import BacktestingEngine
from data_retrieval import StockDataRetriever
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import threading
import time

app = dash.Dash(__name__, title="Perfect Storm Dashboard", external_stylesheets=["https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css"])
retriever = StockDataRetriever()
alerts = []

def monitor_market_regimes(stock_symbol, time_period):
    """Continuously monitors market regimes and sends alerts."""
    while True:
        historical_data = retriever.get_stock_history(stock_symbol, range_period=time_period)
        indicators = PerfectStormIndicators()
        calculated_indicators = indicators.calculate_moving_averages(historical_data)
        analyzer = CorrelationRegimeAnalyzer(calculated_indicators)
        market_regimes = analyzer.detect_market_regimes()
        
        if market_regimes[-1] != market_regimes[-2]:
            alerts.append(f"Market regime shift detected for {stock_symbol}!")
        
        time.sleep(60)  # Check every 60 seconds

# Start monitoring in a separate thread
threading.Thread(target=monitor_market_regimes, args=("AAPL", "1y"), daemon=True).start()

class CorrelationRegimeAnalyzer:
    """
    Analyzes correlations between indicators and detects market regimes.
    """
    
    def __init__(self, data, n_components=3):
        """
        Initializes the analyzer.
        
        Parameters:
        - data: DataFrame with technical indicators
        - n_components: Number of market regimes for HMM
        """
        self.data = data
        self.n_components = n_components
    
    def compute_correlation_matrix(self):
        """Computes and visualizes the correlation matrix of indicators."""
        return self.data.corr()
    
    def detect_market_regimes(self):
        """Identifies market regimes using Hidden Markov Model (HMM)."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_data)
        
        model = GaussianHMM(n_components=self.n_components, covariance_type='full', n_iter=1000)
        model.fit(pca_features)
        states = model.predict(pca_features)
        
        return states

    def compute_weighted_indicator_signals(self, indicators, predictive_power):
        """
        Compute weighted signals based on historical predictive power.
        Parameters:
         - indicators: DataFrame of technical indicators.
         - predictive_power: Dict mapping indicator names to their weight.
        Returns:
         - weighted_signals: Series of weighted signals.
        """
        weighted = indicators.copy()
        for col in weighted.columns:
            weight = predictive_power.get(col, 1)
            weighted[col] = weighted[col] * weight
        weighted_signals = weighted.sum(axis=1)
        return weighted_signals

# Dashboard Layout
app.layout = html.Div([
    html.Div([
        html.H1("Market Regime & Correlation Analysis", className="text-center text-primary mt-4"),
        html.Div([
            html.Label("Stock Symbol:", className="form-label"),
            dcc.Input(id='symbol-input', type='text', value='AAPL', className="form-control"),
        ], className="mb-3"),
        html.Div([
            html.Label("Time Period:", className="form-label"),
            dcc.Dropdown(
                id='time-period',
                options=[
                    {'label': '1 Month', 'value': '1mo'},
                    {'label': '3 Months', 'value': '3mo'},
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '5 Years', 'value': '5y'}
                ],
                value='1y',
                className="form-select"
            ),
        ], className="mb-3"),
        html.Button('Analyze', id='analyze-button', n_clicks=0, className="btn btn-primary mb-4"),
        html.Div(id='alerts', className='alert alert-warning')
    ], className="container shadow p-4 bg-light rounded"),
    
    html.Div([
        dcc.Graph(id='correlation-matrix', className="mb-4"),
        dcc.Graph(id='market-regime', className="mb-4")
    ], className="container")
])

@app.callback(
    [Output('correlation-matrix', 'figure'), Output('market-regime', 'figure'), Output('alerts', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [dash.State('symbol-input', 'value'), dash.State('time-period', 'value')]
)
def update_dashboard(n_clicks, stock_symbol, time_period):
    """Updates the dashboard with correlation matrix and market regimes."""
    historical_data = retriever.get_stock_history(stock_symbol, range_period=time_period)
    indicators = PerfectStormIndicators()
    calculated_indicators = indicators.calculate_moving_averages(historical_data)
    analyzer = CorrelationRegimeAnalyzer(calculated_indicators)
    
    corr_matrix = analyzer.compute_correlation_matrix()
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='coolwarm')
    fig_corr.update_layout(title=f"Indicator Correlation Matrix - {stock_symbol}")
    
    market_regimes = analyzer.detect_market_regimes()
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(y=market_regimes, mode='lines+markers', name='Market Regimes'))
    fig_regime.update_layout(title=f"Market Regime Detection - {stock_symbol}", xaxis_title="Time", yaxis_title="Regime")
    
    return fig_corr, fig_regime, "\n".join(alerts[-5:])

if __name__ == "__main__":
    app.run_server(debug=True)
