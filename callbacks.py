from dash import Output, Input, State, html, ctx, dcc
import plotly.graph_objects as go
import traceback
import pandas as pd
from datetime import datetime
from dash.exceptions import PreventUpdate
import os
# import logging # No longer directly needed, log_with_timestamp handles logging calls
import numpy as np
import random # Keep for alerts example
import os # Retained: Used in save_manual_breadth_data and update_dashboard
import pandas as pd # Retained: Used in various callbacks

# --- Import Core Modules ---
from market_data_retrieval import MarketDataRetriever
from quantitative_strategy import QuantitativeStrategy # NEW Strategy Class
from backtesting_engine import BacktestingEngine
from portfolio_optimization import PortfolioOptimizer
# ML Models for Training Callback are no longer needed here directly
# from ml_anomaly_detection_enhanced import MarketAnomalyDetection
# from ml_clustering_enhanced_completed import PerfectStormClustering
# from ml_pattern_recognition_enhanced import MarketPatternRecognition


# --- Import Dashboard Utility Functions ---
from dashboard_utils import (
    create_market_data_info, create_main_chart, create_indicators_chart,
    create_moving_averages_chart, create_volume_chart, create_oscillators_chart,
    create_sentiment_chart, create_backtesting_results, create_perfect_storm_analysis,
    create_backtesting_chart, create_correlation_report_charts,
    create_correlation_dashboard_component, create_portfolio_report_charts,
    create_portfolio_report_component, log_with_timestamp # Added log_with_timestamp
    # Import ML visualization functions if they exist separately, or they are part of reports
)

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured in dashboard_utils

# --- Default Feature Sets for Model Training (Removed) ---
# DEFAULT_ANOMALY_FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd_line', 'stoch_k', 'cci']
# DEFAULT_CLUSTERING_FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'bb_upper', 'bb_lower', 'adx', 'cmf', 'mfi', 'atr']
# DEFAULT_PATTERN_FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd_line', 'stoch_k', 'cci', 'bb_width', 'atr']
# DEFAULT_PATTERN_TARGET_COL = 'future_return_signal'


def register_callbacks(app):
    @app.callback(
        Output('save-breadth-output', 'children'),
        Input('save-breadth-button', 'n_clicks'),
        State('adv-issues-input', 'value'),
        State('dec-issues-input', 'value'),
        State('adv-volume-input', 'value'),
        State('dec-volume-input', 'value'),
        prevent_initial_call=True
    )
    def save_manual_breadth_data(n_clicks, adv_issues, dec_issues, adv_vol, dec_vol):
        if n_clicks == 0:
            raise PreventUpdate
        file_path = "market_breadth_manual.csv"
        today_date = datetime.now().strftime('%Y-%m-%d')
        try:
             if adv_issues is None or dec_issues is None or adv_vol is None or dec_vol is None:
                  return html.Div("Error: All fields must be filled.", style={'color': 'red'})
             adv_issues = float(adv_issues)
             dec_issues = float(dec_issues)
             adv_vol = float(adv_vol)
             dec_vol = float(dec_vol)
             if adv_issues < 0 or dec_issues < 0 or adv_vol < 0 or dec_vol < 0:
                  return html.Div("Error: Values cannot be negative.", style={'color': 'red'})
             if dec_issues == 0 or dec_vol == 0:
                 log_with_timestamp("Declining issues or volume is zero.", log_level="WARNING")

             new_data = pd.DataFrame([{
                  'date': today_date,
                  'advancing_issues': adv_issues,
                  'declining_issues': dec_issues,
                  'advancing_volume': adv_vol,
                  'declining_volume': dec_vol
             }])

             if os.path.exists(file_path):
                  existing_df = pd.read_csv(file_path, parse_dates=['date'])
                  combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                  combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date
                  combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                  combined_df.to_csv(file_path, index=False, date_format='%Y-%m-%d')
                  message = f"Market breadth data for {today_date} updated."
             else:
                  new_data.to_csv(file_path, index=False, date_format='%Y-%m-%d')
                  message = f"Market breadth file created for {today_date}."
             return html.Div(message, style={'color': 'green'})

        except ValueError:
            return html.Div("Error: Please enter valid numbers.", style={'color': 'red'})
        except Exception as e:
             log_with_timestamp(f"Error saving market breadth: {e}", log_level="ERROR")
             return html.Div(f"Error saving data: {str(e)}", style={'color': 'red'})

    @app.callback(
        [Output('market-data-info', 'children'),
         Output('main-chart', 'figure'),
         Output('indicators-chart', 'figure'),
         Output('moving-averages-chart', 'figure'),
         Output('volume-chart', 'figure'),
         Output('oscillators-chart', 'figure'),
         Output('sentiment-chart', 'figure'),
         Output('prediction-patterns-chart', 'figure'),
         Output('roc-curve-chart', 'figure'),
         Output('prec-recall-chart', 'figure'),
         Output('confusion-matrix-chart', 'figure'),
         Output('clusters-scatter-chart', 'figure'),
         Output('clusters-tsne-chart', 'figure'),
         Output('clusters-umap-chart', 'figure'),
         Output('clusters-over-time-chart', 'figure'),
         Output('cluster-anomalies-chart', 'figure'),
         Output('anomaly-scores-chart', 'figure'),
         Output('price-anomalies-chart', 'figure'),
         Output('market-regime-chart', 'figure'),
         Output('regime-transition-matrix', 'figure'),
         Output('regime-stats-chart', 'figure'),
         Output('returns-distribution-chart', 'figure'),
         Output('backtesting-results-chart', 'figure'),
         Output('correlation-multi-method-charts', 'children'),
         Output('walk-forward-optimization-chart', 'figure'),
         Output('monte-carlo-simulation-chart', 'figure'),
         Output('regime-analysis-summary-chart', 'figure'),
         Output('perfect-storm-analysis', 'children')],
        [Input('update-button', 'n_clicks')],
        [State('symbol-input', 'value'),
         State('period-dropdown', 'value'),
         State('interval-dropdown', 'value')]
    )
    def update_dashboard(n_clicks, symbol, period, interval):
        start_time = datetime.now()
        triggered_id = ctx.triggered_id
        if n_clicks is None and triggered_id is None:
            log_with_timestamp("Preventing initial update.", log_level="INFO")
            raise PreventUpdate

        log_with_timestamp(f"Update triggered for {symbol} ({period}, {interval}) by {triggered_id}. Click count: {n_clicks}", log_level="INFO")

        empty_fig = go.Figure().update_layout(title="Data Unavailable or Processing Error")
        def create_empty_output(num_outputs):
            return tuple([empty_fig] * (num_outputs - 2) + [html.Div("Error processing request."), html.Div("Error processing request.")])

        log_with_timestamp("Retrieving market data...", log_level="INFO")
        try:
            api_key = os.getenv("ALPHAVANTAGE_API_KEY", "25WNVRI1YIXCDIH1")
            data_retriever = MarketDataRetriever(api_key=api_key)
            stock_data = data_retriever.get_stock_history(symbol, interval=interval, period=period)
            market_breadth_data = data_retriever.get_market_breadth_data()
            sentiment_data = data_retriever.get_sentiment_data()

            if stock_data is None or stock_data.empty:
                 log_with_timestamp(f"Failed to retrieve stock data for {symbol}.", log_level="ERROR")
                 num_outputs_on_error = 27
                 return create_empty_output(num_outputs_on_error)

        except Exception as e:
            log_with_timestamp(f"Data Retrieval Error: {e}", log_level="ERROR")
            num_outputs_on_error = 27
            return create_empty_output(num_outputs_on_error)

        log_with_timestamp(f"Data retrieved. Stock data shape: {stock_data.shape}", log_level="INFO")

        log_with_timestamp("Executing quantitative strategy...", log_level="INFO")
        strategy_results = pd.DataFrame()
        strategy_report_dict = {}
        current_market_regime = None

        try:
            ml_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'bb_upper', 'bb_lower', 'adx', 'cmf', 'mfi']
            correlation_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'mfi']
            strategy = QuantitativeStrategy(
                 symbol=symbol,
                 period=period,
                 interval=interval,
                 ml_feature_cols=ml_features,
                 correlation_features=correlation_features
            )
            strategy_results, strategy_report_dict, current_market_regime = strategy.run_strategy(
                df_market_data=stock_data,
                market_breadth=market_breadth_data,
                sentiment=sentiment_data
            )
            if strategy_results.empty:
                raise ValueError("Strategy execution returned empty DataFrame.")
            log_with_timestamp("Strategy execution complete.", log_level="INFO")

        except Exception as e:
            log_with_timestamp(f"Strategy Execution Error: {e}", log_level="ERROR")
            strategy_results = stock_data.copy()
            strategy_results['conviction_score'] = 0.0
            strategy_results['buy_signal'] = 0
            strategy_results['sell_signal'] = 0
            strategy_results['market_regime'] = 'unknown'
            strategy_report_dict = {'error': str(e)}

        log_with_timestamp("Running backtest...", log_level="INFO")
        backtester = BacktestingEngine(initial_capital=100000)
        backtesting_chart = empty_fig
        backtesting_performance = {}

        walk_forward_fig = go.Figure().update_layout(title="Walk-Forward Optimization: Not Run")
        monte_carlo_fig = go.Figure().update_layout(title="Monte Carlo Simulation: Not Run")
        regime_analysis_fig = go.Figure().update_layout(title="Regime Analysis: Not Run")

        try:
            backtester.run_backtest(strategy_results)
            backtesting_performance = backtester.metrics
            backtesting_chart = create_backtesting_chart(backtester.results, backtester.trades, backtester.metrics)

            param_grid = {'stop_loss': [0.02, 0.05], 'take_profit': [0.05, 0.1]}
            wfo_results = backtester.walk_forward_optimization(
                strategy_results, None, param_grid, window_size=50, step_size=25, metric='sharpe_ratio', n_jobs=1, verbose=False
            )
            from dashboard_utils import create_walk_forward_optimization_chart # Keep local import for clarity
            walk_forward_fig = create_walk_forward_optimization_chart(wfo_results)

            mc_results = backtester.run_monte_carlo_simulation(
                strategy_results, None, n_simulations=100, confidence_level=0.95
            )
            from dashboard_utils import create_monte_carlo_simulation_chart # Keep local import
            monte_carlo_fig = create_monte_carlo_simulation_chart(mc_results)

            regime_analysis = backtester.analyze_market_regimes(
                strategy_results, None, regime_col='market_regime', n_regimes=3
            )
            from dashboard_utils import create_regime_analysis_chart # Keep local import
            regime_analysis_fig = create_regime_analysis_chart(regime_analysis)

        except Exception as e:
            log_with_timestamp(f"Backtesting Error: {e}", log_level="ERROR")
            backtesting_chart = go.Figure().update_layout(title="Backtesting Failed")
            backtesting_performance = {'error': str(e)}
            walk_forward_fig = go.Figure().update_layout(title="Walk-Forward Optimization: Failed")
            monte_carlo_fig = go.Figure().update_layout(title="Monte Carlo Simulation: Failed")
            regime_analysis_fig = go.Figure().update_layout(title="Regime Analysis: Failed")

        log_with_timestamp("Backtesting complete.", log_level="INFO")
        log_with_timestamp("Generating visualizations and reports...", log_level="INFO")

        market_data_info = create_market_data_info(strategy_results, symbol, market_breadth_data, sentiment_data)
        main_chart = create_main_chart(strategy_results, symbol)
        indicators_chart = create_indicators_chart(strategy_results)
        moving_averages_chart = create_moving_averages_chart(strategy_results)
        volume_chart = create_volume_chart(strategy_results)
        oscillators_chart = create_oscillators_chart(strategy_results)
        sentiment_chart = create_sentiment_chart(strategy_results, sentiment_data)

        clustering_report = strategy.ml_clusters_report_cache if hasattr(strategy, 'ml_clusters_report_cache') and strategy.ml_clusters_report_cache is not None else {'visualizations': {}}
        pattern_report = strategy.ml_pattern_report_cache if hasattr(strategy, 'ml_pattern_report_cache') and strategy.ml_pattern_report_cache is not None else {'visualizations': {}}
        anomaly_report = strategy.ml_anomaly_report_cache if hasattr(strategy, 'ml_anomaly_report_cache') and strategy.ml_anomaly_report_cache is not None else {'visualizations': {}}
        regime_report = strategy.regime_report_cache if hasattr(strategy, 'regime_report_cache') and strategy.regime_report_cache is not None else {'visualizations': {}}
        # correlation_report_data = strategy.correlation_report_cache if hasattr(strategy, 'correlation_report_cache') and strategy.correlation_report_cache is not None else {}

        corr_df = strategy_results.copy()
        if 'returns' not in corr_df:
            corr_df['returns'] = corr_df['close'].pct_change().fillna(0)
        valid_corr_features = [f for f in strategy.correlation_features if f in corr_df.columns]

        loaded_models = strategy.correlation_model.load_all_available_models(symbol=symbol, period=period, interval=interval)
        correlation_charts = {}
        if loaded_models and len(loaded_models) > 0:
            for method, report_item in loaded_models.items(): # report is a reserved keyword
                if 'visualizations' in report_item:
                    for k, v in report_item['visualizations'].items():
                        correlation_charts[f"{k}__{method}"] = v
            strategy.correlation_model.multi_method_results = loaded_models
            correlation_report_data = {'multi_method_results': loaded_models}
        else:
            correlation_report_data = strategy.correlation_model.generate_correlation_report(
                corr_df, valid_corr_features, target_col='returns', display_dashboard=False,
                symbol=symbol, period=period, interval=interval
            )
            if hasattr(strategy.correlation_model, 'multi_method_results'):
                loaded_models = strategy.correlation_model.multi_method_results
                correlation_report_data = {'multi_method_results': loaded_models}
            if 'visualizations' in correlation_report_data: # Check main report if multi_method_results is empty
                 for k, v in correlation_report_data['visualizations'].items():
                      correlation_charts[f"{k}__default"] = v


        from dashboard_utils import create_correlation_multi_method_charts # Keep local
        correlation_multi_method_comp = create_correlation_multi_method_charts(correlation_charts)
        correlation_summary_comp = create_correlation_dashboard_component(correlation_report_data)

        def get_viz(report, viz_key, default_fig=empty_fig): # report is a reserved keyword
            if report and 'visualizations' in report and viz_key in report['visualizations']:
                return report['visualizations'][viz_key]
            return default_fig

        perfect_storm_text = html.Div([
            html.H4("Strategy Analysis", style={'color': '#2c3e50'}),
            html.P(f"Latest Conviction Score: {strategy_report_dict.get('latest_conviction_score', 'N/A'):.3f}"),
            html.P(f"Current Market Regime: {strategy_report_dict.get('current_regime', 'N/A')}"),
            html.P(f"Latest Anomaly Score: {strategy_report_dict.get('latest_anomaly_score', 'N/A'):.3f}"),
            html.P(f"Latest ML Pattern Confidence: {strategy_report_dict.get('latest_pattern_confidence', 'N/A'):.3f}"),
            html.H5("Backtesting Summary:"),
            create_backtesting_results(backtesting_performance),
            html.Hr(),
            correlation_summary_comp
        ])

        log_with_timestamp("Visualizations generated.", log_level="INFO")
        end_time = datetime.now()
        log_with_timestamp(f"Dashboard update successful. Total time: {end_time - start_time}", log_level="INFO")

        return (market_data_info, main_chart, indicators_chart, moving_averages_chart,
                volume_chart, oscillators_chart, sentiment_chart,
                get_viz(pattern_report, 'predictions'), get_viz(pattern_report, 'roc_curve'),
                get_viz(pattern_report, 'precision_recall_curve'), get_viz(pattern_report, 'confusion_matrix'),
                get_viz(clustering_report, 'cluster_scatter'), get_viz(clustering_report, 'cluster_tsne'),
                get_viz(clustering_report, 'cluster_umap'), get_viz(clustering_report, 'clusters_time_series'),
                get_viz(clustering_report, 'anomaly_scores'), get_viz(anomaly_report, 'anomaly_scores'),
                get_viz(anomaly_report, 'price_anomalies'), get_viz(regime_report, 'regimes'),
                get_viz(regime_report, 'transition_matrix'), get_viz(regime_report, 'regime_statistics'),
                get_viz(regime_report, 'returns_distribution'), backtesting_chart,
                correlation_multi_method_comp, walk_forward_fig, monte_carlo_fig,
                regime_analysis_fig, perfect_storm_text)

    @app.callback(
        Output('alerts-div', 'children'),
        Input('real-time-alerts', 'n_intervals')
    )
    def update_alerts(n):
        alerts = []
        if random.random() > 0.8:
            alerts.append(html.Li("ALERT: Significant market signal detected based on Conviction Score!"))
        if not alerts:
            return html.Ul([html.Li("No critical alerts.")])
        return html.Ul(alerts, style={'color': 'red', 'fontWeight': 'bold'})

    @app.callback(
        [Output('portfolio-report-status', 'children'),
         Output('portfolio-report-container', 'style'),
         Output('portfolio-report-summary', 'children'),
         Output('portfolio-efficient-frontier', 'figure'),
         Output('portfolio-allocation-pie', 'figure'),
         Output('portfolio-allocation-bar', 'figure'),
         Output('portfolio-risk-contribution', 'figure'),
         Output('portfolio-performance-metrics', 'figure')],
        [Input('generate-portfolio-button', 'n_clicks')],
        [State('portfolio-symbols-input', 'value'),
         State('portfolio-period-dropdown', 'value'),
         State('portfolio-capital-input', 'value'),
         State('portfolio-risk-profile-dropdown', 'value')],
        prevent_initial_call=True
    )
    def generate_portfolio_report(n_clicks, symbols_input, period, capital, risk_profile):
        empty_fig = go.Figure().update_layout(title="Data Unavailable or Processing Error")
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        log_with_timestamp(f"Generating portfolio report for symbols: {symbols_input}", log_level="INFO")
        try:
            if not symbols_input or not period or not capital or not risk_profile:
                return (html.Div("Error: All fields required.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})
            symbols_list = [s.strip().upper() for s in symbols_input.split(',') if s.strip()] # Renamed from symbols
            if len(symbols_list) < 2:
                return (html.Div("Error: At least two symbols required.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})
            capital = float(capital)
            if capital <= 0:
                 return (html.Div("Error: Capital must be positive.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})

            data_retriever = MarketDataRetriever()
            price_dfs = {}
            all_returns = {}
            valid_symbols = []
            for sym in symbols_list: # Use the new variable name
                log_with_timestamp(f"Fetching data for portfolio asset: {sym}", log_level="INFO")
                stock_data = data_retriever.get_stock_history(sym, interval='1d', period=period)
                if stock_data is not None and not stock_data.empty:
                    price_dfs[sym] = stock_data['close']
                    all_returns[sym] = stock_data['close'].pct_change().dropna()
                    valid_symbols.append(sym)
                else:
                    log_with_timestamp(f"Could not retrieve data for portfolio symbol: {sym}", log_level="WARNING")

            if len(valid_symbols) < 2:
                 return (html.Div("Error: Need at least two valid symbols with data.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})

            price_df = pd.DataFrame(price_dfs)
            returns_df = pd.DataFrame(all_returns).dropna()

            if len(returns_df) < 60:
                 return (html.Div(f"Error: Insufficient overlapping data ({len(returns_df)} days).", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})

            log_with_timestamp("Running portfolio optimization...", log_level="INFO")
            portfolio_optimizer = PortfolioOptimizer()
            portfolio_report_data = portfolio_optimizer.generate_portfolio_report( # Renamed from portfolio_report
                returns_df,
                total_capital=capital,
                risk_profile=risk_profile
            )

            portfolio_figures = create_portfolio_report_charts(portfolio_report_data, symbol=f"{len(valid_symbols)}-Stock Portfolio")
            summary_component = create_portfolio_report_component(portfolio_report_data, symbol=f"{len(valid_symbols)}-Stock Portfolio")
            log_with_timestamp("Portfolio report generated successfully.", log_level="INFO")

            return (
                 html.Div("Portfolio report generated successfully.", style={'color': 'green'}),
                 {'display': 'block'},
                 summary_component,
                 portfolio_figures['efficient_frontier'],
                 portfolio_figures['portfolio_allocation_pie'],
                 portfolio_figures['portfolio_allocation_bar'],
                 portfolio_figures.get('risk_contribution', empty_fig),
                 portfolio_figures.get('portfolio_performance', empty_fig)
            )
        except Exception as e:
            log_with_timestamp(f"Error generating portfolio report: {e}", log_level="ERROR")
            return (
                 html.Div(f"Error: {str(e)}", style={'color': 'red'}), {'display': 'none'},
                 None, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
             )
