from dash import Output, Input, State, html, ctx, dcc
import plotly.graph_objects as go
import traceback
import pandas as pd
from datetime import datetime
from dash.exceptions import PreventUpdate
import os
import logging
import numpy as np
import random # Keep for alerts example

# --- Import Core Modules ---
from market_data_retrieval import MarketDataRetriever
from quantitative_strategy import QuantitativeStrategy # NEW Strategy Class
from backtesting_engine import BacktestingEngine
from portfolio_optimization import PortfolioOptimizer

# --- Import Dashboard Utility Functions ---
# Assume these are updated to handle potentially different report structures if needed
from dashboard_utils import (
    create_market_data_info, create_main_chart, create_indicators_chart,
    create_moving_averages_chart, create_volume_chart, create_oscillators_chart,
    create_sentiment_chart, create_backtesting_results, create_perfect_storm_analysis,
    create_backtesting_chart, create_correlation_report_charts,
    create_correlation_dashboard_component, create_portfolio_report_charts,
    create_portfolio_report_component
    # Import ML visualization functions if they exist separately, or they are part of reports
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        # (This function remains largely unchanged as it's manual data entry)
        if n_clicks == 0:
            raise PreventUpdate
        # ... (rest of the validation and saving logic) ...
        file_path = "market_breadth_manual.csv"
        today_date = datetime.now().strftime('%Y-%m-%d')
        try:
             # Basic validation
             if adv_issues is None or dec_issues is None or adv_vol is None or dec_vol is None:
                  return html.Div("Error: All fields must be filled.", style={'color': 'red'})
             adv_issues = float(adv_issues)
             dec_issues = float(dec_issues)
             adv_vol = float(adv_vol)
             dec_vol = float(dec_vol)
             if adv_issues < 0 or dec_issues < 0 or adv_vol < 0 or dec_vol < 0:
                  return html.Div("Error: Values cannot be negative.", style={'color': 'red'})
             if dec_issues == 0 or dec_vol == 0:
                 print("Warning: Declining issues or volume is zero.")

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
             logging.error(f"Error saving market breadth: {e}", exc_info=True)
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
        """
        Main callback to update the entire dashboard.
        Orchestrates data retrieval, strategy execution, backtesting, and visualization.
        """
        start_time = datetime.now()
        triggered_id = ctx.triggered_id
        if n_clicks is None and triggered_id is None:
            logging.info("Preventing initial update.")
            raise PreventUpdate

        logging.info(f"Update triggered for {symbol} ({period}, {interval}) by {triggered_id}. Click count: {n_clicks}")

        # --- Create Empty Figures for Fallback ---
        empty_fig = go.Figure().update_layout(title="Data Unavailable or Processing Error")
        def create_empty_output(num_outputs):
            return tuple([empty_fig] * (num_outputs - 2) + [html.Div("Error processing request."), html.Div("Error processing request.")])


        # --- 1. Data Retrieval ---
        logging.info("Retrieving market data...")
        try:
            api_key = os.getenv("ALPHAVANTAGE_API_KEY", "25WNVRI1YIXCDIH1")
            data_retriever = MarketDataRetriever(api_key=api_key)
            stock_data = data_retriever.get_stock_history(symbol, interval=interval, period=period)
            market_breadth_data = data_retriever.get_market_breadth_data() # Reads latest from file
            sentiment_data = data_retriever.get_sentiment_data() # Reads historical file/placeholder

            if stock_data is None or stock_data.empty:
                 logging.error(f"Failed to retrieve stock data for {symbol}.")
                 num_outputs_on_error = 27 # Count of total outputs
                 return create_empty_output(num_outputs_on_error)


        except Exception as e:
            logging.error(f"Data Retrieval Error: {e}", exc_info=True)
            num_outputs_on_error = 27 # Recalculate if outputs change
            return create_empty_output(num_outputs_on_error)

        logging.info(f"Data retrieved. Stock data shape: {stock_data.shape}")

        # --- 2. Strategy Execution ---
        logging.info("Executing quantitative strategy...")
        strategy_results = pd.DataFrame() # Default empty dataframe
        strategy_report_dict = {}
        current_market_regime = None

        try:
            # Define features used by various parts of the strategy
            # These should ideally be loaded from a config
            ml_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'bb_upper', 'bb_lower', 'adx', 'cmf', 'mfi'] # Example set
            correlation_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stoch_k', 'macd_line', 'cci', 'mfi'] # Example set

            # Instantiate the strategy class
            strategy = QuantitativeStrategy(
                 symbol=symbol,
                 period=period,
                 interval=interval,
                 ml_feature_cols=ml_features,
                 correlation_features=correlation_features
                 # Pass initial conviction weights from config if needed
            )

            # Run the strategy
            # This method now encapsulates feature calc, ML, regime, correlation (load/calc), conviction score, and signal generation
            strategy_results, strategy_report_dict, current_market_regime = strategy.run_strategy(
                df_market_data=stock_data,
                market_breadth=market_breadth_data,
                sentiment=sentiment_data # Placeholder for future real sentiment input
            )

            if strategy_results.empty:
                raise ValueError("Strategy execution returned empty DataFrame.")

            logging.info("Strategy execution complete.")

        except Exception as e:
            logging.error(f"Strategy Execution Error: {e}", exc_info=True)
            strategy_results = stock_data.copy() # Fallback: Use original data
             # Add dummy columns if they are essential for backtesting/plotting, but signal generation failed
            strategy_results['conviction_score'] = 0.0
            strategy_results['buy_signal'] = 0
            strategy_results['sell_signal'] = 0
            strategy_results['market_regime'] = 'unknown'
            strategy_report_dict = {'error': str(e)}

        # --- 3. Backtesting ---
        logging.info("Running backtest...")
        backtester = BacktestingEngine(initial_capital=100000) # Configure as needed
        backtesting_chart = empty_fig
        backtesting_performance = {}


        # --- Advanced Backtesting Analytics ---
        walk_forward_fig = go.Figure().update_layout(title="Walk-Forward Optimization: Not Run")
        monte_carlo_fig = go.Figure().update_layout(title="Monte Carlo Simulation: Not Run")
        regime_analysis_fig = go.Figure().update_layout(title="Regime Analysis: Not Run")

        try:
            # Adapt backtester based on current regime?
            # if current_market_regime:
            #    params = get_regime_backtest_params(current_market_regime['regime_name'])
            #    backtester.update_parameters(**params) # Needs a method in Backtester

            # Run backtest using signals generated by the strategy
            backtester.run_backtest(strategy_results) # Assumes strategy_results has buy/sell_signal columns now
            backtesting_performance = backtester.metrics
            # Create the equity curve / trade plot
            backtesting_chart = create_backtesting_chart(backtester.results, backtester.trades, backtester.metrics)

            # --- Walk-Forward Optimization ---
            param_grid = {'stop_loss': [0.02, 0.05], 'take_profit': [0.05, 0.1]} # Example grid, adjust as needed
            wfo_results = backtester.walk_forward_optimization(
                strategy_results, None, param_grid, window_size=50, step_size=25, metric='sharpe_ratio', n_jobs=1, verbose=False
            )
            from dashboard_utils import create_walk_forward_optimization_chart
            walk_forward_fig = create_walk_forward_optimization_chart(wfo_results)

            # --- Monte Carlo Simulation ---
            mc_results = backtester.run_monte_carlo_simulation(
                strategy_results, None, n_simulations=100, confidence_level=0.95
            )
            from dashboard_utils import create_monte_carlo_simulation_chart
            monte_carlo_fig = create_monte_carlo_simulation_chart(mc_results)

            # --- Regime Analysis ---
            regime_analysis = backtester.analyze_market_regimes(
                strategy_results, None, regime_col='market_regime', n_regimes=3
            )
            from dashboard_utils import create_regime_analysis_chart
            regime_analysis_fig = create_regime_analysis_chart(regime_analysis)

        except Exception as e:
            logging.error(f"Backtesting Error: {e}", exc_info=True)
            backtesting_chart = go.Figure().update_layout(title="Backtesting Failed")
            backtesting_performance = {'error': str(e)}
            walk_forward_fig = go.Figure().update_layout(title="Walk-Forward Optimization: Failed")
            monte_carlo_fig = go.Figure().update_layout(title="Monte Carlo Simulation: Failed")
            regime_analysis_fig = go.Figure().update_layout(title="Regime Analysis: Failed")

        logging.info("Backtesting complete.")

        # --- 4. Visualization and Report Generation ---
        # Reuse existing plotting functions where possible, ensure they get the right data
        logging.info("Generating visualizations and reports...")

        # Visualization and Report Generation
        market_data_info = create_market_data_info(strategy_results, symbol, market_breadth_data, sentiment_data) # Use strategy results for latest indicators
        main_chart = create_main_chart(strategy_results, symbol) # Includes basic price, BBands, volume, and FINAL signals
        indicators_chart = create_indicators_chart(strategy_results) # MACD/RSI from final data
        moving_averages_chart = create_moving_averages_chart(strategy_results)
        volume_chart = create_volume_chart(strategy_results) # Includes CMF if available
        oscillators_chart = create_oscillators_chart(strategy_results) # Stoch/CCI from final data
        sentiment_chart = create_sentiment_chart(strategy_results, sentiment_data) # Use strategy_results if indicators derived from sentiment were added

        # Extract ML/Regime/Correlation Visualizations (These come from the individual module reports)
        # It's assumed that the strategy run cached/generated these reports if needed
        # Or, we re-run the report generation here if not stored by strategy.run()
        # Safely get reports or use defaults
        clustering_report = strategy.ml_clusters_report_cache if hasattr(strategy, 'ml_clusters_report_cache') and strategy.ml_clusters_report_cache is not None else {'visualizations': {}}
        pattern_report = strategy.ml_pattern_report_cache if hasattr(strategy, 'ml_pattern_report_cache') and strategy.ml_pattern_report_cache is not None else {'visualizations': {}}
        anomaly_report = strategy.ml_anomaly_report_cache if hasattr(strategy, 'ml_anomaly_report_cache') and strategy.ml_anomaly_report_cache is not None else {'visualizations': {}}
        regime_report = strategy.regime_report_cache if hasattr(strategy, 'regime_report_cache') and strategy.regime_report_cache is not None else {'visualizations': {}}
        correlation_report_data = strategy.correlation_report_cache if hasattr(strategy, 'correlation_report_cache') and strategy.correlation_report_cache is not None else {}



        # Correlation requires more specific features and target
        corr_df = strategy_results.copy() # Avoid modifying the main df
        if 'returns' not in corr_df:
            corr_df['returns'] = corr_df['close'].pct_change().fillna(0)
        valid_corr_features = [f for f in strategy.correlation_features if f in corr_df.columns]

        # --- Load ALL available correlation models for multi-method dashboard ---
        loaded_models = strategy.correlation_model.load_all_available_models(symbol=symbol, period=period, interval=interval)
        correlation_charts = {}
        if loaded_models and len(loaded_models) > 0:
            # Aggregate all visualizations from all loaded models
            for method, report in loaded_models.items():
                if 'visualizations' in report:
                    for k, v in report['visualizations'].items():
                        correlation_charts[f"{k}__{method}"] = v
            # Set multi_method_results so downstream chart code works
            strategy.correlation_model.multi_method_results = loaded_models
            # Use a dummy report structure with multi_method_results for chart generation
            correlation_report_data = {'multi_method_results': loaded_models}
        else:
            # If no models found, generate new report (which will also save models)
            correlation_report_data = strategy.correlation_model.generate_correlation_report(
                corr_df,
                valid_corr_features,
                target_col='returns',
                display_dashboard=False,
                symbol=symbol, period=period, interval=interval
            )
            # After generation, also update multi_method_results for consistency
            if hasattr(strategy.correlation_model, 'multi_method_results'):
                loaded_models = strategy.correlation_model.multi_method_results
                correlation_report_data = {'multi_method_results': loaded_models}
            # Also aggregate visualizations from the generated report
            if 'visualizations' in correlation_report_data:
                for k, v in correlation_report_data['visualizations'].items():
                    correlation_charts[f"{k}__default"] = v

        from dashboard_utils import create_correlation_multi_method_charts
        correlation_multi_method_comp = create_correlation_multi_method_charts(correlation_charts)
        correlation_summary_comp = create_correlation_dashboard_component(correlation_report_data) # For text analysis part

        def get_viz(report, viz_key, default_fig=empty_fig):
            if report and 'visualizations' in report and viz_key in report['visualizations']:
                return report['visualizations'][viz_key]
            return default_fig

        # --- Perfect Storm Analysis Text ---
        # This section needs to be updated to use the conviction score and other factors
        perfect_storm_text = html.Div([
            html.H4("Strategy Analysis", style={'color': '#2c3e50'}),
            html.P(f"Latest Conviction Score: {strategy_report_dict.get('latest_conviction_score', 'N/A'):.3f}"),
            html.P(f"Current Market Regime: {strategy_report_dict.get('current_regime', 'N/A')}"),
            html.P(f"Latest Anomaly Score: {strategy_report_dict.get('latest_anomaly_score', 'N/A'):.3f}"),
            html.P(f"Latest ML Pattern Confidence: {strategy_report_dict.get('latest_pattern_confidence', 'N/A'):.3f}"),
            html.H5("Backtesting Summary:"),
            create_backtesting_results(backtesting_performance), # Use utility for text summary
            html.Hr(),
            correlation_summary_comp # Include correlation summary text/table
        ])

        logging.info("Visualizations generated.")

        # Return all outputs for the dashboard layout
        end_time = datetime.now()
        logging.info(f"Dashboard update successful. Total time: {end_time - start_time}")

        return (market_data_info, main_chart, indicators_chart, moving_averages_chart,
                volume_chart, oscillators_chart, sentiment_chart,
                get_viz(pattern_report, 'predictions'),
                get_viz(pattern_report, 'roc_curve'),
                get_viz(pattern_report, 'precision_recall_curve'),
                get_viz(pattern_report, 'confusion_matrix'),
                get_viz(clustering_report, 'cluster_scatter'),
                get_viz(clustering_report, 'cluster_tsne'),
                get_viz(clustering_report, 'cluster_umap'),
                get_viz(clustering_report, 'clusters_time_series'),
                get_viz(clustering_report, 'anomaly_scores'),
                get_viz(anomaly_report, 'anomaly_scores'),
                get_viz(anomaly_report, 'price_anomalies'),
                get_viz(regime_report, 'regimes'),
                get_viz(regime_report, 'transition_matrix'),
                get_viz(regime_report, 'regime_statistics'),
                get_viz(regime_report, 'returns_distribution'),
                backtesting_chart,
                correlation_multi_method_comp,  # All correlation charts in tabs/grid
                walk_forward_fig,
                monte_carlo_fig,
                regime_analysis_fig,
                perfect_storm_text)

    @app.callback(
        Output('alerts-div', 'children'),
        Input('real-time-alerts', 'n_intervals')
    )
    def update_alerts(n):
        # (This function can remain the same for demo alerts)
        alerts = []
        if random.random() > 0.8: # Example random alert
            alerts.append(html.Li("ALERT: Significant market signal detected based on Conviction Score!"))
        # Add more sophisticated alert logic based on conviction score, anomalies, etc.
        # conviction = get_latest_conviction_score() # Need a way to access latest state
        # if conviction > 0.8: alerts.append(html.Li(f"ALERT: High Buy Conviction ({conviction:.2f})"))
        # elif conviction < -0.8: alerts.append(html.Li(f"ALERT: High Sell Conviction ({conviction:.2f})"))

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
        # (Portfolio optimization callback remains largely unchanged in its own logic)
        empty_fig = go.Figure().update_layout(title="Data Unavailable or Processing Error")
        # Minor Improvement: Log errors here as well.
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        logging.info(f"Generating portfolio report for symbols: {symbols_input}")
        # ... (rest of portfolio generation logic) ...
        # (Ensure error handling logs appropriately)
        try:
            # Validate inputs
            if not symbols_input or not period or not capital or not risk_profile:
                return (html.Div("Error: All fields required.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            if len(symbols) < 2:
                return (html.Div("Error: At least two symbols required.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})
            capital = float(capital)
            if capital <= 0:
                 return (html.Div("Error: Capital must be positive.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})

            # --- Data Retrieval ---
            data_retriever = MarketDataRetriever()
            price_dfs = {}
            all_returns = {}
            valid_symbols = []
            for symbol in symbols:
                logging.info(f"Fetching data for portfolio asset: {symbol}")
                stock_data = data_retriever.get_stock_history(symbol, interval='1d', period=period) # Daily data usually best for portfolio opt.
                if stock_data is not None and not stock_data.empty:
                    price_dfs[symbol] = stock_data['close']
                    all_returns[symbol] = stock_data['close'].pct_change().dropna()
                    valid_symbols.append(symbol)
                else:
                    logging.warning(f"Could not retrieve data for portfolio symbol: {symbol}")

            if len(valid_symbols) < 2:
                 return (html.Div("Error: Need at least two valid symbols with data.", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})

            price_df = pd.DataFrame(price_dfs)
            returns_df = pd.DataFrame(all_returns).dropna()

            if len(returns_df) < 60: # Need sufficient data
                 return (html.Div(f"Error: Insufficient overlapping data ({len(returns_df)} days).", style={'color': 'red'}), {'display': 'none'}, None, {}, {}, {}, {}, {})

            # --- Optimization ---
            logging.info("Running portfolio optimization...")
            portfolio_optimizer = PortfolioOptimizer()
            # You could potentially integrate conviction scores here if available
            # E.g., fetch conviction scores for each asset and use as input/constraint
            portfolio_report = portfolio_optimizer.generate_portfolio_report(
                returns_df,
                total_capital=capital,
                risk_profile=risk_profile
            )

            # --- Visualization ---
            portfolio_figures = create_portfolio_report_charts(portfolio_report, symbol=f"{len(valid_symbols)}-Stock Portfolio")
            summary_component = create_portfolio_report_component(portfolio_report, symbol=f"{len(valid_symbols)}-Stock Portfolio")
            logging.info("Portfolio report generated successfully.")

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
            logging.error(f"Error generating portfolio report: {e}", exc_info=True)
            empty_fig = go.Figure().update_layout(title="Data Unavailable or Processing Error")
            return (
                 html.Div(f"Error: {str(e)}", style={'color': 'red'}), {'display': 'none'},
                 None, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
             )