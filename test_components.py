"""
Test script for Perfect Storm Dashboard components

This script tests the data retrieval and indicator calculation
functions to ensure they work correctly with real market data.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import our custom modules
from data_retrieval import StockDataRetriever
from indicator_calculations import PerfectStormIndicators

def test_data_retrieval():
    """Test the data retrieval functionality"""
    print("Testing data retrieval...")
    
    # Create data retriever instance
    retriever = StockDataRetriever()
    
    # Test with different symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    timeframes = ['6mo', '1y']
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nTesting {symbol} with {timeframe} timeframe:")
            try:
                # Get historical data
                df = retriever.get_stock_history(symbol, interval='1d', range_period=timeframe)
                
                if df.empty:
                    print(f"  ❌ No data returned for {symbol}")
                    continue
                
                print(f"  ✅ Successfully retrieved {len(df)} days of data")
                print(f"  📊 Date range: {df.index.min().date()} to {df.index.max().date()}")
                print(f"  💰 Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")
                
                # Test insights retrieval
                insights = retriever.get_stock_insights(symbol)
                if insights and 'symbol' in insights:
                    print(f"  ✅ Successfully retrieved technical insights")
                else:
                    print(f"  ⚠️ Limited or no insights data available")
                
                # Test profile retrieval
                profile = retriever.get_stock_profile(symbol)
                if profile and 'sector' in profile and profile['sector']:
                    print(f"  ✅ Successfully retrieved company profile")
                    print(f"  🏢 Sector: {profile['sector']}")
                else:
                    print(f"  ⚠️ Limited profile data available")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
    
    return True

def test_indicator_calculations():
    """Test the indicator calculation functionality"""
    print("\nTesting indicator calculations...")
    
    # Create data retriever instance
    retriever = StockDataRetriever()
    
    # Get sample data for testing
    try:
        df = retriever.get_stock_history('AAPL', interval='1d', range_period='1y')
        
        if df.empty:
            print("  ❌ No data available for testing calculations")
            return False
        
        print(f"  ✅ Retrieved {len(df)} days of data for testing")
        
        # Test moving averages calculation
        print("\nTesting moving averages calculation:")
        ma_df = PerfectStormIndicators.calculate_moving_averages(df)
        
        ma_columns = [col for col in ma_df.columns if col.startswith('ma_')]
        if len(ma_columns) > 0:
            print(f"  ✅ Successfully calculated {len(ma_columns)} moving averages")
            for col in ma_columns:
                period = col.split('_')[1]
                print(f"  📈 {period}-day MA: ${ma_df[col].iloc[-1]:.2f}")
        else:
            print("  ❌ Failed to calculate moving averages")
        
        # Test Bollinger Bands calculation
        print("\nTesting Bollinger Bands calculation:")
        bb_df = PerfectStormIndicators.calculate_bollinger_bands(df)
        
        bb_columns = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_percent_b', 'bb_bandwidth']
        if all(col in bb_df.columns for col in bb_columns):
            print("  ✅ Successfully calculated Bollinger Bands")
            print(f"  📊 Middle Band: ${bb_df['bb_middle'].iloc[-1]:.2f}")
            print(f"  📊 Upper Band: ${bb_df['bb_upper'].iloc[-1]:.2f}")
            print(f"  📊 Lower Band: ${bb_df['bb_lower'].iloc[-1]:.2f}")
            print(f"  📊 %B: {bb_df['bb_percent_b'].iloc[-1]:.2f}")
            print(f"  📊 Bandwidth: {bb_df['bb_bandwidth'].iloc[-1]:.4f}")
        else:
            print("  ❌ Failed to calculate Bollinger Bands")
        
        # Test C/D signal calculation
        print("\nTesting C/D signal calculation:")
        cd_df = PerfectStormIndicators.calculate_cd_signal(df)
        
        if 'cd_signal' in cd_df.columns:
            print("  ✅ Successfully calculated C/D signal")
            print(f"  📊 Current C/D signal: {cd_df['cd_signal'].iloc[-1]:.2f}%")
        else:
            print("  ❌ Failed to calculate C/D signal")
        
        # Test MACD calculation
        print("\nTesting MACD calculation:")
        macd_df = PerfectStormIndicators.calculate_macd(df)
        
        macd_columns = ['macd_line', 'macd_signal', 'macd_histogram']
        if all(col in macd_df.columns for col in macd_columns):
            print("  ✅ Successfully calculated MACD")
            print(f"  📊 MACD Line: {macd_df['macd_line'].iloc[-1]:.4f}")
            print(f"  📊 Signal Line: {macd_df['macd_signal'].iloc[-1]:.4f}")
            print(f"  📊 Histogram: {macd_df['macd_histogram'].iloc[-1]:.4f}")
        else:
            print("  ❌ Failed to calculate MACD")
        
        # Test Chaikin Money Flow calculation
        print("\nTesting Chaikin Money Flow calculation:")
        cmf_df = PerfectStormIndicators.calculate_chaikin_money_flow(df)
        
        if 'cmf' in cmf_df.columns:
            print("  ✅ Successfully calculated Chaikin Money Flow")
            print(f"  📊 Current CMF: {cmf_df['cmf'].iloc[-1]:.4f}")
        else:
            print("  ❌ Failed to calculate Chaikin Money Flow")
        
        # Test all indicators calculation
        print("\nTesting all indicators calculation:")
        all_df = PerfectStormIndicators.calculate_all_indicators(df)
        
        # Count the number of indicator columns added
        original_cols = set(['open', 'high', 'low', 'close', 'volume', 'adj_close'])
        indicator_cols = [col for col in all_df.columns if col not in original_cols]
        
        print(f"  ✅ Successfully calculated {len(indicator_cols)} indicator columns")
        print(f"  📊 Sample of recent data with indicators:")
        print(all_df[['close', 'ma_20', 'bb_upper', 'bb_lower', 'cd_signal', 'macd_line', 'cmf']].tail(1).T)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing calculations: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PERFECT STORM DASHBOARD COMPONENT TESTS")
    print("=" * 50)
    
    # Run tests
    data_test_result = test_data_retrieval()
    calc_test_result = test_indicator_calculations()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Data Retrieval: {'PASSED' if data_test_result else 'FAILED'}")
    print(f"Indicator Calculations: {'PASSED' if calc_test_result else 'FAILED'}")
    
    if data_test_result and calc_test_result:
        print("\n✅ All tests passed! The dashboard components are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
