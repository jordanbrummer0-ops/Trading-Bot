#!/usr/bin/env python3
"""
Example Usage of Stock Trading Bot Components

This script demonstrates how to use the trading bot components
independently for custom analysis and strategy development.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import DataFetcher
from trading_strategy import MovingAverageStrategy, RSIStrategy, MACDStrategy, CombinedStrategy
from backtesting_engine import BacktestingEngine


def example_data_fetching():
    """Example: Fetch stock data for multiple symbols."""
    print("\n=== Data Fetching Example ===")
    
    fetcher = DataFetcher()
    
    # Fetch data for a single stock
    print("Fetching AAPL data...")
    aapl_data = fetcher.get_stock_data(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    if not aapl_data.empty:
        print(f"Retrieved {len(aapl_data)} records for AAPL")
        print(f"Date range: {aapl_data.index[0]} to {aapl_data.index[-1]}")
        print(f"Price range: ${aapl_data['Low'].min():.2f} - ${aapl_data['High'].max():.2f}")
    
    # Get current price
    current_price = fetcher.get_current_price('AAPL')
    if current_price:
        print(f"Current AAPL price: ${current_price:.2f}")
    
    # Get stock information
    stock_info = fetcher.get_stock_info('AAPL')
    print(f"Company: {stock_info.get('company_name', 'N/A')}")
    print(f"Sector: {stock_info.get('sector', 'N/A')}")
    
    return aapl_data


def example_strategy_comparison(data):
    """Example: Compare different trading strategies."""
    print("\n=== Strategy Comparison Example ===")
    
    if data.empty:
        print("No data available for strategy comparison")
        return
    
    strategies = {
        'Moving Average': MovingAverageStrategy(short_window=20, long_window=50),
        'RSI': RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        'MACD': MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        'Combined': CombinedStrategy(ma_short=20, ma_long=50, rsi_period=14)
    }
    
    backtester = BacktestingEngine(initial_cash=10000)
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name} Strategy...")
        try:
            result = backtester.run_simple_backtest(data, strategy)
            results[name] = result
            
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Number of Trades: {result['num_trades']}")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"  Error testing {name}: {str(e)}")
    
    # Find best strategy
    if results:
        best_strategy = max(results.items(), key=lambda x: x[1]['total_return'])
        print(f"\nBest performing strategy: {best_strategy[0]} ({best_strategy[1]['total_return']:.2f}% return)")
    
    return results


def example_custom_strategy():
    """Example: Create a custom trading strategy."""
    print("\n=== Custom Strategy Example ===")
    
    from trading_strategy import BaseStrategy
    import pandas as pd
    
    class SimpleBreakoutStrategy(BaseStrategy):
        """Simple breakout strategy based on 20-day high/low."""
        
        def __init__(self, lookback_period=20):
            super().__init__("Simple Breakout")
            self.lookback_period = lookback_period
        
        def generate_signals(self, data):
            data = data.copy()
            
            # Calculate rolling high and low
            data['rolling_high'] = data['High'].rolling(window=self.lookback_period).max()
            data['rolling_low'] = data['Low'].rolling(window=self.lookback_period).min()
            
            # Generate signals
            data['signal'] = 0
            data['position'] = 0
            
            # Buy signal: price breaks above rolling high
            data.loc[data['Close'] > data['rolling_high'].shift(1), 'signal'] = 1
            
            # Sell signal: price breaks below rolling low
            data.loc[data['Close'] < data['rolling_low'].shift(1), 'signal'] = -1
            
            # Generate position
            data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
            
            # Mark entry and exit points
            data['entry'] = (data['position'] != data['position'].shift(1)) & (data['position'] != 0)
            data['exit'] = (data['position'] != data['position'].shift(1)) & (data['position'].shift(1) != 0)
            
            return data
    
    # Test the custom strategy
    fetcher = DataFetcher()
    data = fetcher.get_stock_data('MSFT', '2023-01-01', '2023-12-31')
    
    if not data.empty:
        custom_strategy = SimpleBreakoutStrategy(lookback_period=20)
        backtester = BacktestingEngine(initial_cash=10000)
        
        result = backtester.run_simple_backtest(data, custom_strategy)
        
        print(f"Custom Breakout Strategy Results:")
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"  Number of Trades: {result['num_trades']}")
        print(f"  Win Rate: {result['win_rate']:.2f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    else:
        print("Could not fetch data for custom strategy example")


def example_portfolio_analysis():
    """Example: Analyze multiple stocks for portfolio construction."""
    print("\n=== Portfolio Analysis Example ===")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    fetcher = DataFetcher()
    
    print(f"Analyzing {len(symbols)} stocks for portfolio construction...")
    
    portfolio_results = {}
    strategy = MovingAverageStrategy(short_window=20, long_window=50)
    backtester = BacktestingEngine(initial_cash=10000)
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        try:
            data = fetcher.get_stock_data(symbol, '2023-01-01', '2023-12-31')
            
            if not data.empty:
                result = backtester.run_simple_backtest(data, strategy)
                portfolio_results[symbol] = {
                    'return': result['total_return'],
                    'sharpe': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'num_trades': result['num_trades']
                }
                
                print(f"  Return: {result['total_return']:.2f}%")
                print(f"  Sharpe: {result['sharpe_ratio']:.3f}")
                print(f"  Max DD: {result['max_drawdown']:.2f}%")
            else:
                print(f"  No data available for {symbol}")
                
        except Exception as e:
            print(f"  Error analyzing {symbol}: {str(e)}")
    
    # Rank stocks by Sharpe ratio
    if portfolio_results:
        print("\nRanking by Sharpe Ratio:")
        sorted_stocks = sorted(portfolio_results.items(), 
                             key=lambda x: x[1]['sharpe'], reverse=True)
        
        for i, (symbol, metrics) in enumerate(sorted_stocks, 1):
            print(f"  {i}. {symbol}: Sharpe={metrics['sharpe']:.3f}, "
                  f"Return={metrics['return']:.2f}%")


def main():
    """Run all examples."""
    print("Stock Trading Bot - Example Usage")
    print("=" * 50)
    
    try:
        # Example 1: Data fetching
        data = example_data_fetching()
        
        # Example 2: Strategy comparison
        if not data.empty:
            example_strategy_comparison(data)
        
        # Example 3: Custom strategy
        example_custom_strategy()
        
        # Example 4: Portfolio analysis
        example_portfolio_analysis()
        
        print("\n=== Examples completed successfully! ===")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the main bot: python main.py --symbol AAPL")
        print("3. Modify strategies in src/trading_strategy.py")
        print("4. Create your own custom strategies")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Make sure to install dependencies first: pip install -r requirements.txt")


if __name__ == '__main__':
    main()