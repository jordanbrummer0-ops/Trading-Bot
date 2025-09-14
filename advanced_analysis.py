#!/usr/bin/env python3
"""
Advanced Trading Bot Analysis

This script performs deep analysis of trading strategies with visualization,
parameter optimization, and comprehensive performance evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_fetcher import DataFetcher
from src.trading_strategy import MovingAverageStrategy, RSIStrategy, MACDStrategy
from src.backtesting_engine import BacktestingEngine
from src.visualization import TradingVisualizer, create_analysis_report

def calculate_performance_metrics(data: pd.DataFrame, signals: pd.DataFrame) -> dict:
    """
    Calculate comprehensive performance metrics for a trading strategy.
    
    Args:
        data: Price data
        signals: Trading signals data
    
    Returns:
        Dictionary with performance metrics
    """
    # Merge data and signals
    combined = data.join(signals[['signal', 'position']], how='left')
    combined['position'] = combined['position'].fillna(method='ffill').fillna(0)
    
    # Calculate returns
    combined['returns'] = combined['Close'].pct_change()
    combined['strategy_returns'] = combined['position'].shift(1) * combined['returns']
    
    # Remove NaN values
    strategy_returns = combined['strategy_returns'].dropna()
    
    # Calculate metrics
    total_return = (1 + strategy_returns).prod() - 1
    total_return_pct = total_return * 100
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100
    
    # Number of trades
    position_changes = combined['position'].diff().abs()
    num_trades = (position_changes > 0).sum()
    
    # Win rate
    winning_trades = strategy_returns[strategy_returns > 0]
    win_rate = len(winning_trades) / len(strategy_returns[strategy_returns != 0]) * 100 if len(strategy_returns[strategy_returns != 0]) > 0 else 0
    
    return {
        'total_return': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate
    }

def analyze_strategy_performance(symbol: str, start_date: str = "2023-01-01", 
                               end_date: str = "2024-12-31") -> dict:
    """
    Perform comprehensive analysis of a trading strategy on a specific symbol.
    
    Args:
        symbol: Stock symbol to analyze
        start_date: Start date for analysis
        end_date: End date for analysis
    
    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n🔍 Analyzing {symbol}...")
    
    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.get_stock_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        print(f"❌ No data available for {symbol}")
        return {}
    
    # Initialize strategy
    strategy = MovingAverageStrategy(short_window=20, long_window=50)
    
    # Generate signals
    signals = strategy.generate_signals(data.copy())
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(data, signals)
    
    # Add moving averages to data for visualization
    data['MA_Short'] = data['Close'].rolling(window=20).mean()
    data['MA_Long'] = data['Close'].rolling(window=50).mean()
    
    # Calculate equity curve and drawdown
    equity_curve, drawdown_series = calculate_equity_curve(data, signals, 10000)
    
    # Store all results
    analysis_results = {
        'symbol': symbol,
        'data': data,
        'signals': signals,
        'performance': performance,
        'equity_curve': equity_curve,
        'drawdown_series': drawdown_series,
        'strategy_params': {'short_window': 20, 'long_window': 50}
    }
    
    return analysis_results

def calculate_equity_curve(data: pd.DataFrame, signals: pd.DataFrame, 
                         initial_cash: float = 10000) -> tuple:
    """
    Calculate detailed equity curve and drawdown series.
    
    Args:
        data: Price data
        signals: Trading signals
        initial_cash: Starting portfolio value
    
    Returns:
        Tuple of (equity_curve, drawdown_series)
    """
    # Merge data and signals
    combined = data.join(signals[['signal', 'position']], how='left')
    combined['position'] = combined['position'].fillna(method='ffill').fillna(0)
    
    # Calculate returns
    combined['returns'] = combined['Close'].pct_change()
    combined['strategy_returns'] = combined['position'].shift(1) * combined['returns']
    
    # Calculate cumulative returns and equity curve
    combined['cumulative_returns'] = (1 + combined['strategy_returns']).cumprod()
    equity_curve = initial_cash * combined['cumulative_returns']
    
    # Calculate drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown_series = (equity_curve - rolling_max) / rolling_max * 100
    
    return equity_curve, drawdown_series

def parameter_optimization_analysis(symbol: str, param_ranges: dict,
                                  start_date: str = "2023-01-01",
                                  end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Perform parameter optimization for moving average strategy.
    
    Args:
        symbol: Stock symbol
        param_ranges: Dictionary with parameter ranges
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with optimization results
    """
    print(f"\n🔧 Optimizing parameters for {symbol}...")
    
    fetcher = DataFetcher()
    data = fetcher.get_stock_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        return pd.DataFrame()
    
    results = []
    
    short_windows = param_ranges.get('short_window', [10, 20, 30])
    long_windows = param_ranges.get('long_window', [50, 100, 200])
    
    total_combinations = len(short_windows) * len(long_windows)
    current = 0
    
    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue
                
            current += 1
            print(f"\rTesting combination {current}/{total_combinations}: MA({short_window},{long_window})", end="")
            
            try:
                # Test strategy with these parameters
                strategy = MovingAverageStrategy(short_window=short_window, 
                                               long_window=long_window)
                signals = strategy.generate_signals(data)
                performance = strategy.calculate_performance(data, signals)
                
                results.append({
                    'Short MA': short_window,
                    'Long MA': long_window,
                    'total_return': performance.get('total_return', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'num_trades': performance.get('num_trades', 0),
                    'win_rate': performance.get('win_rate', 0)
                })
            except Exception as e:
                print(f"\nError with MA({short_window},{long_window}): {e}")
                continue
    
    print("\n✅ Parameter optimization completed!")
    return pd.DataFrame(results)

def compare_multiple_strategies(symbol: str, start_date: str = "2023-01-01",
                              end_date: str = "2024-12-31") -> dict:
    """
    Compare different trading strategies on the same symbol.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        Dictionary with strategy comparison results
    """
    print(f"\n📊 Comparing strategies for {symbol}...")
    
    fetcher = DataFetcher()
    data = fetcher.get_stock_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        return {}
    
    strategies = {
        'MA_20_50': MovingAverageStrategy(short_window=20, long_window=50),
        'MA_50_200': MovingAverageStrategy(short_window=50, long_window=200),
        'RSI': RSIStrategy(rsi_period=14, oversold=30, overbought=70),
        'MACD': MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        try:
            print(f"  Testing {strategy_name}...")
            signals = strategy.generate_signals(data)
            performance = strategy.calculate_performance(data, signals)
            
            results[strategy_name] = {
                'performance': performance,
                'signals': signals
            }
        except Exception as e:
            print(f"  ❌ Error with {strategy_name}: {e}")
            continue
    
    return results

def main():
    """
    Main analysis function that performs comprehensive trading strategy analysis.
    """
    print("🚀 Starting Advanced Trading Bot Analysis")
    print("=" * 50)
    
    # Analysis parameters
    symbols_to_analyze = ['MSFT', 'TSLA']  # Winner vs Loser from previous analysis
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    # Create visualizer
    visualizer = TradingVisualizer()
    
    # 1. DETAILED ANALYSIS OF MSFT vs TSLA
    print("\n📈 PHASE 1: Detailed Analysis of MSFT vs TSLA")
    print("-" * 50)
    
    analysis_results = {}
    
    for symbol in symbols_to_analyze:
        results = analyze_strategy_performance(symbol, start_date, end_date)
        if results:
            analysis_results[symbol] = results
            
            # Create comprehensive analysis report
            create_analysis_report(
                symbol, 
                results['data'], 
                results['signals'],
                {
                    'equity_curve': results['equity_curve'],
                    'drawdown_series': results['drawdown_series']
                },
                save_dir="analysis_charts"
            )
    
    # 2. PERFORMANCE COMPARISON
    print("\n📊 PHASE 2: Performance Comparison")
    print("-" * 50)
    
    if len(analysis_results) >= 2:
        comparison_data = {}
        for symbol, results in analysis_results.items():
            comparison_data[symbol] = results['performance']
        
        visualizer.plot_performance_comparison(
            comparison_data, 
            save_path="analysis_charts/performance_comparison.png"
        )
    
    # 3. PARAMETER OPTIMIZATION
    print("\n🔧 PHASE 3: Parameter Optimization")
    print("-" * 50)
    
    param_ranges = {
        'short_window': [10, 20, 30, 40],
        'long_window': [50, 100, 150, 200]
    }
    
    for symbol in symbols_to_analyze:
        optimization_results = parameter_optimization_analysis(
            symbol, param_ranges, start_date, end_date
        )
        
        if not optimization_results.empty:
            # Find best parameters
            best_sharpe = optimization_results.loc[optimization_results['sharpe_ratio'].idxmax()]
            best_return = optimization_results.loc[optimization_results['total_return'].idxmax()]
            
            print(f"\n📈 {symbol} Optimization Results:")
            print(f"  Best Sharpe Ratio: MA({int(best_sharpe['Short MA'])},{int(best_sharpe['Long MA'])}) = {best_sharpe['sharpe_ratio']:.3f}")
            print(f"  Best Return: MA({int(best_return['Short MA'])},{int(best_return['Long MA'])}) = {best_return['total_return']:.2f}%")
            
            # Create optimization heatmap
            visualizer.plot_parameter_optimization(
                optimization_results,
                save_path=f"analysis_charts/{symbol}_parameter_optimization.png"
            )
    
    # 4. STRATEGY COMPARISON
    print("\n🔄 PHASE 4: Strategy Comparison")
    print("-" * 50)
    
    for symbol in symbols_to_analyze:
        strategy_results = compare_multiple_strategies(symbol, start_date, end_date)
        
        if strategy_results:
            print(f"\n📊 {symbol} Strategy Comparison:")
            for strategy_name, results in strategy_results.items():
                perf = results['performance']
                print(f"  {strategy_name:12}: Return={perf.get('total_return', 0):6.2f}%, "
                      f"Sharpe={perf.get('sharpe_ratio', 0):5.3f}, "
                      f"MaxDD={perf.get('max_drawdown', 0):5.2f}%")
    
    # 5. KEY INSIGHTS
    print("\n💡 PHASE 5: Key Insights")
    print("-" * 50)
    
    if 'MSFT' in analysis_results and 'TSLA' in analysis_results:
        msft_perf = analysis_results['MSFT']['performance']
        tsla_perf = analysis_results['TSLA']['performance']
        
        print("\n🎯 Why MSFT outperformed TSLA:")
        print(f"  • MSFT Sharpe Ratio: {msft_perf.get('sharpe_ratio', 0):.3f} vs TSLA: {tsla_perf.get('sharpe_ratio', 0):.3f}")
        print(f"  • MSFT Max Drawdown: {msft_perf.get('max_drawdown', 0):.2f}% vs TSLA: {tsla_perf.get('max_drawdown', 0):.2f}%")
        print(f"  • MSFT showed smoother trends, better suited for MA crossover strategy")
        print(f"  • TSLA's high volatility created many false signals and whipsaws")
        
        print("\n📋 Recommendations:")
        print("  1. Use trend-following strategies (like MA) on smoother, trending stocks")
        print("  2. Consider mean-reversion strategies for highly volatile stocks")
        print("  3. Implement risk management (stop-loss) especially for volatile assets")
        print("  4. Test strategies across different market conditions and sectors")
    
    print("\n✅ Advanced analysis completed!")
    print("📁 All charts saved in 'analysis_charts' directory")
    print("\n🚀 Next Steps:")
    print("  1. Review the generated charts to understand strategy behavior")
    print("  2. Implement risk management features (stop-loss, take-profit)")
    print("  3. Test on different sectors and asset classes")
    print("  4. Consider combining multiple strategies for better performance")

if __name__ == "__main__":
    main()