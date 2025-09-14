#!/usr/bin/env python3
"""
Risk Management Analysis for Trading Bot

This script demonstrates the impact of risk management features on trading performance,
comparing strategies with and without stop-loss, take-profit, and other risk controls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data_fetcher import DataFetcher
from src.trading_strategy import MovingAverageStrategy
from src.enhanced_strategy import (
    EnhancedMovingAverageStrategy, 
    BollingerBandsMeanReversionStrategy,
    compare_strategies_with_risk_management
)
from src.visualization import TradingVisualizer

def analyze_risk_management_impact(symbols: List[str] = ['MSFT', 'TSLA', 'AAPL'],
                                 start_date: str = "2023-01-01",
                                 end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Analyze the impact of risk management on trading performance.
    
    Args:
        symbols: List of symbols to analyze
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        DataFrame with comparison results
    """
    print("🛡️  Starting Risk Management Impact Analysis")
    print("="*60)
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        # Compare strategies with and without risk management
        results = compare_strategies_with_risk_management(symbol, start_date, end_date)
        
        if results:
            for strategy_name, metrics in results.items():
                result_row = {
                    'Symbol': symbol,
                    'Strategy': strategy_name,
                    'Total_Return_Pct': metrics['total_return'],
                    'Sharpe_Ratio': metrics['sharpe_ratio'],
                    'Max_Drawdown_Pct': metrics['max_drawdown'],
                    'Num_Trades': metrics['num_trades']
                }
                
                # Add risk management stats if available
                if 'risk_management_stats' in metrics:
                    risk_stats = metrics['risk_management_stats']
                    result_row.update({
                        'Stop_Loss_Exits': risk_stats.get('stop_loss_exits', 0),
                        'Take_Profit_Exits': risk_stats.get('take_profit_exits', 0),
                        'Signal_Exits': risk_stats.get('signal_exits', 0)
                    })
                else:
                    result_row.update({
                        'Stop_Loss_Exits': 0,
                        'Take_Profit_Exits': 0,
                        'Signal_Exits': 0
                    })
                
                all_results.append(result_row)
    
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        print("\n📈 RISK MANAGEMENT IMPACT SUMMARY")
        print("="*50)
        
        # Calculate average performance by strategy
        strategy_summary = results_df.groupby('Strategy').agg({
            'Total_Return_Pct': 'mean',
            'Sharpe_Ratio': 'mean',
            'Max_Drawdown_Pct': 'mean',
            'Num_Trades': 'mean',
            'Stop_Loss_Exits': 'sum',
            'Take_Profit_Exits': 'sum'
        }).round(2)
        
        print(strategy_summary)
        
        # Show improvement from risk management
        if 'Basic MA' in strategy_summary.index and 'Enhanced MA' in strategy_summary.index:
            basic_sharpe = strategy_summary.loc['Basic MA', 'Sharpe_Ratio']
            enhanced_sharpe = strategy_summary.loc['Enhanced MA', 'Sharpe_Ratio']
            basic_drawdown = strategy_summary.loc['Basic MA', 'Max_Drawdown_Pct']
            enhanced_drawdown = strategy_summary.loc['Enhanced MA', 'Max_Drawdown_Pct']
            
            print(f"\n🎯 RISK MANAGEMENT IMPROVEMENTS:")
            print(f"   Sharpe Ratio: {basic_sharpe:.2f} → {enhanced_sharpe:.2f} ({((enhanced_sharpe/basic_sharpe-1)*100):+.1f}%)")
            print(f"   Max Drawdown: {basic_drawdown:.1f}% → {enhanced_drawdown:.1f}% ({((enhanced_drawdown/basic_drawdown-1)*100):+.1f}%)")
    
    return results_df

def create_risk_management_visualizations(results_df: pd.DataFrame, save_path: str = 'analysis_charts'):
    """
    Create visualizations showing risk management impact.
    
    Args:
        results_df: DataFrame with analysis results
        save_path: Directory to save charts
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance Comparison Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sharpe Ratio comparison
    sharpe_data = results_df.pivot(index='Symbol', columns='Strategy', values='Sharpe_Ratio')
    sharpe_data.plot(kind='bar', ax=axes[0,0], title='Sharpe Ratio by Strategy')
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Max Drawdown comparison
    drawdown_data = results_df.pivot(index='Symbol', columns='Strategy', values='Max_Drawdown_Pct')
    drawdown_data.plot(kind='bar', ax=axes[0,1], title='Maximum Drawdown by Strategy')
    axes[0,1].set_ylabel('Max Drawdown (%)')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Total Return comparison
    return_data = results_df.pivot(index='Symbol', columns='Strategy', values='Total_Return_Pct')
    return_data.plot(kind='bar', ax=axes[1,0], title='Total Return by Strategy')
    axes[1,0].set_ylabel('Total Return (%)')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Number of Trades comparison
    trades_data = results_df.pivot(index='Symbol', columns='Strategy', values='Num_Trades')
    trades_data.plot(kind='bar', ax=axes[1,1], title='Number of Trades by Strategy')
    axes[1,1].set_ylabel('Number of Trades')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/risk_management_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Risk Management Exit Analysis
    enhanced_strategies = results_df[results_df['Strategy'].str.contains('Enhanced|Bollinger')]
    
    if not enhanced_strategies.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Exit reasons by strategy
        exit_data = enhanced_strategies.groupby('Strategy')[['Stop_Loss_Exits', 'Take_Profit_Exits', 'Signal_Exits']].sum()
        exit_data.plot(kind='bar', stacked=True, ax=axes[0], title='Exit Reasons by Strategy')
        axes[0].set_ylabel('Number of Exits')
        axes[0].legend(title='Exit Type')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Exit reasons by symbol
        exit_by_symbol = enhanced_strategies.groupby('Symbol')[['Stop_Loss_Exits', 'Take_Profit_Exits', 'Signal_Exits']].sum()
        exit_by_symbol.plot(kind='bar', stacked=True, ax=axes[1], title='Exit Reasons by Symbol')
        axes[1].set_ylabel('Number of Exits')
        axes[1].legend(title='Exit Type')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/risk_management_exit_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Risk management visualizations saved to {save_path}/")

def demonstrate_tsla_risk_management():
    """
    Demonstrate the impact of risk management on TSLA (high volatility stock).
    """
    print("\n🎯 DEMONSTRATING RISK MANAGEMENT IMPACT ON TSLA")
    print("="*60)
    
    # Fetch TSLA data
    fetcher = DataFetcher()
    data = fetcher.get_stock_data('TSLA', '2023-01-01', '2024-12-31')
    
    if data is None or data.empty:
        print("❌ Could not fetch TSLA data")
        return
    
    # Test basic strategy vs enhanced strategy
    basic_strategy = MovingAverageStrategy(20, 50)
    enhanced_strategy = EnhancedMovingAverageStrategy(
        short_window=20, long_window=50, trend_filter_window=200,
        use_rsi_filter=True, stop_loss_pct=0.05, take_profit_pct=0.15
    )
    
    # Generate signals
    basic_signals = basic_strategy.generate_signals(data.copy())
    enhanced_signals = enhanced_strategy.generate_signals(data.copy())
    
    # Calculate performance
    def calculate_performance(data, signals, position_col='position'):
        returns = data['Close'].pct_change()
        strategy_returns = signals[position_col].shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100
        }
    
    basic_perf = calculate_performance(data, basic_signals)
    enhanced_perf = calculate_performance(data, enhanced_signals, 'risk_managed_position')
    
    print(f"\n📊 TSLA PERFORMANCE COMPARISON:")
    print(f"\n🔸 Basic Moving Average Strategy:")
    print(f"   Total Return: {basic_perf['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {basic_perf['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {basic_perf['max_drawdown']:.2f}%")
    
    print(f"\n🔹 Enhanced Strategy with Risk Management:")
    print(f"   Total Return: {enhanced_perf['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {enhanced_perf['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {enhanced_perf['max_drawdown']:.2f}%")
    
    # Show risk management statistics
    if 'exit_reason' in enhanced_signals.columns:
        exit_reasons = enhanced_signals[enhanced_signals['exit_reason'] != '']['exit_reason'].value_counts()
        print(f"\n🛡️  Risk Management Exits:")
        for reason, count in exit_reasons.items():
            print(f"   {reason.replace('_', ' ').title()}: {count}")
    
    # Calculate improvement
    if basic_perf['max_drawdown'] > 0:
        drawdown_improvement = (basic_perf['max_drawdown'] - enhanced_perf['max_drawdown']) / basic_perf['max_drawdown'] * 100
        print(f"\n🎯 RISK MANAGEMENT IMPACT:")
        print(f"   Drawdown Reduction: {drawdown_improvement:.1f}%")
        print(f"   Sharpe Ratio Change: {((enhanced_perf['sharpe_ratio']/basic_perf['sharpe_ratio']-1)*100):+.1f}%" if basic_perf['sharpe_ratio'] != 0 else "N/A")

def main():
    """
    Main function to run risk management analysis.
    """
    print("🚀 Starting Comprehensive Risk Management Analysis")
    print("="*70)
    
    # 1. Analyze risk management impact across multiple symbols
    symbols = ['MSFT', 'TSLA', 'AAPL', 'GOOGL', 'AMZN']
    results_df = analyze_risk_management_impact(symbols)
    
    if not results_df.empty:
        # 2. Create visualizations
        print("\n📊 Creating risk management visualizations...")
        create_risk_management_visualizations(results_df)
        
        # 3. Save results
        results_df.to_csv('analysis_charts/risk_management_analysis_results.csv', index=False)
        print("💾 Results saved to: analysis_charts/risk_management_analysis_results.csv")
    
    # 4. Demonstrate TSLA specific analysis
    demonstrate_tsla_risk_management()
    
    print("\n🎯 KEY TAKEAWAYS FROM RISK MANAGEMENT ANALYSIS:")
    print("1. 🛡️  Stop-loss mechanisms significantly reduce maximum drawdown")
    print("2. 💰 Take-profit rules help lock in gains during volatile periods")
    print("3. 📊 Risk-adjusted returns (Sharpe ratio) often improve with proper risk management")
    print("4. 🔄 Trailing stops can capture more upside while still protecting downside")
    print("5. ⚖️  There's always a trade-off between risk reduction and return potential")
    
    print("\n✅ Risk management analysis completed!")
    print("\n💡 NEXT STEPS:")
    print("   - Review the generated charts to understand risk management impact")
    print("   - Experiment with different stop-loss and take-profit percentages")
    print("   - Consider implementing position sizing based on volatility")
    print("   - Test risk management on different market conditions")

if __name__ == "__main__":
    main()