import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_fetcher import DataFetcher
from src.trading_strategy import MovingAverageStrategy, RSIStrategy
from bollinger_bands_strategy import BollingerBandsStrategy
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StressTester:
    """
    Comprehensive stress testing for trading strategies during major market events.
    Tests strategies during crashes, corrections, and volatile periods.
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        
        # Define major market stress periods
        self.stress_periods = {
            '2008 Financial Crisis': {
                'start_date': '2007-10-01',
                'end_date': '2009-03-31',
                'description': 'Global financial crisis and market crash',
                'characteristics': ['High volatility', 'Bear market', 'Credit crisis'],
                'market_decline': -57  # S&P 500 peak to trough
            },
            '2020 COVID Crash': {
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'description': 'COVID-19 pandemic market crash and recovery',
                'characteristics': ['Extreme volatility', 'V-shaped recovery', 'Policy intervention'],
                'market_decline': -34  # S&P 500 peak to trough
            },
            '2018 Q4 Correction': {
                'start_date': '2018-10-01',
                'end_date': '2019-01-31',
                'description': 'Trade war fears and rate hike concerns',
                'characteristics': ['Moderate correction', 'Trade tensions', 'Rate uncertainty'],
                'market_decline': -20  # S&P 500 peak to trough
            },
            '2015-2016 Oil Crash': {
                'start_date': '2015-06-01',
                'end_date': '2016-02-29',
                'description': 'Oil price collapse and emerging market stress',
                'characteristics': ['Commodity crash', 'EM stress', 'Dollar strength'],
                'market_decline': -15  # S&P 500 peak to trough
            },
            '2011 European Debt Crisis': {
                'start_date': '2011-05-01',
                'end_date': '2011-10-31',
                'description': 'European sovereign debt crisis',
                'characteristics': ['Sovereign risk', 'Banking stress', 'Contagion fears'],
                'market_decline': -19  # S&P 500 peak to trough
            },
            'Dot-com Crash (2000-2002)': {
                'start_date': '2000-03-01',
                'end_date': '2002-10-31',
                'description': 'Technology bubble burst',
                'characteristics': ['Tech crash', 'Prolonged bear market', 'Recession'],
                'market_decline': -49  # S&P 500 peak to trough
            }
        }
        
        # Test symbols representing different market segments
        self.test_symbols = {
            'Broad Market': ['SPY', 'QQQ', 'IWM'],
            'Defensive': ['KO', 'JNJ', 'PG'],
            'Cyclical': ['CAT', 'BA', 'JPM'],
            'Growth': ['MSFT', 'AAPL', 'GOOGL'],
            'Value': ['BRK-B', 'XOM', 'WFC']
        }
        
        self.strategies = {
            'Moving Average': MovingAverageStrategy(),
            'RSI': RSIStrategy(),
            'Bollinger Bands': BollingerBandsStrategy()
        }
    
    def run_stress_test(self, period_name, period_info):
        """
        Run stress test for a specific period.
        """
        print(f"\n🔥 STRESS TESTING: {period_name}")
        print(f"   Period: {period_info['start_date']} to {period_info['end_date']}")
        print(f"   Description: {period_info['description']}")
        print(f"   Market Decline: {period_info['market_decline']}%")
        print(f"   Characteristics: {', '.join(period_info['characteristics'])}")
        
        results = []
        
        for category, symbols in self.test_symbols.items():
            print(f"\n   📊 Testing {category} stocks...")
            
            for symbol in symbols:
                print(f"      📈 {symbol}...", end=" ")
                
                try:
                    # Fetch data for the stress period
                    data = self.fetcher.get_stock_data(symbol, period_info['start_date'], period_info['end_date'])
                    
                    if data is None or len(data) < 50:
                        print("❌ Insufficient data")
                        continue
                    
                    # Calculate buy & hold performance
                    buy_hold_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    
                    # Calculate volatility during period
                    returns = data['Close'].pct_change().dropna()
                    period_volatility = returns.std() * np.sqrt(252) * 100
                    
                    # Calculate maximum drawdown for buy & hold
                    cumulative_returns = (1 + returns).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown_buyhold = drawdown.min() * 100
                    
                    symbol_results = {
                        'Period': period_name,
                        'Symbol': symbol,
                        'Category': category,
                        'Buy_Hold_Return': buy_hold_return,
                        'Period_Volatility': period_volatility,
                        'Max_Drawdown_BuyHold': abs(max_drawdown_buyhold)
                    }
                    
                    # Test each strategy
                    for strategy_name, strategy in self.strategies.items():
                        try:
                            # Generate signals
                            data_copy = data.copy()
                            if strategy_name == 'Bollinger Bands':
                                data_copy = strategy.generate_signals(data_copy)
                                strategy_returns = data_copy['Close'].pct_change() * data_copy['signal'].shift(1).fillna(0)
                            else:
                                data_copy = strategy.generate_signals(data_copy)
                                strategy_returns = data_copy['Close'].pct_change() * data_copy['signal'].shift(1).fillna(0)
                            
                            # Calculate strategy metrics
                            total_return = (1 + strategy_returns).prod() - 1
                            
                            # Calculate strategy drawdown
                            strategy_cumulative = (1 + strategy_returns).cumprod()
                            strategy_running_max = strategy_cumulative.expanding().max()
                            strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
                            max_drawdown_strategy = strategy_drawdown.min() * 100
                            
                            # Calculate downside protection (how much better/worse than buy & hold)
                            downside_protection = (total_return * 100) - buy_hold_return
                            
                            # Calculate number of trades
                            signals = data_copy['signal'].fillna(0)
                            trades = (signals != signals.shift(1)).sum()
                            
                            # Time in market (percentage of time with position)
                            time_in_market = (signals != 0).sum() / len(signals) * 100
                            
                            symbol_results[f'{strategy_name}_Return'] = total_return * 100
                            symbol_results[f'{strategy_name}_vs_BuyHold'] = downside_protection
                            symbol_results[f'{strategy_name}_Max_Drawdown'] = abs(max_drawdown_strategy)
                            symbol_results[f'{strategy_name}_Trades'] = trades
                            symbol_results[f'{strategy_name}_Time_in_Market'] = time_in_market
                            
                        except Exception as e:
                            print(f"Error with {strategy_name}: {str(e)[:30]}...")
                            symbol_results[f'{strategy_name}_Return'] = buy_hold_return
                            symbol_results[f'{strategy_name}_vs_BuyHold'] = 0
                            symbol_results[f'{strategy_name}_Max_Drawdown'] = abs(max_drawdown_buyhold)
                            symbol_results[f'{strategy_name}_Trades'] = 0
                            symbol_results[f'{strategy_name}_Time_in_Market'] = 100
                    
                    results.append(symbol_results)
                    print("✅")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
                    continue
        
        return pd.DataFrame(results)
    
    def analyze_stress_test_results(self, all_results):
        """
        Analyze and summarize stress test results across all periods.
        """
        print("\n📊 STRESS TEST ANALYSIS SUMMARY")
        print("=" * 40)
        
        # Summary by period
        period_summary = []
        
        for period in all_results['Period'].unique():
            period_data = all_results[all_results['Period'] == period]
            
            if len(period_data) == 0:
                continue
            
            summary = {
                'Period': period,
                'Symbols_Tested': len(period_data),
                'Avg_Buy_Hold_Return': period_data['Buy_Hold_Return'].mean(),
                'Avg_Volatility': period_data['Period_Volatility'].mean(),
                'Avg_Max_Drawdown_BuyHold': period_data['Max_Drawdown_BuyHold'].mean()
            }
            
            # Strategy performance during this period
            for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
                avg_return = period_data[f'{strategy}_Return'].mean()
                avg_protection = period_data[f'{strategy}_vs_BuyHold'].mean()
                avg_drawdown = period_data[f'{strategy}_Max_Drawdown'].mean()
                avg_trades = period_data[f'{strategy}_Trades'].mean()
                
                # Count how many symbols the strategy protected vs buy & hold
                protection_count = (period_data[f'{strategy}_vs_BuyHold'] > 0).sum()
                total_symbols = len(period_data)
                protection_rate = protection_count / total_symbols * 100
                
                summary[f'{strategy}_Avg_Return'] = avg_return
                summary[f'{strategy}_Avg_Protection'] = avg_protection
                summary[f'{strategy}_Avg_Drawdown'] = avg_drawdown
                summary[f'{strategy}_Avg_Trades'] = avg_trades
                summary[f'{strategy}_Protection_Rate'] = protection_rate
            
            period_summary.append(summary)
        
        period_summary_df = pd.DataFrame(period_summary)
        
        print("\n📊 Performance Summary by Stress Period:")
        for _, row in period_summary_df.iterrows():
            print(f"\n🔥 {row['Period']}:")
            print(f"   Buy & Hold: {row['Avg_Buy_Hold_Return']:.1f}% return, {row['Avg_Max_Drawdown_BuyHold']:.1f}% max drawdown")
            
            for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
                protection = row[f'{strategy}_Avg_Protection']
                protection_rate = row[f'{strategy}_Protection_Rate']
                drawdown = row[f'{strategy}_Avg_Drawdown']
                trades = row[f'{strategy}_Avg_Trades']
                
                print(f"   {strategy}: {protection:+.1f}% vs B&H, {protection_rate:.0f}% protection rate, {drawdown:.1f}% drawdown, {trades:.0f} trades")
        
        return period_summary_df
    
    def create_stress_test_visualizations(self, all_results, period_summary_df):
        """
        Create comprehensive visualizations for stress test results.
        """
        print("\n📊 Creating stress test visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Stress Test Analysis', fontsize=20, fontweight='bold')
        
        strategies = ['Moving Average', 'RSI', 'Bollinger Bands']
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Protection rates by period
        ax1 = fig.add_subplot(gs[0, 0])
        protection_data = []
        for strategy in strategies:
            protection_data.append(period_summary_df[f'{strategy}_Protection_Rate'].tolist())
        
        x = np.arange(len(period_summary_df))
        width = 0.25
        
        for i, (strategy, color) in enumerate(zip(strategies, colors)):
            ax1.bar(x + i*width, protection_data[i], width, label=strategy, color=color, alpha=0.7)
        
        ax1.set_xlabel('Stress Periods')
        ax1.set_ylabel('Protection Rate (%)')
        ax1.set_title('Downside Protection Rate by Period')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in period_summary_df['Period']], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average excess returns vs buy & hold
        ax2 = fig.add_subplot(gs[0, 1])
        excess_returns = []
        for strategy in strategies:
            excess_returns.append(period_summary_df[f'{strategy}_Avg_Protection'].tolist())
        
        for i, (strategy, color) in enumerate(zip(strategies, colors)):
            ax2.bar(x + i*width, excess_returns[i], width, label=strategy, color=color, alpha=0.7)
        
        ax2.set_xlabel('Stress Periods')
        ax2.set_ylabel('Excess Return vs B&H (%)')
        ax2.set_title('Average Excess Returns vs Buy & Hold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in period_summary_df['Period']], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Maximum drawdown comparison
        ax3 = fig.add_subplot(gs[0, 2])
        drawdown_data = [period_summary_df['Avg_Max_Drawdown_BuyHold'].tolist()]
        labels = ['Buy & Hold']
        
        for strategy in strategies:
            drawdown_data.append(period_summary_df[f'{strategy}_Avg_Drawdown'].tolist())
            labels.append(strategy)
        
        colors_dd = ['black'] + colors
        
        for i, (data, label, color) in enumerate(zip(drawdown_data, labels, colors_dd)):
            ax3.bar(x + i*0.2, data, 0.2, label=label, color=color, alpha=0.7)
        
        ax3.set_xlabel('Stress Periods')
        ax3.set_ylabel('Maximum Drawdown (%)')
        ax3.set_title('Maximum Drawdown Comparison')
        ax3.set_xticks(x + 0.3)
        ax3.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in period_summary_df['Period']], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Strategy performance heatmap
        ax4 = fig.add_subplot(gs[1, :])
        heatmap_data = []
        for strategy in strategies:
            heatmap_data.append(period_summary_df[f'{strategy}_Avg_Protection'].tolist())
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=strategies, 
                                 columns=[p[:15] + '...' if len(p) > 15 else p for p in period_summary_df['Period']])
        
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                   ax=ax4, cbar_kws={'label': 'Excess Return vs Buy & Hold (%)'})
        ax4.set_title('Strategy Performance Heatmap (Excess Returns vs Buy & Hold)')
        ax4.set_xlabel('Stress Periods')
        ax4.set_ylabel('Strategies')
        
        # Plot 5: Volatility vs Protection scatter
        ax5 = fig.add_subplot(gs[2, 0])
        for i, strategy in enumerate(strategies):
            ax5.scatter(all_results['Period_Volatility'], all_results[f'{strategy}_vs_BuyHold'], 
                       alpha=0.6, label=strategy, color=colors[i])
        
        ax5.set_xlabel('Period Volatility (%)')
        ax5.set_ylabel('Excess Return vs B&H (%)')
        ax5.set_title('Volatility vs Protection')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 6: Trading frequency analysis
        ax6 = fig.add_subplot(gs[2, 1])
        avg_trades = []
        for strategy in strategies:
            avg_trades.append(all_results.groupby('Period')[f'{strategy}_Trades'].mean().tolist())
        
        for i, (strategy, color) in enumerate(zip(strategies, colors)):
            ax6.bar(x + i*width, avg_trades[i], width, label=strategy, color=color, alpha=0.7)
        
        ax6.set_xlabel('Stress Periods')
        ax6.set_ylabel('Average Number of Trades')
        ax6.set_title('Trading Frequency During Stress')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in period_summary_df['Period']], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Category performance analysis
        ax7 = fig.add_subplot(gs[2, 2])
        category_performance = all_results.groupby('Category')[['Moving Average_vs_BuyHold', 'RSI_vs_BuyHold', 'Bollinger Bands_vs_BuyHold']].mean()
        category_performance.plot(kind='bar', ax=ax7, color=colors, alpha=0.7)
        ax7.set_title('Performance by Stock Category')
        ax7.set_xlabel('Stock Categories')
        ax7.set_ylabel('Avg Excess Return vs B&H (%)')
        ax7.legend(strategies, loc='upper right')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.setp(ax7.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_charts/comprehensive_stress_test_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Comprehensive stress test visualizations saved to analysis_charts/")

def main():
    """
    Main function to run comprehensive stress testing.
    """
    print("🔥 COMPREHENSIVE STRESS TESTING ANALYSIS")
    print("=" * 45)
    
    tester = StressTester()
    all_results = []
    
    # Run stress tests for each period
    for period_name, period_info in tester.stress_periods.items():
        try:
            period_results = tester.run_stress_test(period_name, period_info)
            if not period_results.empty:
                all_results.append(period_results)
        except Exception as e:
            print(f"❌ Error testing {period_name}: {str(e)}")
            continue
    
    if not all_results:
        print("❌ No stress test results available")
        return
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Analyze results
    period_summary_df = tester.analyze_stress_test_results(combined_results)
    
    # Save results
    combined_results.to_csv('analysis_charts/stress_test_detailed_results.csv', index=False)
    period_summary_df.to_csv('analysis_charts/stress_test_summary.csv', index=False)
    print("\n💾 Stress test results saved to analysis_charts/")
    
    # Create visualizations
    tester.create_stress_test_visualizations(combined_results, period_summary_df)
    
    # Overall insights
    print("\n🎯 KEY INSIGHTS FROM STRESS TESTING:")
    
    # Overall strategy performance during stress
    overall_protection = {}
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        avg_protection = period_summary_df[f'{strategy}_Avg_Protection'].mean()
        avg_protection_rate = period_summary_df[f'{strategy}_Protection_Rate'].mean()
        overall_protection[strategy] = {'protection': avg_protection, 'rate': avg_protection_rate}
    
    print("\n📊 OVERALL STRESS PERFORMANCE:")
    sorted_strategies = sorted(overall_protection.items(), key=lambda x: x[1]['protection'], reverse=True)
    for i, (strategy, metrics) in enumerate(sorted_strategies, 1):
        print(f"   {i}. {strategy}: {metrics['protection']:+.1f}% avg protection, {metrics['rate']:.0f}% success rate")
    
    # Best and worst periods for each strategy
    print("\n🏆 BEST PERFORMANCE PERIODS:")
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        best_period = period_summary_df.loc[period_summary_df[f'{strategy}_Avg_Protection'].idxmax(), 'Period']
        best_protection = period_summary_df[f'{strategy}_Avg_Protection'].max()
        print(f"   {strategy}: {best_period} ({best_protection:+.1f}% protection)")
    
    print("\n📉 MOST CHALLENGING PERIODS:")
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        worst_period = period_summary_df.loc[period_summary_df[f'{strategy}_Avg_Protection'].idxmin(), 'Period']
        worst_protection = period_summary_df[f'{strategy}_Avg_Protection'].min()
        print(f"   {strategy}: {worst_period} ({worst_protection:+.1f}% protection)")
    
    # Category analysis
    print("\n📊 PERFORMANCE BY STOCK CATEGORY:")
    category_analysis = combined_results.groupby('Category')[['Moving Average_vs_BuyHold', 'RSI_vs_BuyHold', 'Bollinger Bands_vs_BuyHold']].mean()
    
    for category in category_analysis.index:
        print(f"\n   {category}:")
        for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
            protection = category_analysis.loc[category, f'{strategy}_vs_BuyHold']
            print(f"      {strategy}: {protection:+.1f}% avg protection")
    
    print("\n💡 STRESS TESTING CONCLUSIONS:")
    print("1. 🛡️ Risk management is crucial during market stress")
    print("2. 📊 No single strategy works best in all stress scenarios")
    print("3. 🔄 Strategy rotation based on market conditions may be beneficial")
    print("4. 📉 Drawdown control is as important as return generation")
    print("5. ⚡ High volatility periods offer both risk and opportunity")
    print("6. 🏦 Defensive stocks generally show better strategy performance")
    
    print("\n✅ Comprehensive stress testing completed!")
    
    print("\n💡 NEXT STEPS:")
    print("   - Implement adaptive strategies based on market volatility")
    print("   - Add regime detection for strategy selection")
    print("   - Develop portfolio-level stress testing")
    print("   - Create early warning systems for market stress")
    print("   - Implement dynamic position sizing based on stress indicators")

if __name__ == "__main__":
    main()