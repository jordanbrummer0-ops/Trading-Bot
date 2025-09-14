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

class AssetClassAnalyzer:
    """
    Comprehensive asset class analysis for trading strategies.
    Tests strategies across different asset classes to understand performance variations.
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.asset_classes = {
            'Equities': {
                'symbols': ['SPY', 'QQQ', 'IWM', 'VTI', 'SCHB'],
                'characteristics': ['Market beta', 'Economic sensitive', 'Growth potential'],
                'description': 'Stock market ETFs and broad market exposure',
                'typical_volatility': 'Medium (15-25%)'
            },
            'Commodities': {
                'symbols': ['GLD', 'SLV', 'USO', 'DBA', 'PDBC'],
                'characteristics': ['Inflation hedge', 'Supply/demand driven', 'Cyclical'],
                'description': 'Precious metals, energy, and agricultural commodities',
                'typical_volatility': 'High (20-40%)'
            },
            'Cryptocurrency': {
                'symbols': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD'],
                'characteristics': ['High volatility', 'Speculative', '24/7 trading'],
                'description': 'Digital currencies and blockchain assets',
                'typical_volatility': 'Very High (40-100%)'
            },
            'Fixed Income': {
                'symbols': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
                'characteristics': ['Interest rate sensitive', 'Lower volatility', 'Income focused'],
                'description': 'Government and corporate bonds',
                'typical_volatility': 'Low (5-15%)'
            },
            'Real Estate': {
                'symbols': ['VNQ', 'IYR', 'SCHH', 'RWR', 'FREL'],
                'characteristics': ['Interest rate sensitive', 'Income producing', 'Inflation hedge'],
                'description': 'Real Estate Investment Trusts (REITs)',
                'typical_volatility': 'Medium (15-30%)'
            },
            'International': {
                'symbols': ['EFA', 'VEA', 'EEM', 'VWO', 'IEFA'],
                'characteristics': ['Currency exposure', 'Diversification', 'Regional economic factors'],
                'description': 'International developed and emerging markets',
                'typical_volatility': 'Medium-High (18-35%)'
            }
        }
        
        self.strategies = {
            'Moving Average': MovingAverageStrategy(),
            'RSI': RSIStrategy(),
            'Bollinger Bands': BollingerBandsStrategy()
        }
    
    def analyze_asset_class_performance(self, period='2y'):
        """
        Analyze strategy performance across different asset classes.
        """
        print("💰 ASSET CLASS ANALYSIS - STRATEGY PERFORMANCE")
        print("=" * 50)
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        if period == '2y':
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        elif period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        results = []
        asset_class_summaries = {}
        
        for asset_class_name, asset_info in self.asset_classes.items():
            print(f"\n📊 Analyzing {asset_class_name} Asset Class...")
            print(f"   Description: {asset_info['description']}")
            print(f"   Characteristics: {', '.join(asset_info['characteristics'])}")
            print(f"   Typical Volatility: {asset_info['typical_volatility']}")
            
            asset_results = []
            
            for symbol in asset_info['symbols']:
                print(f"   📈 Testing {symbol}...", end=" ")
                
                # Fetch data
                try:
                    data = self.fetcher.get_stock_data(symbol, start_date, end_date)
                    if data is None or len(data) < 100:
                        print("❌ Insufficient data")
                        continue
                    
                    # Calculate basic metrics for the asset
                    returns = data['Close'].pct_change().dropna()
                    asset_volatility = returns.std() * np.sqrt(252) * 100
                    asset_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    
                    # Test each strategy
                    symbol_results = {
                        'Symbol': symbol, 
                        'Asset_Class': asset_class_name,
                        'Asset_Volatility': asset_volatility,
                        'Buy_Hold_Return': asset_return
                    }
                    
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
                            
                            # Calculate metrics
                            total_return = (1 + strategy_returns).prod() - 1
                            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                            max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
                            win_rate = (strategy_returns > 0).sum() / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
                            
                            # Calculate number of trades
                            signals = data_copy['signal'].fillna(0)
                            trades = (signals != signals.shift(1)).sum()
                            
                            symbol_results[f'{strategy_name}_Return'] = total_return * 100
                            symbol_results[f'{strategy_name}_Sharpe'] = sharpe_ratio
                            symbol_results[f'{strategy_name}_Drawdown'] = abs(max_drawdown) * 100
                            symbol_results[f'{strategy_name}_WinRate'] = win_rate * 100
                            symbol_results[f'{strategy_name}_Trades'] = trades
                            symbol_results[f'{strategy_name}_vs_BuyHold'] = total_return * 100 - asset_return
                            
                        except Exception as e:
                            print(f"Error with {strategy_name}: {str(e)[:30]}...")
                            symbol_results[f'{strategy_name}_Return'] = 0
                            symbol_results[f'{strategy_name}_Sharpe'] = 0
                            symbol_results[f'{strategy_name}_Drawdown'] = 0
                            symbol_results[f'{strategy_name}_WinRate'] = 0
                            symbol_results[f'{strategy_name}_Trades'] = 0
                            symbol_results[f'{strategy_name}_vs_BuyHold'] = -asset_return
                    
                    asset_results.append(symbol_results)
                    results.append(symbol_results)
                    print("✅")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
                    continue
            
            # Calculate asset class summary
            if asset_results:
                asset_df = pd.DataFrame(asset_results)
                asset_summary = {
                    'Asset_Class': asset_class_name,
                    'Symbols_Tested': len(asset_results),
                    'Avg_Volatility': asset_df['Asset_Volatility'].mean(),
                    'Avg_BuyHold_Return': asset_df['Buy_Hold_Return'].mean(),
                    'Characteristics': ', '.join(asset_info['characteristics'])
                }
                
                for strategy_name in self.strategies.keys():
                    avg_return = asset_df[f'{strategy_name}_Return'].mean()
                    avg_sharpe = asset_df[f'{strategy_name}_Sharpe'].mean()
                    avg_vs_buyhold = asset_df[f'{strategy_name}_vs_BuyHold'].mean()
                    avg_trades = asset_df[f'{strategy_name}_Trades'].mean()
                    
                    asset_summary[f'{strategy_name}_Avg_Return'] = avg_return
                    asset_summary[f'{strategy_name}_Avg_Sharpe'] = avg_sharpe
                    asset_summary[f'{strategy_name}_vs_BuyHold'] = avg_vs_buyhold
                    asset_summary[f'{strategy_name}_Avg_Trades'] = avg_trades
                
                asset_class_summaries[asset_class_name] = asset_summary
                
                print(f"   📊 Asset Class Summary:")
                print(f"      Average Volatility: {asset_summary['Avg_Volatility']:.1f}%")
                print(f"      Buy & Hold Return: {asset_summary['Avg_BuyHold_Return']:.1f}%")
                for strategy_name in self.strategies.keys():
                    vs_buyhold = asset_summary[f'{strategy_name}_vs_BuyHold']
                    trades = asset_summary[f'{strategy_name}_Avg_Trades']
                    print(f"      {strategy_name}: {asset_summary[f'{strategy_name}_Avg_Return']:.1f}% return ({vs_buyhold:+.1f}% vs B&H, {trades:.0f} trades)")
        
        return pd.DataFrame(results), pd.DataFrame(list(asset_class_summaries.values()))
    
    def create_asset_class_heatmap(self, asset_summary_df):
        """
        Create heatmap visualization of strategy performance by asset class.
        """
        print("\n📊 Creating asset class performance heatmaps...")
        
        # Prepare data for heatmaps
        strategies = ['Moving Average', 'RSI', 'Bollinger Bands']
        asset_classes = asset_summary_df['Asset_Class'].tolist()
        
        # Create return heatmap data
        return_data = []
        vs_buyhold_data = []
        
        for strategy in strategies:
            return_row = []
            vs_buyhold_row = []
            for _, row in asset_summary_df.iterrows():
                return_row.append(row[f'{strategy}_Avg_Return'])
                vs_buyhold_row.append(row[f'{strategy}_vs_BuyHold'])
            return_data.append(return_row)
            vs_buyhold_data.append(vs_buyhold_row)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Strategy Performance by Asset Class', fontsize=16, fontweight='bold')
        
        # Return heatmap
        return_df = pd.DataFrame(return_data, index=strategies, columns=asset_classes)
        sns.heatmap(return_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                   ax=ax1, cbar_kws={'label': 'Average Return (%)'})
        ax1.set_title('Average Returns by Asset Class')
        ax1.set_xlabel('Asset Classes')
        ax1.set_ylabel('Strategies')
        
        # vs Buy & Hold heatmap
        vs_buyhold_df = pd.DataFrame(vs_buyhold_data, index=strategies, columns=asset_classes)
        sns.heatmap(vs_buyhold_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=ax2, cbar_kws={'label': 'Excess Return vs Buy & Hold (%)'})
        ax2.set_title('Strategy Returns vs Buy & Hold')
        ax2.set_xlabel('Asset Classes')
        ax2.set_ylabel('Strategies')
        
        plt.tight_layout()
        plt.savefig('analysis_charts/asset_class_performance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Asset class heatmaps saved to analysis_charts/")
    
    def analyze_volatility_impact(self, results_df):
        """
        Analyze how asset volatility affects strategy performance.
        """
        print("\n📊 VOLATILITY IMPACT ANALYSIS")
        print("=" * 35)
        
        # Create volatility buckets
        results_df['Volatility_Bucket'] = pd.cut(results_df['Asset_Volatility'], 
                                                bins=[0, 15, 25, 40, 100], 
                                                labels=['Low (<15%)', 'Medium (15-25%)', 'High (25-40%)', 'Very High (>40%)'])
        
        volatility_analysis = results_df.groupby('Volatility_Bucket').agg({
            'Moving Average_Return': 'mean',
            'RSI_Return': 'mean',
            'Bollinger Bands_Return': 'mean',
            'Moving Average_Sharpe': 'mean',
            'RSI_Sharpe': 'mean',
            'Bollinger Bands_Sharpe': 'mean',
            'Asset_Volatility': 'mean',
            'Symbol': 'count'
        }).round(2)
        
        print("\n📊 Performance by Volatility Level:")
        print(volatility_analysis)
        
        # Find best strategy for each volatility level
        print("\n🎯 Best Strategy by Volatility Level:")
        for bucket in volatility_analysis.index:
            if pd.isna(bucket):
                continue
            returns = {
                'Moving Average': volatility_analysis.loc[bucket, 'Moving Average_Return'],
                'RSI': volatility_analysis.loc[bucket, 'RSI_Return'],
                'Bollinger Bands': volatility_analysis.loc[bucket, 'Bollinger Bands_Return']
            }
            best_strategy = max(returns, key=returns.get)
            best_return = returns[best_strategy]
            avg_vol = volatility_analysis.loc[bucket, 'Asset_Volatility']
            count = volatility_analysis.loc[bucket, 'Symbol']
            
            print(f"   {bucket} (avg {avg_vol:.1f}% vol, {count} assets): {best_strategy} ({best_return:.1f}% return)")
    
    def create_volatility_performance_chart(self, results_df):
        """
        Create scatter plot showing relationship between volatility and strategy performance.
        """
        print("\n📊 Creating volatility vs performance charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Asset Volatility vs Strategy Performance', fontsize=16, fontweight='bold')
        
        strategies = ['Moving Average', 'RSI', 'Bollinger Bands']
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Volatility vs Returns
        ax1 = axes[0, 0]
        for i, strategy in enumerate(strategies):
            ax1.scatter(results_df['Asset_Volatility'], results_df[f'{strategy}_Return'], 
                       alpha=0.6, label=strategy, color=colors[i])
        ax1.set_xlabel('Asset Volatility (%)')
        ax1.set_ylabel('Strategy Return (%)')
        ax1.set_title('Volatility vs Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volatility vs Sharpe Ratio
        ax2 = axes[0, 1]
        for i, strategy in enumerate(strategies):
            ax2.scatter(results_df['Asset_Volatility'], results_df[f'{strategy}_Sharpe'], 
                       alpha=0.6, label=strategy, color=colors[i])
        ax2.set_xlabel('Asset Volatility (%)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Volatility vs Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volatility vs Number of Trades
        ax3 = axes[1, 0]
        for i, strategy in enumerate(strategies):
            ax3.scatter(results_df['Asset_Volatility'], results_df[f'{strategy}_Trades'], 
                       alpha=0.6, label=strategy, color=colors[i])
        ax3.set_xlabel('Asset Volatility (%)')
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Volatility vs Trading Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Asset Class Performance Comparison
        ax4 = axes[1, 1]
        asset_class_performance = results_df.groupby('Asset_Class')[['Moving Average_Return', 'RSI_Return', 'Bollinger Bands_Return']].mean()
        asset_class_performance.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Average Returns by Asset Class')
        ax4.set_ylabel('Return (%)')
        ax4.set_xlabel('Asset Classes')
        ax4.legend(strategies, loc='upper right')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_charts/volatility_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Volatility analysis charts saved to analysis_charts/")

def main():
    """
    Main function to run comprehensive asset class analysis.
    """
    print("💰 COMPREHENSIVE ASSET CLASS ANALYSIS")
    print("=" * 40)
    
    analyzer = AssetClassAnalyzer()
    
    # Run asset class performance analysis
    results_df, asset_summary_df = analyzer.analyze_asset_class_performance('2y')
    
    if results_df.empty:
        print("❌ No data available for analysis")
        return
    
    print("\n📊 ASSET CLASS SUMMARY RESULTS:")
    print(asset_summary_df.round(2))
    
    # Save detailed results
    results_df.to_csv('analysis_charts/asset_class_analysis_detailed.csv', index=False)
    asset_summary_df.to_csv('analysis_charts/asset_class_analysis_summary.csv', index=False)
    print("\n💾 Results saved to analysis_charts/")
    
    # Create visualizations
    analyzer.create_asset_class_heatmap(asset_summary_df)
    analyzer.analyze_volatility_impact(results_df)
    analyzer.create_volatility_performance_chart(results_df)
    
    # Overall insights
    print("\n🎯 KEY INSIGHTS FROM ASSET CLASS ANALYSIS:")
    
    # Find best performing asset class for each strategy
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        best_asset_class = asset_summary_df.loc[asset_summary_df[f'{strategy}_Avg_Return'].idxmax(), 'Asset_Class']
        best_return = asset_summary_df[f'{strategy}_Avg_Return'].max()
        print(f"📈 {strategy} performs best with {best_asset_class} ({best_return:.1f}% avg return)")
    
    # Strategy vs Buy & Hold analysis
    print("\n📊 STRATEGY vs BUY & HOLD COMPARISON:")
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        avg_excess = asset_summary_df[f'{strategy}_vs_BuyHold'].mean()
        positive_count = (asset_summary_df[f'{strategy}_vs_BuyHold'] > 0).sum()
        total_count = len(asset_summary_df)
        print(f"   {strategy}: {avg_excess:+.1f}% avg excess return ({positive_count}/{total_count} asset classes outperformed)")
    
    # Overall strategy ranking
    overall_performance = {}
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        avg_return = asset_summary_df[f'{strategy}_Avg_Return'].mean()
        avg_sharpe = asset_summary_df[f'{strategy}_Avg_Sharpe'].mean()
        avg_excess = asset_summary_df[f'{strategy}_vs_BuyHold'].mean()
        overall_performance[strategy] = {'return': avg_return, 'sharpe': avg_sharpe, 'excess': avg_excess}
    
    print("\n📊 OVERALL STRATEGY RANKINGS:")
    sorted_by_return = sorted(overall_performance.items(), key=lambda x: x[1]['return'], reverse=True)
    sorted_by_excess = sorted(overall_performance.items(), key=lambda x: x[1]['excess'], reverse=True)
    
    print("   By Average Return:")
    for i, (strategy, metrics) in enumerate(sorted_by_return, 1):
        print(f"      {i}. {strategy}: {metrics['return']:.1f}%")
    
    print("   By Excess Return vs Buy & Hold:")
    for i, (strategy, metrics) in enumerate(sorted_by_excess, 1):
        print(f"      {i}. {strategy}: {metrics['excess']:+.1f}%")
    
    print("\n💡 ASSET CLASS-SPECIFIC RECOMMENDATIONS:")
    print("1. 📈 Equities: Trend-following works well in bull markets")
    print("2. 🥇 Commodities: Mean reversion benefits from high volatility")
    print("3. 💎 Cryptocurrency: High volatility requires careful risk management")
    print("4. 🏛️ Fixed Income: Low volatility limits strategy effectiveness")
    print("5. 🏠 Real Estate: Interest rate sensitivity affects timing")
    print("6. 🌍 International: Currency effects add complexity")
    
    print("\n✅ Comprehensive asset class analysis completed!")
    
    print("\n💡 NEXT STEPS:")
    print("   - Implement asset class rotation strategies")
    print("   - Add volatility-adjusted position sizing")
    print("   - Test strategies during different market regimes")
    print("   - Consider correlation-based portfolio construction")
    print("   - Implement dynamic strategy selection based on asset characteristics")

if __name__ == "__main__":
    main()