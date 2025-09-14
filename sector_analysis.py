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

class SectorAnalyzer:
    """
    Comprehensive sector analysis for trading strategies.
    Tests strategies across different market sectors to understand performance variations.
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.sectors = {
            'Technology': {
                'symbols': ['MSFT', 'AAPL', 'GOOGL', 'NVDA', 'META'],
                'characteristics': ['High growth', 'High volatility', 'Innovation-driven'],
                'description': 'Technology companies with high growth potential'
            },
            'Financial': {
                'symbols': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                'characteristics': ['Interest rate sensitive', 'Economic cycle dependent', 'Dividend paying'],
                'description': 'Banks and financial services companies'
            },
            'Consumer Staples': {
                'symbols': ['KO', 'PG', 'WMT', 'PEP', 'JNJ'],
                'characteristics': ['Defensive', 'Stable earnings', 'Low volatility'],
                'description': 'Essential consumer goods and services'
            },
            'Energy': {
                'symbols': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
                'characteristics': ['Commodity dependent', 'Cyclical', 'High volatility'],
                'description': 'Oil, gas, and energy companies'
            },
            'Healthcare': {
                'symbols': ['UNH', 'JNJ', 'PFE', 'ABT', 'MRK'],
                'characteristics': ['Defensive', 'Regulatory dependent', 'Innovation-driven'],
                'description': 'Healthcare and pharmaceutical companies'
            },
            'Industrial': {
                'symbols': ['BA', 'CAT', 'GE', 'MMM', 'HON'],
                'characteristics': ['Economic sensitive', 'Capital intensive', 'Global exposure'],
                'description': 'Manufacturing and industrial companies'
            }
        }
        
        self.strategies = {
            'Moving Average': MovingAverageStrategy(),
            'RSI': RSIStrategy(),
            'Bollinger Bands': BollingerBandsStrategy()
        }
    
    def analyze_sector_performance(self, period='2y'):
        """
        Analyze strategy performance across different sectors.
        """
        print("🏭 SECTOR ANALYSIS - STRATEGY PERFORMANCE")
        print("=" * 45)
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        if period == '2y':
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        elif period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        results = []
        sector_summaries = {}
        
        for sector_name, sector_info in self.sectors.items():
            print(f"\n📊 Analyzing {sector_name} Sector...")
            print(f"   Description: {sector_info['description']}")
            print(f"   Characteristics: {', '.join(sector_info['characteristics'])}")
            
            sector_results = []
            
            for symbol in sector_info['symbols']:
                print(f"   📈 Testing {symbol}...", end=" ")
                
                # Fetch data
                try:
                    data = self.fetcher.get_stock_data(symbol, start_date, end_date)
                    if data is None or len(data) < 100:
                        print("❌ Insufficient data")
                        continue
                    
                    # Test each strategy
                    symbol_results = {'Symbol': symbol, 'Sector': sector_name}
                    
                    for strategy_name, strategy in self.strategies.items():
                        try:
                            # Generate signals
                            data_copy = data.copy()
                            if strategy_name == 'Bollinger Bands':
                                data_copy = strategy.generate_signals(data_copy)
                                returns = data_copy['Close'].pct_change() * data_copy['signal'].shift(1).fillna(0)
                            else:
                                data_copy = strategy.generate_signals(data_copy)
                                returns = data_copy['Close'].pct_change() * data_copy['signal'].shift(1).fillna(0)
                            
                            # Calculate metrics
                            total_return = (1 + returns).prod() - 1
                            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                            max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
                            win_rate = (returns > 0).sum() / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
                            
                            symbol_results[f'{strategy_name}_Return'] = total_return * 100
                            symbol_results[f'{strategy_name}_Sharpe'] = sharpe_ratio
                            symbol_results[f'{strategy_name}_Drawdown'] = abs(max_drawdown) * 100
                            symbol_results[f'{strategy_name}_WinRate'] = win_rate * 100
                            
                        except Exception as e:
                            print(f"Error with {strategy_name}: {str(e)[:50]}...")
                            symbol_results[f'{strategy_name}_Return'] = 0
                            symbol_results[f'{strategy_name}_Sharpe'] = 0
                            symbol_results[f'{strategy_name}_Drawdown'] = 0
                            symbol_results[f'{strategy_name}_WinRate'] = 0
                    
                    sector_results.append(symbol_results)
                    results.append(symbol_results)
                    print("✅")
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
                    continue
            
            # Calculate sector summary
            if sector_results:
                sector_df = pd.DataFrame(sector_results)
                sector_summary = {
                    'Sector': sector_name,
                    'Symbols_Tested': len(sector_results),
                    'Characteristics': ', '.join(sector_info['characteristics'])
                }
                
                for strategy_name in self.strategies.keys():
                    avg_return = sector_df[f'{strategy_name}_Return'].mean()
                    avg_sharpe = sector_df[f'{strategy_name}_Sharpe'].mean()
                    sector_summary[f'{strategy_name}_Avg_Return'] = avg_return
                    sector_summary[f'{strategy_name}_Avg_Sharpe'] = avg_sharpe
                
                sector_summaries[sector_name] = sector_summary
                
                print(f"   📊 Sector Summary:")
                for strategy_name in self.strategies.keys():
                    print(f"      {strategy_name}: {sector_summary[f'{strategy_name}_Avg_Return']:.1f}% return, {sector_summary[f'{strategy_name}_Avg_Sharpe']:.2f} Sharpe")
        
        return pd.DataFrame(results), pd.DataFrame(list(sector_summaries.values()))
    
    def create_sector_heatmap(self, sector_summary_df):
        """
        Create heatmap visualization of strategy performance by sector.
        """
        print("\n📊 Creating sector performance heatmaps...")
        
        # Prepare data for heatmaps
        strategies = ['Moving Average', 'RSI', 'Bollinger Bands']
        sectors = sector_summary_df['Sector'].tolist()
        
        # Create return heatmap data
        return_data = []
        sharpe_data = []
        
        for strategy in strategies:
            return_row = []
            sharpe_row = []
            for _, row in sector_summary_df.iterrows():
                return_row.append(row[f'{strategy}_Avg_Return'])
                sharpe_row.append(row[f'{strategy}_Avg_Sharpe'])
            return_data.append(return_row)
            sharpe_data.append(sharpe_row)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Strategy Performance by Sector', fontsize=16, fontweight='bold')
        
        # Return heatmap
        return_df = pd.DataFrame(return_data, index=strategies, columns=sectors)
        sns.heatmap(return_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                   ax=ax1, cbar_kws={'label': 'Average Return (%)'})
        ax1.set_title('Average Returns by Sector')
        ax1.set_xlabel('Sectors')
        ax1.set_ylabel('Strategies')
        
        # Sharpe ratio heatmap
        sharpe_df = pd.DataFrame(sharpe_data, index=strategies, columns=sectors)
        sns.heatmap(sharpe_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   ax=ax2, cbar_kws={'label': 'Average Sharpe Ratio'})
        ax2.set_title('Average Sharpe Ratios by Sector')
        ax2.set_xlabel('Sectors')
        ax2.set_ylabel('Strategies')
        
        plt.tight_layout()
        plt.savefig('analysis_charts/sector_performance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Sector heatmaps saved to analysis_charts/")
    
    def analyze_sector_characteristics(self, results_df):
        """
        Analyze how sector characteristics affect strategy performance.
        """
        print("\n🔍 SECTOR CHARACTERISTICS ANALYSIS")
        print("=" * 40)
        
        # Group by sector and calculate statistics
        sector_stats = {}
        
        for sector in results_df['Sector'].unique():
            sector_data = results_df[results_df['Sector'] == sector]
            
            if len(sector_data) == 0:
                continue
            
            stats = {
                'Count': len(sector_data),
                'Volatility': sector_data[['Moving Average_Return', 'RSI_Return', 'Bollinger Bands_Return']].std().mean(),
                'Best_Strategy': '',
                'Best_Return': 0
            }
            
            # Find best performing strategy for this sector
            avg_returns = {
                'Moving Average': sector_data['Moving Average_Return'].mean(),
                'RSI': sector_data['RSI_Return'].mean(),
                'Bollinger Bands': sector_data['Bollinger Bands_Return'].mean()
            }
            
            best_strategy = max(avg_returns, key=avg_returns.get)
            stats['Best_Strategy'] = best_strategy
            stats['Best_Return'] = avg_returns[best_strategy]
            
            sector_stats[sector] = stats
        
        # Print analysis
        for sector, stats in sector_stats.items():
            sector_info = self.sectors.get(sector, {})
            characteristics = sector_info.get('characteristics', [])
            
            print(f"\n📊 {sector}:")
            print(f"   Characteristics: {', '.join(characteristics)}")
            print(f"   Symbols Analyzed: {stats['Count']}")
            print(f"   Return Volatility: {stats['Volatility']:.1f}%")
            print(f"   Best Strategy: {stats['Best_Strategy']} ({stats['Best_Return']:.1f}% avg return)")
            
            # Strategy recommendations based on characteristics
            if 'High volatility' in characteristics:
                print(f"   💡 Recommendation: Consider Bollinger Bands for mean reversion")
            elif 'Defensive' in characteristics:
                print(f"   💡 Recommendation: Moving Average may work well for steady trends")
            elif 'Cyclical' in characteristics:
                print(f"   💡 Recommendation: RSI may help time cyclical turns")
    
    def create_sector_comparison_chart(self, results_df):
        """
        Create detailed comparison charts for sector performance.
        """
        print("\n📊 Creating sector comparison charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sector Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        strategies = ['Moving Average', 'RSI', 'Bollinger Bands']
        
        # Plot 1: Average returns by sector
        ax1 = axes[0, 0]
        sector_returns = results_df.groupby('Sector')[['Moving Average_Return', 'RSI_Return', 'Bollinger Bands_Return']].mean()
        sector_returns.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Average Returns by Sector')
        ax1.set_ylabel('Return (%)')
        ax1.set_xlabel('Sectors')
        ax1.legend(strategies, loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Average Sharpe ratios by sector
        ax2 = axes[0, 1]
        sector_sharpe = results_df.groupby('Sector')[['Moving Average_Sharpe', 'RSI_Sharpe', 'Bollinger Bands_Sharpe']].mean()
        sector_sharpe.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Average Sharpe Ratios by Sector')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xlabel('Sectors')
        ax2.legend(strategies, loc='upper right')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Win rates by sector
        ax3 = axes[1, 0]
        sector_winrate = results_df.groupby('Sector')[['Moving Average_WinRate', 'RSI_WinRate', 'Bollinger Bands_WinRate']].mean()
        sector_winrate.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Average Win Rates by Sector')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_xlabel('Sectors')
        ax3.legend(strategies, loc='upper right')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Max drawdowns by sector
        ax4 = axes[1, 1]
        sector_drawdown = results_df.groupby('Sector')[['Moving Average_Drawdown', 'RSI_Drawdown', 'Bollinger Bands_Drawdown']].mean()
        sector_drawdown.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Average Max Drawdowns by Sector')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.set_xlabel('Sectors')
        ax4.legend(strategies, loc='upper right')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_charts/sector_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Sector comparison charts saved to analysis_charts/")

def main():
    """
    Main function to run comprehensive sector analysis.
    """
    print("🏭 COMPREHENSIVE SECTOR ANALYSIS")
    print("=" * 35)
    
    analyzer = SectorAnalyzer()
    
    # Run sector performance analysis
    results_df, sector_summary_df = analyzer.analyze_sector_performance('2y')
    
    if results_df.empty:
        print("❌ No data available for analysis")
        return
    
    print("\n📊 SECTOR SUMMARY RESULTS:")
    print(sector_summary_df.round(2))
    
    # Save detailed results
    results_df.to_csv('analysis_charts/sector_analysis_detailed.csv', index=False)
    sector_summary_df.to_csv('analysis_charts/sector_analysis_summary.csv', index=False)
    print("\n💾 Results saved to analysis_charts/")
    
    # Create visualizations
    analyzer.create_sector_heatmap(sector_summary_df)
    analyzer.create_sector_comparison_chart(results_df)
    
    # Analyze sector characteristics
    analyzer.analyze_sector_characteristics(results_df)
    
    # Overall insights
    print("\n🎯 KEY INSIGHTS FROM SECTOR ANALYSIS:")
    
    # Find best performing sector for each strategy
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        best_sector = sector_summary_df.loc[sector_summary_df[f'{strategy}_Avg_Return'].idxmax(), 'Sector']
        best_return = sector_summary_df[f'{strategy}_Avg_Return'].max()
        print(f"📈 {strategy} performs best in {best_sector} sector ({best_return:.1f}% avg return)")
    
    # Overall strategy ranking
    overall_performance = {}
    for strategy in ['Moving Average', 'RSI', 'Bollinger Bands']:
        avg_return = sector_summary_df[f'{strategy}_Avg_Return'].mean()
        avg_sharpe = sector_summary_df[f'{strategy}_Avg_Sharpe'].mean()
        overall_performance[strategy] = {'return': avg_return, 'sharpe': avg_sharpe}
    
    print("\n📊 OVERALL STRATEGY RANKINGS:")
    sorted_by_return = sorted(overall_performance.items(), key=lambda x: x[1]['return'], reverse=True)
    sorted_by_sharpe = sorted(overall_performance.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print("   By Average Return:")
    for i, (strategy, metrics) in enumerate(sorted_by_return, 1):
        print(f"      {i}. {strategy}: {metrics['return']:.1f}%")
    
    print("   By Average Sharpe Ratio:")
    for i, (strategy, metrics) in enumerate(sorted_by_sharpe, 1):
        print(f"      {i}. {strategy}: {metrics['sharpe']:.2f}")
    
    print("\n💡 SECTOR-SPECIFIC RECOMMENDATIONS:")
    print("1. 🏦 Financial sector: Interest rate sensitivity makes timing crucial")
    print("2. 🛒 Consumer Staples: Defensive nature suits steady trend-following")
    print("3. ⚡ Energy sector: High volatility benefits from mean reversion")
    print("4. 💻 Technology: Growth trends favor momentum strategies")
    print("5. 🏥 Healthcare: Regulatory events create mean reversion opportunities")
    print("6. 🏭 Industrial: Economic cycles require adaptive strategies")
    
    print("\n✅ Comprehensive sector analysis completed!")
    
    print("\n💡 NEXT STEPS:")
    print("   - Test strategies on sector ETFs for broader exposure")
    print("   - Implement sector rotation strategies")
    print("   - Add economic indicators for sector timing")
    print("   - Consider sector-specific parameter optimization")

if __name__ == "__main__":
    main()