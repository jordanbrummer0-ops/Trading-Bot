import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_fetcher import DataFetcher
from src.trading_strategy import BaseStrategy
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    Philosophy: Prices tend to revert to their mean over time.
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    - Exit positions when price returns to middle band (mean)
    """
    
    def __init__(self, period=20, std_dev=2.0, rsi_period=14):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            period (int): Period for moving average and standard deviation
            std_dev (float): Number of standard deviations for bands
            rsi_period (int): Period for RSI confirmation
        """
        super().__init__("Bollinger Bands Mean Reversion")
        self.period = period
        self.std_dev = std_dev
        self.rsi_period = rsi_period
    
    def calculate_bollinger_bands(self, data):
        """
        Calculate Bollinger Bands.
        """
        # Middle band (Simple Moving Average)
        data['BB_Middle'] = data['Close'].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        data['BB_Std'] = data['Close'].rolling(window=self.period).std()
        
        # Upper and lower bands
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * self.std_dev)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * self.std_dev)
        
        # Calculate band width (volatility measure)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        
        # Calculate %B (position within bands)
        data['BB_PercentB'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
    
    def calculate_rsi(self, data):
        """
        Calculate RSI for confirmation.
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data
    
    def generate_signals(self, data):
        """
        Generate mean reversion signals based on Bollinger Bands.
        """
        # Calculate indicators
        data = self.calculate_bollinger_bands(data)
        data = self.calculate_rsi(data)
        
        # Initialize signals
        data['signal'] = 0
        data['position'] = 0
        
        # Track current position
        position = 0
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            bb_upper = data['BB_Upper'].iloc[i]
            bb_lower = data['BB_Lower'].iloc[i]
            bb_middle = data['BB_Middle'].iloc[i]
            rsi = data['RSI'].iloc[i]
            percent_b = data['BB_PercentB'].iloc[i]
            
            # Skip if we don't have enough data
            if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(rsi):
                data['position'].iloc[i] = position
                continue
            
            # Mean reversion signals
            if position == 0:  # No position
                # Buy signal: Price touches lower band + RSI oversold
                if (current_price <= bb_lower * 1.01 and rsi < 35 and percent_b < 0.1):
                    data['signal'].iloc[i] = 1
                    position = 1
                # Short signal: Price touches upper band + RSI overbought
                elif (current_price >= bb_upper * 0.99 and rsi > 65 and percent_b > 0.9):
                    data['signal'].iloc[i] = -1
                    position = -1
            
            elif position == 1:  # Long position
                # Exit long: Price returns to middle band or touches upper band
                if (current_price >= bb_middle or current_price >= bb_upper * 0.99 or rsi > 70):
                    data['signal'].iloc[i] = 0
                    position = 0
            
            elif position == -1:  # Short position
                # Exit short: Price returns to middle band or touches lower band
                if (current_price <= bb_middle or current_price <= bb_lower * 1.01 or rsi < 30):
                    data['signal'].iloc[i] = 0
                    position = 0
            
            data['position'].iloc[i] = position
        
        return data

class EnhancedBollingerBandsStrategy(BollingerBandsStrategy):
    """
    Enhanced Bollinger Bands strategy with additional filters.
    """
    
    def __init__(self, period=20, std_dev=2.0, rsi_period=14, volume_factor=1.2):
        super().__init__(period, std_dev, rsi_period)
        self.name = "Enhanced Bollinger Bands"
        self.volume_factor = volume_factor
    
    def generate_signals(self, data):
        """
        Generate enhanced signals with volume and volatility filters.
        """
        # Get basic signals
        data = super().generate_signals(data)
        
        # Add volume filter
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Add volatility squeeze detection
        data['BB_Squeeze'] = data['BB_Width'] < data['BB_Width'].rolling(window=20).quantile(0.2)
        
        # Enhanced signal filtering
        enhanced_signals = data['signal'].copy()
        
        for i in range(len(data)):
            if data['signal'].iloc[i] != 0:  # If there's a signal
                # Volume confirmation: Require above-average volume
                volume_ok = data['Volume_Ratio'].iloc[i] > self.volume_factor
                
                # Avoid trading during volatility squeeze
                not_squeezed = not data['BB_Squeeze'].iloc[i]
                
                # Only keep signal if conditions are met
                if not (volume_ok and not_squeezed):
                    enhanced_signals.iloc[i] = 0
        
        data['enhanced_signal'] = enhanced_signals
        return data

def compare_mean_reversion_vs_trend_following(symbols=['MSFT', 'TSLA', 'AAPL'], period='2y'):
    """
    Compare mean reversion (Bollinger Bands) vs trend following (Moving Average) strategies.
    """
    from src.trading_strategy import MovingAverageStrategy
    
    fetcher = DataFetcher()
    results = []
    
    print("🔄 MEAN REVERSION vs TREND FOLLOWING COMPARISON")
    print("=" * 55)
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    if period == '2y':
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    elif period == '1y':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        # Fetch data
        data = fetcher.get_stock_data(symbol, start_date, end_date)
        if data is None or len(data) < 100:
            print(f"❌ Insufficient data for {symbol}")
            continue
        
        # Trend Following Strategy (Moving Average)
        ma_strategy = MovingAverageStrategy()
        data_ma = data.copy()
        data_ma = ma_strategy.generate_signals(data_ma)
        ma_returns = data_ma['Close'].pct_change() * data_ma['signal'].shift(1).fillna(0)
        
        # Mean Reversion Strategy (Bollinger Bands)
        bb_strategy = BollingerBandsStrategy()
        data_bb = data.copy()
        data_bb = bb_strategy.generate_signals(data_bb)
        bb_returns = data_bb['Close'].pct_change() * data_bb['signal'].shift(1).fillna(0)
        
        # Enhanced Bollinger Bands
        ebb_strategy = EnhancedBollingerBandsStrategy()
        data_ebb = data.copy()
        data_ebb = ebb_strategy.generate_signals(data_ebb)
        ebb_returns = data_ebb['Close'].pct_change() * data_ebb['enhanced_signal'].shift(1).fillna(0)
        
        # Calculate performance metrics
        def calculate_metrics(returns, name):
            total_return = (1 + returns).prod() - 1
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
            win_rate = (returns > 0).sum() / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
            num_trades = len(returns[returns != 0])
            
            return {
                'strategy': name,
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown) * 100,
                'win_rate': win_rate * 100,
                'num_trades': num_trades
            }
        
        ma_metrics = calculate_metrics(ma_returns, 'Trend Following (MA)')
        bb_metrics = calculate_metrics(bb_returns, 'Mean Reversion (BB)')
        ebb_metrics = calculate_metrics(ebb_returns, 'Enhanced Mean Reversion')
        
        # Store results
        for metrics in [ma_metrics, bb_metrics, ebb_metrics]:
            result = {'Symbol': symbol}
            result.update(metrics)
            results.append(result)
        
        print(f"   Trend Following: {ma_metrics['total_return']:.1f}% return, {ma_metrics['sharpe_ratio']:.2f} Sharpe")
        print(f"   Mean Reversion: {bb_metrics['total_return']:.1f}% return, {bb_metrics['sharpe_ratio']:.2f} Sharpe")
        print(f"   Enhanced Mean Rev: {ebb_metrics['total_return']:.1f}% return, {ebb_metrics['sharpe_ratio']:.2f} Sharpe")
    
    return pd.DataFrame(results)

def create_bollinger_bands_visualization(symbol='TSLA', period='1y'):
    """
    Create detailed visualization of Bollinger Bands strategy.
    """
    fetcher = DataFetcher()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    if period == '1y':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    elif period == '2y':
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = fetcher.get_stock_data(symbol, start_date, end_date)
    
    if data is None:
        print(f"❌ Could not fetch data for {symbol}")
        return
    
    # Generate signals
    bb_strategy = BollingerBandsStrategy()
    data = bb_strategy.generate_signals(data)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(f'{symbol} - Bollinger Bands Mean Reversion Strategy', fontsize=16, fontweight='bold')
    
    # Plot 1: Price with Bollinger Bands and signals
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Price', linewidth=1.5, color='black')
    ax1.plot(data.index, data['BB_Upper'], label='Upper Band', color='red', alpha=0.7)
    ax1.plot(data.index, data['BB_Middle'], label='Middle Band (SMA)', color='blue', alpha=0.7)
    ax1.plot(data.index, data['BB_Lower'], label='Lower Band', color='green', alpha=0.7)
    ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
    
    # Plot signals
    buy_signals = data.index[data['signal'] == 1]
    sell_signals = data.index[data['signal'] == -1]
    exit_signals = data.index[data['signal'] == 0]
    
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals, data.loc[buy_signals, 'Close'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals, data.loc[sell_signals, 'Close'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title('Price Action with Bollinger Bands')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: %B Oscillator
    ax2 = axes[1]
    ax2.plot(data.index, data['BB_PercentB'], label='%B', color='purple', linewidth=1)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Overbought (1.0)')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Oversold (0.0)')
    ax2.axhline(y=0.5, color='blue', linestyle='-', alpha=0.5, label='Middle (0.5)')
    ax2.fill_between(data.index, 0.8, 1.2, alpha=0.2, color='red')
    ax2.fill_between(data.index, -0.2, 0.2, alpha=0.2, color='green')
    ax2.set_title('%B Oscillator (Position within Bands)')
    ax2.set_ylabel('%B')
    ax2.set_ylim(-0.2, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI and Band Width
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    # RSI on left axis
    ax3.plot(data.index, data['RSI'], label='RSI', color='orange', linewidth=1)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    ax3.fill_between(data.index, 70, 100, alpha=0.2, color='red')
    ax3.fill_between(data.index, 0, 30, alpha=0.2, color='green')
    ax3.set_ylabel('RSI', color='orange')
    ax3.set_ylim(0, 100)
    
    # Band Width on right axis
    ax3_twin.plot(data.index, data['BB_Width'], label='Band Width', color='brown', linewidth=1)
    ax3_twin.set_ylabel('Band Width', color='brown')
    
    ax3.set_title('RSI and Bollinger Band Width')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns
    ax4 = axes[3]
    returns = data['Close'].pct_change() * data['signal'].shift(1).fillna(0)
    buy_hold_returns = data['Close'].pct_change()
    
    strategy_cumulative = (1 + returns).cumprod()
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    
    ax4.plot(data.index, strategy_cumulative, label='Bollinger Bands Strategy', linewidth=2)
    ax4.plot(data.index, buy_hold_cumulative, label='Buy & Hold', linewidth=2, alpha=0.7)
    ax4.set_title('Cumulative Returns Comparison')
    ax4.set_ylabel('Cumulative Return')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_charts/{symbol}_bollinger_bands_strategy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Bollinger Bands strategy chart saved for {symbol}")

def analyze_market_conditions_suitability():
    """
    Analyze which market conditions favor mean reversion vs trend following.
    """
    print("\n🎯 MARKET CONDITIONS ANALYSIS")
    print("=" * 35)
    
    conditions = {
        'Trending Markets': {
            'description': 'Strong directional moves, low volatility',
            'best_strategy': 'Trend Following (Moving Average)',
            'characteristics': ['Clear direction', 'Low noise', 'Momentum persistence'],
            'examples': 'Bull markets, sector rotations'
        },
        'Range-Bound Markets': {
            'description': 'Sideways movement, price oscillation',
            'best_strategy': 'Mean Reversion (Bollinger Bands)',
            'characteristics': ['Price oscillation', 'Support/resistance levels', 'Mean reversion'],
            'examples': 'Consolidation periods, mature markets'
        },
        'High Volatility Markets': {
            'description': 'Large price swings, uncertainty',
            'best_strategy': 'Enhanced Mean Reversion with filters',
            'characteristics': ['High volatility', 'False breakouts', 'Whipsaws'],
            'examples': 'Market crashes, earnings seasons'
        },
        'Low Volatility Markets': {
            'description': 'Stable, predictable movements',
            'best_strategy': 'Either strategy can work',
            'characteristics': ['Low volatility', 'Predictable patterns', 'Lower returns'],
            'examples': 'Stable economic periods, blue-chip stocks'
        }
    }
    
    for condition, details in conditions.items():
        print(f"\n📊 {condition}:")
        print(f"   Description: {details['description']}")
        print(f"   Best Strategy: {details['best_strategy']}")
        print(f"   Characteristics: {', '.join(details['characteristics'])}")
        print(f"   Examples: {details['examples']}")

def main():
    """
    Main function to run Bollinger Bands mean reversion analysis.
    """
    print("🔄 BOLLINGER BANDS MEAN REVERSION STRATEGY")
    print("=" * 45)
    
    # Compare strategies
    results_df = compare_mean_reversion_vs_trend_following(['MSFT', 'TSLA', 'AAPL', 'GOOGL', 'AMZN'])
    
    print("\n📊 STRATEGY COMPARISON RESULTS:")
    print(results_df.round(2))
    
    # Save results
    results_df.to_csv('analysis_charts/mean_reversion_vs_trend_following.csv', index=False)
    print("\n💾 Results saved to: analysis_charts/mean_reversion_vs_trend_following.csv")
    
    # Create detailed visualization for TSLA (volatile stock)
    print("\n📊 Creating detailed visualization for TSLA...")
    create_bollinger_bands_visualization('TSLA', '1y')
    
    # Analyze market conditions
    analyze_market_conditions_suitability()
    
    # Calculate strategy performance by type
    trend_following = results_df[results_df['strategy'] == 'Trend Following (MA)']
    mean_reversion = results_df[results_df['strategy'] == 'Mean Reversion (BB)']
    enhanced_mr = results_df[results_df['strategy'] == 'Enhanced Mean Reversion']
    
    print("\n🎯 STRATEGY PERFORMANCE SUMMARY:")
    print(f"📈 Trend Following Average Return: {trend_following['total_return'].mean():.1f}%")
    print(f"🔄 Mean Reversion Average Return: {mean_reversion['total_return'].mean():.1f}%")
    print(f"✨ Enhanced Mean Reversion Average Return: {enhanced_mr['total_return'].mean():.1f}%")
    
    print(f"\n📊 Sharpe Ratio Comparison:")
    print(f"📈 Trend Following: {trend_following['sharpe_ratio'].mean():.2f}")
    print(f"🔄 Mean Reversion: {mean_reversion['sharpe_ratio'].mean():.2f}")
    print(f"✨ Enhanced Mean Reversion: {enhanced_mr['sharpe_ratio'].mean():.2f}")
    
    print("\n🎯 KEY INSIGHTS FROM MEAN REVERSION ANALYSIS:")
    print("1. 🔄 Mean reversion works best in range-bound, volatile markets")
    print("2. 📈 Trend following excels in strong directional moves")
    print("3. 🎯 Bollinger Bands help identify overbought/oversold conditions")
    print("4. 📊 %B oscillator provides clear entry/exit signals")
    print("5. 🛡️  Volume and volatility filters improve signal quality")
    print("6. ⚖️  Different strategies suit different market regimes")
    print("7. 🔄 Mean reversion often has higher win rates but smaller average wins")
    
    print("\n✅ Bollinger Bands mean reversion analysis completed!")
    
    print("\n💡 NEXT STEPS:")
    print("   - Test different Bollinger Band parameters (period, std dev)")
    print("   - Implement regime detection to switch between strategies")
    print("   - Add position sizing based on volatility")
    print("   - Test on different asset classes and timeframes")

if __name__ == "__main__":
    main()