import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_fetcher import DataFetcher
from src.trading_strategy import MovingAverageStrategy, RSIStrategy
import warnings
warnings.filterwarnings('ignore')

class ConfirmationIndicators:
    """
    Advanced confirmation indicators to reduce false signals in trading strategies.
    Implements trend filters and volatility filters for better signal quality.
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
    
    def calculate_trend_filter(self, data, period=200):
        """
        Calculate long-term trend filter using Simple Moving Average.
        Only allow buy signals when price is above the trend filter.
        """
        return data['Close'].rolling(window=period).mean()
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index for volatility filtering.
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range for volatility measurement.
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def apply_confirmation_filters(self, signals, data, rsi_oversold=30, rsi_overbought=70, 
                                 atr_threshold_percentile=75):
        """
        Apply confirmation filters to trading signals.
        
        Parameters:
        - signals: Original buy/sell signals
        - data: Price data with indicators
        - rsi_oversold: RSI level below which we consider oversold (allow buys)
        - rsi_overbought: RSI level above which we consider overbought (allow sells)
        - atr_threshold_percentile: ATR percentile threshold for volatility filter
        """
        # Calculate indicators
        trend_filter = self.calculate_trend_filter(data)
        rsi = self.calculate_rsi(data)
        atr = self.calculate_atr(data)
        
        # Calculate ATR threshold (dynamic based on historical volatility)
        atr_threshold = atr.quantile(atr_threshold_percentile / 100)
        
        # Create confirmed signals
        confirmed_signals = signals.copy()
        
        # Apply trend filter: Only buy when price is above 200-day SMA
        trend_condition = data['Close'] > trend_filter
        
        # Apply RSI filter: Avoid buying when overbought, avoid selling when oversold
        rsi_buy_condition = rsi < rsi_overbought
        rsi_sell_condition = rsi > rsi_oversold
        
        # Apply ATR filter: Only trade when volatility is reasonable
        volatility_condition = atr < atr_threshold
        
        # Combine all conditions
        for i in range(len(confirmed_signals)):
            if confirmed_signals.iloc[i] == 1:  # Buy signal
                if not (trend_condition.iloc[i] and rsi_buy_condition.iloc[i] and volatility_condition.iloc[i]):
                    confirmed_signals.iloc[i] = 0
            elif confirmed_signals.iloc[i] == -1:  # Sell signal
                if not (rsi_sell_condition.iloc[i] and volatility_condition.iloc[i]):
                    confirmed_signals.iloc[i] = 0
        
        return confirmed_signals, {
            'trend_filter': trend_filter,
            'rsi': rsi,
            'atr': atr,
            'atr_threshold': atr_threshold
        }

class EnhancedMovingAverageStrategy:
    """
    Enhanced Moving Average strategy with confirmation indicators.
    """
    
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        self.confirmation = ConfirmationIndicators()
    
    def generate_signals(self, data):
        """
        Generate trading signals with confirmation filters.
        """
        # Calculate moving averages
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate basic signals
        signals = pd.Series(0, index=data.index)
        signals[data['SMA_short'] > data['SMA_long']] = 1
        signals[data['SMA_short'] < data['SMA_long']] = -1
        
        # Apply confirmation filters
        confirmed_signals, indicators = self.confirmation.apply_confirmation_filters(signals, data)
        
        return confirmed_signals, indicators
    
    def calculate_returns(self, data, signals):
        """
        Calculate strategy returns based on confirmed signals.
        """
        positions = signals.shift(1).fillna(0)
        returns = data['Close'].pct_change() * positions
        return returns.fillna(0)

def analyze_confirmation_impact(symbols=['MSFT', 'TSLA', 'AAPL'], period='2y'):
    """
    Analyze the impact of confirmation indicators on strategy performance.
    """
    fetcher = DataFetcher()
    results = []
    
    print("🔍 ANALYZING CONFIRMATION INDICATORS IMPACT")
    print("=" * 50)
    
    # Calculate date range
    from datetime import datetime, timedelta
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
        if data is None or len(data) < 250:
            print(f"❌ Insufficient data for {symbol}")
            continue
        
        # Basic strategy
        basic_strategy = MovingAverageStrategy()
        data_basic = data.copy()
        data_basic = basic_strategy.generate_signals(data_basic)
        basic_signals = data_basic['signal']
        basic_returns = data_basic['Close'].pct_change() * basic_signals.shift(1).fillna(0)
        
        # Enhanced strategy with confirmation
        enhanced_strategy = EnhancedMovingAverageStrategy()
        enhanced_signals, indicators = enhanced_strategy.generate_signals(data.copy())
        enhanced_returns = enhanced_strategy.calculate_returns(data, enhanced_signals)
        
        # Calculate performance metrics
        def calculate_metrics(returns):
            total_return = (1 + returns).prod() - 1
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
            win_rate = (returns > 0).sum() / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
            return {
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown) * 100,
                'win_rate': win_rate * 100
            }
        
        basic_metrics = calculate_metrics(basic_returns)
        enhanced_metrics = calculate_metrics(enhanced_returns)
        
        # Count signals
        basic_trades = len(basic_signals[basic_signals != 0])
        enhanced_trades = len(enhanced_signals[enhanced_signals != 0])
        
        results.append({
            'Symbol': symbol,
            'Basic_Return': basic_metrics['total_return'],
            'Enhanced_Return': enhanced_metrics['total_return'],
            'Basic_Sharpe': basic_metrics['sharpe_ratio'],
            'Enhanced_Sharpe': enhanced_metrics['sharpe_ratio'],
            'Basic_Drawdown': basic_metrics['max_drawdown'],
            'Enhanced_Drawdown': enhanced_metrics['max_drawdown'],
            'Basic_Trades': basic_trades,
            'Enhanced_Trades': enhanced_trades,
            'Signal_Reduction': (basic_trades - enhanced_trades) / basic_trades * 100 if basic_trades > 0 else 0
        })
        
        print(f"   Basic Strategy: {basic_metrics['total_return']:.1f}% return, {basic_metrics['sharpe_ratio']:.2f} Sharpe")
        print(f"   Enhanced Strategy: {enhanced_metrics['total_return']:.1f}% return, {enhanced_metrics['sharpe_ratio']:.2f} Sharpe")
        print(f"   Signal Reduction: {(basic_trades - enhanced_trades) / basic_trades * 100 if basic_trades > 0 else 0:.1f}%")
    
    return pd.DataFrame(results)

def create_confirmation_visualizations(symbol='MSFT', period='1y'):
    """
    Create visualizations showing the impact of confirmation indicators.
    """
    fetcher = DataFetcher()
    
    # Calculate date range
    from datetime import datetime, timedelta
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
    basic_strategy = MovingAverageStrategy()
    data_basic = data.copy()
    data_basic = basic_strategy.generate_signals(data_basic)
    basic_signals = data_basic['signal']
    
    enhanced_strategy = EnhancedMovingAverageStrategy()
    enhanced_signals, indicators = enhanced_strategy.generate_signals(data.copy())
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(f'{symbol} - Confirmation Indicators Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Price with signals and trend filter
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Price', linewidth=1, alpha=0.8)
    ax1.plot(data.index, indicators['trend_filter'], label='200-day SMA (Trend Filter)', 
             color='orange', linewidth=2, alpha=0.7)
    
    # Plot basic signals
    basic_buys = data.index[basic_signals == 1]
    basic_sells = data.index[basic_signals == -1]
    if len(basic_buys) > 0:
        ax1.scatter(basic_buys, data.loc[basic_buys, 'Close'], 
                   color='lightgreen', marker='^', s=50, alpha=0.6, label='Basic Buy')
    if len(basic_sells) > 0:
        ax1.scatter(basic_sells, data.loc[basic_sells, 'Close'], 
                   color='lightcoral', marker='v', s=50, alpha=0.6, label='Basic Sell')
    
    # Plot enhanced signals
    enhanced_buys = data.index[enhanced_signals == 1]
    enhanced_sells = data.index[enhanced_signals == -1]
    if len(enhanced_buys) > 0:
        ax1.scatter(enhanced_buys, data.loc[enhanced_buys, 'Close'], 
                   color='darkgreen', marker='^', s=100, label='Enhanced Buy')
    if len(enhanced_sells) > 0:
        ax1.scatter(enhanced_sells, data.loc[enhanced_sells, 'Close'], 
                   color='darkred', marker='v', s=100, label='Enhanced Sell')
    
    ax1.set_title('Price Action with Trading Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RSI with overbought/oversold levels
    ax2 = axes[1]
    ax2.plot(data.index, indicators['rsi'], label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.fill_between(data.index, 70, 100, alpha=0.2, color='red')
    ax2.fill_between(data.index, 0, 30, alpha=0.2, color='green')
    ax2.set_title('RSI - Volatility Filter')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ATR with threshold
    ax3 = axes[2]
    ax3.plot(data.index, indicators['atr'], label='ATR', color='brown', linewidth=1)
    ax3.axhline(y=indicators['atr_threshold'], color='red', linestyle='--', 
               alpha=0.7, label=f'ATR Threshold ({indicators["atr_threshold"]:.2f})')
    ax3.fill_between(data.index, indicators['atr_threshold'], indicators['atr'].max(), 
                    alpha=0.2, color='red')
    ax3.set_title('Average True Range - Volatility Measurement')
    ax3.set_ylabel('ATR')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns comparison
    ax4 = axes[3]
    basic_returns = data['Close'].pct_change() * basic_signals.shift(1).fillna(0)
    enhanced_returns = enhanced_strategy.calculate_returns(data, enhanced_signals)
    
    basic_cumulative = (1 + basic_returns).cumprod()
    enhanced_cumulative = (1 + enhanced_returns).cumprod()
    
    ax4.plot(data.index, basic_cumulative, label='Basic Strategy', linewidth=2, alpha=0.8)
    ax4.plot(data.index, enhanced_cumulative, label='Enhanced Strategy', linewidth=2, alpha=0.8)
    ax4.set_title('Cumulative Returns Comparison')
    ax4.set_ylabel('Cumulative Return')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_charts/{symbol}_confirmation_indicators.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Confirmation indicators chart saved for {symbol}")

def main():
    """
    Main function to run confirmation indicators analysis.
    """
    print("🚀 CONFIRMATION INDICATORS ANALYSIS")
    print("=" * 40)
    
    # Analyze impact across multiple symbols
    results_df = analyze_confirmation_impact(['MSFT', 'TSLA', 'AAPL', 'GOOGL', 'AMZN'])
    
    print("\n📊 CONFIRMATION INDICATORS IMPACT SUMMARY:")
    print(results_df.round(2))
    
    # Save results
    results_df.to_csv('analysis_charts/confirmation_indicators_results.csv', index=False)
    print("\n💾 Results saved to: analysis_charts/confirmation_indicators_results.csv")
    
    # Create detailed visualization for MSFT
    print("\n📊 Creating detailed visualization for MSFT...")
    create_confirmation_visualizations('MSFT', '1y')
    
    # Calculate average improvements
    avg_return_improvement = results_df['Enhanced_Return'].mean() - results_df['Basic_Return'].mean()
    avg_sharpe_improvement = results_df['Enhanced_Sharpe'].mean() - results_df['Basic_Sharpe'].mean()
    avg_signal_reduction = results_df['Signal_Reduction'].mean()
    
    print("\n🎯 KEY TAKEAWAYS FROM CONFIRMATION INDICATORS:")
    print(f"1. 📈 Average Return Change: {avg_return_improvement:+.1f}%")
    print(f"2. 📊 Average Sharpe Ratio Change: {avg_sharpe_improvement:+.2f}")
    print(f"3. 🎯 Average Signal Reduction: {avg_signal_reduction:.1f}%")
    print("4. 🛡️  Trend filter prevents buying in downtrends")
    print("5. 📊 RSI filter reduces trades in extreme conditions")
    print("6. 🌊 ATR filter avoids trading in high volatility periods")
    print("7. ✨ Fewer, higher-quality signals often lead to better risk-adjusted returns")
    
    print("\n✅ Confirmation indicators analysis completed!")
    
    print("\n💡 NEXT STEPS:")
    print("   - Fine-tune RSI thresholds (try 25/75 instead of 30/70)")
    print("   - Experiment with different trend filter periods (150-day, 100-day)")
    print("   - Consider adding volume confirmation")
    print("   - Test on different market conditions and timeframes")

if __name__ == "__main__":
    main()