#!/usr/bin/env python3
"""
Enhanced Trading Strategies with Risk Management

This module implements advanced trading strategies with comprehensive risk management
features including stop-loss, take-profit, and confirmation indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

# Import base strategy
from .trading_strategy import BaseStrategy

class RiskManagedStrategy(BaseStrategy):
    """
    Enhanced strategy with risk management features.
    """
    
    def __init__(self, name: str, stop_loss_pct: float = 0.05, 
                 take_profit_pct: float = 0.15, use_trailing_stop: bool = False):
        """
        Initialize risk-managed strategy.
        
        Args:
            name: Strategy name
            stop_loss_pct: Stop loss percentage (e.g., 0.05 = 5%)
            take_profit_pct: Take profit percentage (e.g., 0.15 = 15%)
            use_trailing_stop: Whether to use trailing stop loss
        """
        super().__init__(name)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_trailing_stop = use_trailing_stop
        
    def apply_risk_management(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk management rules to trading signals.
        
        Args:
            data: DataFrame with trading signals
            
        Returns:
            DataFrame with risk-managed signals
        """
        data = data.copy()
        
        # Initialize risk management columns
        data['entry_price'] = np.nan
        data['stop_loss_price'] = np.nan
        data['take_profit_price'] = np.nan
        data['exit_reason'] = ''
        data['risk_managed_signal'] = data['signal'].copy()
        data['risk_managed_position'] = 0
        
        current_position = 0
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        highest_price_since_entry = 0
        
        for i in range(len(data)):
            current_price = data.iloc[i]['Close']
            original_signal = data.iloc[i]['signal']
            
            # Check for new entry signal
            if current_position == 0 and original_signal == 1:
                # Enter long position
                current_position = 1
                entry_price = current_price
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                take_profit_price = entry_price * (1 + self.take_profit_pct)
                highest_price_since_entry = current_price
                
                data.iloc[i, data.columns.get_loc('entry_price')] = entry_price
                data.iloc[i, data.columns.get_loc('stop_loss_price')] = stop_loss_price
                data.iloc[i, data.columns.get_loc('take_profit_price')] = take_profit_price
                data.iloc[i, data.columns.get_loc('risk_managed_signal')] = 1
                data.iloc[i, data.columns.get_loc('risk_managed_position')] = 1
                
            elif current_position == 0 and original_signal == -1:
                # Enter short position
                current_position = -1
                entry_price = current_price
                stop_loss_price = entry_price * (1 + self.stop_loss_pct)
                take_profit_price = entry_price * (1 - self.take_profit_pct)
                highest_price_since_entry = current_price  # For short, track lowest price
                
                data.iloc[i, data.columns.get_loc('entry_price')] = entry_price
                data.iloc[i, data.columns.get_loc('stop_loss_price')] = stop_loss_price
                data.iloc[i, data.columns.get_loc('take_profit_price')] = take_profit_price
                data.iloc[i, data.columns.get_loc('risk_managed_signal')] = -1
                data.iloc[i, data.columns.get_loc('risk_managed_position')] = -1
                
            elif current_position != 0:
                # We're in a position, check for exit conditions
                exit_triggered = False
                exit_reason = ''
                
                if current_position == 1:  # Long position
                    # Update trailing stop if using
                    if self.use_trailing_stop and current_price > highest_price_since_entry:
                        highest_price_since_entry = current_price
                        stop_loss_price = highest_price_since_entry * (1 - self.stop_loss_pct)
                    
                    # Check exit conditions
                    if current_price <= stop_loss_price:
                        exit_triggered = True
                        exit_reason = 'stop_loss'
                    elif current_price >= take_profit_price:
                        exit_triggered = True
                        exit_reason = 'take_profit'
                    elif original_signal == -1:
                        exit_triggered = True
                        exit_reason = 'signal_exit'
                        
                elif current_position == -1:  # Short position
                    # Update trailing stop if using (for short, track lowest price)
                    if self.use_trailing_stop and current_price < highest_price_since_entry:
                        highest_price_since_entry = current_price
                        stop_loss_price = highest_price_since_entry * (1 + self.stop_loss_pct)
                    
                    # Check exit conditions
                    if current_price >= stop_loss_price:
                        exit_triggered = True
                        exit_reason = 'stop_loss'
                    elif current_price <= take_profit_price:
                        exit_triggered = True
                        exit_reason = 'take_profit'
                    elif original_signal == 1:
                        exit_triggered = True
                        exit_reason = 'signal_exit'
                
                if exit_triggered:
                    # Exit position
                    data.iloc[i, data.columns.get_loc('risk_managed_signal')] = -current_position
                    data.iloc[i, data.columns.get_loc('risk_managed_position')] = 0
                    data.iloc[i, data.columns.get_loc('exit_reason')] = exit_reason
                    current_position = 0
                else:
                    # Maintain position
                    data.iloc[i, data.columns.get_loc('risk_managed_position')] = current_position
                    data.iloc[i, data.columns.get_loc('stop_loss_price')] = stop_loss_price
                    data.iloc[i, data.columns.get_loc('take_profit_price')] = take_profit_price
            else:
                # No position, no signal
                data.iloc[i, data.columns.get_loc('risk_managed_position')] = 0
        
        return data

class EnhancedMovingAverageStrategy(RiskManagedStrategy):
    """
    Enhanced Moving Average strategy with risk management and confirmation indicators.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50,
                 trend_filter_window: int = 200, use_rsi_filter: bool = True,
                 rsi_period: int = 14, rsi_oversold: int = 30, rsi_overbought: int = 70,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.15,
                 use_trailing_stop: bool = False):
        """
        Initialize enhanced moving average strategy.
        
        Args:
            short_window: Short-term MA period
            long_window: Long-term MA period
            trend_filter_window: Trend filter MA period (e.g., 200-day)
            use_rsi_filter: Whether to use RSI confirmation
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            use_trailing_stop: Whether to use trailing stop
        """
        super().__init__("Enhanced Moving Average", stop_loss_pct, take_profit_pct, use_trailing_stop)
        self.short_window = short_window
        self.long_window = long_window
        self.trend_filter_window = trend_filter_window
        self.use_rsi_filter = use_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate enhanced trading signals with confirmation indicators.
        
        Args:
            data: Stock price data
            
        Returns:
            DataFrame with enhanced signals
        """
        data = data.copy()
        
        # Calculate moving averages
        data['MA_Short'] = data['Close'].rolling(window=self.short_window).mean()
        data['MA_Long'] = data['Close'].rolling(window=self.long_window).mean()
        data['MA_Trend'] = data['Close'].rolling(window=self.trend_filter_window).mean()
        
        # Calculate RSI if enabled
        if self.use_rsi_filter:
            data['RSI'] = self.calculate_rsi(data['Close'], self.rsi_period)
        
        # Generate basic MA crossover signals
        data['ma_signal'] = 0
        data.loc[data['MA_Short'] > data['MA_Long'], 'ma_signal'] = 1
        data.loc[data['MA_Short'] < data['MA_Long'], 'ma_signal'] = -1
        
        # Apply trend filter
        data['trend_filter'] = data['Close'] > data['MA_Trend']
        
        # Apply RSI filter if enabled
        if self.use_rsi_filter:
            data['rsi_filter_long'] = (data['RSI'] > self.rsi_oversold) & (data['RSI'] < self.rsi_overbought)
            data['rsi_filter_short'] = (data['RSI'] > self.rsi_oversold) & (data['RSI'] < self.rsi_overbought)
        else:
            data['rsi_filter_long'] = True
            data['rsi_filter_short'] = True
        
        # Generate final signals with all filters
        data['signal'] = 0
        data['position'] = 0
        
        # Long signals: MA crossover + trend filter + RSI filter
        long_condition = (
            (data['ma_signal'] == 1) & 
            (data['trend_filter'] == True) & 
            (data['rsi_filter_long'] == True)
        )
        
        # Short signals: MA crossover + opposite trend + RSI filter
        short_condition = (
            (data['ma_signal'] == -1) & 
            (data['trend_filter'] == False) & 
            (data['rsi_filter_short'] == True)
        )
        
        data.loc[long_condition, 'signal'] = 1
        data.loc[short_condition, 'signal'] = -1
        
        # Calculate positions
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Apply risk management
        data = self.apply_risk_management(data)
        
        return data

class BollingerBandsMeanReversionStrategy(RiskManagedStrategy):
    """
    Bollinger Bands mean reversion strategy with risk management.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0,
                 stop_loss_pct: float = 0.03, take_profit_pct: float = 0.08,
                 use_trailing_stop: bool = True):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            window: Moving average window for Bollinger Bands
            num_std: Number of standard deviations for bands
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            use_trailing_stop: Whether to use trailing stop
        """
        super().__init__("Bollinger Bands Mean Reversion", stop_loss_pct, take_profit_pct, use_trailing_stop)
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Bollinger Bands mean reversion signals.
        
        Args:
            data: Stock price data
            
        Returns:
            DataFrame with signals
        """
        data = data.copy()
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=self.window).mean()
        data['BB_Std'] = data['Close'].rolling(window=self.window).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * self.num_std)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * self.num_std)
        
        # Calculate Bollinger Band position
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Generate signals
        data['signal'] = 0
        data['position'] = 0
        
        # Buy when price touches lower band (oversold)
        buy_condition = data['Close'] <= data['BB_Lower']
        
        # Sell when price touches upper band (overbought)
        sell_condition = data['Close'] >= data['BB_Upper']
        
        # Exit when price returns to middle band
        exit_condition = abs(data['Close'] - data['BB_Middle']) < (data['BB_Std'] * 0.5)
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        data.loc[exit_condition, 'signal'] = 0
        
        # Calculate positions (mean reversion logic)
        current_position = 0
        positions = []
        
        for i in range(len(data)):
            signal = data.iloc[i]['signal']
            
            if signal == 1 and current_position <= 0:  # Buy signal
                current_position = 1
            elif signal == -1 and current_position >= 0:  # Sell signal
                current_position = -1
            elif signal == 0:  # Exit signal
                current_position = 0
            
            positions.append(current_position)
        
        data['position'] = positions
        
        # Apply risk management
        data = self.apply_risk_management(data)
        
        return data

def compare_strategies_with_risk_management(symbol: str, start_date: str = "2023-01-01", 
                                          end_date: str = "2024-12-31") -> Dict:
    """
    Compare different strategies with and without risk management.
    
    Args:
        symbol: Stock symbol to analyze
        start_date: Start date
        end_date: End date
        
    Returns:
        Dictionary with comparison results
    """
    from .data_fetcher import DataFetcher
    from .trading_strategy import MovingAverageStrategy
    
    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.get_stock_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        return {}
    
    strategies = {
        'Basic MA': MovingAverageStrategy(20, 50),
        'Enhanced MA': EnhancedMovingAverageStrategy(20, 50, 200, True, 14, 30, 70, 0.05, 0.15, False),
        'Enhanced MA + Trailing': EnhancedMovingAverageStrategy(20, 50, 200, True, 14, 30, 70, 0.05, 0.15, True),
        'Bollinger Bands': BollingerBandsMeanReversionStrategy(20, 2.0, 0.03, 0.08, True)
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        try:
            signals = strategy.generate_signals(data.copy())
            
            # Calculate performance metrics
            if hasattr(strategy, 'apply_risk_management'):
                # Use risk-managed positions
                position_col = 'risk_managed_position' if 'risk_managed_position' in signals.columns else 'position'
            else:
                position_col = 'position'
            
            returns = signals['Close'].pct_change()
            strategy_returns = signals[position_col].shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 0:
                total_return = (1 + strategy_returns).prod() - 1
                sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                
                # Calculate max drawdown
                cumulative = (1 + strategy_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min())
                
                # Count trades
                position_changes = signals[position_col].diff().abs()
                num_trades = (position_changes > 0).sum()
                
                # Risk management stats
                risk_stats = {}
                if 'exit_reason' in signals.columns:
                    exit_reasons = signals[signals['exit_reason'] != '']['exit_reason'].value_counts()
                    risk_stats = {
                        'stop_loss_exits': exit_reasons.get('stop_loss', 0),
                        'take_profit_exits': exit_reasons.get('take_profit', 0),
                        'signal_exits': exit_reasons.get('signal_exit', 0)
                    }
                
                results[name] = {
                    'total_return': total_return * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown * 100,
                    'num_trades': num_trades,
                    'risk_management_stats': risk_stats
                }
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
            continue
    
    return results