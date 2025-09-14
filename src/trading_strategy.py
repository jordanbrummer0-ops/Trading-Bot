#!/usr/bin/env python3
"""
Trading Strategy Module

Implements various trading strategies including moving averages, RSI, and MACD.
Provides a base class for creating custom trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        """Initialize the base strategy.
        
        Args:
            name (str): Name of the strategy
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on the strategy.
        
        Args:
            data (pd.DataFrame): Stock price data with OHLCV columns
        
        Returns:
            pd.DataFrame: Data with additional signal columns
        """
        pass
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns based on the generated signals.
        
        Args:
            data (pd.DataFrame): Data with signals
        
        Returns:
            pd.DataFrame: Data with returns calculated
        """
        if 'signal' not in data.columns:
            raise ValueError("Data must contain 'signal' column")
        
        # Calculate daily returns
        data['daily_return'] = data['Close'].pct_change()
        
        # Calculate strategy returns (signal * next day's return)
        data['strategy_return'] = data['signal'].shift(1) * data['daily_return']
        
        # Calculate cumulative returns
        data['cumulative_return'] = (1 + data['strategy_return']).cumprod()
        data['buy_hold_return'] = (1 + data['daily_return']).cumprod()
        
        return data


class MovingAverageStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the Moving Average strategy.
        
        Args:
            short_window (int): Period for short-term moving average
            long_window (int): Period for long-term moving average
        """
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on moving average crossover.
        
        Buy signal: Short MA crosses above Long MA
        Sell signal: Short MA crosses below Long MA
        
        Args:
            data (pd.DataFrame): Stock price data
        
        Returns:
            pd.DataFrame: Data with signal columns added
        """
        data = data.copy()
        
        # Calculate moving averages
        data[f'MA_{self.short_window}'] = data['Close'].rolling(window=self.short_window).mean()
        data[f'MA_{self.long_window}'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        data['signal'] = 0
        data['position'] = 0
        
        # Buy signal: short MA > long MA
        data.loc[data[f'MA_{self.short_window}'] > data[f'MA_{self.long_window}'], 'signal'] = 1
        
        # Sell signal: short MA < long MA
        data.loc[data[f'MA_{self.short_window}'] < data[f'MA_{self.long_window}'], 'signal'] = -1
        
        # Generate position (1 for long, 0 for no position, -1 for short)
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Mark entry and exit points
        data['entry'] = (data['position'] != data['position'].shift(1)) & (data['position'] != 0)
        data['exit'] = (data['position'] != data['position'].shift(1)) & (data['position'].shift(1) != 0)
        
        self.logger.info(f"Generated {data['entry'].sum()} entry signals and {data['exit'].sum()} exit signals")
        
        return data


class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) Strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize the RSI strategy.
        
        Args:
            rsi_period (int): Period for RSI calculation
            oversold (float): RSI level considered oversold (buy signal)
            overbought (float): RSI level considered overbought (sell signal)
        """
        super().__init__("RSI Strategy")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
        
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on RSI levels.
        
        Buy signal: RSI < oversold level
        Sell signal: RSI > overbought level
        
        Args:
            data (pd.DataFrame): Stock price data
        
        Returns:
            pd.DataFrame: Data with signal columns added
        """
        data = data.copy()
        
        # Calculate RSI
        data['RSI'] = self.calculate_rsi(data['Close'], self.rsi_period)
        
        # Generate signals
        data['signal'] = 0
        data['position'] = 0
        
        # Buy signal: RSI < oversold
        data.loc[data['RSI'] < self.oversold, 'signal'] = 1
        
        # Sell signal: RSI > overbought
        data.loc[data['RSI'] > self.overbought, 'signal'] = -1
        
        # Generate position
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Mark entry and exit points
        data['entry'] = (data['position'] != data['position'].shift(1)) & (data['position'] != 0)
        data['exit'] = (data['position'] != data['position'].shift(1)) & (data['position'].shift(1) != 0)
        
        self.logger.info(f"Generated {data['entry'].sum()} entry signals and {data['exit'].sum()} exit signals")
        
        return data


class MACDStrategy(BaseStrategy):
    """MACD (Moving Average Convergence Divergence) Strategy."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize the MACD strategy.
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line EMA period
        """
        super().__init__("MACD Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD, Signal line, and Histogram.
        
        Args:
            prices (pd.Series): Price series
        
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD, Signal, Histogram
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate Signal line
        signal = macd.ewm(span=self.signal_period).mean()
        
        # Calculate Histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on MACD crossover.
        
        Buy signal: MACD crosses above Signal line
        Sell signal: MACD crosses below Signal line
        
        Args:
            data (pd.DataFrame): Stock price data
        
        Returns:
            pd.DataFrame: Data with signal columns added
        """
        data = data.copy()
        
        # Calculate MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = self.calculate_macd(data['Close'])
        
        # Generate signals
        data['signal'] = 0
        data['position'] = 0
        
        # Buy signal: MACD crosses above Signal line
        data.loc[(data['MACD'] > data['MACD_Signal']) & 
                (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)), 'signal'] = 1
        
        # Sell signal: MACD crosses below Signal line
        data.loc[(data['MACD'] < data['MACD_Signal']) & 
                (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1)), 'signal'] = -1
        
        # Generate position
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Mark entry and exit points
        data['entry'] = (data['position'] != data['position'].shift(1)) & (data['position'] != 0)
        data['exit'] = (data['position'] != data['position'].shift(1)) & (data['position'].shift(1) != 0)
        
        self.logger.info(f"Generated {data['entry'].sum()} entry signals and {data['exit'].sum()} exit signals")
        
        return data


class CombinedStrategy(BaseStrategy):
    """Combined strategy using multiple indicators."""
    
    def __init__(self, ma_short: int = 20, ma_long: int = 50, rsi_period: int = 14, 
                 rsi_oversold: float = 30, rsi_overbought: float = 70):
        """
        Initialize the combined strategy.
        
        Args:
            ma_short (int): Short MA period
            ma_long (int): Long MA period
            rsi_period (int): RSI period
            rsi_oversold (float): RSI oversold level
            rsi_overbought (float): RSI overbought level
        """
        super().__init__("Combined Strategy")
        self.ma_strategy = MovingAverageStrategy(ma_short, ma_long)
        self.rsi_strategy = RSIStrategy(rsi_period, rsi_oversold, rsi_overbought)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using combined MA and RSI strategies.
        
        Buy signal: MA bullish AND RSI oversold
        Sell signal: MA bearish OR RSI overbought
        
        Args:
            data (pd.DataFrame): Stock price data
        
        Returns:
            pd.DataFrame: Data with signal columns added
        """
        # Get signals from individual strategies
        ma_data = self.ma_strategy.generate_signals(data.copy())
        rsi_data = self.rsi_strategy.generate_signals(data.copy())
        
        # Combine the data
        data = data.copy()
        data['MA_signal'] = ma_data['signal']
        data['RSI_signal'] = rsi_data['signal']
        data['RSI'] = rsi_data['RSI']
        data[f'MA_{self.ma_strategy.short_window}'] = ma_data[f'MA_{self.ma_strategy.short_window}']
        data[f'MA_{self.ma_strategy.long_window}'] = ma_data[f'MA_{self.ma_strategy.long_window}']
        
        # Generate combined signals
        data['signal'] = 0
        data['position'] = 0
        
        # Buy signal: MA bullish AND RSI oversold
        buy_condition = (data['MA_signal'] == 1) & (data['RSI_signal'] == 1)
        data.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: MA bearish OR RSI overbought
        sell_condition = (data['MA_signal'] == -1) | (data['RSI_signal'] == -1)
        data.loc[sell_condition, 'signal'] = -1
        
        # Generate position
        data['position'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Mark entry and exit points
        data['entry'] = (data['position'] != data['position'].shift(1)) & (data['position'] != 0)
        data['exit'] = (data['position'] != data['position'].shift(1)) & (data['position'].shift(1) != 0)
        
        self.logger.info(f"Generated {data['entry'].sum()} entry signals and {data['exit'].sum()} exit signals")
        
        return data