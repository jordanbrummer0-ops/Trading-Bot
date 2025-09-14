#!/usr/bin/env python3
"""
Configuration Module for Stock Trading Bot

Centralized configuration management for the trading bot.
Allows easy customization of strategies, parameters, and settings.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List


class TradingBotConfig:
    """Configuration class for the trading bot."""
    
    # Default trading parameters
    DEFAULT_INITIAL_CASH = 10000.0
    DEFAULT_COMMISSION = 0.001  # 0.1%
    DEFAULT_SYMBOL = 'AAPL'
    
    # Default date range (last 365 days)
    DEFAULT_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Logging configuration
    LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    DEFAULT_LOG_LEVEL = 'INFO'
    LOG_FILE = 'trading_bot.log'
    
    # Strategy configurations
    STRATEGY_CONFIGS = {
        'moving_average': {
            'short_window': 20,
            'long_window': 50,
            'description': 'Simple Moving Average Crossover Strategy'
        },
        'rsi': {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'description': 'RSI (Relative Strength Index) Strategy'
        },
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'description': 'MACD (Moving Average Convergence Divergence) Strategy'
        },
        'combined': {
            'ma_short': 20,
            'ma_long': 50,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'description': 'Combined MA and RSI Strategy'
        }
    }
    
    # Popular stock symbols for quick testing
    POPULAR_SYMBOLS = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP'],
        'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT'],
        'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO'],
        'etf': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO']
    }
    
    # Risk management settings
    RISK_MANAGEMENT = {
        'max_position_size': 0.1,  # 10% of portfolio per position
        'stop_loss_pct': 0.05,     # 5% stop loss
        'take_profit_pct': 0.15,   # 15% take profit
        'max_drawdown_pct': 0.20,  # 20% maximum drawdown
    }
    
    # Data fetching settings
    DATA_SETTINGS = {
        'default_interval': '1d',
        'max_retries': 3,
        'timeout_seconds': 30,
        'cache_duration_hours': 1
    }
    
    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        'good_return': 15.0,      # 15% annual return considered good
        'excellent_return': 25.0,  # 25% annual return considered excellent
        'min_sharpe_ratio': 1.0,   # Minimum acceptable Sharpe ratio
        'max_drawdown': 15.0,      # Maximum acceptable drawdown %
        'min_win_rate': 50.0       # Minimum acceptable win rate %
    }
    
    @classmethod
    def get_strategy_config(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy
        
        Returns:
            Dict[str, Any]: Strategy configuration
        """
        return cls.STRATEGY_CONFIGS.get(strategy_name.lower(), {})
    
    @classmethod
    def get_symbols_by_sector(cls, sector: str) -> List[str]:
        """
        Get stock symbols for a specific sector.
        
        Args:
            sector (str): Sector name (tech, finance, healthcare, energy, etf)
        
        Returns:
            List[str]: List of stock symbols
        """
        return cls.POPULAR_SYMBOLS.get(sector.lower(), [])
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        Basic validation for stock symbols.
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if symbol format is valid
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic format validation
        symbol = symbol.upper().strip()
        
        # Check length (most symbols are 1-5 characters)
        if len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # Check for valid characters (letters, numbers, some special chars)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        if not all(c in valid_chars for c in symbol):
            return False
        
        return True
    
    @classmethod
    def validate_date_range(cls, start_date: str, end_date: str) -> bool:
        """
        Validate date range for data fetching.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            bool: True if date range is valid
        """
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Check if start is before end
            if start >= end:
                return False
            
            # Check if dates are not in the future
            now = datetime.now()
            if start > now or end > now:
                return False
            
            # Check if date range is reasonable (not too old)
            max_history = timedelta(days=365 * 20)  # 20 years
            if (now - start) > max_history:
                return False
            
            return True
            
        except ValueError:
            return False
    
    @classmethod
    def get_recommended_parameters(cls, symbol: str, timeframe_days: int) -> Dict[str, Any]:
        """
        Get recommended parameters based on symbol and timeframe.
        
        Args:
            symbol (str): Stock symbol
            timeframe_days (int): Number of days in the analysis period
        
        Returns:
            Dict[str, Any]: Recommended parameters
        """
        recommendations = {
            'initial_cash': cls.DEFAULT_INITIAL_CASH,
            'commission': cls.DEFAULT_COMMISSION
        }
        
        # Adjust parameters based on timeframe
        if timeframe_days <= 90:  # Short-term (3 months)
            recommendations.update({
                'ma_short': 5,
                'ma_long': 20,
                'rsi_period': 7,
                'strategy': 'rsi'  # RSI works better for short-term
            })
        elif timeframe_days <= 365:  # Medium-term (1 year)
            recommendations.update({
                'ma_short': 20,
                'ma_long': 50,
                'rsi_period': 14,
                'strategy': 'moving_average'
            })
        else:  # Long-term (> 1 year)
            recommendations.update({
                'ma_short': 50,
                'ma_long': 200,
                'rsi_period': 21,
                'strategy': 'combined'
            })
        
        # Adjust for volatile stocks (basic heuristic)
        volatile_symbols = ['TSLA', 'NVDA', 'AMD', 'NFLX', 'ZOOM']
        if symbol.upper() in volatile_symbols:
            recommendations['commission'] *= 1.5  # Higher commission for volatile stocks
            recommendations['strategy'] = 'rsi'   # RSI better for volatile stocks
        
        return recommendations
    
    @classmethod
    def create_config_file(cls, filepath: str = 'bot_config.yaml') -> None:
        """
        Create a YAML configuration file with default settings.
        
        Args:
            filepath (str): Path to save the configuration file
        """
        config_content = f"""
# Stock Trading Bot Configuration File
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Trading Parameters
trading:
  initial_cash: {cls.DEFAULT_INITIAL_CASH}
  commission: {cls.DEFAULT_COMMISSION}
  default_symbol: "{cls.DEFAULT_SYMBOL}"

# Date Settings
dates:
  default_start_date: "{cls.DEFAULT_START_DATE}"
  default_end_date: "{cls.DEFAULT_END_DATE}"

# Logging
logging:
  level: "{cls.DEFAULT_LOG_LEVEL}"
  file: "{cls.LOG_FILE}"

# Strategy Parameters
strategies:
  moving_average:
    short_window: {cls.STRATEGY_CONFIGS['moving_average']['short_window']}
    long_window: {cls.STRATEGY_CONFIGS['moving_average']['long_window']}
  
  rsi:
    period: {cls.STRATEGY_CONFIGS['rsi']['rsi_period']}
    oversold: {cls.STRATEGY_CONFIGS['rsi']['oversold']}
    overbought: {cls.STRATEGY_CONFIGS['rsi']['overbought']}
  
  macd:
    fast_period: {cls.STRATEGY_CONFIGS['macd']['fast_period']}
    slow_period: {cls.STRATEGY_CONFIGS['macd']['slow_period']}
    signal_period: {cls.STRATEGY_CONFIGS['macd']['signal_period']}

# Risk Management
risk_management:
  max_position_size: {cls.RISK_MANAGEMENT['max_position_size']}
  stop_loss_pct: {cls.RISK_MANAGEMENT['stop_loss_pct']}
  take_profit_pct: {cls.RISK_MANAGEMENT['take_profit_pct']}
  max_drawdown_pct: {cls.RISK_MANAGEMENT['max_drawdown_pct']}

# Performance Thresholds
performance:
  good_return: {cls.PERFORMANCE_THRESHOLDS['good_return']}
  excellent_return: {cls.PERFORMANCE_THRESHOLDS['excellent_return']}
  min_sharpe_ratio: {cls.PERFORMANCE_THRESHOLDS['min_sharpe_ratio']}
  max_drawdown: {cls.PERFORMANCE_THRESHOLDS['max_drawdown']}
  min_win_rate: {cls.PERFORMANCE_THRESHOLDS['min_win_rate']}
"""
        
        try:
            with open(filepath, 'w') as f:
                f.write(config_content)
            print(f"Configuration file created: {filepath}")
        except Exception as e:
            print(f"Error creating configuration file: {str(e)}")


# Environment-based configuration
class EnvironmentConfig:
    """Environment-specific configuration settings."""
    
    @staticmethod
    def is_development() -> bool:
        """Check if running in development environment."""
        return os.getenv('TRADING_BOT_ENV', 'development').lower() == 'development'
    
    @staticmethod
    def is_production() -> bool:
        """Check if running in production environment."""
        return os.getenv('TRADING_BOT_ENV', 'development').lower() == 'production'
    
    @staticmethod
    def get_log_level() -> str:
        """Get log level from environment or default."""
        return os.getenv('TRADING_BOT_LOG_LEVEL', TradingBotConfig.DEFAULT_LOG_LEVEL)
    
    @staticmethod
    def get_data_cache_dir() -> str:
        """Get data cache directory."""
        return os.getenv('TRADING_BOT_CACHE_DIR', os.path.join(os.getcwd(), 'cache'))


if __name__ == '__main__':
    # Create a sample configuration file
    TradingBotConfig.create_config_file()
    
    # Display some configuration info
    print("\nTrading Bot Configuration:")
    print(f"Default Symbol: {TradingBotConfig.DEFAULT_SYMBOL}")
    print(f"Default Initial Cash: ${TradingBotConfig.DEFAULT_INITIAL_CASH:,.2f}")
    print(f"Available Strategies: {list(TradingBotConfig.STRATEGY_CONFIGS.keys())}")
    print(f"Tech Stocks: {TradingBotConfig.get_symbols_by_sector('tech')[:5]}...")