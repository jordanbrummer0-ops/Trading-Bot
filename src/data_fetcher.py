#!/usr/bin/env python3
"""
Data Fetcher Module

Handles fetching stock data using yfinance library.
Provides methods to retrieve historical stock data and real-time quotes.
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class DataFetcher:
    """Fetches stock data using yfinance."""
    
    def __init__(self):
        """Initialize the DataFetcher."""
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                      interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1h', '5m', etc.)
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data.empty:
                self.logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Clean and prepare data
            data = self._clean_data(data)
            
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price of a stock.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            Optional[float]: Current price or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
            
            for field in price_fields:
                if field in info and info[field] is not None:
                    return float(info[field])
            
            # Fallback: get latest close from 1-day data
            data = ticker.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic information about a stock.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            Dict[str, Any]: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return stock_info
            
        except Exception as e:
            self.logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_stocks(self, symbols: list, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and data as value
        """
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Fetching data for {symbol}")
            data = self.get_stock_data(symbol, start_date, end_date)
            results[symbol] = data
        
        return results
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the stock data.
        
        Args:
            data (pd.DataFrame): Raw stock data
        
        Returns:
            pd.DataFrame: Cleaned stock data
        """
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"Missing column: {col}")
        
        # Round price columns to 2 decimal places
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].round(2)
        
        # Ensure volume is integer
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].astype(int)
        
        return data
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists.
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if symbol exists, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get some basic info
            info = ticker.info
            
            # Check if we got valid data
            if 'symbol' in info or 'shortName' in info or 'longName' in info:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return False