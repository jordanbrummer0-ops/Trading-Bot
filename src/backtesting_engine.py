#!/usr/bin/env python3
"""
Backtesting Engine Module

Implements backtesting functionality using backtrader framework.
Provides comprehensive performance analysis and trade statistics.
"""

import backtrader as bt
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
try:
    from .trading_strategy import BaseStrategy
except ImportError:
    from src.trading_strategy import BaseStrategy


class BacktraderStrategy(bt.Strategy):
    """Backtrader strategy wrapper for our custom strategies."""
    
    params = (
        ('strategy_obj', None),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize the backtrader strategy."""
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trades = []
        
        # Convert data to pandas DataFrame for our strategy
        self.df_data = None
        self.signals_generated = False
        
    def notify_order(self, order):
        """Notify when an order is executed."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notify when a trade is closed."""
        if not trade.isclosed:
            return
        
        trade_info = {
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm,
            'size': trade.size,
            'price': trade.price,
            'value': trade.value,
            'commission': trade.commission
        }
        self.trades.append(trade_info)
        
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    
    def next(self):
        """Execute strategy logic for each bar."""
        if not self.signals_generated:
            self._generate_signals()
            self.signals_generated = True
        
        current_bar = len(self) - 1
        
        if current_bar >= len(self.df_data):
            return
        
        # Check if we have a pending order
        if self.order:
            return
        
        # Get current signal
        current_signal = self.df_data.iloc[current_bar].get('signal', 0)
        
        # Check if not in market
        if not self.position:
            # Buy signal
            if current_signal == 1:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else:
            # Sell signal or exit
            if current_signal == -1:
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell()
    
    def _generate_signals(self):
        """Generate signals using our custom strategy."""
        if self.params.strategy_obj is None:
            return
        
        # Convert backtrader data to pandas DataFrame
        data_list = []
        for i in range(len(self.data)):
            data_list.append({
                'Open': self.data.open[i],
                'High': self.data.high[i],
                'Low': self.data.low[i],
                'Close': self.data.close[i],
                'Volume': self.data.volume[i]
            })
        
        self.df_data = pd.DataFrame(data_list)
        self.df_data.index = pd.date_range(start='2020-01-01', periods=len(self.df_data), freq='D')
        
        # Generate signals using our strategy
        self.df_data = self.params.strategy_obj.generate_signals(self.df_data)
    
    def log(self, txt, dt=None):
        """Log function for the strategy."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')


class BacktestingEngine:
    """Main backtesting engine using backtrader."""
    
    def __init__(self, initial_cash: float = 10000.0, commission: float = 0.001):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_cash (float): Initial cash amount
            commission (float): Commission rate (0.001 = 0.1%)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, data: pd.DataFrame, strategy: BaseStrategy, 
                    printlog: bool = False) -> Dict[str, Any]:
        """
        Run backtest using the provided data and strategy.
        
        Args:
            data (pd.DataFrame): Stock price data
            strategy (BaseStrategy): Trading strategy to test
            printlog (bool): Whether to print trade logs
        
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            # Create cerebro engine
            cerebro = bt.Cerebro()
            
            # Add strategy
            cerebro.addstrategy(BacktraderStrategy, strategy_obj=strategy, printlog=printlog)
            
            # Prepare data for backtrader
            bt_data = self._prepare_data(data)
            cerebro.adddata(bt_data)
            
            # Set initial cash
            cerebro.broker.setcash(self.initial_cash)
            
            # Set commission
            cerebro.broker.setcommission(commission=self.commission)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            self.logger.info(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
            
            # Run backtest
            results = cerebro.run()
            
            final_value = cerebro.broker.getvalue()
            self.logger.info(f'Final Portfolio Value: {final_value:.2f}')
            
            # Extract results
            backtest_results = self._extract_results(results[0], final_value)
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """
        Prepare pandas DataFrame for backtrader.
        
        Args:
            data (pd.DataFrame): Stock price data
        
        Returns:
            bt.feeds.PandasData: Backtrader data feed
        """
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create backtrader data feed
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # Use index as datetime
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=None
        )
        
        return bt_data
    
    def _extract_results(self, strategy_result, final_value: float) -> Dict[str, Any]:
        """
        Extract and format backtest results.
        
        Args:
            strategy_result: Backtrader strategy result
            final_value (float): Final portfolio value
        
        Returns:
            Dict[str, Any]: Formatted results
        """
        results = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': ((final_value - self.initial_cash) / self.initial_cash) * 100,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_trade_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'trades': []
        }
        
        try:
            # Extract trade analysis
            trade_analyzer = strategy_result.analyzers.trades.get_analysis()
            
            if 'total' in trade_analyzer and 'closed' in trade_analyzer['total']:
                results['num_trades'] = trade_analyzer['total']['closed']
                
                if results['num_trades'] > 0:
                    won_trades = trade_analyzer.get('won', {}).get('total', 0)
                    results['win_rate'] = (won_trades / results['num_trades']) * 100
                    
                    # Average trade return
                    if 'pnl' in trade_analyzer and 'net' in trade_analyzer['pnl']:
                        avg_pnl = trade_analyzer['pnl']['net']['average']
                        results['avg_trade_return'] = (avg_pnl / self.initial_cash) * 100
            
            # Extract drawdown
            drawdown_analyzer = strategy_result.analyzers.drawdown.get_analysis()
            if 'max' in drawdown_analyzer and 'drawdown' in drawdown_analyzer['max']:
                results['max_drawdown'] = drawdown_analyzer['max']['drawdown']
            
            # Extract Sharpe ratio
            sharpe_analyzer = strategy_result.analyzers.sharpe.get_analysis()
            if 'sharperatio' in sharpe_analyzer and sharpe_analyzer['sharperatio'] is not None:
                results['sharpe_ratio'] = sharpe_analyzer['sharperatio']
            
            # Extract individual trades
            if hasattr(strategy_result, 'trades'):
                results['trades'] = strategy_result.trades
            
        except Exception as e:
            self.logger.warning(f"Error extracting some results: {str(e)}")
        
        return results
    
    def run_simple_backtest(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """
        Run a simple backtest without backtrader (for quick testing).
        
        Args:
            data (pd.DataFrame): Stock price data
            strategy (BaseStrategy): Trading strategy to test
        
        Returns:
            Dict[str, Any]: Simple backtest results
        """
        try:
            # Generate signals
            data_with_signals = strategy.generate_signals(data.copy())
            
            # Calculate returns
            data_with_returns = strategy.calculate_returns(data_with_signals)
            
            # Calculate performance metrics
            total_return = (data_with_returns['cumulative_return'].iloc[-1] - 1) * 100
            buy_hold_return = (data_with_returns['buy_hold_return'].iloc[-1] - 1) * 100
            
            # Count trades
            num_trades = data_with_returns['entry'].sum()
            
            # Calculate win rate
            trade_returns = []
            position = 0
            entry_price = 0
            
            for idx, row in data_with_returns.iterrows():
                if row['entry'] and position == 0:
                    position = 1
                    entry_price = row['Close']
                elif row['exit'] and position == 1:
                    position = 0
                    trade_return = (row['Close'] - entry_price) / entry_price * 100
                    trade_returns.append(trade_return)
            
            win_rate = 0.0
            avg_trade_return = 0.0
            
            if trade_returns:
                winning_trades = [r for r in trade_returns if r > 0]
                win_rate = (len(winning_trades) / len(trade_returns)) * 100
                avg_trade_return = np.mean(trade_returns)
            
            # Calculate maximum drawdown
            cumulative = data_with_returns['cumulative_return']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            results = {
                'initial_value': self.initial_cash,
                'final_value': self.initial_cash * (1 + total_return / 100),
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'max_drawdown': abs(max_drawdown),
                'sharpe_ratio': self._calculate_sharpe_ratio(data_with_returns['strategy_return']),
                'data': data_with_returns
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running simple backtest: {str(e)}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pd.Series): Strategy returns
            risk_free_rate (float): Risk-free rate (annual)
        
        Returns:
            float: Sharpe ratio
        """
        try:
            # Convert to annual returns
            annual_return = returns.mean() * 252  # Assuming 252 trading days
            annual_volatility = returns.std() * np.sqrt(252)
            
            if annual_volatility == 0:
                return 0.0
            
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            return sharpe_ratio
            
        except Exception:
            return 0.0