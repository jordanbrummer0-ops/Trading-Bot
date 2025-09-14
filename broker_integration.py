#!/usr/bin/env python3
"""
Broker API Integration Framework

This module provides a unified interface for connecting to different brokers
for paper trading and live trading. Currently supports:
- Alpaca Markets
- Interactive Brokers (IBKR)
- Mock broker for testing

Author: Trading Bot System
Version: 1.0
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

# Optional imports for broker APIs
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca Trade API not installed. Run: pip install alpaca-trade-api")

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("Warning: IB-insync not installed. Run: pip install ib-insync")

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data structure"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # "long" or "short"

@dataclass
class Account:
    """Account data structure"""
    buying_power: float
    cash: float
    portfolio_value: float
    day_trade_count: int
    positions: List[Position]

class BrokerInterface(ABC):
    """Abstract base class for broker integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.is_connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker"""
        pass
    
    @abstractmethod
    def get_account(self) -> Optional[Account]:
        """Get account information"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> Optional[str]:
        """Place an order and return order ID"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current market data for a symbol"""
        pass

class AlpacaBroker(BrokerInterface):
    """Alpaca Markets broker integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api = None
        
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca Trade API not available. Install with: pip install alpaca-trade-api")
    
    def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            # Get credentials from config or environment
            api_key = self.config.get('api_key') or os.getenv('ALPACA_API_KEY')
            secret_key = self.config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')
            base_url = self.config.get('base_url', 'https://paper-api.alpaca.markets')  # Paper trading by default
            
            if not api_key or not secret_key:
                self.logger.error("Alpaca API credentials not found")
                return False
            
            self.api = tradeapi.REST(
                api_key,
                secret_key,
                base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.is_connected = True
            self.logger.info(f"Connected to Alpaca. Account status: {account.status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Alpaca"""
        self.is_connected = False
        self.api = None
        self.logger.info("Disconnected from Alpaca")
        return True
    
    def get_account(self) -> Optional[Account]:
        """Get Alpaca account information"""
        if not self.is_connected:
            return None
            
        try:
            account = self.api.get_account()
            positions = self.get_positions()
            
            return Account(
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                day_trade_count=int(account.daytrade_count),
                positions=positions
            )
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self) -> List[Position]:
        """Get current positions from Alpaca"""
        if not self.is_connected:
            return []
            
        try:
            positions = self.api.list_positions()
            result = []
            
            for pos in positions:
                result.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_price=float(pos.avg_cost),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    side="long" if float(pos.qty) > 0 else "short"
                ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def place_order(self, order: Order) -> Optional[str]:
        """Place order with Alpaca"""
        if not self.is_connected:
            return None
            
        try:
            # Convert our order to Alpaca format
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
                type=order.order_type.value,
                time_in_force=order.time_in_force,
                limit_price=order.price if order.order_type == OrderType.LIMIT else None,
                stop_price=order.stop_price if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] else None
            )
            
            self.logger.info(f"Order placed: {alpaca_order.id} for {order.symbol}")
            return alpaca_order.id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca"""
        if not self.is_connected:
            return False
            
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from Alpaca"""
        if not self.is_connected:
            return None
            
        try:
            alpaca_order = self.api.get_order(order_id)
            
            # Convert status
            status_map = {
                'new': OrderStatus.PENDING,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'filled': OrderStatus.FILLED,
                'done_for_day': OrderStatus.CANCELLED,
                'canceled': OrderStatus.CANCELLED,
                'expired': OrderStatus.CANCELLED,
                'replaced': OrderStatus.CANCELLED,
                'pending_cancel': OrderStatus.PENDING,
                'pending_replace': OrderStatus.PENDING,
                'accepted': OrderStatus.PENDING,
                'pending_new': OrderStatus.PENDING,
                'accepted_for_bidding': OrderStatus.PENDING,
                'stopped': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'suspended': OrderStatus.CANCELLED,
                'calculated': OrderStatus.PENDING
            }
            
            return Order(
                symbol=alpaca_order.symbol,
                side=OrderSide(alpaca_order.side),
                quantity=float(alpaca_order.qty),
                order_type=OrderType(alpaca_order.order_type),
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                time_in_force=alpaca_order.time_in_force,
                order_id=alpaca_order.id,
                status=status_map.get(alpaca_order.status, OrderStatus.PENDING),
                filled_quantity=float(alpaca_order.filled_qty or 0),
                filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                timestamp=alpaca_order.created_at
            )
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")
            return None
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current market data from Alpaca"""
        if not self.is_connected:
            return None
            
        try:
            # Get latest trade
            trade = self.api.get_latest_trade(symbol)
            
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            
            return {
                'price': float(trade.price),
                'bid': float(quote.bidprice),
                'ask': float(quote.askprice),
                'bid_size': float(quote.bidsize),
                'ask_size': float(quote.asksize),
                'timestamp': trade.timestamp.timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None

class MockBroker(BrokerInterface):
    """Mock broker for testing and simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.account_balance = config.get('initial_balance', 100000.0)
        self.order_counter = 0
        
    def connect(self) -> bool:
        """Mock connection"""
        self.is_connected = True
        self.logger.info("Connected to Mock Broker")
        return True
    
    def disconnect(self) -> bool:
        """Mock disconnection"""
        self.is_connected = False
        self.logger.info("Disconnected from Mock Broker")
        return True
    
    def get_account(self) -> Optional[Account]:
        """Get mock account information"""
        if not self.is_connected:
            return None
            
        positions = list(self.positions.values())
        portfolio_value = self.account_balance + sum(pos.market_value for pos in positions)
        
        return Account(
            buying_power=self.account_balance,
            cash=self.account_balance,
            portfolio_value=portfolio_value,
            day_trade_count=0,
            positions=positions
        )
    
    def get_positions(self) -> List[Position]:
        """Get mock positions"""
        return list(self.positions.values())
    
    def place_order(self, order: Order) -> Optional[str]:
        """Place mock order"""
        if not self.is_connected:
            return None
            
        self.order_counter += 1
        order_id = f"MOCK_{self.order_counter:06d}"
        
        order.order_id = order_id
        order.timestamp = datetime.now()
        
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = self._get_mock_price(order.symbol)
            
            # Update positions
            self._update_position(order)
        
        self.orders[order_id] = order
        self.logger.info(f"Mock order placed: {order_id} for {order.symbol}")
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel mock order"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            self.logger.info(f"Mock order cancelled: {order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get mock order status"""
        return self.orders.get(order_id)
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get mock market data"""
        price = self._get_mock_price(symbol)
        return {
            'price': price,
            'bid': price - 0.01,
            'ask': price + 0.01,
            'bid_size': 100,
            'ask_size': 100,
            'timestamp': time.time()
        }
    
    def _get_mock_price(self, symbol: str) -> float:
        """Generate mock price based on symbol"""
        # Simple hash-based price generation for consistency
        base_price = hash(symbol) % 1000 + 50
        # Add some randomness
        import random
        random.seed(int(time.time()) + hash(symbol))
        return base_price + random.uniform(-5, 5)
    
    def _update_position(self, order: Order):
        """Update position after order fill"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0,
                market_value=0,
                unrealized_pnl=0,
                side="long"
            )
        
        pos = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            new_quantity = pos.quantity + order.filled_quantity
            if new_quantity != 0:
                pos.avg_price = ((pos.avg_price * pos.quantity) + 
                               (order.filled_price * order.filled_quantity)) / new_quantity
            pos.quantity = new_quantity
        else:  # SELL
            pos.quantity -= order.filled_quantity
        
        # Update market value and PnL
        current_price = self._get_mock_price(symbol)
        pos.market_value = pos.quantity * current_price
        pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
        pos.side = "long" if pos.quantity > 0 else "short" if pos.quantity < 0 else "flat"
        
        # Remove position if quantity is zero
        if pos.quantity == 0:
            del self.positions[symbol]

class BrokerManager:
    """Manager class for broker operations"""
    
    def __init__(self, broker_type: str = "mock", config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.broker = self._create_broker(broker_type)
        self.logger = logging.getLogger('BrokerManager')
        
    def _create_broker(self, broker_type: str) -> BrokerInterface:
        """Create broker instance based on type"""
        if broker_type.lower() == "alpaca":
            return AlpacaBroker(self.config)
        elif broker_type.lower() == "mock":
            return MockBroker(self.config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
    
    def connect(self) -> bool:
        """Connect to broker"""
        return self.broker.connect()
    
    def disconnect(self) -> bool:
        """Disconnect from broker"""
        return self.broker.disconnect()
    
    def execute_signal(self, symbol: str, signal: int, quantity: float, 
                      order_type: OrderType = OrderType.MARKET) -> Optional[str]:
        """Execute trading signal
        
        Args:
            symbol: Trading symbol
            signal: 1 for buy, -1 for sell, 0 for no action
            quantity: Number of shares/units
            order_type: Type of order to place
            
        Returns:
            Order ID if successful, None otherwise
        """
        if signal == 0:
            return None
            
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type=order_type
        )
        
        order_id = self.broker.place_order(order)
        
        if order_id:
            self.logger.info(f"Signal executed: {signal} for {symbol}, Order ID: {order_id}")
        else:
            self.logger.error(f"Failed to execute signal for {symbol}")
            
        return order_id
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        account = self.broker.get_account()
        if not account:
            return {}
            
        return {
            'total_value': account.portfolio_value,
            'cash': account.cash,
            'buying_power': account.buying_power,
            'positions_count': len(account.positions),
            'day_trades': account.day_trade_count,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'value': pos.market_value,
                    'pnl': pos.unrealized_pnl
                }
                for pos in account.positions
            ]
        }

def create_broker_config(broker_type: str = "mock") -> Dict[str, Any]:
    """
    Create broker configuration
    
    Args:
        broker_type: Type of broker ('alpaca', 'mock')
        
    Returns:
        Configuration dictionary
    """
    if broker_type.lower() == "alpaca":
        return {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'base_url': 'https://paper-api.alpaca.markets'  # Paper trading
        }
    elif broker_type.lower() == "mock":
        return {
            'initial_balance': 100000.0
        }
    else:
        return {}

def main():
    """Demo of broker integration"""
    print("🏦 Broker Integration Demo")
    print("=" * 40)
    
    # Test with mock broker
    print("\n📊 Testing Mock Broker...")
    mock_config = create_broker_config("mock")
    mock_manager = BrokerManager("mock", mock_config)
    
    if mock_manager.connect():
        print("✅ Connected to Mock Broker")
        
        # Get account info
        summary = mock_manager.get_portfolio_summary()
        print(f"Portfolio Value: ${summary['total_value']:,.2f}")
        
        # Execute some test trades
        print("\n🔄 Executing test trades...")
        
        # Buy AAPL
        order_id1 = mock_manager.execute_signal("AAPL", 1, 100)
        print(f"Buy order placed: {order_id1}")
        
        # Buy TSLA
        order_id2 = mock_manager.execute_signal("TSLA", 1, 50)
        print(f"Buy order placed: {order_id2}")
        
        # Check portfolio after trades
        summary = mock_manager.get_portfolio_summary()
        print(f"\n📈 Portfolio after trades:")
        print(f"Total Value: ${summary['total_value']:,.2f}")
        print(f"Positions: {summary['positions_count']}")
        
        for pos in summary['positions']:
            print(f"  {pos['symbol']}: {pos['quantity']} shares, Value: ${pos['value']:,.2f}")
        
        mock_manager.disconnect()
    
    # Test Alpaca connection (if credentials available)
    print("\n🦙 Testing Alpaca Connection...")
    if os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_SECRET_KEY'):
        try:
            alpaca_config = create_broker_config("alpaca")
            alpaca_manager = BrokerManager("alpaca", alpaca_config)
            
            if alpaca_manager.connect():
                print("✅ Connected to Alpaca Paper Trading")
                
                summary = alpaca_manager.get_portfolio_summary()
                print(f"Paper Trading Portfolio: ${summary['total_value']:,.2f}")
                
                alpaca_manager.disconnect()
            else:
                print("❌ Failed to connect to Alpaca")
                
        except Exception as e:
            print(f"❌ Alpaca connection error: {str(e)}")
    else:
        print("⚠️  Alpaca credentials not found in environment variables")
        print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY to test Alpaca integration")
    
    print("\n🎯 Broker integration demo completed!")
    print("\n📋 Next Steps:")
    print("• Set up Alpaca paper trading account")
    print("• Configure environment variables for API keys")
    print("• Integrate with main trading bot engine")
    print("• Test with live market data")

if __name__ == "__main__":
    main()