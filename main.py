#!/usr/bin/env python3
"""
Trading Bot Main Engine

This is the core brain of the trading bot that orchestrates:
- Asset selection and data management
- Strategy loading and execution
- Confirmation indicators
- Risk management
- Parameter optimization
- Paper trading integration
- Logging and monitoring

Author: Trading Bot System
Version: 1.0
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_fetcher import DataFetcher
from src.trading_strategy import BaseStrategy, MovingAverageStrategy, RSIStrategy, MACDStrategy, CombinedStrategy
from src.enhanced_strategy import RiskManagedStrategy, EnhancedMovingAverageStrategy, BollingerBandsMeanReversionStrategy
from src.backtesting_engine import BacktestingEngine
from src.visualization import Visualizer
from config import Config

# Import our analysis modules
from risk_management_analysis import RiskManager
from confirmation_indicators import ConfirmationIndicators
from bollinger_bands_strategy import BollingerBandsStrategy, EnhancedBollingerBandsStrategy
from parameter_optimization import ParameterOptimizer

warnings.filterwarnings('ignore')

class AssetType(Enum):
    """Supported asset types"""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"

class StrategyType(Enum):
    """Available trading strategies"""
    MOVING_AVERAGE = "moving_average"
    RSI = "rsi"
    BOLLINGER_BANDS = "bollinger_bands"
    ENHANCED_BOLLINGER = "enhanced_bollinger"
    MACD = "macd"

@dataclass
class TradingConfig:
    """Configuration for trading bot"""
    asset_symbol: str
    asset_type: AssetType
    strategy_type: StrategyType
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    use_confirmation: bool = True
    use_risk_management: bool = True
    optimize_parameters: bool = False
    paper_trading: bool = False
    
class TradingBotEngine:
    """
    Main Trading Bot Engine - The Brain of the System
    
    This class orchestrates all components of the trading bot:
    - Data fetching and management
    - Strategy selection and execution
    - Risk management
    - Parameter optimization
    - Paper trading integration
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.backtesting_engine = BacktestingEngine()
        self.visualizer = Visualizer()
        self.risk_manager = RiskManager()
        self.confirmation_indicators = ConfirmationIndicators()
        self.parameter_optimizer = ParameterOptimizer()
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.strategy: Optional[BaseStrategy] = None
        self.results: Dict[str, Any] = {}
        
        self.logger.info(f"Trading Bot Engine initialized for {config.asset_symbol}")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler('trading_bot_main.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_asset_data(self) -> bool:
        """
        Load and prepare asset data
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            self.logger.info(f"Loading data for {self.config.asset_symbol}")
            
            # Determine the correct symbol format based on asset type
            symbol = self._format_symbol(self.config.asset_symbol, self.config.asset_type)
            
            # Fetch data
            self.data = self.data_fetcher.fetch_data(
                symbol=symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if self.data is None or self.data.empty:
                self.logger.error(f"Failed to load data for {symbol}")
                return False
                
            self.logger.info(f"Successfully loaded {len(self.data)} data points")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading asset data: {str(e)}")
            return False
    
    def _format_symbol(self, symbol: str, asset_type: AssetType) -> str:
        """Format symbol based on asset type"""
        if asset_type == AssetType.CRYPTO:
            return f"{symbol}-USD" if not symbol.endswith('-USD') else symbol
        elif asset_type == AssetType.FOREX:
            return f"{symbol}=X" if not symbol.endswith('=X') else symbol
        else:
            return symbol
    
    def load_strategy(self) -> bool:
        """
        Load and initialize the selected trading strategy
        
        Returns:
            bool: True if strategy loaded successfully
        """
        try:
            self.logger.info(f"Loading strategy: {self.config.strategy_type.value}")
            
            if self.config.strategy_type == StrategyType.MOVING_AVERAGE:
                self.strategy = MovingAverageStrategy()
            elif self.config.strategy_type == StrategyType.RSI:
                self.strategy = EnhancedMovingAverageStrategy()
            elif self.config.strategy_type == StrategyType.BOLLINGER_BANDS:
                self.strategy = BollingerBandsStrategy()
            elif self.config.strategy_type == StrategyType.ENHANCED_BOLLINGER:
                self.strategy = EnhancedBollingerBandsStrategy()
            else:
                self.logger.error(f"Unsupported strategy type: {self.config.strategy_type}")
                return False
                
            self.logger.info(f"Strategy {self.config.strategy_type.value} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading strategy: {str(e)}")
            return False
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Grid Search
        
        Returns:
            Dict containing optimal parameters and performance metrics
        """
        if not self.config.optimize_parameters:
            self.logger.info("Parameter optimization disabled")
            return {}
            
        try:
            self.logger.info("Starting parameter optimization...")
            
            # Define parameter grids based on strategy type
            param_grids = self._get_parameter_grids()
            
            if not param_grids:
                self.logger.warning("No parameter grid defined for this strategy")
                return {}
            
            # Run optimization
            optimization_results = self.parameter_optimizer.optimize_strategy(
                data=self.data,
                strategy_class=type(self.strategy),
                param_grids=param_grids,
                symbols=[self.config.asset_symbol]
            )
            
            self.logger.info("Parameter optimization completed")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error during parameter optimization: {str(e)}")
            return {}
    
    def _get_parameter_grids(self) -> Dict[str, List]:
        """Get parameter grids for different strategies"""
        if self.config.strategy_type == StrategyType.RSI:
            return {
                'rsi_period': [10, 14, 20, 25],
                'rsi_overbought': [65, 70, 75, 80],
                'rsi_oversold': [20, 25, 30, 35]
            }
        elif self.config.strategy_type == StrategyType.MOVING_AVERAGE:
            return {
                'short_window': [10, 15, 20],
                'long_window': [30, 50, 100]
            }
        elif self.config.strategy_type in [StrategyType.BOLLINGER_BANDS, StrategyType.ENHANCED_BOLLINGER]:
            return {
                'period': [15, 20, 25],
                'std_dev': [1.5, 2.0, 2.5]
            }
        else:
            return {}
    
    def apply_confirmation_indicators(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply confirmation indicators to filter signals
        
        Args:
            signals: DataFrame with trading signals
            
        Returns:
            DataFrame with confirmed signals
        """
        if not self.config.use_confirmation:
            self.logger.info("Confirmation indicators disabled")
            return signals
            
        try:
            self.logger.info("Applying confirmation indicators...")
            
            # Apply volume confirmation
            confirmed_signals = self.confirmation_indicators.apply_volume_confirmation(
                self.data, signals
            )
            
            # Apply trend confirmation
            confirmed_signals = self.confirmation_indicators.apply_trend_confirmation(
                self.data, confirmed_signals
            )
            
            # Calculate signal reduction
            original_signals = signals['signal'].abs().sum()
            confirmed_signal_count = confirmed_signals['signal'].abs().sum()
            reduction_pct = ((original_signals - confirmed_signal_count) / original_signals * 100) if original_signals > 0 else 0
            
            self.logger.info(f"Signal reduction: {reduction_pct:.1f}%")
            return confirmed_signals
            
        except Exception as e:
            self.logger.error(f"Error applying confirmation indicators: {str(e)}")
            return signals
    
    def apply_risk_management(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk management rules
        
        Args:
            signals: DataFrame with trading signals
            
        Returns:
            DataFrame with risk-managed signals
        """
        if not self.config.use_risk_management:
            self.logger.info("Risk management disabled")
            return signals
            
        try:
            self.logger.info("Applying risk management rules...")
            
            # Apply stop-loss and take-profit
            risk_managed_signals = self.risk_manager.apply_stop_loss_take_profit(
                self.data, signals
            )
            
            # Apply position sizing
            risk_managed_signals = self.risk_manager.apply_position_sizing(
                self.data, risk_managed_signals
            )
            
            self.logger.info("Risk management applied successfully")
            return risk_managed_signals
            
        except Exception as e:
            self.logger.error(f"Error applying risk management: {str(e)}")
            return signals
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run complete backtesting with all components
        
        Returns:
            Dictionary containing backtest results
        """
        try:
            self.logger.info("Starting comprehensive backtest...")
            
            # Generate base signals
            if hasattr(self.strategy, 'generate_signals'):
                signals = self.strategy.generate_signals(self.data)
            else:
                # Fallback for basic strategies
                signals = self.backtesting_engine.run_backtest(
                    self.data, self.strategy, self.config.initial_capital
                )
            
            # Apply confirmation indicators
            confirmed_signals = self.apply_confirmation_indicators(signals)
            
            # Apply risk management
            final_signals = self.apply_risk_management(confirmed_signals)
            
            # Run final backtest
            results = self.backtesting_engine.run_backtest(
                self.data, final_signals, self.config.initial_capital
            )
            
            # Store results
            self.results = {
                'backtest_results': results,
                'original_signals': signals,
                'confirmed_signals': confirmed_signals,
                'final_signals': final_signals,
                'asset_symbol': self.config.asset_symbol,
                'strategy_type': self.config.strategy_type.value,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            return {}
    
    def generate_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            String containing formatted report
        """
        if not self.results:
            return "No results available. Run backtest first."
            
        try:
            report = []
            report.append("=" * 60)
            report.append("TRADING BOT PERFORMANCE REPORT")
            report.append("=" * 60)
            report.append(f"Asset: {self.config.asset_symbol}")
            report.append(f"Strategy: {self.config.strategy_type.value}")
            report.append(f"Period: {self.config.start_date} to {self.config.end_date}")
            report.append(f"Initial Capital: ${self.config.initial_capital:,.2f}")
            report.append("")
            
            # Performance metrics
            if 'backtest_results' in self.results:
                results = self.results['backtest_results']
                if isinstance(results, dict):
                    report.append("PERFORMANCE METRICS:")
                    report.append("-" * 30)
                    
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            if 'return' in key.lower() or 'ratio' in key.lower():
                                report.append(f"{key}: {value:.4f}")
                            else:
                                report.append(f"{key}: {value:.2f}")
                        else:
                            report.append(f"{key}: {value}")
            
            report.append("")
            report.append("CONFIGURATION:")
            report.append("-" * 20)
            report.append(f"Confirmation Indicators: {'Enabled' if self.config.use_confirmation else 'Disabled'}")
            report.append(f"Risk Management: {'Enabled' if self.config.use_risk_management else 'Disabled'}")
            report.append(f"Parameter Optimization: {'Enabled' if self.config.optimize_parameters else 'Disabled'}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def save_results(self, filename: Optional[str] = None) -> bool:
        """
        Save results to file
        
        Args:
            filename: Optional custom filename
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trading_bot_results_{self.config.asset_symbol}_{timestamp}.json"
            
            # Ensure analysis_charts directory exists
            os.makedirs('analysis_charts', exist_ok=True)
            filepath = os.path.join('analysis_charts', filename)
            
            # Convert results to JSON-serializable format
            serializable_results = self._make_serializable(self.results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Results saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete trading bot analysis pipeline
        
        Returns:
            bool: True if analysis completed successfully
        """
        try:
            self.logger.info("Starting complete trading bot analysis...")
            
            # Step 1: Load asset data
            if not self.load_asset_data():
                return False
            
            # Step 2: Load strategy
            if not self.load_strategy():
                return False
            
            # Step 3: Optimize parameters (if enabled)
            if self.config.optimize_parameters:
                optimization_results = self.optimize_parameters()
                if optimization_results:
                    self.results['optimization'] = optimization_results
            
            # Step 4: Run backtest
            backtest_results = self.run_backtest()
            if not backtest_results:
                return False
            
            # Step 5: Generate visualizations
            self._generate_visualizations()
            
            # Step 6: Save results
            self.save_results()
            
            # Step 7: Print report
            print(self.generate_report())
            
            self.logger.info("Complete analysis finished successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {str(e)}")
            return False
    
    def _generate_visualizations(self):
        """Generate and save visualizations"""
        try:
            if not self.results or 'final_signals' not in self.results:
                return
            
            # Create visualization
            fig = self.visualizer.plot_strategy_performance(
                self.data,
                self.results['final_signals'],
                f"{self.config.asset_symbol} - {self.config.strategy_type.value}"
            )
            
            # Save plot
            os.makedirs('analysis_charts', exist_ok=True)
            filename = f"{self.config.asset_symbol}_{self.config.strategy_type.value}_main_analysis.png"
            filepath = os.path.join('analysis_charts', filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"Visualization saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")

def create_trading_config(
    asset_symbol: str,
    asset_type: str = "stock",
    strategy_type: str = "rsi",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 10000.0,
    use_confirmation: bool = True,
    use_risk_management: bool = True,
    optimize_parameters: bool = False
) -> TradingConfig:
    """
    Helper function to create trading configuration
    
    Args:
        asset_symbol: Symbol to trade (e.g., 'TSLA', 'BTC-USD')
        asset_type: Type of asset ('stock', 'crypto', 'etf', 'forex')
        strategy_type: Strategy to use ('moving_average', 'rsi', 'bollinger_bands')
        start_date: Start date for analysis
        end_date: End date for analysis
        initial_capital: Starting capital amount
        use_confirmation: Whether to use confirmation indicators
        use_risk_management: Whether to use risk management
        optimize_parameters: Whether to optimize strategy parameters
        
    Returns:
        TradingConfig object
    """
    return TradingConfig(
        asset_symbol=asset_symbol,
        asset_type=AssetType(asset_type),
        strategy_type=StrategyType(strategy_type),
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        use_confirmation=use_confirmation,
        use_risk_management=use_risk_management,
        optimize_parameters=optimize_parameters
    )

def main():
    """
    Main function demonstrating the Trading Bot Engine
    """
    print("🤖 Trading Bot Engine - Main Analysis")
    print("=" * 50)
    
    # Example configurations for different scenarios
    configs = [
        # Tesla with RSI strategy and full optimization
        create_trading_config(
            asset_symbol="TSLA",
            asset_type="stock",
            strategy_type="rsi",
            start_date="2022-01-01",
            end_date="2023-12-31",
            use_confirmation=True,
            use_risk_management=True,
            optimize_parameters=True
        ),
        
        # Bitcoin with Bollinger Bands
        create_trading_config(
            asset_symbol="BTC-USD",
            asset_type="crypto",
            strategy_type="bollinger_bands",
            start_date="2022-01-01",
            end_date="2023-12-31",
            use_confirmation=True,
            use_risk_management=True,
            optimize_parameters=False
        ),
        
        # Apple with Moving Average
        create_trading_config(
            asset_symbol="AAPL",
            asset_type="stock",
            strategy_type="moving_average",
            start_date="2022-01-01",
            end_date="2023-12-31",
            use_confirmation=False,
            use_risk_management=True,
            optimize_parameters=False
        )
    ]
    
    # Run analysis for each configuration
    for i, config in enumerate(configs, 1):
        print(f"\n🔄 Running Analysis {i}/3: {config.asset_symbol} with {config.strategy_type.value}")
        print("-" * 60)
        
        # Create and run trading bot engine
        engine = TradingBotEngine(config)
        success = engine.run_complete_analysis()
        
        if success:
            print(f"✅ Analysis {i} completed successfully")
        else:
            print(f"❌ Analysis {i} failed")
        
        print("-" * 60)
    
    print("\n🎯 All analyses completed!")
    print("\n📊 Key Takeaways:")
    print("• Check 'analysis_charts/' for detailed visualizations")
    print("• Review 'trading_bot_main.log' for detailed logs")
    print("• Results saved as JSON files for further analysis")
    print("\n🚀 Next Steps:")
    print("• Fine-tune parameters based on optimization results")
    print("• Test with paper trading integration")
    print("• Deploy to cloud for live trading")

if __name__ == "__main__":
    main()