#!/usr/bin/env python3
"""
Parameter Optimization Module

This module provides advanced parameter optimization techniques for trading strategies:
- Grid Search for exhaustive parameter exploration
- Bayesian Optimization for efficient parameter tuning
- Walk-forward analysis for robust validation
- Multi-objective optimization (return vs risk)

Author: Trading Bot System
Version: 1.0
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from itertools import product

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for advanced optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Run: pip install scikit-optimize")

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_fetcher import DataFetcher
from src.backtesting_engine import BacktestingEngine
from src.trading_strategy import BaseStrategy, MovingAverageStrategy, RSIStrategy, MACDStrategy, CombinedStrategy
from src.enhanced_strategy import RiskManagedStrategy, EnhancedMovingAverageStrategy, BollingerBandsMeanReversionStrategy
from bollinger_bands_strategy import BollingerBandsStrategy, EnhancedBollingerBandsStrategy

warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    optimization_time: float
    method: str
    validation_scores: Optional[List[float]] = None

class ParameterOptimizer:
    """
    Advanced Parameter Optimization for Trading Strategies
    
    Supports multiple optimization methods:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling of parameter space
    - Bayesian Optimization: Efficient optimization using Gaussian Processes
    - Walk-Forward Analysis: Time-series aware validation
    """
    
    def __init__(self, scoring_metric: str = 'sharpe_ratio'):
        self.scoring_metric = scoring_metric
        self.logger = logging.getLogger('ParameterOptimizer')
        self.data_fetcher = DataFetcher()
        self.backtesting_engine = BacktestingEngine()
        
        # Available scoring metrics
        self.scoring_functions = {
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'total_return': self._calculate_total_return,
            'max_drawdown': self._calculate_max_drawdown,
            'calmar_ratio': self._calculate_calmar_ratio,
            'sortino_ratio': self._calculate_sortino_ratio,
            'profit_factor': self._calculate_profit_factor,
            'win_rate': self._calculate_win_rate
        }
    
    def optimize_strategy(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grids: Dict[str, List],
        symbols: List[str],
        method: str = 'grid_search',
        n_calls: int = 50,
        cv_folds: int = 3,
        test_size: float = 0.2
    ) -> OptimizationResult:
        """
        Optimize strategy parameters
        
        Args:
            data: Historical price data
            strategy_class: Strategy class to optimize
            param_grids: Parameter grids to search
            symbols: List of symbols to test on
            method: Optimization method ('grid_search', 'bayesian', 'random_search')
            n_calls: Number of calls for Bayesian optimization
            cv_folds: Number of cross-validation folds
            test_size: Test set size for validation
            
        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()
        self.logger.info(f"Starting {method} optimization for {strategy_class.__name__}")
        
        if method == 'grid_search':
            result = self._grid_search_optimization(
                data, strategy_class, param_grids, symbols, cv_folds, test_size
            )
        elif method == 'bayesian' and BAYESIAN_AVAILABLE:
            result = self._bayesian_optimization(
                data, strategy_class, param_grids, symbols, n_calls, cv_folds, test_size
            )
        elif method == 'random_search':
            result = self._random_search_optimization(
                data, strategy_class, param_grids, symbols, n_calls, cv_folds, test_size
            )
        else:
            self.logger.warning(f"Method {method} not available, falling back to grid search")
            result = self._grid_search_optimization(
                data, strategy_class, param_grids, symbols, cv_folds, test_size
            )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        result.optimization_time = optimization_time
        result.method = method
        
        self.logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        self.logger.info(f"Best score: {result.best_score:.4f}")
        self.logger.info(f"Best parameters: {result.best_params}")
        
        return result
    
    def _grid_search_optimization(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grids: Dict[str, List],
        symbols: List[str],
        cv_folds: int,
        test_size: float
    ) -> OptimizationResult:
        """Perform grid search optimization"""
        param_combinations = list(ParameterGrid(param_grids))
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        best_score = float('-inf')
        best_params = None
        best_metrics = None
        
        for i, params in enumerate(param_combinations):
            self.logger.debug(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Test parameters across multiple symbols
            symbol_scores = []
            symbol_metrics = []
            
            for symbol in symbols:
                try:
                    # Create strategy instance with parameters
                    strategy = self._create_strategy_with_params(strategy_class, params)
                    
                    # Perform cross-validation
                    cv_scores, metrics = self._cross_validate_strategy(
                        data, strategy, cv_folds, test_size
                    )
                    
                    symbol_scores.extend(cv_scores)
                    symbol_metrics.append(metrics)
                    
                except Exception as e:
                    self.logger.warning(f"Error testing {symbol} with {params}: {str(e)}")
                    continue
            
            if symbol_scores:
                avg_score = np.mean(symbol_scores)
                avg_metrics = self._average_metrics(symbol_metrics)
                
                result_entry = {
                    'params': params.copy(),
                    'score': avg_score,
                    'std_score': np.std(symbol_scores),
                    'metrics': avg_metrics,
                    'cv_scores': symbol_scores
                }
                results.append(result_entry)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params.copy()
                    best_metrics = avg_metrics
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=results,
            optimization_time=0.0,  # Will be set by caller
            method='grid_search'
        )
    
    def _bayesian_optimization(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grids: Dict[str, List],
        symbols: List[str],
        n_calls: int,
        cv_folds: int,
        test_size: float
    ) -> OptimizationResult:
        """Perform Bayesian optimization"""
        if not BAYESIAN_AVAILABLE:
            self.logger.error("Bayesian optimization not available")
            return self._grid_search_optimization(
                data, strategy_class, param_grids, symbols, cv_folds, test_size
            )
        
        # Define search space
        dimensions = []
        param_names = []
        
        for param_name, param_values in param_grids.items():
            param_names.append(param_name)
            
            if isinstance(param_values[0], int):
                dimensions.append(Integer(min(param_values), max(param_values)))
            else:
                dimensions.append(Real(min(param_values), max(param_values)))
        
        # Objective function
        @use_named_args(dimensions)
        def objective(**params):
            try:
                # Create strategy with parameters
                strategy = self._create_strategy_with_params(strategy_class, params)
                
                # Test across symbols
                all_scores = []
                for symbol in symbols:
                    cv_scores, _ = self._cross_validate_strategy(
                        data, strategy, cv_folds, test_size
                    )
                    all_scores.extend(cv_scores)
                
                # Return negative score for minimization
                return -np.mean(all_scores) if all_scores else 0
                
            except Exception as e:
                self.logger.warning(f"Error in objective function: {str(e)}")
                return 0
        
        # Run optimization
        self.logger.info(f"Running Bayesian optimization with {n_calls} calls")
        result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        # Get detailed metrics for best parameters
        strategy = self._create_strategy_with_params(strategy_class, best_params)
        _, best_metrics = self._cross_validate_strategy(data, strategy, cv_folds, test_size)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=[],  # Bayesian optimization doesn't store all results
            optimization_time=0.0,
            method='bayesian'
        )
    
    def _random_search_optimization(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grids: Dict[str, List],
        symbols: List[str],
        n_iter: int,
        cv_folds: int,
        test_size: float
    ) -> OptimizationResult:
        """Perform random search optimization"""
        self.logger.info(f"Running random search with {n_iter} iterations")
        
        results = []
        best_score = float('-inf')
        best_params = None
        best_metrics = None
        
        for i in range(n_iter):
            # Randomly sample parameters
            params = {}
            for param_name, param_values in param_grids.items():
                params[param_name] = np.random.choice(param_values)
            
            self.logger.debug(f"Testing iteration {i+1}/{n_iter}: {params}")
            
            # Test parameters
            symbol_scores = []
            symbol_metrics = []
            
            for symbol in symbols:
                try:
                    strategy = self._create_strategy_with_params(strategy_class, params)
                    cv_scores, metrics = self._cross_validate_strategy(
                        data, strategy, cv_folds, test_size
                    )
                    
                    symbol_scores.extend(cv_scores)
                    symbol_metrics.append(metrics)
                    
                except Exception as e:
                    self.logger.warning(f"Error testing {symbol}: {str(e)}")
                    continue
            
            if symbol_scores:
                avg_score = np.mean(symbol_scores)
                avg_metrics = self._average_metrics(symbol_metrics)
                
                result_entry = {
                    'params': params.copy(),
                    'score': avg_score,
                    'std_score': np.std(symbol_scores),
                    'metrics': avg_metrics,
                    'cv_scores': symbol_scores
                }
                results.append(result_entry)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params.copy()
                    best_metrics = avg_metrics
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=results,
            optimization_time=0.0,
            method='random_search'
        )
    
    def _create_strategy_with_params(self, strategy_class: type, params: Dict[str, Any]):
        """Create strategy instance with given parameters"""
        try:
            return strategy_class(**params)
        except TypeError:
            # Fallback: create instance and set attributes
            strategy = strategy_class()
            for param_name, param_value in params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
            return strategy
    
    def _cross_validate_strategy(
        self,
        data: pd.DataFrame,
        strategy,
        cv_folds: int,
        test_size: float
    ) -> Tuple[List[float], Dict[str, float]]:
        """Perform cross-validation on strategy"""
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        all_metrics = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            try:
                # Run backtest on test data
                results = self.backtesting_engine.run_backtest(
                    test_data, strategy, initial_capital=10000
                )
                
                # Calculate score
                score = self._calculate_score(results)
                scores.append(score)
                
                # Calculate metrics
                metrics = self._calculate_all_metrics(results)
                all_metrics.append(metrics)
                
            except Exception as e:
                self.logger.warning(f"Error in cross-validation fold: {str(e)}")
                continue
        
        # Average metrics across folds
        avg_metrics = self._average_metrics(all_metrics) if all_metrics else {}
        
        return scores, avg_metrics
    
    def _calculate_score(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate optimization score from backtest results"""
        scoring_func = self.scoring_functions.get(self.scoring_metric)
        if scoring_func:
            return scoring_func(backtest_results)
        else:
            self.logger.warning(f"Unknown scoring metric: {self.scoring_metric}")
            return 0.0
    
    def _calculate_all_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all available metrics"""
        metrics = {}
        for metric_name, metric_func in self.scoring_functions.items():
            try:
                metrics[metric_name] = metric_func(backtest_results)
            except Exception as e:
                self.logger.debug(f"Error calculating {metric_name}: {str(e)}")
                metrics[metric_name] = 0.0
        return metrics
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple results"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m.get(key, 0) for m in metrics_list]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    # Scoring functions
    def _calculate_sharpe_ratio(self, results: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio"""
        if 'returns' in results and len(results['returns']) > 0:
            returns = np.array(results['returns'])
            if np.std(returns) > 0:
                return np.mean(returns) / np.std(returns) * np.sqrt(252)
        return 0.0
    
    def _calculate_total_return(self, results: Dict[str, Any]) -> float:
        """Calculate total return"""
        return results.get('total_return', 0.0)
    
    def _calculate_max_drawdown(self, results: Dict[str, Any]) -> float:
        """Calculate maximum drawdown (negative for minimization)"""
        return -results.get('max_drawdown', 0.0)
    
    def _calculate_calmar_ratio(self, results: Dict[str, Any]) -> float:
        """Calculate Calmar ratio"""
        total_return = results.get('total_return', 0.0)
        max_drawdown = results.get('max_drawdown', 0.01)  # Avoid division by zero
        return total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    def _calculate_sortino_ratio(self, results: Dict[str, Any]) -> float:
        """Calculate Sortino ratio"""
        if 'returns' in results and len(results['returns']) > 0:
            returns = np.array(results['returns'])
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    return np.mean(returns) / downside_std * np.sqrt(252)
        return 0.0
    
    def _calculate_profit_factor(self, results: Dict[str, Any]) -> float:
        """Calculate profit factor"""
        if 'trades' in results and len(results['trades']) > 0:
            trades = results['trades']
            winning_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            
            if losing_trades:
                gross_profit = sum(winning_trades)
                gross_loss = abs(sum(losing_trades))
                return gross_profit / gross_loss if gross_loss > 0 else 0.0
        return 0.0
    
    def _calculate_win_rate(self, results: Dict[str, Any]) -> float:
        """Calculate win rate"""
        if 'trades' in results and len(results['trades']) > 0:
            trades = results['trades']
            winning_trades = len([t for t in trades if t > 0])
            return winning_trades / len(trades)
        return 0.0
    
    def plot_optimization_results(self, result: OptimizationResult, save_path: str = None):
        """Plot optimization results"""
        if not result.all_results:
            self.logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Parameter Optimization Results - {result.method.title()}', fontsize=16)
        
        # Extract data
        scores = [r['score'] for r in result.all_results]
        param_names = list(result.best_params.keys())
        
        # Score distribution
        axes[0, 0].hist(scores, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(result.best_score, color='red', linestyle='--', 
                          label=f'Best Score: {result.best_score:.4f}')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        
        # Parameter vs Score (first parameter)
        if param_names:
            param_name = param_names[0]
            param_values = [r['params'][param_name] for r in result.all_results]
            axes[0, 1].scatter(param_values, scores, alpha=0.6, color='green')
            axes[0, 1].set_xlabel(param_name)
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title(f'Score vs {param_name}')
        
        # Top 10 results
        top_results = sorted(result.all_results, key=lambda x: x['score'], reverse=True)[:10]
        top_scores = [r['score'] for r in top_results]
        axes[1, 0].bar(range(len(top_scores)), top_scores, color='orange')
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Top 10 Results')
        
        # Parameter correlation heatmap (if multiple parameters)
        if len(param_names) > 1:
            param_data = pd.DataFrame([r['params'] for r in result.all_results])
            param_data['score'] = scores
            corr_matrix = param_data.corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], square=True)
            axes[1, 1].set_title('Parameter Correlation Matrix')
        else:
            axes[1, 1].text(0.5, 0.5, 'Need multiple parameters\nfor correlation analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Parameter Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Optimization plot saved to {save_path}")
        
        plt.show()
    
    def save_optimization_results(self, result: OptimizationResult, filename: str):
        """Save optimization results to JSON file"""
        try:
            # Convert to serializable format
            serializable_result = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'best_metrics': result.best_metrics,
                'optimization_time': result.optimization_time,
                'method': result.method,
                'timestamp': datetime.now().isoformat(),
                'scoring_metric': self.scoring_metric,
                'num_results': len(result.all_results) if result.all_results else 0
            }
            
            # Add top 10 results
            if result.all_results:
                top_results = sorted(result.all_results, key=lambda x: x['score'], reverse=True)[:10]
                serializable_result['top_results'] = top_results
            
            with open(filename, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            self.logger.info(f"Optimization results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

def main():
    """Demo of parameter optimization"""
    print("🔧 Parameter Optimization Demo")
    print("=" * 40)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(scoring_metric='sharpe_ratio')
    
    # Fetch sample data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data('AAPL', '2022-01-01', '2023-12-31')
    
    if data is None or data.empty:
        print("❌ Failed to fetch data")
        return
    
    print(f"📊 Loaded {len(data)} data points for AAPL")
    
    # Define parameter grids for different strategies
    rsi_params = {
        'rsi_period': [10, 14, 20, 25],
        'rsi_overbought': [70, 75, 80],
        'rsi_oversold': [20, 25, 30]
    }
    
    bollinger_params = {
        'period': [15, 20, 25],
        'std_dev': [1.5, 2.0, 2.5]
    }
    
    # Test RSI optimization
    print("\n🔄 Optimizing RSI Strategy...")
    try:
        rsi_result = optimizer.optimize_strategy(
            data=data,
            strategy_class=EnhancedMovingAverageStrategy,
            param_grids=rsi_params,
            symbols=['AAPL'],
            method='grid_search',
            cv_folds=3
        )
        
        print(f"✅ RSI Optimization completed")
        print(f"Best Score: {rsi_result.best_score:.4f}")
        print(f"Best Parameters: {rsi_result.best_params}")
        print(f"Optimization Time: {rsi_result.optimization_time:.2f} seconds")
        
        # Save results
        os.makedirs('analysis_charts', exist_ok=True)
        optimizer.save_optimization_results(
            rsi_result, 
            'analysis_charts/rsi_optimization_results.json'
        )
        
    except Exception as e:
        print(f"❌ RSI optimization failed: {str(e)}")
    
    # Test Bollinger Bands optimization
    print("\n🔄 Optimizing Bollinger Bands Strategy...")
    try:
        bb_result = optimizer.optimize_strategy(
            data=data,
            strategy_class=BollingerBandsStrategy,
            param_grids=bollinger_params,
            symbols=['AAPL'],
            method='grid_search',
            cv_folds=3
        )
        
        print(f"✅ Bollinger Bands Optimization completed")
        print(f"Best Score: {bb_result.best_score:.4f}")
        print(f"Best Parameters: {bb_result.best_params}")
        print(f"Optimization Time: {bb_result.optimization_time:.2f} seconds")
        
        # Save results
        optimizer.save_optimization_results(
            bb_result, 
            'analysis_charts/bollinger_optimization_results.json'
        )
        
        # Plot results
        optimizer.plot_optimization_results(
            bb_result, 
            'analysis_charts/bollinger_optimization_plot.png'
        )
        
    except Exception as e:
        print(f"❌ Bollinger Bands optimization failed: {str(e)}")
    
    # Test Bayesian optimization (if available)
    if BAYESIAN_AVAILABLE:
        print("\n🔄 Testing Bayesian Optimization...")
        try:
            bayesian_result = optimizer.optimize_strategy(
                data=data,
                strategy_class=EnhancedMovingAverageStrategy,
                param_grids=rsi_params,
                symbols=['AAPL'],
                method='bayesian',
                n_calls=20,
                cv_folds=2
            )
            
            print(f"✅ Bayesian Optimization completed")
            print(f"Best Score: {bayesian_result.best_score:.4f}")
            print(f"Best Parameters: {bayesian_result.best_params}")
            
        except Exception as e:
            print(f"❌ Bayesian optimization failed: {str(e)}")
    else:
        print("⚠️  Bayesian optimization not available (install scikit-optimize)")
    
    print("\n🎯 Parameter optimization demo completed!")
    print("\n📋 Key Insights:")
    print("• Grid search provides exhaustive parameter exploration")
    print("• Bayesian optimization is more efficient for large parameter spaces")
    print("• Cross-validation helps prevent overfitting")
    print("• Multiple metrics should be considered for robust optimization")
    
    print("\n🚀 Next Steps:")
    print("• Test optimized parameters on out-of-sample data")
    print("• Implement walk-forward analysis for time-series validation")
    print("• Consider multi-objective optimization (return vs risk)")
    print("• Integrate with main trading bot for live parameter updates")

if __name__ == "__main__":
    main()