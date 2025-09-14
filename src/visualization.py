#!/usr/bin/env python3
"""
Advanced Visualization Module for Trading Bot Analysis

This module provides comprehensive visualization capabilities for analyzing
trading strategies, including price charts with signal overlays, performance
metrics visualization, and comparative analysis tools.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradingVisualizer:
    """
    Advanced visualization class for trading strategy analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.colors = {
            'price': '#2E86AB',
            'ma_short': '#A23B72',
            'ma_long': '#F18F01',
            'buy': '#43AA8B',
            'sell': '#F8333C',
            'volume': '#90A959',
            'background': '#F5F5F5'
        }
    
    def plot_strategy_analysis(self, data: pd.DataFrame, signals: pd.DataFrame, 
                             symbol: str, strategy_name: str = "Moving Average",
                             save_path: Optional[str] = None) -> None:
        """
        Create comprehensive strategy analysis plot with price, signals, and indicators.
        
        Args:
            data: Stock price data with OHLCV columns
            signals: DataFrame with buy/sell signals and positions
            symbol: Stock symbol
            strategy_name: Name of the trading strategy
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'{symbol} - {strategy_name} Strategy Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Main price chart with signals
        ax1 = axes[0]
        
        # Plot price and moving averages
        ax1.plot(data.index, data['Close'], label='Close Price', 
                color=self.colors['price'], linewidth=1.5)
        
        if 'MA_Short' in data.columns:
            ax1.plot(data.index, data['MA_Short'], label='MA Short', 
                    color=self.colors['ma_short'], alpha=0.7)
        if 'MA_Long' in data.columns:
            ax1.plot(data.index, data['MA_Long'], label='MA Long', 
                    color=self.colors['ma_long'], alpha=0.7)
        
        # Plot buy and sell signals
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       color=self.colors['buy'], marker='^', s=100, 
                       label='Buy Signal', zorder=5)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       color=self.colors['sell'], marker='v', s=100, 
                       label='Sell Signal', zorder=5)
        
        ax1.set_ylabel('Price ($)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Position chart
        ax2 = axes[1]
        ax2.fill_between(signals.index, 0, signals['position'], 
                        alpha=0.3, color=self.colors['buy'], 
                        label='Position (1=Long, 0=Cash)')
        ax2.set_ylabel('Position', fontweight='bold')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Volume chart
        ax3 = axes[2]
        ax3.bar(data.index, data['Volume'], alpha=0.6, 
               color=self.colors['volume'], width=1)
        ax3.set_ylabel('Volume', fontweight='bold')
        ax3.set_xlabel('Date', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, results: Dict[str, Dict], 
                                  save_path: Optional[str] = None) -> None:
        """
        Create performance comparison charts for multiple stocks/strategies.
        
        Args:
            results: Dictionary with stock symbols as keys and performance metrics as values
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        symbols = list(results.keys())
        returns = [results[s].get('total_return', 0) for s in symbols]
        sharpe_ratios = [results[s].get('sharpe_ratio', 0) for s in symbols]
        max_drawdowns = [results[s].get('max_drawdown', 0) for s in symbols]
        num_trades = [results[s].get('num_trades', 0) for s in symbols]
        
        # Returns comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(symbols, returns, color=[self.colors['buy'] if r > 0 else self.colors['sell'] for r in returns])
        ax1.set_title('Total Returns (%)', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Sharpe ratio comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(symbols, sharpe_ratios, color=self.colors['ma_short'])
        ax2.set_title('Sharpe Ratios', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, sharpe_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Max drawdown comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(symbols, max_drawdowns, color=self.colors['sell'])
        ax3.set_title('Maximum Drawdown (%)', fontweight='bold')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, max_drawdowns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Number of trades
        ax4 = axes[1, 1]
        bars4 = ax4.bar(symbols, num_trades, color=self.colors['ma_long'])
        ax4.set_title('Number of Trades', fontweight='bold')
        ax4.set_ylabel('Trades Count')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, num_trades):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{int(value)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to: {save_path}")
        
        plt.show()
    
    def plot_parameter_optimization(self, optimization_results: pd.DataFrame,
                                  param1_name: str = "Short MA", 
                                  param2_name: str = "Long MA",
                                  metric: str = "sharpe_ratio",
                                  save_path: Optional[str] = None) -> None:
        """
        Create heatmap for parameter optimization results.
        
        Args:
            optimization_results: DataFrame with parameter combinations and results
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            metric: Metric to visualize (sharpe_ratio, total_return, max_drawdown)
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Pivot the data for heatmap
        pivot_data = optimization_results.pivot(param1_name, param2_name, metric)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0 if metric == 'total_return' else None,
                   cbar_kws={'label': metric.replace('_', ' ').title()})
        
        plt.title(f'Parameter Optimization: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(param2_name, fontweight='bold')
        plt.ylabel(param1_name, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization heatmap saved to: {save_path}")
        
        plt.show()
    
    def plot_interactive_chart(self, data: pd.DataFrame, signals: pd.DataFrame,
                             symbol: str, strategy_name: str = "Moving Average") -> None:
        """
        Create interactive plotly chart with price, signals, and indicators.
        
        Args:
            data: Stock price data with OHLCV columns
            signals: DataFrame with buy/sell signals and positions
            symbol: Stock symbol
            strategy_name: Name of the trading strategy
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Signals', 'Position', 'Volume'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Close Price',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Moving averages
        if 'MA_Short' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MA_Short'], name='MA Short',
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'MA_Long' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MA_Long'], name='MA Long',
                          line=dict(color='red', width=1)),
                row=1, col=1
            )
        
        # Buy signals
        buy_signals = signals[signals['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                          mode='markers', name='Buy Signal',
                          marker=dict(symbol='triangle-up', size=12, color='green')),
                row=1, col=1
            )
        
        # Sell signals
        sell_signals = signals[signals['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                          mode='markers', name='Sell Signal',
                          marker=dict(symbol='triangle-down', size=12, color='red')),
                row=1, col=1
            )
        
        # Position chart
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['position'], name='Position',
                      fill='tonexty', line=dict(color='lightgreen')),
            row=2, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume',
                  marker_color='lightblue'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {strategy_name} Strategy Analysis (Interactive)',
            xaxis_title='Date',
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def create_risk_analysis_chart(self, equity_curve: pd.Series, 
                                 drawdown_series: pd.Series,
                                 symbol: str,
                                 save_path: Optional[str] = None) -> None:
        """
        Create detailed risk analysis chart showing equity curve and drawdowns.
        
        Args:
            equity_curve: Portfolio value over time
            drawdown_series: Drawdown percentage over time
            symbol: Stock symbol
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Equity curve
        ax1.plot(equity_curve.index, equity_curve.values, 
                color=self.colors['price'], linewidth=2, label='Portfolio Value')
        ax1.set_title(f'{symbol} - Portfolio Performance & Risk Analysis', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax2.fill_between(drawdown_series.index, 0, drawdown_series.values, 
                        color=self.colors['sell'], alpha=0.7, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk analysis chart saved to: {save_path}")
        
        plt.show()

def create_analysis_report(symbol: str, data: pd.DataFrame, signals: pd.DataFrame,
                         performance_metrics: Dict, save_dir: str = "charts") -> None:
    """
    Create a comprehensive analysis report with multiple visualizations.
    
    Args:
        symbol: Stock symbol
        data: Stock price data
        signals: Trading signals data
        performance_metrics: Dictionary with performance metrics
        save_dir: Directory to save charts
    """
    import os
    
    # Create charts directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = TradingVisualizer()
    
    print(f"\n=== Creating Analysis Report for {symbol} ===")
    
    # 1. Strategy analysis chart
    print("📊 Generating strategy analysis chart...")
    visualizer.plot_strategy_analysis(
        data, signals, symbol, 
        save_path=f"{save_dir}/{symbol}_strategy_analysis.png"
    )
    
    # 2. Interactive chart (if in Jupyter)
    print("🔄 Generating interactive chart...")
    try:
        visualizer.plot_interactive_chart(data, signals, symbol)
    except Exception as e:
        print(f"Interactive chart not available: {e}")
    
    # 3. Risk analysis
    if 'equity_curve' in performance_metrics and 'drawdown_series' in performance_metrics:
        print("⚠️  Generating risk analysis chart...")
        visualizer.create_risk_analysis_chart(
            performance_metrics['equity_curve'],
            performance_metrics['drawdown_series'],
            symbol,
            save_path=f"{save_dir}/{symbol}_risk_analysis.png"
        )
    
    print(f"✅ Analysis report completed! Charts saved in '{save_dir}' directory.")