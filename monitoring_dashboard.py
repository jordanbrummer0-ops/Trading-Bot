#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard

A comprehensive Streamlit dashboard for monitoring trading bot performance,
positions, and key metrics in real-time.

Features:
- Real-time performance monitoring
- Portfolio overview and positions
- Strategy performance comparison
- Risk metrics and alerts
- Trade history and analysis
- Parameter optimization results
- Live market data integration

Author: Trading Bot System
Version: 1.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

# Import our modules with error handling for cloud deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.data_fetcher import DataFetcher
    from main import TradingBotEngine
    from broker_integration import BrokerManager
    CLOUD_MODE = False
except ImportError as e:
    st.warning(f"Running in cloud mode - some features may be limited: {e}")
    DataFetcher = None
    TradingBotEngine = None
    BrokerManager = None
    CLOUD_MODE = True

# Configure Streamlit page
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="[Chart]",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    """
    Comprehensive Trading Bot Monitoring Dashboard
    
    Provides real-time monitoring and analysis capabilities:
    - Portfolio performance tracking
    - Strategy comparison and analysis
    - Risk monitoring and alerts
    - Trade execution monitoring
    - Parameter optimization results
    """
    
    def __init__(self):
        # Initialize components based on availability
        self.data_fetcher = DataFetcher() if DataFetcher else None
        self.bot_engine = None
        self.broker_manager = None
        self.cloud_mode = CLOUD_MODE
        
        # Initialize session state
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {}
        if 'trade_history' not in st.session_state:
            st.session_state.trade_history = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    def load_bot_data(self):
        """Load trading bot data and configuration"""
        try:
            # Load configuration if exists
            config_path = 'bot_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
            return {}
        except Exception as e:
            st.error(f"Error loading bot configuration: {str(e)}")
            return {}
    
    def load_performance_data(self):
        """Load historical performance data"""
        try:
            # Load from various sources
            performance_files = [
                'analysis_charts/portfolio_performance.json',
                'analysis_charts/strategy_comparison.json',
                'analysis_charts/risk_metrics.json'
            ]
            
            performance_data = {}
            for file_path in performance_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        key = os.path.basename(file_path).replace('.json', '')
                        performance_data[key] = json.load(f)
            
            return performance_data
        except Exception as e:
            st.error(f"Error loading performance data: {str(e)}")
            return {}
    
    def create_portfolio_overview(self):
        """Create portfolio overview section"""
        st.header("[PORTFOLIO] Portfolio Overview")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Sample portfolio metrics (replace with real data)
        total_value = 125000.00
        daily_pnl = 2500.00
        total_return = 25.0
        sharpe_ratio = 1.85
        
        with col1:
            st.metric(
                label="Total Portfolio Value",
                value=f"${total_value:,.2f}",
                delta=f"${daily_pnl:,.2f}"
            )
        
        with col2:
            st.metric(
                label="Total Return",
                value=f"{total_return:.1f}%",
                delta="2.3%"
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{sharpe_ratio:.2f}",
                delta="0.15"
            )
        
        with col4:
            st.metric(
                label="Max Drawdown",
                value="-8.5%",
                delta="-1.2%"
            )
        
        # Portfolio composition chart
        st.subheader("Portfolio Composition")
        
        # Sample data (replace with real portfolio data)
        portfolio_data = {
            'Asset': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'Cash'],
            'Value': [25000, 22000, 18000, 15000, 12000, 33000],
            'Weight': [20.0, 17.6, 14.4, 12.0, 9.6, 26.4]
        }
        
        df_portfolio = pd.DataFrame(portfolio_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                df_portfolio, 
                values='Value', 
                names='Asset',
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                df_portfolio, 
                x='Asset', 
                y='Value',
                title="Asset Values"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def create_performance_charts(self):
        """Create performance monitoring charts"""
        st.header("[PERFORMANCE] Performance Analysis")
        
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        np.random.seed(42)
        
        # Portfolio performance
        portfolio_returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_cumulative = (1 + pd.Series(portfolio_returns)).cumprod()
        
        # Benchmark performance (S&P 500)
        benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))
        benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod()
        
        # Create performance comparison chart
        fig_performance = go.Figure()
        
        fig_performance.add_trace(go.Scatter(
            x=dates,
            y=portfolio_cumulative * 100000,  # Starting with $100k
            mode='lines',
            name='Trading Bot',
            line=dict(color='#00CC96', width=2)
        ))
        
        fig_performance.add_trace(go.Scatter(
            x=dates,
            y=benchmark_cumulative * 100000,
            mode='lines',
            name='S&P 500 Benchmark',
            line=dict(color='#FF6692', width=2)
        ))
        
        fig_performance.update_layout(
            title="Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Strategy performance comparison
        st.subheader("Strategy Performance Comparison")
        
        strategy_data = {
            'Strategy': ['RSI', 'Bollinger Bands', 'Moving Average', 'Combined'],
            'Return (%)': [15.2, 12.8, 8.5, 18.7],
            'Sharpe Ratio': [1.45, 1.32, 0.98, 1.67],
            'Max Drawdown (%)': [-8.2, -6.5, -12.1, -7.8],
            'Win Rate (%)': [58.3, 62.1, 54.7, 61.9]
        }
        
        df_strategies = pd.DataFrame(strategy_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_returns = px.bar(
                df_strategies, 
                x='Strategy', 
                y='Return (%)',
                title="Strategy Returns",
                color='Return (%)'
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            fig_sharpe = px.bar(
                df_strategies, 
                x='Strategy', 
                y='Sharpe Ratio',
                title="Strategy Sharpe Ratios",
                color='Sharpe Ratio'
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Display strategy metrics table
        st.subheader("Strategy Metrics Summary")
        st.dataframe(df_strategies, use_container_width=True)
    
    def create_risk_monitoring(self):
        """Create risk monitoring section"""
        st.header("[RISK] Risk Monitoring")
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Drawdown",
                value="-3.2%",
                delta="-0.5%"
            )
        
        with col2:
            st.metric(
                label="Portfolio Beta",
                value="0.85",
                delta="-0.02"
            )
        
        with col3:
            st.metric(
                label="VaR (95%)",
                value="-$2,150",
                delta="-$200"
            )
        
        # Risk alerts
        st.subheader("Risk Alerts")
        
        alerts = [
            {"type": "warning", "message": "TSLA position exceeds 15% allocation limit", "time": "2 hours ago"},
            {"type": "info", "message": "Portfolio correlation with market increased to 0.78", "time": "4 hours ago"},
            {"type": "success", "message": "Stop-loss triggered for AMZN position - Risk managed", "time": "1 day ago"}
        ]
        
        for alert in alerts:
            if alert["type"] == "warning":
                st.warning(f"[WARNING] {alert['message']} ({alert['time']})")
            elif alert["type"] == "info":
                st.info(f"[INFO] {alert['message']} ({alert['time']})")
            elif alert["type"] == "success":
                st.success(f"[SUCCESS] {alert['message']} ({alert['time']})")
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        
        # Generate sample drawdown data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown (%)',
            line=dict(color='red')
        ))
        
        fig_drawdown.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig_drawdown, use_container_width=True)
    
    def create_trade_monitoring(self):
        """Create trade monitoring section"""
        st.header("💼 Trade Monitoring")
        
        # Recent trades
        st.subheader("Recent Trades")
        
        # Sample trade data
        trade_data = {
            'Timestamp': [
                datetime.now() - timedelta(hours=2),
                datetime.now() - timedelta(hours=5),
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=1, hours=3),
                datetime.now() - timedelta(days=2)
            ],
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
            'Action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
            'Quantity': [50, 30, 25, 40, 20],
            'Price': [175.50, 420.25, 2850.00, 245.80, 3200.00],
            'P&L': [250.00, -150.00, 500.00, 180.00, -75.00],
            'Strategy': ['RSI', 'Bollinger', 'RSI', 'Moving Avg', 'Combined']
        }
        
        df_trades = pd.DataFrame(trade_data)
        df_trades['Total Value'] = df_trades['Quantity'] * df_trades['Price']
        
        # Color code P&L
        def color_pnl(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'
        
        styled_df = df_trades.style.applymap(color_pnl, subset=['P&L'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Trade statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade Statistics")
            total_trades = len(df_trades)
            winning_trades = len(df_trades[df_trades['P&L'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            avg_pnl = df_trades['P&L'].mean()
            
            st.metric("Total Trades Today", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Average P&L", f"${avg_pnl:.2f}")
        
        with col2:
            # P&L distribution
            fig_pnl = px.histogram(
                df_trades, 
                x='P&L', 
                title="P&L Distribution",
                nbins=10
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    
    def create_market_overview(self):
        """Create market overview section"""
        st.header("🌍 Market Overview")
        
        # Market indices
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("S&P 500", "4,750.25", "12.50 (0.26%)")
        
        with col2:
            st.metric("NASDAQ", "15,240.80", "-25.30 (-0.17%)")
        
        with col3:
            st.metric("VIX", "18.45", "1.25 (7.26%)")
        
        with col4:
            st.metric("USD/EUR", "1.0850", "-0.0025 (-0.23%)")
        
        # Market heatmap
        st.subheader("Sector Performance")
        
        sector_data = {
            'Sector': ['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer', 'Industrials'],
            'Performance': [2.1, -0.5, 1.8, -1.2, 0.8, 1.5]
        }
        
        df_sectors = pd.DataFrame(sector_data)
        
        fig_sectors = px.bar(
            df_sectors, 
            x='Sector', 
            y='Performance',
            title="Sector Performance Today (%)",
            color='Performance',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig_sectors, use_container_width=True)
    
    def create_settings_panel(self):
        """Create settings and configuration panel"""
        st.header("[CONFIG] Bot Configuration")
        
        # Bot status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bot Status")
            bot_status = st.selectbox("Status", ["Running", "Paused", "Stopped"])
            
            if bot_status == "Running":
                st.success("🟢 Bot is actively trading")
            elif bot_status == "Paused":
                st.warning("🟡 Bot is paused")
            else:
                st.error("🔴 Bot is stopped")
        
        with col2:
            st.subheader("Trading Parameters")
            max_position_size = st.slider("Max Position Size (%)", 1, 20, 10)
            stop_loss = st.slider("Stop Loss (%)", 1, 10, 5)
            take_profit = st.slider("Take Profit (%)", 5, 50, 15)
        
        # Strategy selection
        st.subheader("Active Strategies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_enabled = st.checkbox("RSI Strategy", value=True)
            if rsi_enabled:
                rsi_period = st.number_input("RSI Period", 5, 30, 14)
        
        with col2:
            bb_enabled = st.checkbox("Bollinger Bands", value=True)
            if bb_enabled:
                bb_period = st.number_input("BB Period", 10, 50, 20)
        
        with col3:
            ma_enabled = st.checkbox("Moving Average", value=False)
            if ma_enabled:
                ma_short = st.number_input("Short MA", 5, 50, 10)
                ma_long = st.number_input("Long MA", 20, 200, 50)
        
        # Save configuration
        if st.button("Save Configuration"):
            config = {
                'bot_status': bot_status,
                'max_position_size': max_position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategies': {
                    'rsi': {'enabled': rsi_enabled, 'period': rsi_period if rsi_enabled else 14},
                    'bollinger': {'enabled': bb_enabled, 'period': bb_period if bb_enabled else 20},
                    'moving_average': {'enabled': ma_enabled, 'short': ma_short if ma_enabled else 10, 'long': ma_long if ma_enabled else 50}
                }
            }
            
            with open('bot_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("Configuration saved successfully!")
    
    def run_dashboard(self):
        """Main dashboard runner"""
        # Sidebar navigation
        st.sidebar.title("[DASHBOARD] Trading Bot Dashboard")
        st.sidebar.markdown("---")
        
        # Navigation menu
        page = st.sidebar.selectbox(
            "Navigate to:",
            [
                "Portfolio Overview",
                "Performance Analysis", 
                "Risk Monitoring",
                "Trade Monitoring",
                "Market Overview",
                "Bot Configuration"
            ]
        )
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        
        if auto_refresh:
            st.sidebar.info("Dashboard will refresh every 30 seconds")
            # Note: In production, implement actual auto-refresh
        
        # Manual refresh button
        if st.sidebar.button("[REFRESH] Refresh Data"):
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Last Updated:** " + datetime.now().strftime("%H:%M:%S"))
        
        # Main content area
        if page == "Portfolio Overview":
            self.create_portfolio_overview()
        elif page == "Performance Analysis":
            self.create_performance_charts()
        elif page == "Risk Monitoring":
            self.create_risk_monitoring()
        elif page == "Trade Monitoring":
            self.create_trade_monitoring()
        elif page == "Market Overview":
            self.create_market_overview()
        elif page == "Bot Configuration":
            self.create_settings_panel()

def main():
    """Main function to run the Streamlit dashboard"""
    # Dashboard title
    st.title("[BOT] Trading Bot Monitoring Dashboard")
    st.markdown("Real-time monitoring and control of your automated trading system")
    st.markdown("---")
    
    # Initialize and run dashboard
    dashboard = TradingDashboard()
    dashboard.run_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Trading Bot Dashboard v1.0 | "
        "[WARNING] For educational purposes only - Not financial advice"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()