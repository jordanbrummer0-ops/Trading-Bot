#!/usr/bin/env python3
"""
Trading Bot Startup Script

Main orchestration script for starting and managing the trading bot system.
Provides unified interface for running different components and modes.

Features:
- Unified startup for all components
- Environment validation
- Configuration management
- Process monitoring
- Graceful shutdown handling
- Development vs production modes

Usage:
    python start_bot.py --mode production
    python start_bot.py --mode development --paper-trading
    python start_bot.py --mode dashboard-only
    python start_bot.py --mode backtest --symbol AAPL

Author: Trading Bot System
Version: 1.0
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class TradingBotOrchestrator:
    """
    Main orchestrator for the trading bot system
    
    Manages startup, shutdown, and monitoring of all bot components:
    - Main trading engine
    - Streamlit dashboard
    - Background workers
    - Monitoring services
    """
    
    def __init__(self, mode: str, config: Dict):
        self.mode = mode
        self.config = config
        self.processes = {}
        self.running = False
        self.project_root = Path(__file__).parent
        
        # Setup logging
        self.setup_logging()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'bot_orchestrator_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('TradingBotOrchestrator')
    
    def validate_environment(self) -> bool:
        """Validate environment and prerequisites"""
        self.logger.info("[VALIDATE] Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("[ERROR] Python 3.8+ required")
            return False
        
        # Check required files
        required_files = [
            'main.py',
            'monitoring_dashboard.py',
            'requirements.txt',
            'src/data_fetcher.py'
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                self.logger.error(f"[ERROR] Missing required file: {file_path}")
                return False
        
        # Check environment variables for production mode
        if self.mode == 'production':
            required_env_vars = [
                'ALPACA_API_KEY',
                'ALPACA_SECRET_KEY'
            ]
            
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                self.logger.error(f"[ERROR] Missing environment variables: {', '.join(missing_vars)}")
                return False
        
        # Check dependencies
        try:
            import pandas
            import numpy
            import yfinance
            import streamlit
            import plotly
            self.logger.info("[SUCCESS] Core dependencies available")
        except ImportError as e:
            self.logger.error(f"[ERROR] Missing dependency: {e}")
            return False
        
        self.logger.info("[SUCCESS] Environment validation passed")
        return True
    
    def load_configuration(self) -> Dict:
        """Load bot configuration"""
        config_file = self.project_root / 'config' / f'{self.mode}_config.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'trading': {
                'max_position_size': 0.10,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'paper_trading': self.mode != 'production'
            },
            'strategies': {
                'rsi': {'enabled': True, 'period': 14},
                'bollinger': {'enabled': True, 'period': 20},
                'moving_average': {'enabled': False}
            },
            'monitoring': {
                'dashboard_port': 8501,
                'auto_refresh': True,
                'alerts_enabled': True
            }
        }
    
    def start_main_engine(self):
        """Start the main trading engine"""
        self.logger.info("[START] Starting main trading engine...")
        
        cmd = [sys.executable, 'main.py']
        
        # Add mode-specific arguments
        if self.config['trading']['paper_trading']:
            cmd.extend(['--paper-trading'])
        
        if self.mode == 'development':
            cmd.extend(['--debug'])
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes['main_engine'] = process
            self.logger.info("[SUCCESS] Main trading engine started")
            
            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_process_output,
                args=('main_engine', process),
                daemon=True
            ).start()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to start main engine: {e}")
    
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        self.logger.info("[DASHBOARD] Starting dashboard...")
        
        port = self.config['monitoring']['dashboard_port']
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'monitoring_dashboard.py',
            '--server.port', str(port),
            '--server.address', '0.0.0.0',
            '--server.headless', 'true'
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes['dashboard'] = process
            self.logger.info(f"[SUCCESS] Dashboard started on port {port}")
            
            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_process_output,
                args=('dashboard', process),
                daemon=True
            ).start()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to start dashboard: {e}")
    
    def start_parameter_optimization(self, symbol: str = None):
        """Start parameter optimization process"""
        self.logger.info("[OPTIMIZE] Starting parameter optimization...")
        
        cmd = [sys.executable, 'parameter_optimization.py']
        
        if symbol:
            cmd.extend(['--symbol', symbol])
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes['optimization'] = process
            self.logger.info("[SUCCESS] Parameter optimization started")
            
            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_process_output,
                args=('optimization', process),
                daemon=True
            ).start()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to start optimization: {e}")
    
    def run_backtest(self, symbol: str, strategy: str = None):
        """Run backtesting for specific symbol and strategy"""
        self.logger.info(f"[BACKTEST] Running backtest for {symbol}...")
        
        cmd = [sys.executable, 'main.py', '--backtest', '--symbol', symbol]
        
        if strategy:
            cmd.extend(['--strategy', strategy])
        
        try:
            process = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if process.returncode == 0:
                self.logger.info(f"[SUCCESS] Backtest completed for {symbol}")
                print(process.stdout)
            else:
                self.logger.error(f"[ERROR] Backtest failed for {symbol}")
                print(process.stderr)
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"[ERROR] Backtest timeout for {symbol}")
        except Exception as e:
            self.logger.error(f"[ERROR] Backtest error: {e}")
    
    def monitor_process_output(self, name: str, process: subprocess.Popen):
        """Monitor process output and log it"""
        while process.poll() is None:
            try:
                output = process.stdout.readline()
                if output:
                    self.logger.info(f"[{name}] {output.strip()}")
                
                error = process.stderr.readline()
                if error:
                    self.logger.error(f"[{name}] {error.strip()}")
                    
            except Exception as e:
                self.logger.error(f"Error monitoring {name}: {e}")
                break
    
    def monitor_system_health(self):
        """Monitor system health and performance"""
        self.logger.info("[HEALTH] Starting system health monitoring...")
        
        while self.running:
            try:
                # Check process health
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        self.logger.warning(f"[WARNING] Process {name} has stopped")
                        
                        # Restart critical processes
                        if name in ['main_engine', 'dashboard'] and self.running:
                            self.logger.info(f"[RESTART] Restarting {name}...")
                            if name == 'main_engine':
                                self.start_main_engine()
                            elif name == 'dashboard':
                                self.start_dashboard()
                
                # System resource monitoring
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    
                    if cpu_percent > 80:
                        self.logger.warning(f"[WARNING] High CPU usage: {cpu_percent}%")
                    
                    if memory_percent > 80:
                        self.logger.warning(f"[WARNING] High memory usage: {memory_percent}%")
                        
                except ImportError:
                    pass  # psutil not available
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of all processes"""
        self.logger.info("🛑 Shutting down trading bot system...")
        self.running = False
        
        # Stop all processes
        for name, process in self.processes.items():
            if process.poll() is None:
                self.logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    self.logger.info(f"[SUCCESS] {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"[WARNING] Force killing {name}")
                    process.kill()
        
        self.logger.info("[SUCCESS] All processes stopped")
    
    def run(self):
        """Main run method"""
        self.logger.info(f"[START] Starting Trading Bot in {self.mode} mode...")
        
        # Validate environment
        if not self.validate_environment():
            return False
        
        # Load configuration
        self.config.update(self.load_configuration())
        
        self.running = True
        
        try:
            if self.mode == 'dashboard-only':
                # Only start dashboard
                self.start_dashboard()
                
            elif self.mode == 'optimization':
                # Only run parameter optimization
                symbol = self.config.get('optimization_symbol', 'AAPL')
                self.start_parameter_optimization(symbol)
                
            elif self.mode == 'backtest':
                # Run backtest mode
                symbol = self.config.get('backtest_symbol', 'AAPL')
                strategy = self.config.get('backtest_strategy')
                self.run_backtest(symbol, strategy)
                return True
                
            else:
                # Full system startup
                self.start_main_engine()
                time.sleep(5)  # Wait for main engine to initialize
                self.start_dashboard()
            
            # Start health monitoring
            health_thread = threading.Thread(
                target=self.monitor_system_health,
                daemon=True
            )
            health_thread.start()
            
            # Print startup summary
            self.print_startup_summary()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.shutdown()
        
        return True
    
    def print_startup_summary(self):
        """Print startup summary"""
        print("\n" + "="*60)
        print("[BOT] TRADING BOT SYSTEM STARTED")
        print("="*60)
        print(f"Mode: {self.mode.upper()}")
        print(f"Paper Trading: {'Yes' if self.config['trading']['paper_trading'] else 'No'}")
        
        if 'dashboard' in self.processes:
            port = self.config['monitoring']['dashboard_port']
            print(f"[DASHBOARD] Dashboard: http://localhost:{port}")
        
        print(f"[LOGS] Logs: {self.project_root / 'logs'}")
        print(f"[CHARTS] Charts: {self.project_root / 'analysis_charts'}")
        print("\nPress Ctrl+C to stop the system")
        print("="*60 + "\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Trading Bot System Orchestrator')
    parser.add_argument(
        '--mode',
        choices=['development', 'production', 'dashboard-only', 'optimization', 'backtest'],
        default='development',
        help='Operating mode'
    )
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        help='Force paper trading mode'
    )
    parser.add_argument(
        '--symbol',
        default='AAPL',
        help='Symbol for backtest or optimization'
    )
    parser.add_argument(
        '--strategy',
        choices=['rsi', 'bollinger', 'moving_average', 'combined'],
        help='Strategy for backtest'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'trading': {
            'paper_trading': args.paper_trading or args.mode != 'production'
        },
        'backtest_symbol': args.symbol,
        'backtest_strategy': args.strategy,
        'optimization_symbol': args.symbol
    }
    
    # Create and run orchestrator
    orchestrator = TradingBotOrchestrator(args.mode, config)
    success = orchestrator.run()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()