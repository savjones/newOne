"""
Comprehensive Backtest Pipeline

This module provides a complete backtesting pipeline that can run multiple
strategies against multiple datasets with comprehensive reporting and analysis.

Features:
- Multi-strategy, multi-dataset backtesting
- Parallel execution for performance
- Comprehensive performance metrics
- Risk analysis and drawdown tracking
- Strategy comparison and ranking
- Detailed reporting with visualizations
- Walk-forward analysis support
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
from enum import Enum

from strategies import STRATEGY_REGISTRY, get_strategy
from utils.config import ConfigManager
from utils.risk import RiskManager
from utils.logging import setup_logging
from backtests.engine import BacktestEngine
from backtests.portfolio import Portfolio
from backtests.performance import PerformanceAnalyzer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.csv_loader import CSVDataLoader, NormalizedData

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestJob:
    """Individual backtest job configuration"""
    job_id: str
    strategy_name: str
    strategy_config: Dict
    dataset_name: str
    dataset: NormalizedData
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: BacktestStatus = BacktestStatus.PENDING
    error_message: Optional[str] = None


@dataclass 
class BacktestResult:
    """Backtest execution result"""
    job_id: str
    strategy_name: str
    dataset_name: str
    symbol: str
    timeframe: str
    performance_metrics: Dict[str, float]
    trades: List[Any] = field(default_factory=list)
    portfolio_history: List[Any] = field(default_factory=list)
    execution_time: float = 0.0
    status: BacktestStatus = BacktestStatus.COMPLETED
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Data settings
    data_directory: str = "data/csvs"
    min_data_quality: float = 0.90
    min_data_points: int = 100
    
    # Execution settings
    parallel_execution: bool = True
    max_workers: Optional[int] = None
    timeout_minutes: int = 30
    
    # Backtesting settings
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # Strategy settings
    strategy_configs_dir: str = "configs/strategies"
    enabled_strategies: Optional[List[str]] = None
    
    # Output settings
    results_directory: str = "backtest_results"
    save_trades: bool = True
    save_portfolio_history: bool = True
    generate_reports: bool = True
    
    # Risk management
    max_drawdown_threshold: float = 0.3
    min_sharpe_ratio: float = 0.0
    
    # Performance filters
    min_total_return: float = -1.0  # Allow losses for analysis
    max_volatility: float = 2.0


class BacktestPipeline:
    """
    Comprehensive backtesting pipeline
    
    This class orchestrates the execution of multiple strategies across
    multiple datasets, providing comprehensive analysis and reporting.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize backtest pipeline
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.data_loader = CSVDataLoader(self.config.data_directory)
        self.loaded_datasets: Dict[str, NormalizedData] = {}
        self.strategy_configs: Dict[str, Dict] = {}
        self.results: List[BacktestResult] = []
        self.jobs: List[BacktestJob] = []
        
        # Setup results directory
        self.results_dir = Path(self.config.results_directory)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(level=logging.INFO)
        
    def load_data(self) -> Dict[str, NormalizedData]:
        """
        Load and validate all datasets
        
        Returns:
            Dictionary of loaded and validated datasets
        """
        logger.info("Loading datasets...")
        
        # Load all CSV files
        all_data = self.data_loader.load_all_csvs()
        
        # Filter datasets based on quality criteria
        validated_data = {}
        
        for filename, dataset in all_data.items():
            # Validate data quality
            quality_metrics = self.data_loader.validate_data_quality(
                dataset, self.config.min_data_quality
            )
            
            if not quality_metrics['passes_threshold']:
                logger.warning(f"Dataset {filename} failed quality check: "
                             f"quality={quality_metrics['quality_score']:.2f}")
                continue
            
            if len(dataset.data) < self.config.min_data_points:
                logger.warning(f"Dataset {filename} has insufficient data points: "
                             f"{len(dataset.data)} < {self.config.min_data_points}")
                continue
            
            validated_data[filename] = dataset
            logger.info(f"✓ Validated dataset {filename}: "
                       f"{len(dataset.data)} rows, quality={quality_metrics['quality_score']:.2f}")
        
        self.loaded_datasets = validated_data
        logger.info(f"Loaded {len(validated_data)} validated datasets")
        
        return validated_data
    
    def load_strategy_configs(self) -> Dict[str, Dict]:
        """
        Load strategy configurations
        
        Returns:
            Dictionary of strategy configurations
        """
        logger.info("Loading strategy configurations...")
        
        config_dir = Path(self.config.strategy_configs_dir)
        logger.info(f"Loading strategy configs from: {config_dir}")
        
        if not config_dir.exists():
            logger.error(f"Strategy config directory not found: {config_dir}")
            return {}
        
        configs = {}
        config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
        
        for config_file in config_files:
            strategy_name = config_file.stem
            logger.info(f"Found config file: {config_file.name} -> strategy: {strategy_name}")
            
            # Skip if not in enabled strategies list
            if (self.config.enabled_strategies and 
                strategy_name not in self.config.enabled_strategies):
                logger.info(f"Skipping {strategy_name} - not in enabled list: {self.config.enabled_strategies}")
                continue
            
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                logger.info(f"Loaded config for {strategy_name}: {list(config.keys())}")
                
                # Validate strategy exists
                if get_strategy(strategy_name) is None:
                    logger.warning(f"Strategy {strategy_name} not found in registry")
                    continue
                
                configs[strategy_name] = config
                logger.info(f"✓ Loaded config for {strategy_name}")
                
            except Exception as e:
                logger.error(f"Failed to load config for {strategy_name}: {e}")
        
        self.strategy_configs = configs
        logger.info(f"Loaded {len(configs)} strategy configurations")
        
        return configs
    
    def create_jobs(self) -> List[BacktestJob]:
        """
        Create backtest jobs for all strategy-dataset combinations
        
        Returns:
            List of backtest jobs
        """
        jobs = []
        job_counter = 0
        
        for strategy_name, strategy_config in self.strategy_configs.items():
            for dataset_name, dataset in self.loaded_datasets.items():
                job_id = f"{strategy_name}_{dataset.symbol}_{dataset.timeframe}_{job_counter:03d}"
                
                job = BacktestJob(
                    job_id=job_id,
                    strategy_name=strategy_name,
                    strategy_config=strategy_config,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    initial_capital=self.config.initial_capital,
                    commission_rate=self.config.commission_rate,
                    slippage_rate=self.config.slippage_rate
                )
                
                jobs.append(job)
                job_counter += 1
        
        self.jobs = jobs
        logger.info(f"Created {len(jobs)} backtest jobs")
        
        return jobs
    
    def run_single_backtest(self, job: BacktestJob) -> BacktestResult:
        """
        Execute a single backtest job
        
        Args:
            job: Backtest job to execute
            
        Returns:
            Backtest result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Running backtest: {job.job_id}")
            job.status = BacktestStatus.RUNNING
            
            # Get strategy class
            strategy_class = get_strategy(job.strategy_name)
            if strategy_class is None:
                raise ValueError(f"Strategy {job.strategy_name} not found")
            
            # Initialize risk manager
            risk_config = job.strategy_config.get('risk', {})
            risk_manager = RiskManager(risk_config)
            
            # Initialize strategy
            strategy = strategy_class(job.strategy_config, risk_manager)
            
            # Initialize portfolio
            portfolio = Portfolio(job.initial_capital)
            
            # Initialize backtest engine
            engine = BacktestEngine(
                initial_capital=job.initial_capital,
                commission=job.commission_rate,
                slippage=job.slippage_rate
            )
            
            # Add strategy to engine
            engine.add_strategy(strategy)
            
            # Prepare data
            data = job.dataset.data.copy()
            
            # Ensure data has the right format for the engine
            if 'timestamp' not in data.columns:
                data = data.reset_index()
                data = data.rename(columns={'datetime': 'timestamp'})
            
            # Convert data index to timezone-naive if needed
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_convert(None)
            
            # Add market data to engine
            engine.add_market_data(job.dataset.symbol, data)
            
            # Apply date filters if specified
            if job.start_date:
                data = data[data.index >= job.start_date]
            if job.end_date:
                data = data[data.index <= job.end_date]
            
            if len(data) < self.config.min_data_points:
                raise ValueError(f"Insufficient data after filtering: {len(data)} rows")
            
            # Run backtest
            start_date = data.index.min()
            end_date = data.index.max()
            timeframe = job.dataset.timeframe
            
            # Convert timezone-aware timestamps to timezone-naive datetime objects
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_convert(None).to_pydatetime()
            else:
                start_date = start_date.to_pydatetime() if hasattr(start_date, 'to_pydatetime') else start_date
            
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_convert(None).to_pydatetime()
            else:
                end_date = end_date.to_pydatetime() if hasattr(end_date, 'to_pydatetime') else end_date
            
            backtest_results = engine.run_backtest(start_date, end_date, timeframe)
            
            # Extract performance metrics (already calculated by the engine)
            performance_metrics = {
                **backtest_results['portfolio_performance'],
                **backtest_results['strategy_performance'].get(job.strategy_name, {})
            }
            
            # Apply performance filters
            if not self._passes_performance_filters(performance_metrics):
                logger.warning(f"Backtest {job.job_id} failed performance filters")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = BacktestResult(
                job_id=job.job_id,
                strategy_name=job.strategy_name,
                dataset_name=job.dataset_name,
                symbol=job.dataset.symbol,
                timeframe=job.dataset.timeframe,
                performance_metrics=performance_metrics,
                trades=[],  # TODO: Extract trades from strategy results
                portfolio_history=[],  # TODO: Portfolio history not directly available
                execution_time=execution_time,
                status=BacktestStatus.COMPLETED,
                metadata={
                    'data_points': len(data),
                    'date_range': [str(data.index.min()), str(data.index.max())],
                    'initial_capital': job.initial_capital,
                    'commission_rate': job.commission_rate,
                    'slippage_rate': job.slippage_rate
                }
            )
            
            logger.info(f"✓ Completed backtest {job.job_id}: "
                       f"Return={performance_metrics.get('total_return', 0):.2%}, "
                       f"Sharpe={performance_metrics.get('sharpe_ratio', 0):.2f}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(f"✗ Failed backtest {job.job_id}: {error_msg}")
            
            return BacktestResult(
                job_id=job.job_id,
                strategy_name=job.strategy_name,
                dataset_name=job.dataset_name,
                symbol=job.dataset.symbol if hasattr(job.dataset, 'symbol') else 'UNKNOWN',
                timeframe=job.dataset.timeframe if hasattr(job.dataset, 'timeframe') else 'UNKNOWN',
                performance_metrics={},
                execution_time=execution_time,
                status=BacktestStatus.FAILED,
                error_message=error_msg
            )
    
    def _passes_performance_filters(self, metrics: Dict[str, float]) -> bool:
        """Check if performance metrics pass filters"""
        filters = [
            metrics.get('total_return', -999) >= self.config.min_total_return,
            metrics.get('max_drawdown', 999) <= self.config.max_drawdown_threshold,
            metrics.get('sharpe_ratio', -999) >= self.config.min_sharpe_ratio,
            metrics.get('volatility', 999) <= self.config.max_volatility
        ]
        
        return all(filters)
    
    def run_all_backtests(self) -> List[BacktestResult]:
        """
        Execute all backtest jobs
        
        Returns:
            List of backtest results
        """
        if not self.jobs:
            logger.error("No backtest jobs created. Call create_jobs() first.")
            return []
        
        logger.info(f"Starting execution of {len(self.jobs)} backtest jobs...")
        
        results = []
        
        if self.config.parallel_execution and len(self.jobs) > 1:
            # Parallel execution
            max_workers = self.config.max_workers or min(4, len(self.jobs))
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(self.run_single_backtest, job): job
                    for job in self.jobs
                }
                
                # Collect results
                completed = 0
                for future in as_completed(future_to_job, timeout=self.config.timeout_minutes * 60):
                    job = future_to_job[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        logger.info(f"Progress: {completed}/{len(self.jobs)} jobs completed")
                        
                    except Exception as e:
                        logger.error(f"Job {job.job_id} failed with exception: {e}")
                        
                        # Create failed result
                        failed_result = BacktestResult(
                            job_id=job.job_id,
                            strategy_name=job.strategy_name,
                            dataset_name=job.dataset_name,
                            symbol=job.dataset.symbol,
                            timeframe=job.dataset.timeframe,
                            performance_metrics={},
                            status=BacktestStatus.FAILED,
                            error_message=str(e)
                        )
                        results.append(failed_result)
                        completed += 1
        else:
            # Sequential execution
            for i, job in enumerate(self.jobs, 1):
                logger.info(f"Progress: {i}/{len(self.jobs)} - Running {job.job_id}")
                result = self.run_single_backtest(job)
                results.append(result)
        
        self.results = results
        
        # Summary
        successful = sum(1 for r in results if r.status == BacktestStatus.COMPLETED)
        failed = len(results) - successful
        
        logger.info(f"Backtest execution completed: {successful} successful, {failed} failed")
        
        return results
    
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate summary report of all backtest results
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for result in self.results:
            if result.status != BacktestStatus.COMPLETED:
                continue
            
            metrics = result.performance_metrics
            
            summary_data.append({
                'Strategy': result.strategy_name,
                'Symbol': result.symbol,
                'Timeframe': result.timeframe,
                'Total Return (%)': metrics.get('total_return', 0) * 100,
                'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Volatility (%)': metrics.get('volatility', 0) * 100,
                'Win Rate (%)': metrics.get('win_rate', 0) * 100,
                'Profit Factor': metrics.get('profit_factor', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Avg Trade Duration': metrics.get('avg_trade_duration', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'Execution Time (s)': result.execution_time
            })
        
        df = pd.DataFrame(summary_data)
        
        if not df.empty:
            # Sort by Sharpe ratio descending
            df = df.sort_values('Sharpe Ratio', ascending=False)
            df = df.reset_index(drop=True)
        
        return df
    
    def get_best_strategies(self, metric: str = 'sharpe_ratio', top_n: int = 10) -> pd.DataFrame:
        """
        Get top performing strategies by specified metric
        
        Args:
            metric: Performance metric to rank by
            top_n: Number of top strategies to return
            
        Returns:
            DataFrame with top strategies
        """
        summary = self.generate_summary_report()
        
        if summary.empty:
            return pd.DataFrame()
        
        # Map metric names
        metric_mapping = {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': 'Total Return (%)',
            'sortino_ratio': 'Sortino Ratio',
            'calmar_ratio': 'Calmar Ratio',
            'win_rate': 'Win Rate (%)',
            'profit_factor': 'Profit Factor'
        }
        
        column_name = metric_mapping.get(metric, metric)
        
        if column_name not in summary.columns:
            logger.error(f"Metric {metric} not found in summary")
            return summary.head(top_n)
        
        return summary.nlargest(top_n, column_name)
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save backtest results to file
        
        Args:
            filename: Optional filename (generates timestamp-based name if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.pkl"
        
        filepath = self.results_dir / filename
        
        # Save results
        save_data = {
            'config': self.config,
            'results': self.results,
            'summary': self.generate_summary_report().to_dict('records'),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_jobs': len(self.jobs),
                'successful_jobs': sum(1 for r in self.results if r.status == BacktestStatus.COMPLETED),
                'datasets': list(self.loaded_datasets.keys()),
                'strategies': list(self.strategy_configs.keys())
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save summary as CSV
        summary_path = filepath.with_suffix('.csv')
        summary_df = self.generate_summary_report()
        if not summary_df.empty:
            summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def load_results(self, filepath: str) -> None:
        """
        Load backtest results from file
        
        Args:
            filepath: Path to results file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.config = data['config']
        self.results = data['results']
        
        logger.info(f"Loaded {len(self.results)} results from {filepath}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete backtesting pipeline
        
        Returns:
            Dictionary with pipeline results and summary
        """
        logger.info("Starting full backtesting pipeline...")
        
        pipeline_start = datetime.now()
        
        # Step 1: Load data
        datasets = self.load_data()
        if not datasets:
            logger.error("No valid datasets loaded. Pipeline aborted.")
            return {'status': 'failed', 'error': 'No valid datasets'}
        
        # Step 2: Load strategy configs
        configs = self.load_strategy_configs()
        if not configs:
            logger.error("No strategy configurations loaded. Pipeline aborted.")
            return {'status': 'failed', 'error': 'No strategy configs'}
        
        # Step 3: Create jobs
        jobs = self.create_jobs()
        if not jobs:
            logger.error("No backtest jobs created. Pipeline aborted.")
            return {'status': 'failed', 'error': 'No jobs created'}
        
        # Step 4: Execute backtests
        results = self.run_all_backtests()
        
        # Step 5: Generate reports
        summary = self.generate_summary_report()
        best_strategies = self.get_best_strategies()
        
        # Step 6: Save results
        results_file = self.save_results()
        
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        
        # Pipeline summary
        successful_results = [r for r in results if r.status == BacktestStatus.COMPLETED]
        
        pipeline_summary = {
            'status': 'completed',
            'execution_time': pipeline_duration,
            'datasets_loaded': len(datasets),
            'strategies_configured': len(configs),
            'jobs_created': len(jobs),
            'jobs_successful': len(successful_results),
            'jobs_failed': len(results) - len(successful_results),
            'results_file': results_file,
            'best_strategy': best_strategies.iloc[0].to_dict() if not best_strategies.empty else None,
            'summary_stats': {
                'avg_sharpe': summary['Sharpe Ratio'].mean() if not summary.empty else 0,
                'avg_return': summary['Total Return (%)'].mean() if not summary.empty else 0,
                'max_drawdown': summary['Max Drawdown (%)'].max() if not summary.empty else 0
            }
        }
        
        logger.info(f"Pipeline completed in {pipeline_duration:.1f} seconds")
        logger.info(f"Results summary: {len(successful_results)} successful backtests")
        
        if not best_strategies.empty:
            best = best_strategies.iloc[0]
            logger.info(f"Best strategy: {best['Strategy']} on {best['Symbol']} "
                       f"(Sharpe: {best['Sharpe Ratio']:.2f}, Return: {best['Total Return (%)']:.1f}%)")
        
        return pipeline_summary
