"""
Comprehensive Backtest Execution Script

This script runs comprehensive backtests for all strategies against all available CSV datasets.
It demonstrates the Ultimate Oscillator + ADX + MFI strategy along with all other implemented
strategies using the reusable backtest pipeline.

Features:
- Automatic CSV loading with schema normalization
- Multi-strategy backtesting across all datasets
- Comprehensive performance analysis and reporting
- Strategy comparison and ranking
- Detailed execution logging

Usage:
    python run_comprehensive_backtests.py [--strategies STRATEGIES] [--parallel] [--config CONFIG]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backtests.pipeline import BacktestPipeline, PipelineConfig
from data.csv_loader import CSVDataLoader
from utils.logging import setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive backtests for all strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        help='Specific strategies to test (default: all available)',
        choices=['momentum', 'mean_reversion', 'breakout', 'microstructure', 
                'pairs_trading', 'ultimate_oscillator']
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel execution of backtests'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/csvs',
        help='Directory containing CSV data files'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='backtest_results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000,
        help='Initial capital for backtests'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (default: 0.1%)'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0005,
        help='Slippage rate (default: 0.05%)'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.90,
        help='Minimum data quality threshold'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout in minutes for individual backtests'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top strategies to display'
    )
    
    return parser.parse_args()


def create_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Create pipeline configuration from arguments"""
    return PipelineConfig(
        data_directory=args.data_dir,
        min_data_quality=args.min_quality,
        min_data_points=100,
        parallel_execution=args.parallel,
        max_workers=args.max_workers,
        timeout_minutes=args.timeout,
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        strategy_configs_dir="configs/strategies",
        enabled_strategies=args.strategies,
        results_directory=args.results_dir,
        save_trades=True,
        save_portfolio_history=True,
        generate_reports=True,
        max_drawdown_threshold=0.5,
        min_sharpe_ratio=-2.0,
        min_total_return=-0.8,
        max_volatility=3.0
    )


def print_data_summary(loader: CSVDataLoader) -> None:
    """Print summary of loaded data"""
    summary = loader.get_data_summary()
    
    if summary.empty:
        print("No data loaded.")
        return
    
    print("\n" + "="*80)
    print("LOADED DATASETS SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print(f"\nTotal datasets: {len(summary)}")
    print(f"Average data quality: {summary['Data Quality (%)'].mean():.1f}%")
    print(f"Date range: {summary['Start Date'].min()} to {summary['End Date'].max()}")


def print_strategy_summary(pipeline: BacktestPipeline) -> None:
    """Print summary of configured strategies"""
    if not pipeline.strategy_configs:
        print("No strategies configured.")
        return
    
    print("\n" + "="*80)
    print("CONFIGURED STRATEGIES")
    print("="*80)
    
    for strategy_name in pipeline.strategy_configs.keys():
        print(f"✓ {strategy_name.replace('_', ' ').title()}")
    
    print(f"\nTotal strategies: {len(pipeline.strategy_configs)}")


def print_execution_plan(pipeline: BacktestPipeline) -> None:
    """Print execution plan summary"""
    if not pipeline.jobs:
        print("No jobs created.")
        return
    
    print("\n" + "="*80)
    print("EXECUTION PLAN")
    print("="*80)
    
    job_summary = {}
    for job in pipeline.jobs:
        key = f"{job.strategy_name} × {job.dataset.symbol} ({job.dataset.timeframe})"
        job_summary[key] = job_summary.get(key, 0) + 1
    
    for job_desc, count in job_summary.items():
        print(f"• {job_desc}")
    
    print(f"\nTotal backtest jobs: {len(pipeline.jobs)}")
    print(f"Estimated execution time: {len(pipeline.jobs) * 2:.0f}-{len(pipeline.jobs) * 10:.0f} seconds")


def print_results_summary(pipeline: BacktestPipeline, top_n: int = 10) -> None:
    """Print comprehensive results summary"""
    summary = pipeline.generate_summary_report()
    
    if summary.empty:
        print("No results to display.")
        return
    
    print("\n" + "="*100)
    print("BACKTEST RESULTS SUMMARY")
    print("="*100)
    
    # Overall statistics
    successful = len(summary)
    total_jobs = len(pipeline.jobs)
    
    print(f"Execution Summary:")
    print(f"  • Total jobs: {total_jobs}")
    print(f"  • Successful: {successful}")
    print(f"  • Failed: {total_jobs - successful}")
    print(f"  • Success rate: {successful/total_jobs*100:.1f}%")
    
    if successful == 0:
        return
    
    print(f"\nOverall Performance Statistics:")
    print(f"  • Average Return: {summary['Total Return (%)'].mean():.2f}%")
    print(f"  • Average Sharpe Ratio: {summary['Sharpe Ratio'].mean():.2f}")
    print(f"  • Best Return: {summary['Total Return (%)'].max():.2f}%")
    print(f"  • Best Sharpe: {summary['Sharpe Ratio'].max():.2f}")
    print(f"  • Worst Drawdown: {summary['Max Drawdown (%)'].max():.2f}%")
    
    # Top strategies
    print(f"\n🏆 TOP {min(top_n, len(summary))} STRATEGIES BY SHARPE RATIO:")
    print("-" * 100)
    
    top_strategies = summary.head(top_n)
    
    # Format columns for better display
    display_columns = [
        'Strategy', 'Symbol', 'Timeframe', 'Total Return (%)', 
        'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades'
    ]
    
    display_df = top_strategies[display_columns].copy()
    
    # Format numeric columns
    for col in ['Total Return (%)', 'Max Drawdown (%)', 'Win Rate (%)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
    
    display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    display_df['Total Trades'] = display_df['Total Trades'].apply(lambda x: f"{x:.0f}")
    
    print(display_df.to_string(index=False))
    
    # Strategy performance by type
    print(f"\n📊 PERFORMANCE BY STRATEGY TYPE:")
    print("-" * 60)
    
    strategy_performance = summary.groupby('Strategy').agg({
        'Total Return (%)': ['mean', 'std', 'count'],
        'Sharpe Ratio': ['mean', 'max'],
        'Max Drawdown (%)': 'mean'
    }).round(2)
    
    strategy_performance.columns = [
        'Avg Return (%)', 'Return Std (%)', 'Count',
        'Avg Sharpe', 'Best Sharpe', 'Avg Drawdown (%)'
    ]
    
    print(strategy_performance.to_string())
    
    # Asset performance
    if 'Symbol' in summary.columns:
        print(f"\n📈 PERFORMANCE BY ASSET:")
        print("-" * 60)
        
        asset_performance = summary.groupby('Symbol').agg({
            'Total Return (%)': ['mean', 'count'],
            'Sharpe Ratio': 'mean',
            'Max Drawdown (%)': 'mean'
        }).round(2)
        
        asset_performance.columns = [
            'Avg Return (%)', 'Strategy Count', 'Avg Sharpe', 'Avg Drawdown (%)'
        ]
        
        print(asset_performance.to_string())


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🚀 COMPREHENSIVE CRYPTO TRADING STRATEGY BACKTESTS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Initial capital: ${args.initial_capital:,.0f}")
    print(f"Commission rate: {args.commission:.3f}")
    print(f"Slippage rate: {args.slippage:.4f}")
    
    if args.strategies:
        print(f"Selected strategies: {', '.join(args.strategies)}")
    else:
        print("Testing all available strategies")
    
    print(f"Parallel execution: {'Enabled' if args.parallel else 'Disabled'}")
    
    try:
        # Create pipeline configuration
        config = create_pipeline_config(args)
        
        # Initialize pipeline
        logger.info("Initializing backtest pipeline...")
        pipeline = BacktestPipeline(config)
        
        # Load and validate data
        logger.info("Loading CSV datasets...")
        datasets = pipeline.load_data()
        
        if not datasets:
            print("\n❌ ERROR: No valid datasets found!")
            print("Please ensure CSV files are present in the data directory with proper OHLCV format.")
            return 1
        
        print_data_summary(pipeline.data_loader)
        
        # Load strategy configurations
        logger.info("Loading strategy configurations...")
        configs = pipeline.load_strategy_configs()
        
        if not configs:
            print("\n❌ ERROR: No strategy configurations found!")
            print("Please ensure strategy config files are present in configs/strategies/")
            return 1
        
        print_strategy_summary(pipeline)
        
        # Create execution jobs
        logger.info("Creating backtest jobs...")
        jobs = pipeline.create_jobs()
        
        if not jobs:
            print("\n❌ ERROR: No backtest jobs created!")
            return 1
        
        print_execution_plan(pipeline)
        
        # Confirm execution
        print(f"\n⚠️  About to execute {len(jobs)} backtest jobs.")
        
        if not args.parallel and len(jobs) > 20:
            response = input("This may take a while. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Execution cancelled.")
                return 0
        
        # Execute backtests
        print("\n🔄 EXECUTING BACKTESTS...")
        print("=" * 80)
        
        pipeline_start = datetime.now()
        results = pipeline.run_full_pipeline()
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        
        if results['status'] == 'failed':
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return 1
        
        # Display results
        print_results_summary(pipeline, args.top_n)
        
        # Save results
        results_file = results['results_file']
        print(f"\n💾 Results saved to: {results_file}")
        print(f"📊 Summary CSV: {results_file.replace('.pkl', '.csv')}")
        
        # Final summary
        print(f"\n✅ EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {pipeline_duration:.1f} seconds")
        print(f"Average time per job: {pipeline_duration/len(jobs):.1f} seconds")
        
        if results.get('best_strategy'):
            best = results['best_strategy']
            print(f"\n🏆 BEST PERFORMING STRATEGY:")
            print(f"   {best['Strategy']} on {best['Symbol']} ({best['Timeframe']})")
            print(f"   Return: {best['Total Return (%)']:.1f}% | Sharpe: {best['Sharpe Ratio']:.2f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user.")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
