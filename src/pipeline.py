import argparse
import sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import time
import os
# import schedule  # pip install schedule - for automated scheduling

# Import new modules
try:
    from .logger import main_logger, ErrorTracker, setup_logger
    from .macro import MacroDataFetcher, fetch_fred_data
    from .validation import DataValidator, validate_tickers
    from .database import DatabaseManager, generate_request_id
    from .output_manager import create_request_outputs
    modules_available = True
except ImportError as e:
    print(f"Warning: Enhanced modules not available: {e}")
    modules_available = False
    # Fallback to basic logging
    import logging
    logging.basicConfig(level=logging.INFO)
    main_logger = logging.getLogger(__name__)

def fetch_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False, threads=True)
    frames = []
    for t in tickers:
        df = raw[t].copy()
        df = df.reset_index().rename(columns=str.title)
        df['Ticker'] = t
        # standardize column names
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
        frames.append(df[['date','Ticker','open','high','low','close','adj_close','volume']])
    return pd.concat(frames, ignore_index=True)

def validate(df: pd.DataFrame):
    report = {}

    # basic nulls
    report['null_counts'] = df.isna().sum().to_dict()

    # duplicates on (ticker, date)
    dup_mask = df.duplicated(subset=['Ticker','date'], keep=False)
    report['duplicate_rows'] = int(dup_mask.sum())
    df = df.drop_duplicates(subset=['Ticker','date'], keep='last')

    # negative or impossible values
    report['neg_close'] = int((df['close'] <= 0).sum())
    report['neg_volume'] = int((df['volume'] < 0).sum())

    # returns & extreme jumps flag (z-score > 6)
    df = df.sort_values(['Ticker','date'])
    df['ret'] = df.groupby('Ticker')['close'].pct_change()
    z = (df['ret'] - df['ret'].mean(skipna=True)) / df['ret'].std(skipna=True)
    report['extreme_moves'] = int((z.abs() > 6).sum())

    # approximate missing business days check (per ticker)
    missing_days = {}
    for t, g in df.groupby('Ticker'):
        if g.empty: 
            missing_days[t] = 0
            continue
        cal = pd.bdate_range(g['date'].min(), g['date'].max())
        missing = len(set(cal.date) - set(pd.to_datetime(g['date']).dt.date))
        missing_days[t] = int(missing)
    report['approx_missing_bdays'] = missing_days

    return df, report

def transform(df: pd.DataFrame):
    # features commonly used by traders/researchers
    df = df.sort_values(['Ticker','date']).copy()
    df['return'] = df.groupby('Ticker')['close'].pct_change()
    df['ma20'] = df.groupby('Ticker')['close'].transform(lambda s: s.rolling(20).mean())
    df['ma50'] = df.groupby('Ticker')['close'].transform(lambda s: s.rolling(50).mean())
    df['vol20'] = df.groupby('Ticker')['return'].transform(lambda s: s.rolling(20).std())
    
    # Add update timestamp for tracking when data was processed
    df['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # tidy types
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def save_sqlite(df: pd.DataFrame, db_path: Path, table: str = 'prices'):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists='replace', index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_ticker_date ON {table}(Ticker, date);')

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_validation(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def main():
    ap = argparse.ArgumentParser(
        description="Enhanced YFinance Pipeline with multi-source validation and macro data"
    )
    ap.add_argument('--tickers', nargs='+', default=['AAPL','TSLA','SPY','GLD','USO','UUP'])
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'))
    ap.add_argument('--db', default='data/market_data.db')
    
    # Enhanced features - macro is now automatic when API key available
    ap.add_argument('--enable-validation', action='store_true', 
                    help='Enable multi-source validation with Alpha Vantage')
    ap.add_argument('--alpha-vantage-key', type=str,
                    help='Alpha Vantage API key for validation')
    ap.add_argument('--fred-api-key', type=str,
                    help='FRED API key for macroeconomic data')
    ap.add_argument('--tolerance', type=float, default=0.5,
                    help='Price tolerance percentage for cross-validation')
    ap.add_argument('--macro-categories', nargs='+',
                    help='Macro categories to fetch: rates, inflation, growth, employment, etc.')
    
    args = ap.parse_args()

    # Initialize enhanced components if available
    if modules_available:
        logger = main_logger
        error_tracker = ErrorTracker(logger)
        logger.info("🚀 Starting enhanced YFinance pipeline")
    else:
        logger = main_logger
        error_tracker = None
        logger.info("Starting basic YFinance pipeline (enhanced features unavailable)")

    # Generate unique request ID
    request_id = generate_request_id(args.tickers, args.start, args.end) if modules_available else "basic_request"
    logger.info(f"📋 Request ID: {request_id}")

    try:
        # Initialize database manager
        if modules_available:
            db_manager = DatabaseManager(Path(args.db))
            db_manager.create_request_log(request_id, args.tickers, args.start, args.end)

        # 1. FETCH TICKER DATA
        logger.info(f"📈 Fetching ticker data for {len(args.tickers)} tickers: {args.tickers}")
        df_raw = fetch_data(args.tickers, args.start, args.end)
        df_valid, basic_report = validate(df_raw)
        df_feat = transform(df_valid)

        # 2. CROSS-VALIDATION (Ticker data only)
        validation_report = None
        cross_validation_results = None
        if args.enable_validation and modules_available:
            logger.info("🔍 Starting multi-source validation for ticker data")
            try:
                alpha_key = args.alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
                if alpha_key:
                    validator = DataValidator(alpha_key)
                    validated_df, validation_report = validator.validate_multiple_tickers(
                        args.tickers, args.start, args.end, args.tolerance
                    )
                    
                    # Merge validation flags with main data
                    if not validated_df.empty:
                        validated_df['date'] = pd.to_datetime(validated_df['date']).dt.date
                        df_feat = df_feat.merge(
                            validated_df[['date', 'ticker', 'discrepancy_flag']],
                            left_on=['date', 'Ticker'],
                            right_on=['date', 'ticker'],
                            how='left'
                        )
                        df_feat = df_feat.drop('ticker', axis=1)
                        df_feat['discrepancy_flag'] = df_feat['discrepancy_flag'].fillna(False)
                        
                        # Extract cross-validation results
                        cross_validation_results = validation_report.get('all_discrepancies', [])
                        logger.info(f"✅ Cross-validation completed: {len(cross_validation_results)} discrepancies found")
                else:
                    logger.warning("⚠️ No Alpha Vantage API key provided, skipping cross-validation")
            except Exception as e:
                logger.error(f"❌ Multi-source validation failed: {str(e)}")
                if error_tracker:
                    error_tracker.log_error("validation", "multi_source", str(e))

        # 3. FETCH MACROECONOMIC DATA (Automatic when API key available)
        macro_df = None
        macro_validation = None
        fred_key = args.fred_api_key or os.getenv('FRED_API_KEY')
        if fred_key and modules_available:
            logger.info("📊 Fetching macroeconomic data for same time period")
            try:
                fetcher = MacroDataFetcher(fred_key)
                macro_df = fetcher.fetch_key_indicators(
                    args.start, args.end, args.macro_categories
                )
                
                if not macro_df.empty:
                    # Validate macro data (basic checks only)
                    macro_validation = fetcher.validate_macro_data(macro_df)
                    logger.info(f"📈 Fetched {len(macro_df)} macro records, {macro_df['series_id'].nunique()} series")
                else:
                    logger.warning("⚠️ No macroeconomic data was fetched")
            except Exception as e:
                logger.error(f"❌ Macro data fetch failed: {str(e)}")
                if error_tracker:
                    error_tracker.log_error("macro_fetch", "fred", str(e))
        elif not fred_key:
            logger.info("💡 Tip: Add FRED API key to include economic context")

        # 4. SAVE TO DATABASE (with duplicate checking)
        if modules_available:
            logger.info("💾 Saving data to database (checking for duplicates)")
            market_records = db_manager.save_market_data(df_feat, request_id)
            macro_records = db_manager.save_macro_data(macro_df, request_id) if macro_df is not None else 0
            
            # Save cross-validation results
            if cross_validation_results:
                db_manager.save_cross_validation(request_id, cross_validation_results)
            
            # Update request status
            db_manager.update_request_status(
                request_id, 
                'completed',
                total_records_fetched=market_records,
                macro_records_fetched=macro_records,
                validation_performed=bool(validation_report)
            )
        else:
            # Fallback for basic mode
            save_sqlite(df_feat, Path(args.db))

        # 5. CREATE OUTPUT FILES
        if modules_available:
            logger.info("📄 Creating output files")
            
            # Create all output files
            file_paths = create_request_outputs(
                df_feat,
                request_id,
                args.tickers,
                args.start,
                args.end,
                basic_report,
                validation_report,
                macro_validation,
                execution_summary={
                    "total_market_records": len(df_feat),
                    "total_macro_records": len(macro_df) if macro_df is not None else 0,
                    "cross_validation_performed": bool(validation_report),
                    "discrepancies_found": len(cross_validation_results) if cross_validation_results else 0
                }
            )
            
            logger.info(f"📄 Created CSV: {file_paths['csv'].name}")
            logger.info(f"📋 Created validation report: {file_paths['validation'].name}")
            if 'log' in file_paths:
                logger.info(f"📜 Created execution log: {file_paths['log'].name}")
        else:
            # Fallback for basic mode
            save_csv(df_feat, Path("outputs/prices.csv"))
            save_validation(basic_report, Path("outputs/validation_report.json"))

        # 6. FINAL SUMMARY
        logger.info("="*70)
        logger.info("🎉 PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        logger.info(f"📋 Request ID: {request_id}")
        logger.info(f"✅ Processed {len(df_feat)} market records for {df_feat['Ticker'].nunique()} tickers")
        logger.info(f"📊 Date range: {args.start} to {args.end}")
        
        if validation_report:
            summary = validation_report.get('summary', {})
            logger.info(f"🔍 Cross-validation: {summary.get('cross_validated_tickers', 0)} tickers validated, "
                       f"{summary.get('total_discrepancies', 0)} discrepancies found")
        
        if macro_df is not None and not macro_df.empty:
            logger.info(f"📈 Macro data: {len(macro_df)} records, {macro_df['series_id'].nunique()} economic series")
        
        if modules_available:
            db_stats = db_manager.get_database_stats()
            logger.info(f"🗄️ Database now contains {db_stats['market_data_count']} total market records")
            logger.info(f"📄 Files saved in outputs/ with timestamp {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        logger.info("="*70)

        # Console output for backward compatibility
        print(f"\n🎉 SUCCESS! Request {request_id}")
        print(f"📊 Processed: {len(df_feat)} records, {df_feat['Ticker'].nunique()} tickers")
        if validation_report:
            print("✅ Enhanced validation completed with multi-source cross-validation")
        else:
            print(f"✅ Basic validation completed: {basic_report}")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        if error_tracker:
            error_tracker.log_error("pipeline_execution", "main", str(e))
        
        # Update request status as failed
        if modules_available and 'db_manager' in locals():
            db_manager.update_request_status(request_id, 'failed', error_count=1)
        
        raise

# =============================================================================
# AUTOMATED SCHEDULING FEATURES (COMMENTED OUT FOR FUTURE USE)
# =============================================================================

# def setup_logging():
#     """Set up production-ready logging"""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler('logs/pipeline.log'),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# def run_daily_update():
#     """Run daily data update - called by scheduler"""
#     logger = setup_logging()
#     logger.info("Starting automated daily update...")
    
#     try:
#         # Get yesterday's data (markets are usually T+1)
#         end_date = datetime.now().strftime('%Y-%m-%d')
#         start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Last week
        
#         # Default tickers for daily updates
#         tickers = ['AAPL', 'TSLA', 'SPY', 'QQQ', 'VTI', 'GLD', 'UUP']
        
#         logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
        
#         # Run pipeline
#         df_raw = fetch_data(tickers, start_date, end_date)
#         df_valid, report = validate(df_raw)
#         df_feat = transform(df_valid)
        
#         # Save with timestamp in filename for daily tracking
#         timestamp = datetime.now().strftime('%Y%m%d')
#         save_sqlite(df_feat, Path(f"data/market_data_{timestamp}.db"))
#         save_csv(df_feat, Path(f"outputs/prices_{timestamp}.csv"))
#         save_validation(report, Path(f"outputs/validation_{timestamp}.json"))
        
#         # Also update the main files
#         save_sqlite(df_feat, Path("data/market_data.db"))
#         save_csv(df_feat, Path("outputs/prices.csv"))
#         save_validation(report, Path("outputs/validation_report.json"))
        
#         logger.info(f"Daily update completed successfully. Processed {len(df_feat)} rows")
        
#     except Exception as e:
#         logger.error(f"Daily update failed: {str(e)}")
#         # Could send email/slack notification here
#         raise

# def run_weekly_full_update():
#     """Run comprehensive weekly update with extended history"""
#     logger = setup_logging()
#     logger.info("Starting weekly full update...")
    
#     try:
#         # Get last 3 months of data for comprehensive update
#         end_date = datetime.now().strftime('%Y-%m-%d')
#         start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
#         # Extended ticker list for weekly updates
#         tickers = [
#             'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',  # Tech
#             'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO',        # ETFs
#             'GLD', 'SLV', 'USO', 'UUP', 'TLT', 'HYG'         # Commodities/Bonds
#         ]
        
#         logger.info(f"Running weekly update for {len(tickers)} tickers")
        
#         df_raw = fetch_data(tickers, start_date, end_date)
#         df_valid, report = validate(df_raw)
#         df_feat = transform(df_valid)
        
#         # Save weekly backup
#         timestamp = datetime.now().strftime('%Y%m%d')
#         save_sqlite(df_feat, Path(f"data/weekly_backup_{timestamp}.db"))
        
#         # Update main files
#         save_sqlite(df_feat, Path("data/market_data.db"))
#         save_csv(df_feat, Path("outputs/prices.csv"))
#         save_validation(report, Path("outputs/validation_report.json"))
        
#         logger.info(f"Weekly update completed. Processed {len(df_feat)} rows for {len(tickers)} tickers")
        
#     except Exception as e:
#         logger.error(f"Weekly update failed: {str(e)}")
#         raise

# def setup_scheduler():
#     """Set up automated scheduling"""
#     # Daily updates at 6 PM EST (after market close)
#     schedule.every().day.at("18:00").do(run_daily_update)
    
#     # Weekly comprehensive update on Sundays at 8 AM
#     schedule.every().sunday.at("08:00").do(run_weekly_full_update)
    
#     print("📅 Scheduler configured:")
#     print("   Daily updates: 6:00 PM EST")
#     print("   Weekly updates: Sunday 8:00 AM EST")
#     print("   Press Ctrl+C to stop...")

# def run_scheduler():
#     """Run the automated scheduler - call this to start automation"""
#     setup_scheduler()
    
#     while True:
#         schedule.run_pending()
#         time.sleep(60)  # Check every minute

# def run_manual_update(tickers=None, days_back=30):
#     """Manual update function for ad-hoc runs"""
#     if tickers is None:
#         tickers = ['AAPL', 'TSLA', 'SPY', 'GLD', 'USO', 'UUP']
    
#     end_date = datetime.now().strftime('%Y-%m-%d')
#     start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
#     print(f"Manual update: {tickers} from {start_date} to {end_date}")
    
#     df_raw = fetch_data(tickers, start_date, end_date)
#     df_valid, report = validate(df_raw)
#     df_feat = transform(df_valid)
    
#     save_sqlite(df_feat, Path("data/market_data.db"))
#     save_csv(df_feat, Path("outputs/prices.csv"))
#     save_validation(report, Path("outputs/validation_report.json"))
    
#     print(f"Manual update completed: {len(df_feat)} rows processed")

# =============================================================================
# USAGE EXAMPLES FOR AUTOMATION (COMMENTED OUT)
# =============================================================================

# if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--schedule':
#     # Run automated scheduler
#     run_scheduler()

# if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--daily':
#     # Run one-time daily update
#     run_daily_update()

# if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--weekly':
#     # Run one-time weekly update
#     run_weekly_full_update()

if __name__ == '__main__':
    main()
