"""
Multi-Source Cross-Validation Module for YFinance Pipeline
Validates financial data by comparing multiple sources (Yahoo Finance vs Alpha Vantage)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import json

try:
    import alpha_vantage
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    alpha_vantage = None

try:
    from .logger import main_logger, retry_with_backoff, ErrorTracker, log_function_call
except ImportError:
    # Fallback for standalone execution
    import logging
    logging.basicConfig(level=logging.INFO)
    main_logger = logging.getLogger(__name__)
    
    def retry_with_backoff(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def log_function_call(logger):
        def decorator(func):
            return func
        return decorator
    
    class ErrorTracker:
        def __init__(self, logger):
            self.logger = logger
        def log_error(self, *args): pass
        def log_warning(self, *args): pass


class DataValidator:
    """
    Cross-validates financial data between Yahoo Finance and Alpha Vantage
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize data validator with multiple sources
        
        Args:
            alpha_vantage_key: Alpha Vantage API key for secondary validation
        """
        self.logger = main_logger
        self.error_tracker = ErrorTracker(self.logger)
        
        # Initialize Alpha Vantage client if available
        self.alpha_vantage_available = False
        if alpha_vantage and alpha_vantage_key:
            try:
                self.av_client = TimeSeries(key=alpha_vantage_key, output_format='pandas')
                self.alpha_vantage_available = True
                self.logger.info("Alpha Vantage client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Alpha Vantage: {str(e)}")
        else:
            if not alpha_vantage:
                self.logger.warning("alpha_vantage library not available. Install with: pip install alpha-vantage")
            else:
                self.logger.warning("No Alpha Vantage API key provided")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0, logger=main_logger)
    @log_function_call(main_logger)
    def fetch_yahoo_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance (primary source)
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with Yahoo Finance data
        """
        try:
            self.logger.debug(f"Fetching Yahoo data for {ticker}")
            
            # Fetch data using yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"No Yahoo data returned for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names and structure
            df = df.reset_index()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df['ticker'] = ticker
            df['source'] = 'yahoo'
            
            # Ensure we have the required columns
            required_cols = ['date', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.logger.info(f"Successfully fetched {len(df)} records from Yahoo for {ticker}")
            return df
            
        except Exception as e:
            error_msg = f"Failed to fetch Yahoo data for {ticker}: {str(e)}"
            self.error_tracker.log_error("yahoo_fetch", ticker, error_msg)
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0, logger=main_logger)
    @log_function_call(main_logger)
    def fetch_alpha_vantage_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage (secondary source)
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with Alpha Vantage data
        """
        if not self.alpha_vantage_available:
            self.logger.warning("Alpha Vantage not available, skipping")
            return pd.DataFrame()
        
        try:
            self.logger.debug(f"Fetching Alpha Vantage data for {ticker}")
            
            # Fetch daily data from Alpha Vantage
            data, meta_data = self.av_client.get_daily(symbol=ticker, outputsize='full')
            
            if data.empty:
                self.logger.warning(f"No Alpha Vantage data returned for {ticker}")
                return pd.DataFrame()
            
            # Process and standardize the data
            df = data.reset_index()
            df.columns = ['date'] + [col.split('. ')[1].lower().replace(' ', '_') for col in df.columns[1:]]
            df['ticker'] = ticker
            df['source'] = 'alpha_vantage'
            
            # Filter by date range
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            # Ensure we have the required columns
            if 'close' not in df.columns:
                raise ValueError("Missing 'close' column in Alpha Vantage data")
            
            self.logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage for {ticker}")
            
            # Add rate limiting to respect Alpha Vantage API limits
            time.sleep(12)  # Alpha Vantage free tier: 5 requests per minute
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to fetch Alpha Vantage data for {ticker}: {str(e)}"
            self.error_tracker.log_error("alpha_vantage_fetch", ticker, error_msg)
            raise
    
    @log_function_call(main_logger)
    def compare_sources(
        self,
        yahoo_df: pd.DataFrame,
        alpha_df: pd.DataFrame,
        tolerance_pct: float = 0.5
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Compare data between Yahoo Finance and Alpha Vantage
        
        Args:
            yahoo_df: DataFrame with Yahoo Finance data
            alpha_df: DataFrame with Alpha Vantage data
            tolerance_pct: Tolerance percentage for price differences
        
        Returns:
            Tuple of (merged DataFrame, list of discrepancy records)
        """
        if yahoo_df.empty or alpha_df.empty:
            self.logger.warning("One or both data sources are empty, cannot compare")
            return pd.DataFrame(), []
        
        try:
            # Prepare data for merging
            yahoo_merge = yahoo_df[['date', 'ticker', 'close']].copy()
            yahoo_merge['date'] = pd.to_datetime(yahoo_merge['date']).dt.date
            yahoo_merge = yahoo_merge.rename(columns={'close': 'yahoo_close'})
            
            alpha_merge = alpha_df[['date', 'ticker', 'close']].copy()
            alpha_merge['date'] = pd.to_datetime(alpha_merge['date']).dt.date
            alpha_merge = alpha_merge.rename(columns={'close': 'alpha_close'})
            
            # Merge on date and ticker
            merged = pd.merge(
                yahoo_merge,
                alpha_merge,
                on=['date', 'ticker'],
                how='inner'
            )
            
            if merged.empty:
                self.logger.warning("No overlapping data between sources")
                return pd.DataFrame(), []
            
            # Calculate differences
            merged['yahoo_close'] = pd.to_numeric(merged['yahoo_close'], errors='coerce')
            merged['alpha_close'] = pd.to_numeric(merged['alpha_close'], errors='coerce')
            
            # Remove rows where either price is NaN
            merged = merged.dropna(subset=['yahoo_close', 'alpha_close'])
            
            if merged.empty:
                self.logger.warning("No valid price data after cleaning")
                return merged, []
            
            # Calculate percentage difference
            merged['price_diff'] = merged['yahoo_close'] - merged['alpha_close']
            merged['diff_pct'] = abs(merged['price_diff'] / merged['yahoo_close'] * 100)
            
            # Identify discrepancies
            discrepancies = merged[merged['diff_pct'] > tolerance_pct]
            
            # Create discrepancy records
            discrepancy_records = []
            for _, row in discrepancies.iterrows():
                record = {
                    "ticker": row['ticker'],
                    "date": str(row['date']),
                    "yahoo_close": float(row['yahoo_close']),
                    "alpha_close": float(row['alpha_close']),
                    "price_diff": float(row['price_diff']),
                    "diff_pct": float(row['diff_pct']),
                    "resolved_as": "Yahoo"  # Default to Yahoo as source of truth
                }
                discrepancy_records.append(record)
            
            # Add resolved price (defaulting to Yahoo)
            merged['resolved_close'] = merged['yahoo_close']
            merged['discrepancy_flag'] = merged['diff_pct'] > tolerance_pct
            
            self.logger.info(
                f"Compared {len(merged)} records, found {len(discrepancy_records)} "
                f"discrepancies above {tolerance_pct}% threshold"
            )
            
            return merged, discrepancy_records
            
        except Exception as e:
            error_msg = f"Failed to compare data sources: {str(e)}"
            self.error_tracker.log_error("data_comparison", "comparison", error_msg)
            raise
    
    @log_function_call(main_logger)
    def validate_ticker_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        tolerance_pct: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate data for a single ticker across multiple sources
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tolerance_pct: Tolerance percentage for price differences
        
        Returns:
            Tuple of (validated DataFrame, validation report)
        """
        validation_report = {
            "ticker": ticker,
            "date_range": f"{start_date} to {end_date}",
            "sources_attempted": ["yahoo"],
            "sources_successful": [],
            "total_records": 0,
            "discrepancies": [],
            "validation_status": "pending"
        }
        
        try:
            # Fetch Yahoo data (primary source)
            yahoo_df = self.fetch_yahoo_data(ticker, start_date, end_date)
            
            if not yahoo_df.empty:
                validation_report["sources_successful"].append("yahoo")
                validation_report["total_records"] = len(yahoo_df)
                validation_report["yahoo_records"] = len(yahoo_df)
            
            # Attempt to fetch Alpha Vantage data if available
            alpha_df = pd.DataFrame()
            if self.alpha_vantage_available:
                validation_report["sources_attempted"].append("alpha_vantage")
                try:
                    alpha_df = self.fetch_alpha_vantage_data(ticker, start_date, end_date)
                    
                    if not alpha_df.empty:
                        validation_report["sources_successful"].append("alpha_vantage")
                        validation_report["alpha_records"] = len(alpha_df)
                    
                except Exception as e:
                    self.logger.warning(f"Alpha Vantage fetch failed for {ticker}: {str(e)}")
            
            # If we have data from both sources, compare them
            if not yahoo_df.empty and not alpha_df.empty:
                merged_df, discrepancies = self.compare_sources(
                    yahoo_df, alpha_df, tolerance_pct
                )
                
                validation_report["discrepancies"] = discrepancies
                validation_report["discrepancy_count"] = len(discrepancies)
                validation_report["comparison_records"] = len(merged_df)
                
                # Return the merged data with validation flags
                result_df = merged_df[['date', 'ticker', 'resolved_close', 'discrepancy_flag']]
                result_df = result_df.rename(columns={'resolved_close': 'close'})
                
                validation_report["validation_status"] = "cross_validated"
                
            elif not yahoo_df.empty:
                # Only Yahoo data available
                result_df = yahoo_df[['date', 'ticker', 'close']].copy()
                result_df['discrepancy_flag'] = False
                
                validation_report["validation_status"] = "single_source_yahoo"
                
            else:
                # No data available
                result_df = pd.DataFrame()
                validation_report["validation_status"] = "no_data"
                validation_report["error"] = "No data available from any source"
            
            self.logger.info(f"Validation completed for {ticker}: {validation_report['validation_status']}")
            
            return result_df, validation_report
            
        except Exception as e:
            error_msg = f"Validation failed for {ticker}: {str(e)}"
            validation_report["validation_status"] = "failed"
            validation_report["error"] = error_msg
            self.error_tracker.log_error("validation", ticker, error_msg)
            
            return pd.DataFrame(), validation_report
    
    @log_function_call(main_logger)
    def validate_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        tolerance_pct: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate data for multiple tickers
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            tolerance_pct: Tolerance percentage for price differences
        
        Returns:
            Tuple of (combined validated DataFrame, comprehensive validation report)
        """
        all_data = []
        ticker_reports = {}
        all_discrepancies = []
        
        summary_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "date_range": f"{start_date} to {end_date}",
            "total_tickers": len(tickers),
            "successful_tickers": 0,
            "failed_tickers": 0,
            "cross_validated_tickers": 0,
            "single_source_tickers": 0,
            "total_discrepancies": 0,
            "tolerance_pct": tolerance_pct
        }
        
        for ticker in tickers:
            try:
                self.logger.info(f"Validating ticker {ticker}")
                
                ticker_df, ticker_report = self.validate_ticker_data(
                    ticker, start_date, end_date, tolerance_pct
                )
                
                ticker_reports[ticker] = ticker_report
                
                if not ticker_df.empty:
                    all_data.append(ticker_df)
                    summary_report["successful_tickers"] += 1
                    
                    if ticker_report["validation_status"] == "cross_validated":
                        summary_report["cross_validated_tickers"] += 1
                    elif ticker_report["validation_status"] == "single_source_yahoo":
                        summary_report["single_source_tickers"] += 1
                
                # Collect discrepancies
                if "discrepancies" in ticker_report:
                    all_discrepancies.extend(ticker_report["discrepancies"])
                
                if ticker_report["validation_status"] == "failed":
                    summary_report["failed_tickers"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to validate ticker {ticker}: {str(e)}")
                summary_report["failed_tickers"] += 1
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            summary_report["total_records"] = len(combined_df)
        else:
            combined_df = pd.DataFrame()
            summary_report["total_records"] = 0
        
        summary_report["total_discrepancies"] = len(all_discrepancies)
        
        # Create comprehensive report
        comprehensive_report = {
            "summary": summary_report,
            "ticker_details": ticker_reports,
            "all_discrepancies": all_discrepancies,
            "error_summary": self.error_tracker.get_summary()
        }
        
        self.logger.info(
            f"Validation completed for {summary_report['successful_tickers']}/{len(tickers)} tickers. "
            f"Found {len(all_discrepancies)} total discrepancies."
        )
        
        return combined_df, comprehensive_report
    
    @log_function_call(main_logger)
    def save_validation_report(
        self,
        report: Dict,
        filepath: Path
    ):
        """
        Save validation report to JSON file
        
        Args:
            report: Validation report dictionary
            filepath: Path to save the report
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to save validation report: {str(e)}"
            self.error_tracker.log_error("report_save", "validation_report", error_msg)
            raise
    
    @log_function_call(main_logger)
    def save_anomaly_report(
        self,
        discrepancies: List[Dict],
        filepath: Path
    ):
        """
        Save anomaly/discrepancy report as CSV
        
        Args:
            discrepancies: List of discrepancy records
            filepath: Path to save the CSV report
        """
        try:
            if not discrepancies:
                self.logger.info("No discrepancies to save")
                return
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(discrepancies)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Anomaly report with {len(discrepancies)} records saved to {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to save anomaly report: {str(e)}"
            self.error_tracker.log_error("report_save", "anomaly_report", error_msg)
            raise


# Convenience functions
def validate_tickers(
    tickers: List[str],
    start_date: str,
    end_date: str,
    alpha_vantage_key: str = None,
    tolerance_pct: float = 0.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to validate multiple tickers
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        alpha_vantage_key: Optional Alpha Vantage API key
        tolerance_pct: Tolerance percentage for price differences
    
    Returns:
        Tuple of (validated DataFrame, validation report)
    """
    validator = DataValidator(alpha_vantage_key)
    return validator.validate_multiple_tickers(tickers, start_date, end_date, tolerance_pct)


# Example usage and testing
if __name__ == "__main__":
    # Test the validation system
    import os
    
    # Test with a small set of tickers
    test_tickers = ['AAPL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Testing validation for {test_tickers} from {start_date} to {end_date}")
    
    # Test without Alpha Vantage (Yahoo only)
    validator = DataValidator()
    
    for ticker in test_tickers:
        df, report = validator.validate_ticker_data(ticker, start_date, end_date)
        print(f"{ticker}: {report['validation_status']} - {len(df)} records")
    
    # Test multiple tickers
    combined_df, comprehensive_report = validator.validate_multiple_tickers(
        test_tickers, start_date, end_date
    )
    
    print(f"Combined validation: {len(combined_df)} total records")
    print(f"Summary: {comprehensive_report['summary']}")
    
    print("Validation system test completed")
