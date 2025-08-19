"""
Macroeconomic Data Module for YFinance Pipeline
Fetches macroeconomic indicators from FRED (Federal Reserve Economic Data) API
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

try:
    import fredapi
except ImportError:
    fredapi = None

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


class MacroDataFetcher:
    """
    Fetches and manages macroeconomic data from FRED API
    """
    
    # Key economic indicators with their FRED series IDs
    FRED_SERIES = {
        # Interest Rates
        'fed_funds_rate': 'FEDFUNDS',           # Federal Funds Rate
        'treasury_10y': 'GS10',                # 10-Year Treasury Rate
        'treasury_2y': 'GS2',                  # 2-Year Treasury Rate
        'treasury_3m': 'GS3M',                 # 3-Month Treasury Rate
        
        # Inflation
        'cpi_all': 'CPIAUCSL',                 # Consumer Price Index
        'cpi_core': 'CPILFESL',               # Core CPI (excluding food & energy)
        'pce_inflation': 'PCEPI',             # PCE Price Index
        'pce_core': 'PCEPILFE',               # Core PCE Price Index
        
        # Economic Growth
        'gdp': 'GDP',                          # Gross Domestic Product
        'gdp_real': 'GDPC1',                  # Real GDP
        'gdp_deflator': 'GDPDEF',             # GDP Deflator
        
        # Employment
        'unemployment_rate': 'UNRATE',         # Unemployment Rate
        'nonfarm_payrolls': 'PAYEMS',         # Total Nonfarm Payrolls
        'labor_force_participation': 'CIVPART', # Labor Force Participation Rate
        
        # Money Supply & Credit
        'money_supply_m1': 'M1SL',            # M1 Money Supply
        'money_supply_m2': 'M2SL',            # M2 Money Supply
        
        # Market Indicators
        'vix': 'VIXCLS',                      # VIX Volatility Index
        'dollar_index': 'DTWEXBGS',           # Trade Weighted USD Index
        
        # Commodities
        'oil_price': 'DCOILWTICO',            # WTI Crude Oil Price
        'gold_price': 'GOLDAMGBD228NLBM',     # Gold Price
        
        # Housing
        'housing_starts': 'HOUST',            # Housing Starts
        'case_shiller': 'CSUSHPISA',          # Case-Shiller Home Price Index
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED API connection
        
        Args:
            api_key: FRED API key. If None, will attempt to use environment variable
        """
        self.logger = main_logger
        self.error_tracker = ErrorTracker(self.logger)
        
        if fredapi is None:
            raise ImportError(
                "fredapi library is required. Install with: pip install fredapi"
            )
        
        try:
            self.fred = fredapi.Fred(api_key=api_key)
            self.logger.info("FRED API connection initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize FRED API: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0, logger=main_logger)
    @log_function_call(main_logger)
    def fetch_series(
        self, 
        series_id: str, 
        start_date: str, 
        end_date: str,
        frequency: str = None
    ) -> pd.DataFrame:
        """
        Fetch a single economic time series from FRED
        
        Args:
            series_id: FRED series identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Optional frequency (d=daily, w=weekly, m=monthly, q=quarterly, a=annual)
        
        Returns:
            DataFrame with date index and series values
        """
        try:
            self.logger.debug(f"Fetching FRED series {series_id} from {start_date} to {end_date}")
            
            # Fetch data from FRED
            data = self.fred.get_series(
                series_id,
                start=start_date,
                end=end_date,
                frequency=frequency
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for series {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame with proper structure
            df = data.to_frame(name='value')
            df.index.name = 'date'
            df = df.reset_index()
            df['series_id'] = series_id
            df['series_name'] = self._get_series_name(series_id)
            
            # Handle missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            self.logger.info(f"Successfully fetched {len(df)} records for series {series_id}")
            return df
            
        except Exception as e:
            error_msg = f"Failed to fetch FRED series {series_id}: {str(e)}"
            self.error_tracker.log_error("fred_fetch", series_id, error_msg)
            raise
    
    def _get_series_name(self, series_id: str) -> str:
        """Get human-readable name for a series ID"""
        name_mapping = {v: k for k, v in self.FRED_SERIES.items()}
        return name_mapping.get(series_id, series_id)
    
    @log_function_call(main_logger)
    def fetch_multiple_series(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
        frequency: str = None
    ) -> pd.DataFrame:
        """
        Fetch multiple economic time series
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Optional frequency for all series
        
        Returns:
            Combined DataFrame with all series data
        """
        all_data = []
        successful_series = []
        
        for series_id in series_ids:
            try:
                df = self.fetch_series(series_id, start_date, end_date, frequency)
                if not df.empty:
                    all_data.append(df)
                    successful_series.append(series_id)
                else:
                    self.error_tracker.log_warning(
                        "fred_fetch", 
                        series_id, 
                        "Empty dataset returned"
                    )
            
            except Exception as e:
                self.logger.error(f"Failed to fetch series {series_id}: {str(e)}")
                continue
            
            # Add small delay to be respectful to FRED API
            time.sleep(0.1)
        
        if not all_data:
            self.logger.error("No series data was successfully fetched")
            return pd.DataFrame()
        
        # Combine all series data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        self.logger.info(
            f"Successfully fetched {len(successful_series)}/{len(series_ids)} series "
            f"with {len(combined_df)} total records"
        )
        
        return combined_df
    
    @log_function_call(main_logger)
    def fetch_key_indicators(
        self,
        start_date: str,
        end_date: str,
        categories: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch key economic indicators
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            categories: List of category names to fetch. If None, fetches all.
                       Options: 'rates', 'inflation', 'growth', 'employment', 
                       'money', 'market', 'commodities', 'housing'
        
        Returns:
            DataFrame with key economic indicators
        """
        # Define category mappings
        category_series = {
            'rates': ['fed_funds_rate', 'treasury_10y', 'treasury_2y', 'treasury_3m'],
            'inflation': ['cpi_all', 'cpi_core', 'pce_inflation', 'pce_core'],
            'growth': ['gdp', 'gdp_real', 'gdp_deflator'],
            'employment': ['unemployment_rate', 'nonfarm_payrolls', 'labor_force_participation'],
            'money': ['money_supply_m1', 'money_supply_m2'],
            'market': ['vix', 'dollar_index'],
            'commodities': ['oil_price', 'gold_price'],
            'housing': ['housing_starts', 'case_shiller']
        }
        
        # Determine which series to fetch
        if categories is None:
            # Fetch all series
            series_to_fetch = list(self.FRED_SERIES.values())
        else:
            series_to_fetch = []
            for category in categories:
                if category in category_series:
                    for series_name in category_series[category]:
                        if series_name in self.FRED_SERIES:
                            series_to_fetch.append(self.FRED_SERIES[series_name])
                else:
                    self.logger.warning(f"Unknown category: {category}")
        
        self.logger.info(f"Fetching {len(series_to_fetch)} economic indicators")
        
        return self.fetch_multiple_series(series_to_fetch, start_date, end_date)
    
    @log_function_call(main_logger)
    def validate_macro_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate macroeconomic data quality
        
        Args:
            df: DataFrame with macro data
        
        Returns:
            Validation report dictionary
        """
        if df.empty:
            return {"error": "Empty dataset provided for validation"}
        
        report = {
            "total_records": len(df),
            "unique_series": df['series_id'].nunique() if 'series_id' in df.columns else 0,
            "date_range": {
                "start": df['date'].min() if 'date' in df.columns else None,
                "end": df['date'].max() if 'date' in df.columns else None
            },
            "null_values": {},
            "series_summary": {}
        }
        
        # Check for null values
        for column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                report["null_values"][column] = int(null_count)
        
        # Series-specific validation
        if 'series_id' in df.columns:
            for series_id in df['series_id'].unique():
                series_data = df[df['series_id'] == series_id]
                
                series_report = {
                    "record_count": len(series_data),
                    "null_values": int(series_data['value'].isnull().sum()),
                    "min_value": float(series_data['value'].min()) if not series_data['value'].isnull().all() else None,
                    "max_value": float(series_data['value'].max()) if not series_data['value'].isnull().all() else None,
                    "first_date": series_data['date'].min() if 'date' in series_data.columns else None,
                    "last_date": series_data['date'].max() if 'date' in series_data.columns else None
                }
                
                report["series_summary"][series_id] = series_report
        
        self.logger.info(f"Macro data validation completed: {report['total_records']} records, {report['unique_series']} series")
        
        return report
    
    @log_function_call(main_logger)
    def save_to_sqlite(
        self,
        df: pd.DataFrame,
        db_path: Path,
        table_name: str = 'macro_data'
    ):
        """
        Save macroeconomic data to SQLite database
        
        Args:
            df: DataFrame with macro data
            db_path: Path to SQLite database
            table_name: Name of the table to create/update
        """
        if df.empty:
            self.logger.warning("No data to save to database")
            return
        
        try:
            # Ensure directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata columns
            df_to_save = df.copy()
            df_to_save['updated_at'] = datetime.now().isoformat()
            
            # Save to SQLite
            with sqlite3.connect(db_path) as conn:
                df_to_save.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Create indices for efficient querying
                conn.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_series_date 
                    ON {table_name}(series_id, date)
                ''')
                
                conn.execute(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_date 
                    ON {table_name}(date)
                ''')
            
            self.logger.info(f"Saved {len(df_to_save)} macro records to {db_path}/{table_name}")
            
        except Exception as e:
            error_msg = f"Failed to save macro data to database: {str(e)}"
            self.error_tracker.log_error("database_save", "macro_data", error_msg)
            raise


# Convenience functions for easy usage
def fetch_fred_data(
    api_key: str,
    start_date: str,
    end_date: str,
    categories: List[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to fetch FRED data with validation
    
    Args:
        api_key: FRED API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        categories: List of category names to fetch
    
    Returns:
        Tuple of (DataFrame with data, validation report)
    """
    try:
        fetcher = MacroDataFetcher(api_key)
        df = fetcher.fetch_key_indicators(start_date, end_date, categories)
        report = fetcher.validate_macro_data(df)
        
        return df, report
        
    except Exception as e:
        main_logger.error(f"Failed to fetch FRED data: {str(e)}")
        return pd.DataFrame(), {"error": str(e)}


def get_default_macro_series() -> List[str]:
    """Get list of default macroeconomic series for basic analysis"""
    return [
        'FEDFUNDS',    # Fed Funds Rate
        'GS10',        # 10-Year Treasury
        'CPIAUCSL',    # CPI
        'UNRATE',      # Unemployment Rate
        'GDP',         # GDP
        'VIXCLS',      # VIX
        'DTWEXBGS'     # Dollar Index
    ]


# Example usage and testing
if __name__ == "__main__":
    # Test the macro data fetcher (requires FRED API key)
    import os
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("Set FRED_API_KEY environment variable to test")
        exit(1)
    
    # Test fetching data
    fetcher = MacroDataFetcher(api_key)
    
    # Fetch basic economic indicators for the last year
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Fetching macro data from {start_date} to {end_date}")
    
    df = fetcher.fetch_key_indicators(
        start_date=start_date,
        end_date=end_date,
        categories=['rates', 'inflation', 'employment']
    )
    
    if not df.empty:
        print(f"Fetched {len(df)} records for {df['series_id'].nunique()} series")
        
        # Validate data
        report = fetcher.validate_macro_data(df)
        print("Validation report:", report)
        
        # Save to test database
        test_db = Path("test_macro_data.db")
        fetcher.save_to_sqlite(df, test_db)
        print(f"Data saved to {test_db}")
        
        # Clean up test file
        test_db.unlink(missing_ok=True)
    else:
        print("No data was fetched")
