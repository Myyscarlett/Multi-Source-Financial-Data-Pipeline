"""
Database management module for YFinance Enhanced Pipeline
Handles database schema creation, data storage, and duplicate prevention
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import hashlib

from .logger import main_logger, log_function_call


class DatabaseManager:
    """
    Manages all database operations for the YFinance pipeline
    """
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.logger = main_logger
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
    
    @log_function_call(main_logger)
    def _initialize_schema(self):
        """Create all required tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            # Request log table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS request_log (
                    request_id TEXT PRIMARY KEY,
                    request_timestamp DATETIME NOT NULL,
                    tickers TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    status TEXT NOT NULL,
                    total_records_fetched INTEGER DEFAULT 0,
                    macro_records_fetched INTEGER DEFAULT 0,
                    validation_performed BOOLEAN DEFAULT FALSE,
                    error_count INTEGER DEFAULT 0
                )
            ''')
            
            # Market data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(10,4),
                    high DECIMAL(10,4),
                    low DECIMAL(10,4),
                    close DECIMAL(10,4),
                    adj_close DECIMAL(10,4),
                    volume BIGINT,
                    return_pct DECIMAL(8,6),
                    ma20 DECIMAL(10,4),
                    ma50 DECIMAL(10,4),
                    vol20 DECIMAL(8,6),
                    fetch_timestamp DATETIME NOT NULL,
                    request_id TEXT NOT NULL,
                    discrepancy_flag BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (request_id) REFERENCES request_log(request_id),
                    UNIQUE(ticker, date, request_id)
                )
            ''')
            
            # Macro data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS macro_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    value DECIMAL(15,6),
                    series_name TEXT,
                    category TEXT,
                    fetch_timestamp DATETIME NOT NULL,
                    request_id TEXT NOT NULL,
                    FOREIGN KEY (request_id) REFERENCES request_log(request_id),
                    UNIQUE(series_id, date, request_id)
                )
            ''')
            
            # Validation log table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS validation_log (
                    validation_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    ticker TEXT,
                    validation_date DATETIME NOT NULL,
                    validation_type TEXT NOT NULL,
                    issue_type TEXT,
                    description TEXT,
                    severity_score DECIMAL(3,2),
                    details TEXT,
                    FOREIGN KEY (request_id) REFERENCES request_log(request_id)
                )
            ''')
            
            # Cross validation table (ticker data only)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cross_validation (
                    validation_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    yahoo_close DECIMAL(10,4),
                    alpha_close DECIMAL(10,4),
                    diff_pct DECIMAL(6,3),
                    resolved_as TEXT DEFAULT 'Yahoo',
                    request_id TEXT NOT NULL,
                    FOREIGN KEY (request_id) REFERENCES request_log(request_id)
                )
            ''')
            
            # Create indices for better performance
            self._create_indices(conn)
            
            self.logger.info("Database schema initialized successfully")
    
    def _create_indices(self, conn):
        """Create database indices for better query performance"""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON market_data(ticker, date)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_request ON market_data(request_id)",
            "CREATE INDEX IF NOT EXISTS idx_macro_data_series_date ON macro_data(series_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_macro_data_request ON macro_data(request_id)",
            "CREATE INDEX IF NOT EXISTS idx_validation_request ON validation_log(request_id)",
            "CREATE INDEX IF NOT EXISTS idx_cross_validation_ticker ON cross_validation(ticker, date)"
        ]
        
        for index in indices:
            conn.execute(index)
    
    @log_function_call(main_logger)
    def create_request_log(self, request_id: str, tickers: List[str], start_date: str, end_date: str) -> str:
        """Create a new request log entry"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO request_log (
                    request_id, request_timestamp, tickers, start_date, end_date, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                request_id,
                datetime.now().isoformat(),
                ','.join(tickers),
                start_date,
                end_date,
                'started'
            ))
        
        self.logger.info(f"Created request log: {request_id}")
        return request_id
    
    @log_function_call(main_logger)
    def update_request_status(self, request_id: str, status: str, **kwargs):
        """Update request log with completion status and metrics"""
        with sqlite3.connect(self.db_path) as conn:
            update_fields = ["status = ?"]
            values = [status]
            
            for key, value in kwargs.items():
                if key in ['total_records_fetched', 'macro_records_fetched', 'validation_performed', 'error_count']:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            values.append(request_id)
            
            query = f"UPDATE request_log SET {', '.join(update_fields)} WHERE request_id = ?"
            conn.execute(query, values)
        
        self.logger.info(f"Updated request {request_id} status: {status}")
    
    @log_function_call(main_logger)
    def save_market_data(self, df: pd.DataFrame, request_id: str) -> int:
        """Save market data with duplicate checking"""
        if df.empty:
            self.logger.warning("No market data to save")
            return 0
        
        # Add request metadata
        df_to_save = df.copy()
        df_to_save['fetch_timestamp'] = datetime.now().isoformat()
        df_to_save['request_id'] = request_id
        df_to_save['discrepancy_flag'] = df_to_save.get('discrepancy_flag', False)
        
        with sqlite3.connect(self.db_path) as conn:
            # Check for existing data to prevent duplicates
            existing_check = f'''
                SELECT COUNT(*) FROM market_data 
                WHERE ticker = ? AND date = ? AND request_id != ?
            '''
            
            records_saved = 0
            for _, row in df_to_save.iterrows():
                # Check if this ticker/date combination already exists (from different request)
                existing_count = conn.execute(
                    existing_check, 
                    (row['Ticker'], row['date'], request_id)
                ).fetchone()[0]
                
                if existing_count > 0:
                    self.logger.debug(f"Skipping duplicate: {row['Ticker']} {row['date']}")
                    continue
                
                # Insert new record
                try:
                    conn.execute('''
                        INSERT OR IGNORE INTO market_data (
                            ticker, date, open, high, low, close, adj_close, volume,
                            return_pct, ma20, ma50, vol20, fetch_timestamp, request_id, discrepancy_flag
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['Ticker'], row['date'], row['open'], row['high'], row['low'],
                        row['close'], row['adj_close'], row['volume'], row.get('return'),
                        row.get('ma20'), row.get('ma50'), row.get('vol20'),
                        row['fetch_timestamp'], row['request_id'], row['discrepancy_flag']
                    ))
                    records_saved += 1
                except Exception as e:
                    self.logger.error(f"Error saving market data record: {e}")
        
        self.logger.info(f"Saved {records_saved} market data records (duplicates skipped)")
        return records_saved
    
    @log_function_call(main_logger)
    def save_macro_data(self, df: pd.DataFrame, request_id: str) -> int:
        """Save macro data with duplicate checking"""
        if df.empty:
            self.logger.warning("No macro data to save")
            return 0
        
        # Add request metadata
        df_to_save = df.copy()
        df_to_save['fetch_timestamp'] = datetime.now().isoformat()
        df_to_save['request_id'] = request_id
        
        with sqlite3.connect(self.db_path) as conn:
            records_saved = 0
            for _, row in df_to_save.iterrows():
                try:
                    conn.execute('''
                        INSERT OR IGNORE INTO macro_data (
                            series_id, date, value, series_name, category,
                            fetch_timestamp, request_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['series_id'], row['date'], row['value'], 
                        row.get('series_name'), row.get('category'),
                        row['fetch_timestamp'], row['request_id']
                    ))
                    records_saved += 1
                except Exception as e:
                    self.logger.error(f"Error saving macro data record: {e}")
        
        self.logger.info(f"Saved {records_saved} macro data records")
        return records_saved
    
    @log_function_call(main_logger)
    def save_validation_log(self, request_id: str, validations: List[Dict]):
        """Save validation results"""
        with sqlite3.connect(self.db_path) as conn:
            for validation in validations:
                validation_id = f"{request_id}_{validation.get('type', 'unknown')}_{datetime.now().strftime('%H%M%S')}"
                
                conn.execute('''
                    INSERT INTO validation_log (
                        validation_id, request_id, ticker, validation_date,
                        validation_type, issue_type, description, severity_score, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validation_id, request_id, validation.get('ticker'),
                    datetime.now().isoformat(), validation.get('type'),
                    validation.get('issue_type'), validation.get('description'),
                    validation.get('severity', 0), json.dumps(validation.get('details', {}))
                ))
        
        self.logger.info(f"Saved {len(validations)} validation log entries")
    
    @log_function_call(main_logger)
    def save_cross_validation(self, request_id: str, discrepancies: List[Dict]):
        """Save cross-validation results"""
        if not discrepancies:
            self.logger.info("No cross-validation discrepancies to save")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for disc in discrepancies:
                validation_id = f"{request_id}_cross_{disc['ticker']}_{disc['date'].replace('-', '')}"
                
                conn.execute('''
                    INSERT OR REPLACE INTO cross_validation (
                        validation_id, ticker, date, yahoo_close, alpha_close,
                        diff_pct, resolved_as, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validation_id, disc['ticker'], disc['date'],
                    disc['yahoo_close'], disc['alpha_close'],
                    disc['diff_pct'], disc['resolved_as'], request_id
                ))
        
        self.logger.info(f"Saved {len(discrepancies)} cross-validation records")
    
    @log_function_call(main_logger)
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count records in each table
            tables = ['request_log', 'market_data', 'macro_data', 'validation_log', 'cross_validation']
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f"{table}_count"] = count
            
            # Get unique tickers
            unique_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM market_data").fetchone()[0]
            stats["unique_tickers"] = unique_tickers
            
            # Get date range
            date_range = conn.execute(
                "SELECT MIN(date) as start_date, MAX(date) as end_date FROM market_data"
            ).fetchone()
            stats["date_range"] = {"start": date_range[0], "end": date_range[1]}
            
        return stats


def generate_request_id(tickers: List[str], start_date: str, end_date: str) -> str:
    """Generate unique request ID based on parameters and timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tickers_str = '_'.join(sorted(tickers))
    
    # Create hash for very long ticker lists
    if len(tickers_str) > 50:
        tickers_hash = hashlib.md5(tickers_str.encode()).hexdigest()[:8]
        tickers_str = f"{tickers[0]}_and_{len(tickers)-1}_more_{tickers_hash}"
    
    request_id = f"{timestamp}_{tickers_str}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    return request_id


# Example usage and testing
if __name__ == "__main__":
    # Test database functionality
    db = DatabaseManager(Path("test_database.db"))
    
    # Test request creation
    request_id = generate_request_id(['AAPL', 'MSFT'], '2024-01-01', '2024-01-31')
    db.create_request_log(request_id, ['AAPL', 'MSFT'], '2024-01-01', '2024-01-31')
    
    print(f"Test request created: {request_id}")
    print("Database stats:", db.get_database_stats())
    
    # Clean up test file
    Path("test_database.db").unlink(missing_ok=True)
