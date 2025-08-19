"""
Centralized logging system for YFinance pipeline
Provides logging utilities with error handling and retry mechanisms
"""

import logging
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Optional
import json


def setup_logger(name: str = "yfinance_pipeline", log_level: str = "INFO") -> logging.Logger:
    """
    Set up centralized logger with file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Error file handler for critical issues
    error_handler = logging.FileHandler(
        log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    
    return logger


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exception types to catch and retry
        logger: Logger instance for logging retry attempts
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0 and logger:
                        logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0 and logger:
                        logger.info(f"Successfully completed {func.__name__} on attempt {attempt + 1}")
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_factor ** attempt
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                                f"Retrying in {delay:.1f} seconds..."
                            )
                        time.sleep(delay)
                    else:
                        if logger:
                            logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}. "
                                f"Final error: {str(e)}"
                            )
            
            # If we get here, all retries failed
            raise last_exception
            
        return wrapper
    return decorator


class ErrorTracker:
    """
    Tracks and manages errors throughout the pipeline execution
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.errors = []
        self.warnings = []
        self.start_time = datetime.now()
    
    def log_error(self, operation: str, ticker: str, error: str, details: dict = None):
        """Log an error and add to tracking"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "ticker": ticker,
            "error": error,
            "details": details or {}
        }
        
        self.errors.append(error_record)
        self.logger.error(f"{operation} failed for {ticker}: {error}")
    
    def log_warning(self, operation: str, ticker: str, warning: str, details: dict = None):
        """Log a warning and add to tracking"""
        warning_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "ticker": ticker,
            "warning": warning,
            "details": details or {}
        }
        
        self.warnings.append(warning_record)
        self.logger.warning(f"{operation} warning for {ticker}: {warning}")
    
    def log_ticker_skipped(self, ticker: str, reason: str):
        """Log when a ticker is skipped"""
        self.log_error("data_fetch", ticker, f"Ticker skipped: {reason}")
    
    def log_data_anomaly(self, ticker: str, date: str, anomaly_type: str, details: dict):
        """Log data anomalies like price discrepancies"""
        self.log_warning(
            "data_validation", 
            ticker, 
            f"{anomaly_type} detected on {date}",
            details
        )
    
    def get_summary(self) -> dict:
        """Get summary of all errors and warnings"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            "execution_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration
            },
            "error_summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "errors_by_operation": self._group_by_operation(self.errors),
                "warnings_by_operation": self._group_by_operation(self.warnings)
            },
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def _group_by_operation(self, records: list) -> dict:
        """Group records by operation type"""
        grouped = {}
        for record in records:
            operation = record["operation"]
            if operation not in grouped:
                grouped[operation] = 0
            grouped[operation] += 1
        return grouped
    
    def save_report(self, filepath: Path):
        """Save error tracking report to file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        
        self.logger.info(f"Error tracking report saved to {filepath}")


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function entry/exit with parameters and timing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            # Log function start (avoid logging sensitive data)
            logger.debug(f"Starting {func.__name__} with {len(args)} args, {len(kwargs)} kwargs")
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                duration = time.time() - start_time
                logger.debug(f"Completed {func.__name__} successfully in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                # Log failure
                duration = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Pre-configured logger instances
main_logger = setup_logger("yfinance_pipeline")
data_logger = setup_logger("yfinance_data", "DEBUG")
validation_logger = setup_logger("yfinance_validation")
macro_logger = setup_logger("yfinance_macro")


# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    logger = setup_logger("test_logger")
    
    logger.info("Testing logging system")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test error tracker
    error_tracker = ErrorTracker(logger)
    error_tracker.log_error("test_operation", "TEST", "Test error message")
    error_tracker.log_warning("test_operation", "TEST", "Test warning message")
    
    # Test retry decorator
    @retry_with_backoff(max_retries=2, logger=logger)
    def test_function_that_fails():
        raise ValueError("Test error")
    
    try:
        test_function_that_fails()
    except ValueError:
        logger.info("Expected error caught and handled")
    
    logger.info("Logging system test completed")
