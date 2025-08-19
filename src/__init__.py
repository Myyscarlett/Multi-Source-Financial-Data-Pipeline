"""
YFinance Enhanced Pipeline Package

This package provides a comprehensive financial data pipeline with:
- Multi-source data validation
- Macroeconomic data integration
- Automated reporting and visualization
- Centralized logging and error handling
"""

__version__ = "1.0.0"
__author__ = "YFinance Pipeline Team"

# Core modules
from .pipeline import main, fetch_data, validate, transform
from .logger import setup_logger, main_logger, retry_with_backoff, ErrorTracker

# Enhanced modules (optional imports)
try:
    from .macro import MacroDataFetcher, fetch_fred_data
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

try:
    from .validation import DataValidator, validate_tickers
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# Reporting functionality removed
REPORTING_AVAILABLE = False

# Feature availability status
FEATURES = {
    "macro_data": MACRO_AVAILABLE,
    "multi_source_validation": VALIDATION_AVAILABLE,
    "automated_reporting": REPORTING_AVAILABLE
}

__all__ = [
    # Core functionality
    "main", "fetch_data", "validate", "transform",
    
    # Logging
    "setup_logger", "main_logger", "retry_with_backoff", "ErrorTracker",
    
    # Enhanced features (conditional)
    "MacroDataFetcher", "fetch_fred_data",
    "DataValidator", "validate_tickers",
    
    # Metadata
    "FEATURES", "__version__"
]
