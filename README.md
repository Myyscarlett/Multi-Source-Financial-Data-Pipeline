# YFinance Enhanced Data Pipeline

A production-ready financial data platform that fetches, validates, and processes market data with enterprise-grade features including multi-source validation, macroeconomic data integration, and comprehensive request tracking.

## Overview

This project provides a comprehensive financial data pipeline that:
- Fetches historical stock data from Yahoo Finance
- Cross-validates prices with Alpha Vantage for reliability
- Integrates macroeconomic data from FRED API
- Maintains a persistent SQLite database with request tracking
- Generates timestamped outputs for each data request
- Provides comprehensive validation and quality assessment

## Key Features

### Comprehensive Data Collection
- Multi-ticker support for simultaneous stock data download
- Automatic macroeconomic context via FRED integration
- Multi-source validation using Alpha Vantage cross-referencing
- Technical indicators calculation (MA20, MA50, volatility, returns)

### Enterprise-Grade Architecture
- Unique request ID tracking for full traceability
- Append-only database with duplicate prevention
- Timestamped outputs with descriptive naming
- Combined basic and cross-source validation
- Professional logging with centralized error handling

### Automated Reporting
- Data reliability metrics and quality scores
- Combined validation reports (basic + cross-validation)
- Complete execution logs with performance metrics
- Price discrepancy flagging and anomaly detection

## Project Structure

```
YFinance/
├─ src/
│   ├─ __init__.py           # Package initialization
│   ├─ pipeline.py           # Main enhanced pipeline orchestrator
│   ├─ database.py           # Database management with duplicate prevention
│   ├─ output_manager.py     # CSV and report generation
│   ├─ validation.py         # Multi-source cross-validation (ticker data)
│   ├─ macro.py              # FRED macroeconomic data integration
│   └─ logger.py             # Professional logging and error handling
├─ data/
│   └─ market_data.db        # SQLite database with organized tables
├─ outputs/                  # Generated files for each request
│   ├─ prices_TICKER_DATERANGE_TIMESTAMP.csv
│   ├─ validation_report_REQUEST_ID_TIMESTAMP.json
│   └─ execution_log_REQUEST_ID_TIMESTAMP.json
├─ logs/                     # System logs (auto-generated)
├─ run_pipeline.py           # Enhanced pipeline runner
├─ requirements.txt          # Python dependencies
├─ README.md                 # Project overview (this file)
└─ USER_GUIDE.md            # Complete usage instructions and examples
```

## Database Schema

The system automatically creates and manages these tables:

### request_log
Tracks every pipeline execution with metadata
- `request_id`, `timestamp`, `tickers`, `date_range`, `status`, `records_fetched`

### market_data
Accumulated stock price data with technical indicators
- `ticker`, `date`, `ohlcv`, `technical_indicators`, `fetch_timestamp`, `request_id`, `discrepancy_flag`

### macro_data 
Economic indicators from FRED API
- `series_id`, `date`, `value`, `series_name`, `category`, `fetch_timestamp`, `request_id`

### cross_validation
Price discrepancies between data sources
- `ticker`, `date`, `yahoo_close`, `alpha_close`, `diff_pct`, `resolved_as`, `request_id`

### validation_log
Data quality issues and resolutions
- `request_id`, `validation_type`, `issue_type`, `description`, `severity_score`

## Data Quality Features

### Basic Validation (Always Performed)
- Null value detection and counting
- Duplicate record identification and removal
- Negative price/volume flagging
- Extreme price movement detection (z-score > 6)
- Missing business day estimation

### Cross-Validation (When Enabled)
- Yahoo Finance vs Alpha Vantage price comparison
- Configurable tolerance threshold (default 0.5%)
- Automatic discrepancy flagging and logging
- Yahoo Finance prioritized as "source of truth"
- Detailed anomaly reporting with resolution

### Macro Data Validation (Basic Only)
- Data completeness checks
- Missing value identification
- Series availability verification
- Date range coverage analysis

## Dependencies

```txt
yfinance              # Yahoo Finance data
pandas                # Data manipulation
numpy                 # Numerical computing
python-dateutil       # Date handling

# Enhanced features
fredapi>=0.5.0        # FRED economic data
alpha-vantage>=2.3.0  # Multi-source validation
```

## API Integration

### Supported Data Sources
- **Yahoo Finance**: Primary source for stock price data
- **Alpha Vantage**: Secondary source for price validation (optional)
- **FRED**: Federal Reserve Economic Data for macroeconomic context (optional)

### Output Generation
- **Timestamped CSVs**: Format `prices_TICKER1-TICKER2_STARTDATE-ENDDATE_TIMESTAMP.csv`
- **Validation Reports**: Comprehensive JSON reports with quality scores
- **Execution Logs**: Complete request summaries with performance metrics

## Technical Architecture

### Request Workflow
1. Request registration with unique ID generation
2. Multi-source data fetching (Yahoo Finance, Alpha Vantage, FRED)
3. Comprehensive validation (basic checks + cross-validation)
4. Technical indicator calculation and data processing
5. Database storage with duplicate prevention
6. Timestamped output file generation

### Error Handling
- Centralized logging system with multiple log levels
- Exponential backoff for API rate limit handling
- Graceful degradation when optional features fail
- Comprehensive error tracking and reporting

## License & Disclaimer

This project is for educational and research purposes only. Not intended for financial advice or live trading. Market data accuracy depends on underlying providers (Yahoo Finance, Alpha Vantage, FRED).

---

## Getting Started

For complete installation instructions, usage examples, and troubleshooting, see **[USER_GUIDE.md](USER_GUIDE.md)**.

Basic usage:
```bash
pip install -r requirements.txt
python run_pipeline.py --tickers AAPL MSFT
```