# YFinance Pipeline User Guide

Complete instructions and examples for using the YFinance Enhanced Data Pipeline.

## Quick Start

### 1. Installation
```bash
# Clone/download the project
cd YFinance

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
# Simple request - creates timestamped files
python run_pipeline.py --tickers AAPL MSFT --start 2024-01-01 --end 2024-01-31
```

**What you get:**
- Unique CSV: `prices_AAPL-MSFT_20240101-20240131_20250818_143022.csv`
- Validation report with quality scores
- Database accumulation with request tracking
- Professional logging

### 3. Enhanced Features
```bash
# With cross-validation (requires API key)
python run_pipeline.py --tickers AAPL TSLA --enable-validation

# Automatic macro data (when API key available)
export FRED_API_KEY="your_key"
python run_pipeline.py --tickers SPY --start 2024-01-01

# Full enhanced pipeline
export ALPHA_VANTAGE_API_KEY="your_av_key"
export FRED_API_KEY="your_fred_key"
python run_pipeline.py --tickers AAPL MSFT GOOGL --enable-validation
```

## Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--tickers` | Stock symbols (space-separated) | `--tickers AAPL MSFT GOOGL` |
| `--start` | Start date (YYYY-MM-DD) | `--start 2024-01-01` |
| `--end` | End date (YYYY-MM-DD) | `--end 2024-12-31` |
| `--enable-validation` | Enable Alpha Vantage cross-validation | `--enable-validation` |
| `--alpha-vantage-key` | Alpha Vantage API key | `--alpha-vantage-key YOUR_KEY` |
| `--fred-api-key` | FRED API key (auto-enables macro data) | `--fred-api-key YOUR_KEY` |
| `--tolerance` | Price difference tolerance (%) | `--tolerance 0.3` |
| `--macro-categories` | Macro data categories | `--macro-categories rates inflation` |

## API Keys Setup

### Alpha Vantage (For Price Validation)
- **Get Free Key**: [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
- **Purpose**: Cross-validate Yahoo Finance prices for reliability
- **Benefit**: Detect price discrepancies and improve data confidence

### FRED (For Economic Context)
- **Get Free Key**: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **Purpose**: Automatic macroeconomic data for market context
- **Benefit**: Fed rates, inflation, GDP, unemployment alongside market data

```bash
# Set environment variables
export ALPHA_VANTAGE_API_KEY="your_key_here"
export FRED_API_KEY="your_key_here"

# Or use command line flags
python run_pipeline.py --alpha-vantage-key YOUR_KEY --fred-api-key YOUR_KEY
```

## Understanding Outputs

### Timestamped CSV Files
Format: `prices_TICKER1-TICKER2_STARTDATE-ENDDATE_TIMESTAMP.csv`

**Example**: `prices_AAPL-MSFT_20240101-20240131_20250818_143022.csv`
- **Tickers**: AAPL-MSFT (sorted, hyphen-separated)
- **Date Range**: 20240101-20240131
- **Fetch Time**: 20250818_143022
- **Content**: Enhanced price data with technical indicators, request metadata

### Validation Reports
**Example**: `validation_report_20250818_210058_AAPL_MSFT_20240101_20240110_20250818_210058.json`

```json
{
  "request_details": {
    "tickers": ["AAPL", "MSFT"],
    "date_range": {"start": "2024-01-01", "end": "2024-01-10"}
  },
  "ticker_validation": {
    "basic_checks": { "null_counts": {...}, "duplicates": 0 },
    "cross_validation": { "discrepancies_found": 3, "details": [...] }
  },
  "quality_assessment": {
    "overall_score": 95.2,
    "basic_data_quality": 100,
    "cross_validation_reliability": 92.5
  },
  "recommendations": [
    "Excellent data quality! No issues detected.",
    "3 minor price discrepancies found between sources."
  ]
}
```

### Execution Logs
Complete request summaries with performance metrics and file paths.

## Usage Examples

### Investment Research
```bash
# Analyze tech giants with economic context
python run_pipeline.py --tickers AAPL MSFT GOOGL AMZN NVDA --start 2024-01-01
```

### Portfolio Tracking
```bash
# Track your portfolio with validation
python run_pipeline.py --tickers AAPL TSLA BTC-USD --enable-validation --start 2023-01-01
```

### Market Analysis
```bash
# Market indices with macro context
python run_pipeline.py --tickers SPY QQQ IWM --start 2024-01-01 --macro-categories rates inflation market
```

### Risk Assessment
```bash
# High-reliability data for critical decisions
python run_pipeline.py --tickers SPY ^VIX --enable-validation --tolerance 0.1
```

## Request Workflow

### What Happens When You Run a Request:

1. **Request Registration**
   - Generate unique request ID
   - Log request parameters
   - Initialize database tracking

2. **Data Fetching**
   - Yahoo Finance (primary source)
   - Alpha Vantage (validation, if enabled)
   - FRED API (macro data, if key available)

3. **Data Validation**
   - Basic quality checks (nulls, duplicates, extremes)
   - Cross-validation between sources (ticker data only)
   - Macro data completeness verification

4. **Data Processing**
   - Technical indicator calculation
   - Return and volatility metrics
   - Data standardization and cleaning

5. **Database Storage**
   - Append to database with duplicate prevention
   - Save cross-validation results
   - Update request status and metrics

6. **Output Generation**
   - Create timestamped CSV with request metadata
   - Generate comprehensive validation report
   - Create execution summary log

### Example Console Output:
```
Starting enhanced YFinance pipeline
Request ID: 20250818_210058_AAPL_MSFT_20240101_20240110
Fetching ticker data for 2 tickers: ['AAPL', 'MSFT']
Starting multi-source validation for ticker data
Cross-validation completed: 0 discrepancies found
Fetching macroeconomic data for same time period
Fetched 1,250 macro records, 7 economic series
Saving data to database (checking for duplicates)
Creating output files

PIPELINE EXECUTION SUMMARY
Request ID: 20250818_210058_AAPL_MSFT_20240101_20240110
Processed 12 market records for 2 tickers
Cross-validation: 2 tickers validated, 0 discrepancies found
Macro data: 1,250 records, 7 economic series
Database now contains 28 total market records
```

## Advanced Configuration

### Macro Data Categories
- **`rates`**: Fed Funds Rate, Treasury yields (2Y, 10Y, 3M)
- **`inflation`**: CPI, Core CPI, PCE indices
- **`growth`**: GDP, Real GDP, GDP Deflator
- **`employment`**: Unemployment Rate, Nonfarm Payrolls
- **`market`**: VIX, Dollar Index
- **`commodities`**: Oil (WTI), Gold prices
- **`housing`**: Housing Starts, Case-Shiller Index

### Cross-Validation Settings
- **Default tolerance**: 0.5% price difference
- **Resolution method**: Always prioritize Yahoo Finance
- **Rate limiting**: Built-in delays for API compliance
- **Error handling**: Graceful degradation if validation fails

## Database Analysis

Check your accumulated data:

```python
from src.database import DatabaseManager
from pathlib import Path

db = DatabaseManager(Path('data/market_data.db'))
stats = db.get_database_stats()

print(f"Total requests: {stats['request_log_count']}")
print(f"Market records: {stats['market_data_count']}")
print(f"Unique tickers: {stats['unique_tickers']}")
print(f"Date range: {stats['date_range']}")
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Use the enhanced runner
python run_pipeline.py  # Correct
# instead of
python src/pipeline.py  # Missing enhanced features
```

**2. API Rate Limits**
- Built-in delays respect API constraints
- Alpha Vantage: 5 requests/minute (free tier)
- FRED: 120 requests/minute

**3. Database Locked**
- Only one pipeline instance should run at a time
- Database automatically handles concurrent access

**4. Missing Data**
- Check ticker symbol format
- Verify date ranges are reasonable
- Some tickers may not have data for specified periods

## Production Deployment

### Environment Setup
```bash
# Production environment variables
export ALPHA_VANTAGE_API_KEY="production_key"
export FRED_API_KEY="production_key"
export PIPELINE_LOG_LEVEL="INFO"
```

### Automated Scheduling
```bash
# Daily market data updates (example cron job)
0 18 * * 1-5 cd /path/to/YFinance && python run_pipeline.py --tickers SPY QQQ --enable-validation

# Weekly comprehensive analysis
0 8 * * 0 cd /path/to/YFinance && python run_pipeline.py --tickers AAPL MSFT GOOGL AMZN --enable-validation --start 2023-01-01
```

### Monitoring
- **Logs**: Monitor `logs/` directory for errors
- **Database**: Check `market_data_count` growth
- **Validation**: Review discrepancy reports for data quality

## Example: Complete Workflow

### 1. Run Enhanced Pipeline
```bash
export ALPHA_VANTAGE_API_KEY="your_av_key"
export FRED_API_KEY="your_fred_key"

python run_pipeline.py \
  --tickers AAPL MSFT GOOGL \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --enable-validation \
  --tolerance 0.3
```

### 2. Generated Files
- `prices_AAPL-GOOGL-MSFT_20240101-20240331_20250818_143022.csv`
- `validation_report_REQUEST_ID_TIMESTAMP.json`
- `execution_log_REQUEST_ID_TIMESTAMP.json`

### 3. Database Accumulation
- Market data appended with duplicate prevention
- Macro data for Q1 2024 economic context
- Cross-validation results logged
- Request fully traceable by ID

### 4. Quality Assurance
- Data reliability scores calculated
- Price discrepancies flagged and resolved
- Comprehensive recommendations provided
- Professional logging for audit trails

## Ready to Get Started?

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run first request**: `python run_pipeline.py --tickers AAPL`
3. **Check outputs**: Review generated CSV and validation report
4. **Get API keys** for enhanced features (optional)
5. **Scale up**: Add more tickers and enable cross-validation

**Your enterprise-grade financial data platform is ready!**
