"""
Output file management for YFinance Enhanced Pipeline
Handles CSV generation, validation reports, and file organization
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

from .logger import main_logger, log_function_call


class OutputManager:
    """
    Manages all output file generation for the pipeline
    """
    
    def __init__(self, base_output_dir: Path = None):
        self.logger = main_logger
        self.base_dir = Path(base_output_dir) if base_output_dir else Path("outputs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    @log_function_call(main_logger)
    def create_timestamped_csv(
        self, 
        df: pd.DataFrame, 
        tickers: List[str], 
        start_date: str, 
        end_date: str,
        request_id: str
    ) -> Path:
        """
        Create timestamped CSV file with descriptive filename
        Format: prices_TICKER1-TICKER2_STARTDATE-ENDDATE_TIMESTAMP.csv
        """
        if df.empty:
            self.logger.warning("Cannot create CSV from empty DataFrame")
            return None
        
        # Create filename components
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tickers_str = '-'.join(sorted(tickers))
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')
        
        # Handle very long ticker lists
        if len(tickers_str) > 50:
            tickers_hash = hashlib.md5(tickers_str.encode()).hexdigest()[:6]
            tickers_str = f"{tickers[0]}-and-{len(tickers)-1}more-{tickers_hash}"
        
        # Create filename
        filename = f"prices_{tickers_str}_{start_clean}-{end_clean}_{timestamp}.csv"
        filepath = self.base_dir / filename
        
        # Add metadata columns
        df_output = df.copy()
        df_output['request_id'] = request_id
        df_output['export_timestamp'] = datetime.now().isoformat()
        
        # Save CSV
        df_output.to_csv(filepath, index=False)
        
        self.logger.info(f"📄 Created timestamped CSV: {filename}")
        return filepath
    
    @log_function_call(main_logger)
    def create_validation_report(
        self,
        request_id: str,
        basic_validation: Dict,
        cross_validation: Dict = None,
        macro_validation: Dict = None,
        tickers: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Path:
        """
        Create comprehensive validation report
        """
        timestamp = datetime.now()
        
        # Build comprehensive report
        report = {
            "report_metadata": {
                "request_id": request_id,
                "report_type": "comprehensive_validation",
                "generated_at": timestamp.isoformat(),
                "report_version": "1.0"
            },
            "request_details": {
                "tickers": tickers,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "execution_timestamp": timestamp.isoformat()
            },
            "ticker_validation": {
                "basic_checks": basic_validation,
                "cross_validation": cross_validation if cross_validation else {
                    "status": "not_performed",
                    "reason": "No Alpha Vantage API key provided or validation not enabled"
                }
            },
            "macro_validation": macro_validation if macro_validation else {
                "status": "not_fetched",
                "reason": "Macro data not requested or FRED API key not provided"
            }
        }
        
        # Calculate overall quality scores
        report["quality_assessment"] = self._calculate_quality_scores(
            basic_validation, cross_validation, macro_validation
        )
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(
            basic_validation, cross_validation, macro_validation
        )
        
        # Save report
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"validation_report_{request_id}_{timestamp_str}.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Created validation report: {filename}")
        return filepath
    
    def _calculate_quality_scores(
        self, 
        basic_validation: Dict, 
        cross_validation: Dict = None,
        macro_validation: Dict = None
    ) -> Dict:
        """Calculate data quality scores"""
        scores = {
            "basic_data_quality": 100,
            "cross_validation_reliability": "N/A",
            "macro_data_completeness": "N/A",
            "overall_score": 100
        }
        
        # Basic data quality score
        if basic_validation:
            issues = (
                basic_validation.get("duplicate_rows", 0) +
                basic_validation.get("neg_close", 0) +
                basic_validation.get("neg_volume", 0) +
                basic_validation.get("extreme_moves", 0)
            )
            
            # Count null values
            null_count = sum(basic_validation.get("null_counts", {}).values())
            
            # Calculate score (deduct points for issues)
            total_issues = issues + (null_count / 10)  # Weight nulls less than other issues
            scores["basic_data_quality"] = max(0, 100 - (total_issues * 5))
        
        # Cross-validation reliability
        if cross_validation and cross_validation.get("status") == "performed":
            total_discrepancies = len(cross_validation.get("all_discrepancies", []))
            total_comparisons = cross_validation.get("summary", {}).get("comparison_records", 1)
            
            if total_comparisons > 0:
                reliability = max(0, 100 - ((total_discrepancies / total_comparisons) * 100))
                scores["cross_validation_reliability"] = round(reliability, 1)
        
        # Macro data completeness
        if macro_validation and "series_coverage" in macro_validation:
            series_coverage = macro_validation["series_coverage"]
            if series_coverage:
                avg_completeness = sum(
                    series.get("record_count", 0) for series in series_coverage.values()
                ) / len(series_coverage)
                scores["macro_data_completeness"] = min(100, avg_completeness)
        
        # Overall score (weighted average)
        reliability_score = scores["cross_validation_reliability"]
        if reliability_score != "N/A":
            scores["overall_score"] = round(
                (scores["basic_data_quality"] * 0.7) + (reliability_score * 0.3), 1
            )
        else:
            scores["overall_score"] = scores["basic_data_quality"]
        
        return scores
    
    def _generate_recommendations(
        self, 
        basic_validation: Dict, 
        cross_validation: Dict = None,
        macro_validation: Dict = None
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Basic validation recommendations
        if basic_validation:
            null_counts = basic_validation.get("null_counts", {})
            if any(count > 0 for count in null_counts.values()):
                recommendations.append(
                    "⚠️ Missing data detected. Consider data imputation or extended date ranges."
                )
            
            if basic_validation.get("duplicate_rows", 0) > 0:
                recommendations.append(
                    "🔄 Duplicate records found and removed. Check data source for consistency."
                )
            
            if basic_validation.get("extreme_moves", 0) > 0:
                recommendations.append(
                    "📈 Extreme price movements detected. Verify if these are legitimate market events."
                )
        
        # Cross-validation recommendations
        if cross_validation:
            if cross_validation.get("status") == "not_performed":
                recommendations.append(
                    "🔍 Enable cross-validation with Alpha Vantage API key for improved data reliability."
                )
            elif cross_validation.get("status") == "performed":
                discrepancies = len(cross_validation.get("all_discrepancies", []))
                if discrepancies > 0:
                    recommendations.append(
                        f"⚡ {discrepancies} price discrepancies found between sources. "
                        "Review anomaly details for trading decisions."
                    )
                else:
                    recommendations.append(
                        "✅ Perfect cross-validation match. High confidence in data accuracy."
                    )
        
        # Macro data recommendations
        if macro_validation:
            if macro_validation.get("status") == "not_fetched":
                recommendations.append(
                    "📊 Add macroeconomic context with FRED API key for comprehensive market analysis."
                )
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append(
                "✅ Excellent data quality! No issues detected. Data ready for analysis."
            )
        
        return recommendations
    
    @log_function_call(main_logger)
    def create_summary_log(
        self,
        request_id: str,
        execution_summary: Dict,
        file_paths: Dict
    ) -> Path:
        """
        Create execution summary log
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"execution_log_{request_id}_{timestamp}.json"
        filepath = self.base_dir / filename
        
        log_data = {
            "request_id": request_id,
            "execution_summary": execution_summary,
            "generated_files": {
                "csv_file": str(file_paths.get("csv")),
                "validation_report": str(file_paths.get("validation")),
                "log_file": str(file_paths.get("log"))
            },
            "log_created_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"📜 Created execution log: {filename}")
        return filepath
    
    @log_function_call(main_logger)
    def list_output_files(self) -> Dict:
        """List all output files by type"""
        files = {
            "csv_files": list(self.base_dir.glob("prices_*.csv")),
            "validation_reports": list(self.base_dir.glob("validation_report_*.json")),
            "execution_logs": list(self.base_dir.glob("execution_log_*.json"))
        }
        
        # Add file counts and latest files
        summary = {}
        for file_type, file_list in files.items():
            summary[file_type] = {
                "count": len(file_list),
                "latest": max(file_list, key=lambda x: x.stat().st_mtime) if file_list else None
            }
        
        return summary


# Convenience functions
def create_request_outputs(
    df: pd.DataFrame,
    request_id: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    basic_validation: Dict,
    cross_validation: Dict = None,
    macro_validation: Dict = None,
    execution_summary: Dict = None
) -> Dict[str, Path]:
    """
    Create all output files for a request
    
    Returns:
        Dictionary with paths to created files
    """
    output_manager = OutputManager()
    
    # Create CSV
    csv_path = output_manager.create_timestamped_csv(
        df, tickers, start_date, end_date, request_id
    )
    
    # Create validation report
    validation_path = output_manager.create_validation_report(
        request_id, basic_validation, cross_validation, macro_validation,
        tickers, start_date, end_date
    )
    
    # Create execution log
    file_paths = {
        "csv": csv_path,
        "validation": validation_path
    }
    
    if execution_summary:
        log_path = output_manager.create_summary_log(
            request_id, execution_summary, file_paths
        )
        file_paths["log"] = log_path
    
    return file_paths


# Example usage and testing
if __name__ == "__main__":
    # Test output manager
    output_manager = OutputManager(Path("test_outputs"))
    
    # Create sample data
    sample_df = pd.DataFrame({
        'Ticker': ['AAPL', 'AAPL'],
        'date': ['2024-01-01', '2024-01-02'],
        'close': [180.0, 185.0]
    })
    
    # Test CSV creation
    csv_path = output_manager.create_timestamped_csv(
        sample_df, ['AAPL'], '2024-01-01', '2024-01-02', 'test_request_123'
    )
    
    # Test validation report
    basic_validation = {"null_counts": {"close": 0}, "duplicate_rows": 0}
    validation_path = output_manager.create_validation_report(
        'test_request_123', basic_validation
    )
    
    print(f"Test CSV created: {csv_path}")
    print(f"Test validation report created: {validation_path}")
    
    # Clean up test files
    import shutil
    if Path("test_outputs").exists():
        shutil.rmtree("test_outputs")
