# reporting/validator.py
"""
Data Integrity Validator - Category 1 Tests for iterative multi-agent system
Checks fundamental DataFrame health after each subtask execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    test_name: str
    message: str
    details: Dict[str, Any] = None


class UnitTester:
    """
    Validates fundamental DataFrame integrity after subtask execution.
    Prevents data corruption from propagating through the pipeline.
    """
    
    def __init__(self, expected_shape: Tuple[int, int], 
                 expected_columns: List[str], 
                 target_column: str):
        """
        Initialize validator with expected DataFrame characteristics.
        
        Args:
            expected_shape: (rows, cols) - original DataFrame shape
            expected_columns: List of essential column names
            target_column: Name of target variable column
        """
        self.expected_rows, self.expected_cols = expected_shape
        self.expected_columns = expected_columns
        self.target_column = target_column
        
        # Configurable thresholds
        self.min_row_retention = 0.95  # Must retain 95% of rows
        self.max_nan_percentage = 0.10  # Max 10% new NaN values per column
        
    def validate_dataframe_integrity(self, df: pd.DataFrame, 
                                   subtask_name: str = "") -> List[ValidationResult]:
        """
        Run complete data integrity validation suite.
        
        Args:
            df: DataFrame to validate
            subtask_name: Name of subtask for context in error messages
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        context = f" after iterative process '{subtask_name}'" if subtask_name else ""
        
        # Test 1: DataFrame exists and is valid
        results.append(self._test_dataframe_exists(df, context))
        
        if not results[-1].passed:
            return results  # Can't continue if DataFrame doesn't exist
        
        # Test 2: Shape integrity
        results.append(self._test_shape_integrity(df, context))
        
        # Test 3: Essential columns present
        results.append(self._test_essential_columns(df, context))
        
        # Test 4: Target variable integrity
        results.append(self._test_target_integrity(df, context))
        
        # Test 5: Data type consistency
        results.append(self._test_data_types(df, context))
        
        # Test 6: Data corruption detection
        results.append(self._test_data_corruption(df, context))
        
        return results
    
    def _test_dataframe_exists(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Test 1: DataFrame exists and is valid."""
        if df is None:
            return ValidationResult(
                passed=False,
                test_name="DataFrame Existence",
                message=f"DataFrame is None{context}",
                details={"error_type": "missing_dataframe"}
            )
        
        if not isinstance(df, pd.DataFrame):
            return ValidationResult(
                passed=False,
                test_name="DataFrame Existence", 
                message=f"Object is not a DataFrame{context}, got {type(df)}",
                details={"actual_type": str(type(df))}
            )
        
        if df.empty:
            return ValidationResult(
                passed=False,
                test_name="DataFrame Existence",
                message=f"DataFrame is empty{context}",
                details={"shape": df.shape}
            )
        
        return ValidationResult(
            passed=True,
            test_name="DataFrame Existence",
            message=f"DataFrame exists and is valid{context}"
        )
    
    def _test_shape_integrity(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Test 2: Shape integrity - reasonable row/column counts."""
        current_rows, current_cols = df.shape
        min_rows = int(self.expected_rows * self.min_row_retention)
        
        # Check row count
        if current_rows < min_rows:
            return ValidationResult(
                passed=False,
                test_name="Shape Integrity",
                message=f"Excessive row loss{context}: {current_rows}/{self.expected_rows} rows remaining "
                       f"(below {self.min_row_retention*100}% threshold)",
                details={
                    "expected_rows": self.expected_rows,
                    "actual_rows": current_rows,
                    "rows_lost": self.expected_rows - current_rows,
                    "retention_rate": current_rows / self.expected_rows
                }
            )
        
        # Check column count (shouldn't decrease during EDA)
        if current_cols < self.expected_cols:
            return ValidationResult(
                passed=False,
                test_name="Shape Integrity",
                message=f"Column loss detected{context}: {current_cols}/{self.expected_cols} columns remaining",
                details={
                    "expected_cols": self.expected_cols,
                    "actual_cols": current_cols,
                    "cols_lost": self.expected_cols - current_cols
                }
            )
        
        return ValidationResult(
            passed=True,
            test_name="Shape Integrity",
            message=f"Shape integrity maintained{context}: {current_rows} rows, {current_cols} columns",
            details={"shape": df.shape, "retention_rate": current_rows / self.expected_rows}
        )
    
    def _test_essential_columns(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Test 3: Essential columns still present."""
        missing_columns = set(self.expected_columns) - set(df.columns)
        
        if missing_columns:
            return ValidationResult(
                passed=False,
                test_name="Essential Columns",
                message=f"Missing essential columns{context}: {list(missing_columns)}",
                details={
                    "missing_columns": list(missing_columns),
                    "present_columns": list(df.columns),
                    "expected_columns": self.expected_columns
                }
            )
        
        return ValidationResult(
            passed=True,
            test_name="Essential Columns",
            message=f"All essential columns present{context}",
            details={"columns_count": len(df.columns)}
        )
    
    def _test_target_integrity(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Test 4: Target variable integrity."""
        if self.target_column not in df.columns:
            return ValidationResult(
                passed=False,
                test_name="Target Integrity",
                message=f"Target column '{self.target_column}' missing{context}",
                details={"missing_target": self.target_column}
            )
        
        target_series = df[self.target_column]
        unique_values = target_series.nunique()
        
        # Check for NaN values in target
        nan_count = target_series.isna().sum()
        if nan_count > 0:
            return ValidationResult(
                passed=False,
                test_name="Target Integrity",
                message=f"Target variable contains {nan_count} NaN values{context}",
                details={"nan_count": nan_count, "total_rows": len(target_series)}
            )
        
        # Check for completely empty target (no valid values)
        if unique_values == 0:
            return ValidationResult(
                passed=False,
                test_name="Target Integrity",
                message=f"Target variable has no valid values{context}",
                details={"actual_unique": unique_values}
            )
        
        # Create appropriate target distribution summary
        target_details = {"unique_values": unique_values}
        
        # For categorical targets (low unique count), include value counts
        # For numeric targets (high unique count), include basic statistics
        if unique_values <= 20:  # Likely categorical
            target_details["target_distribution"] = target_series.value_counts().to_dict()
        else:  # Likely numeric/continuous
            if target_series.dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                target_details["target_stats"] = {
                    "min": float(target_series.min()),
                    "max": float(target_series.max()),
                    "mean": float(target_series.mean()),
                    "median": float(target_series.median())
                }
            else:
                # For non-numeric with many unique values, just show sample
                target_details["sample_values"] = list(target_series.unique()[:10])
        
        return ValidationResult(
            passed=True,
            test_name="Target Integrity",
            message=f"Target variable integrity maintained{context}",
            details=target_details
        )

    def _test_data_types(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Test 5: Data type consistency."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        
        issues = []
        
        # Check for unexpected string values in numeric columns
        for col in numeric_columns:
            if df.dtypes[col] == 'object':  # Numeric column became object type
                issues.append(f"Column '{col}' became object type (was numeric)")
        
        # Check if target column is still categorical/object
        if self.target_column in df.columns:
            target_dtype = df[self.target_column].dtype
            valid_target_types = ['object', 'category', 'int8', 'int16', 'int32', 'int64', 
                                'float16', 'float32', 'float64', 'bool']
            if not any(str(target_dtype).startswith(vtype) for vtype in valid_target_types):
                issues.append(f"Target column '{self.target_column}' has unexpected type: {target_dtype}")
        
        if issues:
            return ValidationResult(
                passed=False,
                test_name="Data Types",
                message=f"Data type issues detected{context}: {'; '.join(issues)}",
                details={
                    "issues": issues,
                    "dtypes": df.dtypes.to_dict()
                }
            )
        
        return ValidationResult(
            passed=True,
            test_name="Data Types",
            message=f"Data types consistent{context}",
            details={
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(non_numeric_columns)
            }
        )
    
    def _test_data_corruption(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Test 6: Detect data corruption indicators."""
        issues = []
        
        # Check for excessive NaN introduction
        for col in df.columns:
            if col in self.expected_columns:
                nan_percentage = df[col].isna().mean()
                if nan_percentage > self.max_nan_percentage:
                    issues.append(f"Column '{col}' has {nan_percentage:.1%} NaN values "
                                f"(above {self.max_nan_percentage:.1%} threshold)")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isna().all()].tolist()
        if empty_columns:
            issues.append(f"Completely empty columns: {empty_columns}")
            
        # Check for infinite values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                issues.append(f"Column '{col}' has {infinite_count} infinite values")

        if issues:
            return ValidationResult(
                passed=False,
                test_name="Data Corruption",
                message=f"Data corruption indicators detected{context}: {'; '.join(issues)}",
                details={"issues": issues}
            )
        
        return ValidationResult(
            passed=True,
            test_name="Data Corruption",
            message=f"No data corruption detected{context}"
        )
    
    def format_validation_report(self, results: List[ValidationResult]) -> str:
        """Format validation results into a readable report."""
        passed_tests = [r for r in results if r.passed]
        failed_tests = [r for r in results if not r.passed]
        
        report = f"Data Integrity Validation Report (Iterative Process)\n"
        report += f"{'='*60}\n"
        report += f"Total Tests: {len(results)} | Passed: {len(passed_tests)} | Failed: {len(failed_tests)}\n\n"
        
        if failed_tests:
            report += "❌ FAILED TESTS:\n"
            for result in failed_tests:
                report += f"  • {result.test_name}: {result.message}\n"
            report += "\n"
        
        if passed_tests:
            report += "✅ PASSED TESTS:\n"
            for result in passed_tests:
                report += f"  • {result.test_name}: {result.message}\n"
        
        return report
    
    def get_failure_summary(self, results: List[ValidationResult]) -> str:
        """Get a concise summary of failures for retry prompts."""
        failed_tests = [r for r in results if not r.passed]
        
        if not failed_tests:
            return "All data integrity tests passed."
        
        summary = "Data integrity issues detected:\n"
        for result in failed_tests:
            summary += f"- {result.message}\n"
        
        return summary
    
def unit_test_report(validation_log, phase_name, project_name, filename):
    """Create a comprehensive HTML validation report for iterative processes."""
    
    if not validation_log:
        print("📊 No validation data to report")
        return
    
    # Enhanced HTML template for iterative processes
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Validation Report - {phase_name} (Iterative Process)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f6; }}
            .pass {{ color: #27ae60; font-weight: bold; }}
            .fail {{ color: #e74c3c; font-weight: bold; }}
            .iterative-info {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
            .summary-stats {{ background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Validation Report: {phase_name}</h1>
        
        <div class="iterative-info">
            <h3>🔄 Iterative Process Validation</h3>
            <p><strong>Project:</strong> {project_name}</p>
            <p><strong>Architecture:</strong> 3-Agent Iterative (Planner → Developer → Auditor → Developer)</p>
            <p><strong>Validation Level:</strong> Data Integrity Tests</p>
        </div>
        
        <div class="summary-stats">
            <h3>📊 Summary Statistics</h3>
            <p><strong>Total Subtasks Validated:</strong> {len(validation_log)}</p>
    """
    
    # Calculate overall statistics
    total_tests = 0
    total_passed = 0
    
    for subtask_entry in validation_log:
        tests = subtask_entry['tests']
        total_tests += len(tests)
        total_passed += sum(1 for test in tests if test["status"] == "PASS")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    html += f"""
            <p><strong>Total Tests Run:</strong> {total_tests}</p>
            <p><strong>Tests Passed:</strong> <span class="pass">{total_passed}</span></p>
            <p><strong>Tests Failed:</strong> <span class="fail">{total_tests - total_passed}</span></p>
            <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
        </div>
    """
    
    # Add detailed results for each subtask
    for subtask_entry in validation_log:
        subtask_name = subtask_entry['subtask_name']
        subtask_index = subtask_entry['subtask_index']
        tests = subtask_entry['tests']
        
        # Calculate subtask-specific stats
        subtask_passed = sum(1 for test in tests if test["status"] == "PASS")
        subtask_total = len(tests)
        subtask_rate = (subtask_passed / subtask_total * 100) if subtask_total > 0 else 0
        
        html += f"""
        <h2>Subtask {subtask_index}: {subtask_name}</h2>
        <p><strong>Validation Success Rate:</strong> {subtask_rate:.1f}% ({subtask_passed}/{subtask_total})</p>
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Details</th>
            </tr>
        """
        
        # Add each test result
        for test in tests:
            status_class = "pass" if test["status"] == "PASS" else "fail"
            html += f"""
            <tr>
                <td>{test['test']}</td>
                <td class="{status_class}">{test['status']}</td>
                <td>{test['reason']}</td>
            </tr>
            """
        
        html += "</table><br>"
    
    html += """
        <div class="iterative-info">
            <h3>🔍 About These Validations</h3>
            <p>These tests validate data integrity throughout the iterative process:</p>
            <ul>
                <li><strong>DataFrame Existence:</strong> Ensures data structure remains valid</li>
                <li><strong>Shape Integrity:</strong> Monitors row/column count changes</li>
                <li><strong>Essential Columns:</strong> Verifies critical columns are preserved</li>
                <li><strong>Target Integrity:</strong> Validates target variable consistency</li>
                <li><strong>Data Types:</strong> Checks for unexpected type changes</li>
                <li><strong>Data Corruption:</strong> Detects potential data quality issues</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📊 Iterative validation report saved: {filename}")