"""
RiskSense AI - Data Ingestion Module

Handles data loading, schema validation, and initial quality checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

from . import config

logger = logging.getLogger(__name__)


# -----------------------------
# Expected Schema Definition
# -----------------------------
EXPECTED_SCHEMA: Dict[str, str] = {
    "loan_id": "object",
    "loan_amnt": "float64",
    "term": "object",
    "int_rate": "float64",
    "installment": "float64",
    "grade": "object",
    "sub_grade": "object",
    "emp_title": "object",
    "emp_length": "object",
    "home_ownership": "object",
    "annual_inc": "float64",
    "verification_status": "object",
    "issue_date": "datetime64[ns]",
    "purpose": "object",
    "dti": "float64",
    "delinq_2yrs": "float64",
    "open_acc": "float64",
    "pub_rec": "float64",
    "revol_bal": "float64",
    "revol_util": "float64",
    "total_acc": "float64",
    "default": "int64",
}


def load_data(
    filepath: Optional[Path] = None,
    sample_frac: Optional[float] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        filepath: Path to the data file. Defaults to config.DEFAULT_DATA_PATH
        sample_frac: Optional fraction of data to sample (for development)
        sample_n: Optional number of rows to sample (alternative to sample_frac)
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If data file does not exist
        ValueError: If loaded data is empty
    """
    if filepath is None:
        filepath = config.DEFAULT_DATA_PATH
    
    logger.info(f"Loading data from: {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    if df.empty:
        raise ValueError("Loaded data is empty")
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    # Sample if requested
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=config.RANDOM_STATE)
        logger.info(f"Sampled to {len(df):,} rows (n={sample_n})")
    elif sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=config.RANDOM_STATE)
        logger.info(f"Sampled to {len(df):,} rows ({sample_frac:.0%})")
    
    # Create binary target from loan_status if needed
    df = create_target_column(df)
    
    return df


def create_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary default target from loan_status if not present.
    
    Lending Club data uses loan_status with values like:
    - 'Fully Paid', 'Current' -> default=0
    - 'Charged Off', 'Default', 'Late (31-120 days)' -> default=1
    """
    df = df.copy()
    
    # If target already exists, return as-is
    if config.TARGET_COLUMN in df.columns:
        logger.info("Target column already exists")
        return df
    
    # Check for raw target column
    if config.RAW_TARGET_COLUMN not in df.columns:
        logger.warning(f"No target column found ({config.RAW_TARGET_COLUMN})")
        return df
    
    # Create binary target
    logger.info(f"Creating binary target from {config.RAW_TARGET_COLUMN}")
    
    # Filter to only completed loans (not current/in grace period)
    completed_statuses = config.DEFAULT_STATUSES + ["Fully Paid"]
    df = df[df[config.RAW_TARGET_COLUMN].isin(completed_statuses)]
    
    # Map to binary
    df[config.TARGET_COLUMN] = df[config.RAW_TARGET_COLUMN].isin(
        config.DEFAULT_STATUSES
    ).astype(int)
    
    logger.info(f"After filtering: {len(df):,} completed loans")
    logger.info(f"Default rate: {df[config.TARGET_COLUMN].mean():.2%}")
    
    return df


def validate_schema(
    df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema against expected schema.
    
    Args:
        df: DataFrame to validate
        expected_schema: Expected column names and dtypes
        strict: If True, raise error on validation failure
    
    Returns:
        Tuple of (is_valid, list of issues)
    """
    if expected_schema is None:
        expected_schema = EXPECTED_SCHEMA
    
    issues = []
    
    # Check for missing columns
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for unexpected columns
    extra_cols = set(df.columns) - set(expected_schema.keys())
    if extra_cols:
        logger.warning(f"Unexpected columns found: {extra_cols}")
    
    # Check dtypes for present columns
    for col, expected_dtype in expected_schema.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if not actual_dtype.startswith(expected_dtype.split("[")[0]):
                issues.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        for issue in issues:
            logger.warning(f"Schema validation issue: {issue}")
        
        if strict:
            raise ValueError(f"Schema validation failed: {issues}")
    
    return is_valid, issues


def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Perform data quality checks and return summary statistics.
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": df.duplicated().sum(),
        "missing_values": {},
        "target_distribution": None,
    }
    
    # Calculate missing value percentages
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 0:
            quality_report["missing_values"][col] = round(missing_pct, 2)
    
    # Target distribution (if present)
    if config.TARGET_COLUMN in df.columns:
        target_dist = df[config.TARGET_COLUMN].value_counts(normalize=True).to_dict()
        quality_report["target_distribution"] = target_dist
    
    logger.info(f"Data quality check complete: {quality_report['total_rows']:,} rows")
    
    return quality_report


def ingest_pipeline(
    filepath: Optional[Path] = None,
    validate: bool = True,
    save_processed: bool = True,
) -> pd.DataFrame:
    """
    Full ingestion pipeline: load, validate, quality check, and optionally save.
    
    Args:
        filepath: Path to raw data file
        validate: Whether to run schema validation
        save_processed: Whether to save processed data
    
    Returns:
        Processed DataFrame
    """
    logger.info("Starting ingestion pipeline")
    
    # Load data
    df = load_data(filepath)
    
    # Validate schema
    if validate:
        is_valid, issues = validate_schema(df, strict=False)
        if not is_valid:
            logger.warning(f"Schema validation found {len(issues)} issues")
    
    # Quality check
    quality_report = check_data_quality(df)
    
    # Log target distribution
    if quality_report["target_distribution"]:
        logger.info(f"Target distribution: {quality_report['target_distribution']}")
    
    # Save processed data
    if save_processed:
        output_path = config.PROCESSED_DATA_DIR / "loans_ingested.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to: {output_path}")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    df = ingest_pipeline()
    print(f"Ingestion complete: {len(df):,} rows")
