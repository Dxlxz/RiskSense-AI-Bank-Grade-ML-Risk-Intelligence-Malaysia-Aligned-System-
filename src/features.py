"""
RiskSense AI - Feature Engineering Module

Handles feature creation, transformation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

from . import config

logger = logging.getLogger(__name__)


# -----------------------------
# Feature Definitions
# -----------------------------
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

CATEGORICAL_FEATURES = [
    "term",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "emp_length",
]


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw data.
    
    Features include ratios, interactions, and behavioral signals.
    """
    df = df.copy()
    
    # Loan-to-income ratio
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    
    # Installment burden
    df["installment_to_income"] = df["installment"] * 12 / (df["annual_inc"] + 1)
    
    # Credit utilization category
    df["high_utilization"] = (df["revol_util"] > 80).astype(int)
    
    # Delinquency flag
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
    
    # Public records flag
    df["has_public_rec"] = (df["pub_rec"] > 0).astype(int)
    
    # Account ratio (revolving balance to total accounts)
    df["revol_per_account"] = df["revol_bal"] / (df["total_acc"] + 1)
    
    logger.info(f"Created {6} derived features")
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for early warning signals.
    
    Note: Requires temporal data with date columns.
    """
    df = df.copy()
    
    if config.DATE_COLUMN in df.columns:
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        
        # Extract date components
        df["issue_month"] = df[config.DATE_COLUMN].dt.month
        df["issue_quarter"] = df[config.DATE_COLUMN].dt.quarter
        df["issue_year"] = df[config.DATE_COLUMN].dt.year
        
        logger.info("Created temporal features from date column")
    else:
        logger.warning(f"Date column '{config.DATE_COLUMN}' not found, skipping temporal features")
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Handle missing values with specified strategies.
    
    Args:
        df: Input DataFrame
        numeric_strategy: 'mean', 'median', or 'zero'
        categorical_strategy: 'mode' or 'unknown'
    """
    df = df.copy()
    
    # Numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            if numeric_strategy == "median":
                fill_value = df[col].median()
            elif numeric_strategy == "mean":
                fill_value = df[col].mean()
            else:
                fill_value = 0
            
            df[col] = df[col].fillna(fill_value)
            logger.debug(f"Filled {col} missing values with {fill_value:.2f}")
    
    # Categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isna().any():
            if categorical_strategy == "mode":
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            else:
                fill_value = "Unknown"
            
            df[col] = df[col].fillna(fill_value)
    
    logger.info("Missing value handling complete")
    
    return df


def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical variables using label encoding.
    
    For production, consider target encoding or one-hot encoding.
    """
    df = df.copy()
    
    if encoders is None:
        encoders = {}
        fit_mode = True
    else:
        fit_mode = False
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if fit_mode:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder
            else:
                # Handle unseen categories
                known_classes = set(encoders[col].classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known_classes else "Unknown"
                )
                df[col] = encoders[col].transform(df[col])
    
    logger.info(f"Encoded {len(CATEGORICAL_FEATURES)} categorical features")
    
    return df, encoders


def split_data(
    df: pd.DataFrame,
    target_col: str = None,
    test_size: float = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with optional stratification.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    
    if test_size is None:
        test_size = config.TEST_SIZE
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c not in config.EXCLUDE_FEATURES]
    X = df[feature_cols]
    y = df[target_col]
    
    # Stratified split
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=config.RANDOM_STATE,
    )
    
    logger.info(f"Split data: train={len(X_train):,}, test={len(X_test):,}")
    logger.info(f"Train default rate: {y_train.mean():.2%}")
    logger.info(f"Test default rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def feature_pipeline(
    df: pd.DataFrame,
    fit_mode: bool = True,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Full feature engineering pipeline.
    
    Args:
        df: Raw DataFrame
        fit_mode: True for training, False for inference
        encoders: Pre-fitted encoders (for inference)
    
    Returns:
        Processed DataFrame and encoders
    """
    logger.info("Starting feature engineering pipeline")
    logger.info(f"Input shape: {df.shape}")
    
    # Select only relevant columns (keep ID, target, numerics, categoricals)
    keep_cols = (
        [config.ID_COLUMN, config.TARGET_COLUMN] + 
        NUMERIC_FEATURES + 
        CATEGORICAL_FEATURES
    )
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()
    logger.info(f"Selected {len(available_cols)} relevant columns")
    
    # Convert numeric columns that might be strings (like int_rate with %)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            # Handle percentage strings like "10.5%"
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived features
    df = create_derived_features(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categoricals
    df, encoders = encode_categoricals(df, encoders)
    
    logger.info(f"Feature pipeline complete: {len(df.columns)} features")
    
    return df, encoders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    print("Feature engineering module ready")
