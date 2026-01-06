"""
RiskSense AI - Configuration Module

Global settings, paths, thresholds, and hyperparameters for the risk modeling system.
"""

from pathlib import Path
from typing import Dict, Any
import os

# -----------------------------
# Path Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data sources (inside project data/raw folder)
LENDING_CLUB_DATA = RAW_DATA_DIR / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"
LOAN_DEFAULT_DATA = RAW_DATA_DIR / "Loan_Default.csv"

# Default data source
DEFAULT_DATA_PATH = LENDING_CLUB_DATA

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Data Configuration
# -----------------------------
# Lending Club column mappings
TARGET_COLUMN = "default"  # Will be derived from loan_status
RAW_TARGET_COLUMN = "loan_status"  # Original column in Lending Club data
ID_COLUMN = "id"  # Lending Club uses 'id', not 'loan_id'
DATE_COLUMN = "issue_d"  # Lending Club uses 'issue_d'

# Default statuses (will be mapped to default=1)
DEFAULT_STATUSES = ["Charged Off", "Default", "Late (31-120 days)"]

# Features to exclude from modeling
EXCLUDE_FEATURES = [
    ID_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    RAW_TARGET_COLUMN,
    "member_id",
    "url",
    "desc",
    "title",
]

# -----------------------------
# Model Configuration
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Class imbalance handling
SCALE_POS_WEIGHT = 10  # Approximate ratio of negatives to positives

# -----------------------------
# Thresholds
# -----------------------------
# Risk band thresholds (PD score)
RISK_BANDS: Dict[str, tuple] = {
    "A": (0.00, 0.05),   # Very Low Risk
    "B": (0.05, 0.10),   # Low Risk
    "C": (0.10, 0.20),   # Medium Risk
    "D": (0.20, 0.35),   # High Risk
    "E": (0.35, 1.00),   # Very High Risk
}

# Decision thresholds
AUTO_APPROVE_THRESHOLD = 0.05   # PD <= 5%
AUTO_DECLINE_THRESHOLD = 0.50   # PD >= 50%
MANUAL_REVIEW_BAND = (0.05, 0.50)

# Confidence threshold for scoring
MIN_CONFIDENCE_THRESHOLD = 0.70

# -----------------------------
# Monitoring Thresholds
# -----------------------------
PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25

DRIFT_WARNING_THRESHOLD = 0.05
DRIFT_CRITICAL_THRESHOLD = 0.10

# -----------------------------
# XGBoost Hyperparameters
# -----------------------------
XGBOOST_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# -----------------------------
# Logging Configuration
# -----------------------------
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# -----------------------------
# API Configuration
# -----------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
