"""
RiskSense AI - Model Training Module

Handles model training, evaluation, and persistence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from . import config
from . import ingestion
from . import features

logger = logging.getLogger(__name__)


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Train baseline Logistic Regression model.
    
    Baseline is essential for:
    - Regulatory compliance (interpretability)
    - Champion/challenger comparisons
    - Feature sanity checks
    """
    logger.info("Training baseline Logistic Regression model")
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        solver="lbfgs",
    )
    
    model.fit(X_train, y_train)
    
    # Log feature importances
    importance = pd.Series(
        np.abs(model.coef_[0]),
        index=X_train.columns,
    ).sort_values(ascending=False)
    
    logger.info(f"Top 5 features: {importance.head().to_dict()}")
    
    return model


def train_champion_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """
    Train champion XGBoost model.
    
    Uses gradient boosting for improved predictive power
    while maintaining explainability through SHAP.
    """
    logger.info("Training champion XGBoost model")
    
    if params is None:
        params = config.XGBOOST_PARAMS.copy()
    
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    
    # Log feature importances
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)
    
    logger.info(f"Top 5 features (gain): {importance.head().to_dict()}")
    
    return model


def calibrate_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Calibrate probability outputs using isotonic regression.
    
    Critical for banking: raw probabilities are often overconfident.
    """
    logger.info(f"Calibrating model using {method} regression")
    
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)
    
    return calibrated


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Evaluate model performance with banking-relevant metrics.
    """
    logger.info(f"Evaluating {model_name}")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "avg_precision": average_precision_score(y_test, y_pred_proba),
        "gini": 2 * roc_auc_score(y_test, y_pred_proba) - 1,
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["true_positive_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Cost-sensitive: assume false negative (missed default) costs 10x false positive
    cost_ratio = 10
    metrics["weighted_cost"] = (fp * 1 + fn * cost_ratio) / len(y_test)
    
    logger.info(f"{model_name} AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"{model_name} Gini: {metrics['gini']:.4f}")
    
    return metrics


def save_model(
    model: Any,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save model with metadata for governance.
    """
    output_path = config.MODELS_DIR / f"{model_name}.joblib"
    
    artifact = {
        "model": model,
        "metadata": metadata or {},
    }
    
    joblib.dump(artifact, output_path)
    logger.info(f"Saved model to: {output_path}")
    
    return output_path


def load_model(model_name: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and its metadata.
    """
    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    artifact = joblib.load(model_path)
    
    return artifact["model"], artifact.get("metadata", {})


def train_pipeline(
    data_path: Optional[Path] = None,
    save_models: bool = True,
    sample_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full training pipeline: ingest → features → train → evaluate → save.
    
    Args:
        data_path: Path to data file (optional, uses config default)
        save_models: Whether to save models to disk
        sample_n: Optional number of samples for faster development
    """
    logger.info("=" * 50)
    logger.info("Starting training pipeline")
    logger.info("=" * 50)
    
    # Load and process data
    df = ingestion.load_data(data_path, sample_n=sample_n)
    df, encoders = features.feature_pipeline(df, fit_mode=True)
    
    # Split data
    X_train, X_test, y_train, y_test = features.split_data(df)
    
    # Train baseline
    baseline = train_baseline_model(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test, "Baseline LR")
    
    # Train champion
    champion = train_champion_model(X_train, y_train)
    champion_metrics = evaluate_model(champion, X_test, y_test, "Champion XGB")
    
    # Compare and select
    if champion_metrics["auc_roc"] > baseline_metrics["auc_roc"]:
        logger.info("Champion model outperforms baseline")
        selected_model = champion
        selected_name = "champion_xgb"
    else:
        logger.warning("Baseline model selected (champion did not improve)")
        selected_model = baseline
        selected_name = "baseline_lr"
    
    # Save models
    if save_models:
        save_model(baseline, "baseline_lr", {"metrics": baseline_metrics})
        save_model(champion, "champion_xgb", {"metrics": champion_metrics})
        joblib.dump(encoders, config.MODELS_DIR / "encoders.joblib")
    
    logger.info("=" * 50)
    logger.info("Training pipeline complete")
    logger.info("=" * 50)
    
    return {
        "baseline": {"model": baseline, "metrics": baseline_metrics},
        "champion": {"model": champion, "metrics": champion_metrics},
        "selected": selected_name,
        "encoders": encoders,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    results = train_pipeline()
    print(f"\nSelected model: {results['selected']}")
    print(f"Champion AUC: {results['champion']['metrics']['auc_roc']:.4f}")
