"""
RiskSense AI - Scoring Module

Handles batch scoring, risk band assignment, and decision routing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import logging
from datetime import datetime

from . import config
from . import features as feat_module
from . import rules

logger = logging.getLogger(__name__)


def load_scoring_artifacts() -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and encoders for scoring.
    """
    model_path = config.MODELS_DIR / "champion_xgb.joblib"
    encoder_path = config.MODELS_DIR / "encoders.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    artifact = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    
    return artifact["model"], encoders


def assign_risk_band(pd_score: float) -> str:
    """
    Assign risk band based on PD score.
    
    Bands are defined in config.RISK_BANDS.
    """
    for band, (lower, upper) in config.RISK_BANDS.items():
        if lower <= pd_score < upper:
            return band
    return "E"  # Default to highest risk


def assign_decision(
    pd_score: float,
    rule_flags: Dict[str, bool],
) -> Tuple[str, str]:
    """
    Assign decision recommendation based on PD and rule flags.
    
    Returns:
        Tuple of (decision, reason)
    """
    # Check for KO rules first
    if rule_flags.get("ko_triggered", False):
        return "DECLINE", rule_flags.get("ko_reason", "Rule-based decline")
    
    # Check for exclusions
    if rule_flags.get("excluded", False):
        return "EXCLUDE", rule_flags.get("exclusion_reason", "Excluded from scoring")
    
    # PD-based decisions
    if pd_score <= config.AUTO_APPROVE_THRESHOLD:
        return "APPROVE", f"PD score {pd_score:.2%} below threshold"
    
    if pd_score >= config.AUTO_DECLINE_THRESHOLD:
        return "DECLINE", f"PD score {pd_score:.2%} above threshold"
    
    return "MANUAL_REVIEW", f"PD score {pd_score:.2%} requires review"


def score_batch(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    encoders: Optional[Dict] = None,
    apply_rules: bool = True,
) -> pd.DataFrame:
    """
    Score a batch of records.
    
    Returns DataFrame with:
    - pd_score: Probability of default
    - risk_band: A-E classification
    - decision: APPROVE/DECLINE/MANUAL_REVIEW/EXCLUDE
    - decision_reason: Explanation for decision
    - confidence: Model confidence (1 - uncertainty)
    """
    logger.info(f"Scoring batch of {len(df):,} records")
    
    # Load artifacts if not provided
    if model is None or encoders is None:
        model, encoders = load_scoring_artifacts()
    
    # Process features
    df_processed, _ = feat_module.feature_pipeline(df, fit_mode=False, encoders=encoders)
    
    # Get feature columns (exclude target and ID)
    feature_cols = [c for c in df_processed.columns if c not in config.EXCLUDE_FEATURES]
    X = df_processed[feature_cols]
    
    # Score
    pd_scores = model.predict_proba(X)[:, 1]
    
    # Create output DataFrame
    output = pd.DataFrame({
        config.ID_COLUMN: df[config.ID_COLUMN] if config.ID_COLUMN in df.columns else range(len(df)),
        "pd_score": pd_scores,
        "risk_band": [assign_risk_band(s) for s in pd_scores],
        "score_timestamp": datetime.now().isoformat(),
    })
    
    # Confidence (simple heuristic: distance from 0.5)
    output["confidence"] = 1 - 2 * np.abs(pd_scores - 0.5)
    
    # Apply rules
    if apply_rules:
        rule_results = [rules.apply_rules(row) for _, row in df.iterrows()]
        
        decisions = []
        reasons = []
        for pd_score, rule_flags in zip(pd_scores, rule_results):
            decision, reason = assign_decision(pd_score, rule_flags)
            decisions.append(decision)
            reasons.append(reason)
        
        output["decision"] = decisions
        output["decision_reason"] = reasons
    else:
        output["decision"] = None
        output["decision_reason"] = None
    
    # Log summary
    logger.info(f"Scoring complete. Risk band distribution:")
    logger.info(output["risk_band"].value_counts().to_dict())
    
    if "decision" in output.columns and output["decision"].notna().any():
        logger.info(f"Decision distribution: {output['decision'].value_counts().to_dict()}")
    
    return output


def score_single(
    record: Dict[str, Any],
    model: Optional[Any] = None,
    encoders: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Score a single record.
    
    Convenience wrapper around score_batch.
    """
    df = pd.DataFrame([record])
    result = score_batch(df, model, encoders)
    return result.iloc[0].to_dict()


def score_pipeline(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full scoring pipeline: load data → score → save results.
    """
    logger.info("=" * 50)
    logger.info("Starting scoring pipeline")
    logger.info("=" * 50)
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records from {input_path}")
    
    # Score
    results = score_batch(df)
    
    # Save
    if output_path is None:
        output_path = config.PROCESSED_DATA_DIR / "scored_output.csv"
    
    results.to_csv(output_path, index=False)
    logger.info(f"Saved scored output to: {output_path}")
    
    logger.info("=" * 50)
    logger.info("Scoring pipeline complete")
    logger.info("=" * 50)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    print("Scoring module ready")
