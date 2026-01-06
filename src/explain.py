"""
RiskSense AI - Explainability Module

Provides SHAP explanations for model decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
from . import config

logger = logging.getLogger(__name__)


# -----------------------------
# Reason Code Mapping
# -----------------------------
REASON_CODE_MAP: Dict[str, str] = {
    "loan_to_income": "Loan amount is high relative to income",
    "installment_to_income": "Monthly installment burden is significant",
    "dti": "Debt-to-income ratio indicates existing obligations",
    "revol_util": "Credit utilization is elevated",
    "int_rate": "Interest rate reflects higher risk tier",
    "delinq_2yrs": "Recent delinquency on credit file",
    "pub_rec": "Public records indicate past issues",
    "annual_inc": "Income level affects repayment capacity",
    "grade": "Credit grade indicates risk profile",
    "emp_length": "Employment history affects stability assessment",
    "has_delinquency": "Past delinquency detected",
    "has_public_rec": "Public records present",
    "high_utilization": "Credit utilization above threshold",
}


def create_explainer(
    model: Any,
    X_background: pd.DataFrame,
    explainer_type: str = "tree",
) -> Any:
    """
    Create SHAP explainer for the model.
    
    Args:
        model: Trained model
        X_background: Background data for explainer
        explainer_type: 'tree' for tree-based, 'kernel' for any model
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Run: pip install shap")
    
    logger.info(f"Creating {explainer_type} explainer")
    
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        # Sample background for kernel explainer (expensive)
        background = shap.sample(X_background, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    return explainer


def explain_single(
    model: Any,
    explainer: Any,
    X: pd.DataFrame,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Generate SHAP explanation for a single record.
    
    Returns dict with:
    - shap_values: raw SHAP values
    - top_factors: top N contributing features
    - reason_codes: human-readable reasons
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")
    
    # Get SHAP values (for positive class)
    shap_values = explainer.shap_values(X)
    
    # Handle tree explainer output format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    # Get single record values
    if len(shap_values.shape) > 1:
        values = shap_values[0]
    else:
        values = shap_values
    
    # Create feature importance DataFrame
    importance = pd.DataFrame({
        "feature": X.columns,
        "shap_value": values,
        "abs_shap": np.abs(values),
    }).sort_values("abs_shap", ascending=False)
    
    # Top factors
    top_factors = importance.head(top_n).to_dict("records")
    
    # Generate reason codes
    reason_codes = []
    for factor in top_factors:
        feature = factor["feature"]
        shap_val = factor["shap_value"]
        
        direction = "increases" if shap_val > 0 else "decreases"
        reason = REASON_CODE_MAP.get(feature, f"Feature '{feature}' {direction} risk")
        
        reason_codes.append({
            "feature": feature,
            "direction": direction,
            "impact": abs(shap_val),
            "reason": reason,
        })
    
    return {
        "shap_values": values.tolist(),
        "top_factors": top_factors,
        "reason_codes": reason_codes,
    }


def explain_batch(
    model: Any,
    explainer: Any,
    X: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate SHAP explanations for a batch of records.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")
    
    logger.info(f"Generating explanations for {len(X):,} records")
    
    # Get all SHAP values at once (more efficient)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    explanations = []
    for i in range(len(X)):
        values = shap_values[i]
        
        importance = pd.DataFrame({
            "feature": X.columns,
            "shap_value": values,
            "abs_shap": np.abs(values),
        }).sort_values("abs_shap", ascending=False)
        
        top_factors = importance.head(top_n)[["feature", "shap_value"]].to_dict("records")
        
        explanations.append({
            "index": i,
            "top_factors": top_factors,
        })
    
    logger.info("Batch explanation complete")
    
    return explanations


def global_feature_importance(
    explainer: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate global feature importance using SHAP.
    
    Returns DataFrame with mean absolute SHAP values per feature.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed")
    
    logger.info("Calculating global feature importance")
    
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    
    importance["rank"] = range(1, len(importance) + 1)
    
    return importance


def format_explanation_for_officer(
    reason_codes: List[Dict[str, Any]],
) -> str:
    """
    Format explanation for bank officer consumption.
    
    Produces human-readable text suitable for decision support.
    """
    lines = ["Risk Assessment Factors:"]
    lines.append("-" * 40)
    
    for i, rc in enumerate(reason_codes, 1):
        direction_symbol = "↑" if rc["direction"] == "increases" else "↓"
        impact_level = "High" if rc["impact"] > 0.1 else "Medium" if rc["impact"] > 0.05 else "Low"
        
        lines.append(f"{i}. {rc['reason']}")
        lines.append(f"   Impact: {impact_level} {direction_symbol}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    print(f"Explainability module ready. SHAP available: {SHAP_AVAILABLE}")
