"""
RiskSense AI - Monitoring Module

Handles drift detection, score stability, and model health monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import logging

from . import config

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures shift in score distribution between two populations.
    
    Interpretation:
    - PSI < 0.10: No significant change
    - 0.10 <= PSI < 0.25: Moderate change, investigation needed
    - PSI >= 0.25: Significant change, action required
    
    Args:
        expected: Reference distribution (training scores)
        actual: Current distribution (production scores)
        n_bins: Number of bins for discretization
    
    Returns:
        PSI value
    """
    # Create bins from expected distribution
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calculate proportions
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]
    
    # Avoid division by zero
    expected_pct = (expected_counts + 0.0001) / len(expected)
    actual_pct = (actual_counts + 0.0001) / len(actual)
    
    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi


def calculate_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate drift for each feature using PSI.
    
    Returns dict of {feature_name: psi_value}
    """
    if features is None:
        features = [c for c in reference.columns if c in current.columns]
    
    drift_scores = {}
    
    for feature in features:
        if reference[feature].dtype in ['float64', 'int64']:
            try:
                psi = calculate_psi(
                    reference[feature].dropna().values,
                    current[feature].dropna().values,
                )
                drift_scores[feature] = psi
            except Exception as e:
                logger.warning(f"Could not calculate drift for {feature}: {e}")
                drift_scores[feature] = np.nan
    
    return drift_scores


def calculate_score_stability(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze stability of model scores over time.
    
    Returns comprehensive stability metrics.
    """
    psi = calculate_psi(reference_scores, current_scores)
    
    stability = {
        "psi": psi,
        "psi_status": (
            "GREEN" if psi < config.PSI_WARNING_THRESHOLD
            else "AMBER" if psi < config.PSI_CRITICAL_THRESHOLD
            else "RED"
        ),
        "reference_mean": float(np.mean(reference_scores)),
        "current_mean": float(np.mean(current_scores)),
        "mean_shift": float(np.mean(current_scores) - np.mean(reference_scores)),
        "reference_std": float(np.std(reference_scores)),
        "current_std": float(np.std(current_scores)),
    }
    
    return stability


def generate_monitoring_report(
    reference_data: pd.DataFrame,
    reference_scores: np.ndarray,
    current_data: pd.DataFrame,
    current_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Generate comprehensive monitoring report.
    
    Includes:
    - Score stability (PSI)
    - Feature drift analysis
    - Distribution comparisons
    - Alerts and recommendations
    """
    logger.info("Generating monitoring report")
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "reference_period": {
            "n_records": len(reference_data),
            "score_mean": float(np.mean(reference_scores)),
        },
        "current_period": {
            "n_records": len(current_data),
            "score_mean": float(np.mean(current_scores)),
        },
        "score_stability": {},
        "feature_drift": {},
        "alerts": [],
        "recommendations": [],
    }
    
    # Score stability
    stability = calculate_score_stability(reference_scores, current_scores)
    report["score_stability"] = stability
    
    if stability["psi_status"] == "RED":
        report["alerts"].append({
            "level": "CRITICAL",
            "message": f"Score distribution has shifted significantly (PSI={stability['psi']:.3f})",
            "action": "Investigate score drift and consider model retraining",
        })
    elif stability["psi_status"] == "AMBER":
        report["alerts"].append({
            "level": "WARNING",
            "message": f"Score distribution shows moderate shift (PSI={stability['psi']:.3f})",
            "action": "Monitor closely and prepare for potential retraining",
        })
    
    # Feature drift
    drift_scores = calculate_feature_drift(reference_data, current_data)
    report["feature_drift"] = drift_scores
    
    # Check for drifting features
    drifting_features = [
        f for f, psi in drift_scores.items()
        if not np.isnan(psi) and psi > config.DRIFT_WARNING_THRESHOLD
    ]
    
    if drifting_features:
        report["alerts"].append({
            "level": "WARNING",
            "message": f"Feature drift detected in: {drifting_features}",
            "action": "Review feature pipelines and data quality",
        })
    
    # Generate recommendations
    if stability["mean_shift"] > 0.05:
        report["recommendations"].append(
            "Average risk score has increased. Consider stricter approval thresholds."
        )
    elif stability["mean_shift"] < -0.05:
        report["recommendations"].append(
            "Average risk score has decreased. Review for potential data quality issues."
        )
    
    if len(report["alerts"]) == 0:
        report["recommendations"].append(
            "No significant drift detected. Continue monitoring."
        )
    
    logger.info(f"Report generated with {len(report['alerts'])} alerts")
    
    return report


def save_monitoring_report(
    report: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save monitoring report to file.
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.LOGS_DIR / f"monitoring_report_{timestamp}.json"
    
    import json
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Saved monitoring report to: {output_path}")
    
    return output_path


def print_report_summary(report: Dict[str, Any]) -> None:
    """
    Print human-readable report summary.
    """
    print("\n" + "=" * 60)
    print("RISKSENSE AI - MONITORING REPORT")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print()
    
    # Score stability
    stability = report["score_stability"]
    print(f"📊 Score Stability: {stability['psi_status']}")
    print(f"   PSI: {stability['psi']:.4f}")
    print(f"   Mean Shift: {stability['mean_shift']:+.4f}")
    print()
    
    # Alerts
    if report["alerts"]:
        print("⚠️  ALERTS:")
        for alert in report["alerts"]:
            print(f"   [{alert['level']}] {alert['message']}")
    else:
        print("✅ No alerts")
    print()
    
    # Recommendations
    print("📋 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    print("Monitoring module ready")
    print(f"PSI thresholds: Warning={config.PSI_WARNING_THRESHOLD}, Critical={config.PSI_CRITICAL_THRESHOLD}")
