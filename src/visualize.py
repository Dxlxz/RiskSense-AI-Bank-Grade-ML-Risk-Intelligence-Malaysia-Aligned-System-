"""
RiskSense AI - Visualization Module

Reusable plotting functions for model analysis and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Any
import logging

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
except ImportError:
    roc_curve = None

logger = logging.getLogger(__name__)

# Default style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'info': '#3498DB',
}


def plot_roc_curves(
    y_true: np.ndarray,
    predictions: dict,
    title: str = "ROC Curve Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        y_true: True labels
        predictions: Dict of {model_name: predicted_probabilities}
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = list(COLORS.values())
    for i, (name, y_pred) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        ax.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
                label=f'{name} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC curve to: {save_path}")
    
    return fig


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[dict] = None,
    title: str = "Score Distribution by Outcome",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot score distribution by outcome class.
    
    Args:
        scores: Predicted probabilities
        labels: True labels (0/1)
        thresholds: Optional dict with 'approve' and 'decline' thresholds
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(scores[labels == 0], bins=50, alpha=0.6, 
            label='Non-Default', color=COLORS['success'], density=True)
    ax.hist(scores[labels == 1], bins=50, alpha=0.6, 
            label='Default', color=COLORS['danger'], density=True)
    
    if thresholds:
        if 'approve' in thresholds:
            ax.axvline(thresholds['approve'], color=COLORS['success'], 
                      linestyle='--', linewidth=2, label=f"Auto-Approve (<{thresholds['approve']:.0%})")
        if 'decline' in thresholds:
            ax.axvline(thresholds['decline'], color=COLORS['danger'], 
                      linestyle='--', linewidth=2, label=f"Auto-Decline (>{thresholds['decline']:.0%})")
    
    ax.set_xlabel('Predicted Default Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance: pd.DataFrame,
    feature_col: str = 'feature',
    importance_col: str = 'importance',
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort and take top N
    data = importance.sort_values(importance_col, ascending=True).tail(top_n)
    
    # Create color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
    
    ax.barh(data[feature_col], data[importance_col], color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Non-Default', 'Default'],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_risk_band_distribution(
    risk_bands: pd.Series,
    title: str = "Risk Band Distribution",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot distribution of applications across risk bands.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    band_counts = risk_bands.value_counts().sort_index()
    colors = [COLORS['success'], COLORS['info'], COLORS['warning'], 
              COLORS['secondary'], COLORS['danger']][:len(band_counts)]
    
    bars = ax.bar(band_counts.index, band_counts.values, color=colors, 
                  edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, band_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Risk Band', fontsize=12)
    ax.set_ylabel('Number of Applications', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_monitoring_dashboard(
    psi: float,
    feature_drift: dict,
    psi_threshold_warning: float = 0.10,
    psi_threshold_critical: float = 0.25,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot monitoring dashboard with PSI gauge and feature drift.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: PSI Gauge
    ax1 = axes[0]
    
    # Determine status
    if psi < psi_threshold_warning:
        status_color = COLORS['success']
        status_text = 'GREEN'
    elif psi < psi_threshold_critical:
        status_color = COLORS['warning']
        status_text = 'AMBER'
    else:
        status_color = COLORS['danger']
        status_text = 'RED'
    
    ax1.barh(['PSI'], [psi], color=status_color, height=0.4)
    ax1.axvline(psi_threshold_warning, color=COLORS['warning'], linestyle='--', label=f'Warning ({psi_threshold_warning})')
    ax1.axvline(psi_threshold_critical, color=COLORS['danger'], linestyle='--', label=f'Critical ({psi_threshold_critical})')
    ax1.set_xlim(0, max(0.4, psi * 1.2))
    ax1.set_title(f'Population Stability Index: {status_text}', fontsize=14, fontweight='bold')
    ax1.text(psi + 0.01, 0, f'{psi:.4f}', va='center', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right: Feature Drift
    ax2 = axes[1]
    
    if feature_drift:
        drift_df = pd.DataFrame([
            {'feature': k, 'psi': v} 
            for k, v in feature_drift.items() 
            if not np.isnan(v)
        ]).sort_values('psi')
        
        colors = [COLORS['danger'] if v > psi_threshold_warning else COLORS['success'] 
                 for v in drift_df['psi']]
        
        ax2.barh(drift_df['feature'], drift_df['psi'], color=colors, edgecolor='black', alpha=0.8)
        ax2.axvline(psi_threshold_warning, color=COLORS['warning'], linestyle='--')
        ax2.set_xlabel('Feature PSI')
        ax2.set_title('Feature Drift Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_model_report(
    model_name: str,
    metrics: dict,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate a complete visual report for a model.
    
    Creates multiple plots saved to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating model report for: {model_name}")
    
    # ROC Curve
    plot_roc_curves(
        y_true, 
        {model_name: y_pred_proba},
        save_path=output_dir / 'roc_curve.png'
    )
    
    # Score Distribution
    plot_score_distribution(
        y_pred_proba, 
        y_true,
        thresholds={'approve': 0.05, 'decline': 0.50},
        save_path=output_dir / 'score_distribution.png'
    )
    
    # Feature Importance
    plot_feature_importance(
        feature_importance,
        save_path=output_dir / 'feature_importance.png'
    )
    
    # Confusion Matrix
    y_pred = (y_pred_proba >= 0.5).astype(int)
    plot_confusion_matrix(
        y_true, 
        y_pred,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    logger.info(f"Model report saved to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Visualization module ready")
    print(f"Available functions:")
    print("  - plot_roc_curves()")
    print("  - plot_score_distribution()")
    print("  - plot_feature_importance()")
    print("  - plot_confusion_matrix()")
    print("  - plot_risk_band_distribution()")
    print("  - plot_monitoring_dashboard()")
    print("  - create_model_report()")
