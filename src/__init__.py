"""
RiskSense AI - Bank-Grade ML Risk Intelligence Platform

This package provides credit risk scoring, early warning detection,
explainability, and drift monitoring capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import config
from . import ingestion
from . import features
from . import train
from . import score
from . import explain
from . import rules
from . import monitor
from . import visualize

__all__ = [
    "config",
    "ingestion",
    "features",
    "train",
    "score",
    "explain",
    "rules",
    "monitor",
    "visualize",
]
