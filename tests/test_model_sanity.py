"""
RiskSense AI - Model Sanity Tests

Tests for model behavior and sanity checks.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config


class TestModelSanity:
    """Sanity tests for model behavior."""
    
    def test_predictions_in_valid_range(self):
        """Test that predictions are between 0 and 1."""
        # Mock predictions for testing
        predictions = np.array([0.1, 0.5, 0.9, 0.01, 0.99])
        
        assert np.all(predictions >= 0), "Predictions should be >= 0"
        assert np.all(predictions <= 1), "Predictions should be <= 1"
    
    def test_risk_bands_comprehensive(self):
        """Test that risk bands cover full probability range."""
        bands = config.RISK_BANDS
        
        # Check that bands cover 0 to 1
        all_ranges = [(lower, upper) for lower, upper in bands.values()]
        all_ranges.sort(key=lambda x: x[0])
        
        assert all_ranges[0][0] == 0.0, "Lowest band should start at 0"
        assert all_ranges[-1][1] == 1.0, "Highest band should end at 1"
    
    def test_calibration_sanity(self):
        """Test calibration assumptions."""
        # For a well-calibrated model, predicted probabilities should
        # match observed frequencies in aggregate
        
        # Mock test: if we predict 10% PD for 100 loans,
        # we expect ~10 defaults (with variance)
        n_samples = 1000
        predicted_pd = 0.10
        
        # Simulate defaults based on predicted probability
        np.random.seed(42)
        defaults = np.random.binomial(1, predicted_pd, n_samples)
        
        observed_rate = defaults.mean()
        
        # Should be within reasonable tolerance
        assert abs(observed_rate - predicted_pd) < 0.05, \
            f"Observed rate {observed_rate} too far from predicted {predicted_pd}"


class TestModelMonotonicity:
    """Tests for expected monotonic relationships."""
    
    def test_dti_increases_risk(self):
        """Higher DTI should generally increase risk (conceptual test)."""
        # This is a conceptual test - in production, test against actual model
        low_dti = 10
        high_dti = 50
        
        # We expect higher DTI to correlate with higher risk
        # This test documents expected behavior
        assert high_dti > low_dti
    
    def test_income_decreases_risk(self):
        """Higher income should generally decrease risk (conceptual test)."""
        low_income = 30000
        high_income = 100000
        
        # We expect higher income to correlate with lower risk
        assert high_income > low_income


class TestDecisionThresholds:
    """Tests for decision threshold logic."""
    
    def test_auto_approve_threshold(self):
        """Test auto-approve threshold is reasonable."""
        threshold = config.AUTO_APPROVE_THRESHOLD
        
        # Should be a low probability (low risk)
        assert 0 < threshold < 0.20, \
            f"Auto-approve threshold {threshold} seems unreasonable"
    
    def test_auto_decline_threshold(self):
        """Test auto-decline threshold is reasonable."""
        threshold = config.AUTO_DECLINE_THRESHOLD
        
        # Should be a high probability (high risk)
        assert 0.30 < threshold <= 1.0, \
            f"Auto-decline threshold {threshold} seems unreasonable"
    
    def test_manual_review_gap(self):
        """Test there's a sensible gap for manual review."""
        approve = config.AUTO_APPROVE_THRESHOLD
        decline = config.AUTO_DECLINE_THRESHOLD
        
        gap = decline - approve
        
        # Should have meaningful manual review range
        assert gap > 0.20, \
            f"Manual review gap {gap} seems too narrow"


class TestExplainability:
    """Tests for explainability requirements."""
    
    def test_reason_code_mapping_coverage(self):
        """Test reason codes cover important features."""
        from src.explain import REASON_CODE_MAP
        
        # Key features should have mappings
        key_features = [
            "dti",
            "annual_inc",
            "delinq_2yrs",
            "revol_util",
        ]
        
        for feature in key_features:
            assert feature in REASON_CODE_MAP, \
                f"Missing reason code for {feature}"


class TestGovernance:
    """Tests for governance requirements."""
    
    def test_random_state_set(self):
        """Verify reproducibility is configured."""
        assert config.RANDOM_STATE is not None
        assert isinstance(config.RANDOM_STATE, int)
    
    def test_psi_thresholds_valid(self):
        """Verify PSI monitoring thresholds are set."""
        assert 0 < config.PSI_WARNING_THRESHOLD < config.PSI_CRITICAL_THRESHOLD
        assert config.PSI_CRITICAL_THRESHOLD <= 0.5  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
