"""
RiskSense AI - Pipeline Tests

Unit and integration tests for the data and model pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src import ingestion
from src import features
from src import rules


class TestConfig:
    """Test configuration module."""
    
    def test_paths_exist(self):
        """Verify critical paths are defined."""
        assert config.PROJECT_ROOT.exists()
        assert config.DATA_DIR is not None
        
    def test_risk_bands_valid(self):
        """Verify risk bands are properly defined."""
        assert len(config.RISK_BANDS) > 0
        
        for band, (lower, upper) in config.RISK_BANDS.items():
            assert lower < upper, f"Band {band} has invalid range"
            assert 0 <= lower <= 1, f"Band {band} lower bound out of range"
            assert 0 <= upper <= 1, f"Band {band} upper bound out of range"
    
    def test_thresholds_valid(self):
        """Verify decision thresholds are sensible."""
        assert 0 <= config.AUTO_APPROVE_THRESHOLD < config.AUTO_DECLINE_THRESHOLD <= 1


class TestIngestion:
    """Test data ingestion module."""
    
    def test_schema_validation_with_valid_data(self):
        """Test schema validation passes for valid data."""
        valid_df = pd.DataFrame({
            "loan_id": ["L001"],
            "loan_amnt": [10000.0],
            "annual_inc": [50000.0],
            "default": [0],
        })
        
        # Should not raise
        is_valid, issues = ingestion.validate_schema(
            valid_df,
            expected_schema={"loan_id": "object", "loan_amnt": "float64"},
            strict=False,
        )
        
        # May have issues due to missing columns, but should not crash
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_quality_check_returns_dict(self):
        """Test quality check returns expected structure."""
        df = pd.DataFrame({
            "col1": [1, 2, np.nan],
            "col2": ["a", "b", "c"],
            config.TARGET_COLUMN: [0, 1, 0],
        })
        
        report = ingestion.check_data_quality(df)
        
        assert "total_rows" in report
        assert "total_columns" in report
        assert "missing_values" in report
        assert report["total_rows"] == 3


class TestFeatures:
    """Test feature engineering module."""
    
    def test_derived_features_creation(self):
        """Test derived features are created correctly."""
        df = pd.DataFrame({
            "loan_amnt": [10000.0],
            "annual_inc": [50000.0],
            "installment": [300.0],
            "revol_util": [85.0],
            "delinq_2yrs": [1],
            "pub_rec": [0],
            "revol_bal": [5000.0],
            "total_acc": [10],
        })
        
        result = features.create_derived_features(df)
        
        assert "loan_to_income" in result.columns
        assert "installment_to_income" in result.columns
        assert "high_utilization" in result.columns
        assert "has_delinquency" in result.columns
    
    def test_missing_value_handling(self):
        """Test missing values are handled."""
        df = pd.DataFrame({
            "loan_amnt": [10000.0, np.nan, 15000.0],
            "annual_inc": [50000.0, 60000.0, np.nan],
        })
        
        result = features.handle_missing_values(df)
        
        assert not result["loan_amnt"].isna().any()
        assert not result["annual_inc"].isna().any()


class TestRules:
    """Test business rules module."""
    
    def test_exclusion_zero_income(self):
        """Test zero income triggers exclusion."""
        record = {"annual_inc": 0}
        
        excluded, reason = rules.check_exclusions(record)
        
        assert excluded is True
        assert reason is not None
    
    def test_ko_excessive_delinquency(self):
        """Test excessive delinquency triggers KO."""
        record = {"delinq_2yrs": 5}
        
        ko, reason = rules.check_ko_rules(record)
        
        assert ko is True
        assert "delinquencies" in reason.lower()
    
    def test_warning_high_utilization(self):
        """Test high utilization triggers warning."""
        record = {"revol_util": 90}
        
        warnings = rules.check_warnings(record)
        
        assert "HIGH_UTILIZATION" in warnings
    
    def test_apply_rules_returns_expected_structure(self):
        """Test apply_rules returns complete structure."""
        record = {"annual_inc": 50000, "delinq_2yrs": 0, "revol_util": 50}
        
        result = rules.apply_rules(record)
        
        assert "excluded" in result
        assert "ko_triggered" in result
        assert "warnings" in result
        assert isinstance(result["warnings"], list)
    
    def test_rule_summary(self):
        """Test rule summary is valid."""
        summary = rules.get_rule_summary()
        
        assert "exclusion_rules" in summary
        assert "ko_rules" in summary
        assert "warning_rules" in summary
        assert summary["total_rules"] > 0


class TestIntegration:
    """Integration tests for pipeline components."""
    
    def test_feature_pipeline_end_to_end(self):
        """Test feature pipeline processes data correctly."""
        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "loan_amnt": [10000.0, 15000.0],
            "term": ["36 months", "60 months"],
            "int_rate": [10.0, 15.0],
            "installment": [300.0, 400.0],
            "grade": ["B", "C"],
            "sub_grade": ["B1", "C2"],
            "emp_title": ["Engineer", "Manager"],
            "emp_length": ["5 years", "10+ years"],
            "home_ownership": ["RENT", "OWN"],
            "annual_inc": [50000.0, 75000.0],
            "verification_status": ["Verified", "Not Verified"],
            "purpose": ["debt_consolidation", "credit_card"],
            "dti": [15.0, 20.0],
            "delinq_2yrs": [0.0, 1.0],
            "open_acc": [5.0, 8.0],
            "pub_rec": [0.0, 0.0],
            "revol_bal": [3000.0, 5000.0],
            "revol_util": [30.0, 50.0],
            "total_acc": [10.0, 15.0],
            "default": [0, 1],
        })
        
        processed, encoders = features.feature_pipeline(df, fit_mode=True)
        
        assert len(processed) == 2
        assert encoders is not None
        assert len(processed.columns) > len(df.columns)  # Derived features added


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
