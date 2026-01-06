"""
RiskSense AI - Business Rules Module

Implements exclusions, knock-out (KO) rules, and routing logic.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

from . import config

logger = logging.getLogger(__name__)


# -----------------------------
# Exclusion Rules
# -----------------------------
# Records that should not be scored at all

EXCLUSION_RULES = [
    {
        "name": "insufficient_history",
        "description": "Less than 6 months employment history",
        "condition": lambda row: row.get("emp_length", "10+ years") in ["< 1 year", "n/a", None],
        "reason": "Insufficient employment history for scoring",
    },
    {
        "name": "zero_income",
        "description": "Reported income is zero or missing",
        "condition": lambda row: row.get("annual_inc", 1) <= 0,
        "reason": "Cannot score without valid income",
    },
]


# -----------------------------
# Knock-Out (KO) Rules
# -----------------------------
# Hard declines regardless of PD score

KO_RULES = [
    {
        "name": "excessive_delinquency",
        "description": "More than 3 delinquencies in past 2 years",
        "condition": lambda row: row.get("delinq_2yrs", 0) > 3,
        "reason": "Excessive recent delinquencies",
    },
    {
        "name": "bankruptcy",
        "description": "Public bankruptcy record",
        "condition": lambda row: row.get("pub_rec_bankruptcies", 0) > 0,
        "reason": "Bankruptcy on record",
    },
    {
        "name": "extreme_dti",
        "description": "DTI ratio exceeds 50%",
        "condition": lambda row: row.get("dti", 0) > 50,
        "reason": "Debt-to-income ratio too high",
    },
]


# -----------------------------
# Warning Rules
# -----------------------------
# Flags that require attention but don't auto-decline

WARNING_RULES = [
    {
        "name": "high_utilization",
        "description": "Credit utilization above 80%",
        "condition": lambda row: row.get("revol_util", 0) > 80,
        "flag": "HIGH_UTILIZATION",
    },
    {
        "name": "recent_inquiries",
        "description": "Multiple recent credit inquiries",
        "condition": lambda row: row.get("inq_last_6mths", 0) > 3,
        "flag": "MULTIPLE_INQUIRIES",
    },
    {
        "name": "high_loan_amount",
        "description": "Loan amount exceeds 3x annual income",
        "condition": lambda row: (
            row.get("loan_amnt", 0) > 3 * row.get("annual_inc", 1)
            if row.get("annual_inc", 0) > 0 else False
        ),
        "flag": "HIGH_LOAN_TO_INCOME",
    },
]


def check_exclusions(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if record should be excluded from scoring.
    
    Returns:
        Tuple of (is_excluded, reason)
    """
    for rule in EXCLUSION_RULES:
        try:
            if rule["condition"](record):
                logger.debug(f"Exclusion triggered: {rule['name']}")
                return True, rule["reason"]
        except Exception as e:
            logger.warning(f"Error evaluating exclusion rule '{rule['name']}': {e}")
    
    return False, None


def check_ko_rules(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if record triggers any knock-out rules.
    
    Returns:
        Tuple of (is_ko, reason)
    """
    for rule in KO_RULES:
        try:
            if rule["condition"](record):
                logger.debug(f"KO rule triggered: {rule['name']}")
                return True, rule["reason"]
        except Exception as e:
            logger.warning(f"Error evaluating KO rule '{rule['name']}': {e}")
    
    return False, None


def check_warnings(record: Dict[str, Any]) -> List[str]:
    """
    Check for warning flags on a record.
    
    Returns:
        List of warning flags
    """
    flags = []
    
    for rule in WARNING_RULES:
        try:
            if rule["condition"](record):
                flags.append(rule["flag"])
                logger.debug(f"Warning flag: {rule['flag']}")
        except Exception as e:
            logger.warning(f"Error evaluating warning rule '{rule['name']}': {e}")
    
    return flags


def apply_rules(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply all rules to a record and return flag summary.
    
    Returns dict with:
    - excluded: bool
    - exclusion_reason: str or None
    - ko_triggered: bool
    - ko_reason: str or None
    - warnings: list of warning flags
    """
    result = {
        "excluded": False,
        "exclusion_reason": None,
        "ko_triggered": False,
        "ko_reason": None,
        "warnings": [],
    }
    
    # Check exclusions first
    excluded, exclusion_reason = check_exclusions(record)
    if excluded:
        result["excluded"] = True
        result["exclusion_reason"] = exclusion_reason
        return result  # No further processing needed
    
    # Check KO rules
    ko, ko_reason = check_ko_rules(record)
    result["ko_triggered"] = ko
    result["ko_reason"] = ko_reason
    
    # Check warnings (always)
    result["warnings"] = check_warnings(record)
    
    return result


def get_rule_summary() -> Dict[str, Any]:
    """
    Get summary of all configured rules.
    
    Useful for governance and documentation.
    """
    return {
        "exclusion_rules": [
            {"name": r["name"], "description": r["description"]}
            for r in EXCLUSION_RULES
        ],
        "ko_rules": [
            {"name": r["name"], "description": r["description"]}
            for r in KO_RULES
        ],
        "warning_rules": [
            {"name": r["name"], "description": r["description"], "flag": r["flag"]}
            for r in WARNING_RULES
        ],
        "total_rules": len(EXCLUSION_RULES) + len(KO_RULES) + len(WARNING_RULES),
    }


def validate_rule_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze rule coverage on a dataset.
    
    Returns statistics on how many records are affected by each rule.
    """
    logger.info(f"Analyzing rule coverage on {len(df):,} records")
    
    coverage = {
        "total_records": len(df),
        "exclusions": {},
        "ko_rules": {},
        "warnings": {},
    }
    
    # Count exclusions
    for rule in EXCLUSION_RULES:
        count = sum(1 for _, row in df.iterrows() if rule["condition"](row.to_dict()))
        coverage["exclusions"][rule["name"]] = {
            "count": count,
            "percentage": count / len(df) * 100,
        }
    
    # Count KO triggers
    for rule in KO_RULES:
        count = sum(1 for _, row in df.iterrows() if rule["condition"](row.to_dict()))
        coverage["ko_rules"][rule["name"]] = {
            "count": count,
            "percentage": count / len(df) * 100,
        }
    
    # Count warnings
    for rule in WARNING_RULES:
        count = sum(1 for _, row in df.iterrows() if rule["condition"](row.to_dict()))
        coverage["warnings"][rule["name"]] = {
            "count": count,
            "percentage": count / len(df) * 100,
        }
    
    return coverage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    summary = get_rule_summary()
    print(f"Rules module ready. Total rules: {summary['total_rules']}")
