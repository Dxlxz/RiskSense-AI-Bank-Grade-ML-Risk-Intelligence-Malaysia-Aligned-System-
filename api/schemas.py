"""
RiskSense AI - API Schemas

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class LoanApplication(BaseModel):
    """Input schema for a single loan application."""
    
    id: Optional[str] = Field(None, description="Unique loan identifier")
    loan_amnt: float = Field(..., description="Loan amount requested", ge=0)
    term: str = Field(..., description="Loan term (e.g., '36 months', '60 months')")
    int_rate: float = Field(..., description="Interest rate", ge=0, le=100)
    installment: float = Field(..., description="Monthly installment amount", ge=0)
    grade: str = Field(..., description="Credit grade (A-G)")
    sub_grade: Optional[str] = Field(None, description="Credit sub-grade")
    emp_title: Optional[str] = Field(None, description="Employment title")
    emp_length: Optional[str] = Field(None, description="Employment length")
    home_ownership: str = Field(..., description="Home ownership status")
    annual_inc: float = Field(..., description="Annual income", ge=0)
    verification_status: str = Field(..., description="Income verification status")
    purpose: str = Field(..., description="Loan purpose")
    dti: float = Field(..., description="Debt-to-income ratio", ge=0)
    delinq_2yrs: Optional[float] = Field(0, description="Delinquencies in past 2 years")
    open_acc: Optional[float] = Field(None, description="Open credit accounts")
    pub_rec: Optional[float] = Field(0, description="Public records")
    revol_bal: Optional[float] = Field(None, description="Revolving balance")
    revol_util: Optional[float] = Field(None, description="Revolving utilization %")
    total_acc: Optional[float] = Field(None, description="Total credit accounts")
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "LOAN001",
                "loan_amnt": 15000.0,
                "term": "36 months",
                "int_rate": 12.5,
                "installment": 500.0,
                "grade": "B",
                "sub_grade": "B3",
                "emp_title": "Software Engineer",
                "emp_length": "5 years",
                "home_ownership": "RENT",
                "annual_inc": 75000.0,
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "dti": 18.5,
                "delinq_2yrs": 0,
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": 35.0,
                "total_acc": 15,
            }
        }


class ScoringResponse(BaseModel):
    """Output schema for a single scoring result."""
    
    loan_id: str = Field(..., description="Loan identifier")
    pd_score: float = Field(..., description="Probability of default (0-1)")
    risk_band: str = Field(..., description="Risk band classification (A-E)")
    decision: str = Field(..., description="Decision recommendation")
    decision_reason: str = Field(..., description="Reason for decision")
    confidence: float = Field(..., description="Model confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "LOAN001",
                "pd_score": 0.12,
                "risk_band": "C",
                "decision": "MANUAL_REVIEW",
                "decision_reason": "PD score 12.00% requires review",
                "confidence": 0.76,
            }
        }


class BatchScoringRequest(BaseModel):
    """Input schema for batch scoring."""
    
    applications: List[LoanApplication] = Field(
        ...,
        description="List of loan applications to score",
    )


class BatchScoringResponse(BaseModel):
    """Output schema for batch scoring results."""
    
    total_scored: int = Field(..., description="Total applications scored")
    results: List[Dict[str, Any]] = Field(..., description="Scoring results")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ExplanationRequest(BaseModel):
    """Request for model explanation."""
    
    application: LoanApplication
    top_n: int = Field(5, description="Number of top factors to return", ge=1, le=20)


class ExplanationResponse(BaseModel):
    """Response with model explanation."""
    
    loan_id: str
    pd_score: float
    top_factors: List[Dict[str, Any]]
    reason_codes: List[Dict[str, Any]]
    formatted_explanation: str
