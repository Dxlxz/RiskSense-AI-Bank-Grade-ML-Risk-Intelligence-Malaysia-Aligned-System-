"""
RiskSense AI - API Module

FastAPI application for real-time scoring and explanations.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
API_DIR = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any

from api.schemas import (
    LoanApplication,
    ScoringResponse,
    HealthResponse,
    BatchScoringRequest,
    BatchScoringResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RiskSense AI",
    description="Bank-Grade ML Risk Intelligence Platform API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for loaded model
model_state: Dict[str, Any] = {
    "model": None,
    "encoders": None,
    "loaded": False,
}


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    try:
        from src import score as score_module
        model, encoders = score_module.load_scoring_artifacts()
        model_state["model"] = model
        model_state["encoders"] = encoders
        model_state["loaded"] = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_state["loaded"] = False


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy" if model_state["loaded"] else "degraded",
        model_loaded=model_state["loaded"],
        version="0.1.0",
    )


@app.post("/score", response_model=ScoringResponse)
async def score_application(application: LoanApplication):
    """
    Score a single loan application.
    
    Returns PD score, risk band, decision recommendation, and explanations.
    """
    if not model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )
    
    try:
        from src import score as score_module
        
        # Convert to dict and score
        record = application.dict()
        result = score_module.score_single(
            record,
            model=model_state["model"],
            encoders=model_state["encoders"],
        )
        
        return ScoringResponse(
            loan_id=result.get("loan_id", "unknown"),
            pd_score=result["pd_score"],
            risk_band=result["risk_band"],
            decision=result["decision"],
            decision_reason=result["decision_reason"],
            confidence=result["confidence"],
        )
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}",
        )


@app.post("/score/batch", response_model=BatchScoringResponse)
async def score_batch(request: BatchScoringRequest):
    """
    Score a batch of loan applications.
    
    More efficient for bulk scoring operations.
    """
    if not model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable.",
        )
    
    try:
        import pandas as pd
        from src import score as score_module
        
        # Convert to DataFrame
        records = [app.dict() for app in request.applications]
        df = pd.DataFrame(records)
        
        # Score batch
        results = score_module.score_batch(
            df,
            model=model_state["model"],
            encoders=model_state["encoders"],
        )
        
        return BatchScoringResponse(
            total_scored=len(results),
            results=results.to_dict("records"),
        )
        
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch scoring failed: {str(e)}",
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not model_state["loaded"]:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "model_type": type(model_state["model"]).__name__,
        "n_features": getattr(model_state["model"], "n_features_in_", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
