"""
Job Fraud Shield - Production REST API
======================================
FastAPI microservice providing real-time inference, batch scoring,
heuristic rule scanning, and explainability for job fraud detection.

Endpoints:
- GET  /health         : Readiness and liveness probe
- GET  /info           : Model metadata, architecture, and benchmark stats
- POST /predict        : Single posting analysis with risk level and token attribution
- POST /batch-predict  : High-throughput batch processing for job boards / scrapers
"""

import os
import re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from model import JobFraudModelPipeline
from heuristics import scan_risk_indicators

app = FastAPI(
    title="Job Fraud Shield API",
    description="Production-ready REST API for AI-powered scam and fraudulent job posting detection.",
    version="1.0.0"
)

# Initialize and load pipeline on startup
pipeline = JobFraudModelPipeline()
models_dir = os.path.join(os.path.dirname(__file__), "models")

@app.on_event("startup")
def load_model_pipeline():
    print(f"Loading JobFraudModelPipeline from: {models_dir}")
    pipeline.load(models_dir)
    print("Model pipeline loaded and ready for production inference.")

# Request & Response Schemas
class JobPredictionRequest(BaseModel):
    text: str = Field(..., description="Raw job posting text (title, description, requirements, benefits, etc.)", min_length=10)
    threshold: Optional[float] = Field(0.50, description="Decision threshold for fraud classification (0.10 to 0.90)", ge=0.10, le=0.90)
    model_type: Optional[str] = Field("primary", description="'primary' (Balanced LR), 'baseline' (Naive Bayes), or 'standard' (Default LR)")
    include_explanation: Optional[bool] = Field(True, description="Whether to include token-level feature attribution in the response")

class HeuristicFlag(BaseModel):
    category: str
    severity: str
    detail: str
    tip: str
    matched: List[str]

class TokenContribution(BaseModel):
    term: str
    contribution: float
    weight: float

class JobPredictionResponse(BaseModel):
    is_fraud: bool
    risk_level: str
    fraud_probability: float
    legitimate_probability: float
    decision_threshold: float
    model_used: str
    heuristic_flags: List[HeuristicFlag]
    flag_count: int
    top_fraud_tokens: Optional[List[TokenContribution]] = None
    top_legit_tokens: Optional[List[TokenContribution]] = None

class BatchPredictionRequest(BaseModel):
    jobs: List[str] = Field(..., description="List of raw job postings to score", max_items=100)
    threshold: Optional[float] = Field(0.50, ge=0.10, le=0.90)

class BatchPredictionItem(BaseModel):
    index: int
    is_fraud: bool
    risk_level: str
    fraud_probability: float
    flag_count: int

class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_fraud_count: int
    results: List[BatchPredictionItem]

# Endpoints
@app.get("/health", tags=["System"])
def health_check():
    """Health and readiness probe for container orchestrators (Kubernetes/ECS/Cloud Run)."""
    is_loaded = pipeline.primary_model is not None and pipeline.tfidf is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "service": "Job Fraud Shield API",
        "version": "1.0.0"
    }

@app.get("/info", tags=["System"])
def model_info():
    """Returns architecture specifications, benchmarks, and training metadata."""
    return {
        "primary_model": "Logistic Regression (Class-Balanced)",
        "baseline_model": "Multinomial Naive Bayes (alpha=0.1)",
        "feature_extractor": "TF-IDF N-Gram Vectorizer (10,000 unigrams/bigrams)",
        "test_set_benchmarks": {
            "fraud_recall": "88.44%",
            "fraud_precision": "67.70%",
            "fraud_f1_score": "0.7669",
            "roc_auc": "0.9842"
        },
        "training_dataset": "EMScAD (17,880 real-world postings)"
    }

@app.post("/predict", response_model=JobPredictionResponse, tags=["Inference"])
def predict_job_fraud(payload: JobPredictionRequest):
    """
    Analyzes a single job posting for employment scam and phishing risk.
    Returns fraud probability, risk tier, rule-based red flags, and token attributions.
    """
    if pipeline.tfidf is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready yet.")

    # 1. ML Classifier Inference
    pred = pipeline.predict(
        raw_text=payload.text,
        threshold=payload.threshold,
        model_type=payload.model_type
    )

    if 'error' in pred:
        raise HTTPException(status_code=422, detail=pred['error'])

    # 2. Rule-Based Heuristic Scan
    raw_flags = scan_risk_indicators(payload.text)
    heuristic_flags = [
        HeuristicFlag(
            category=f['category'],
            severity=f['severity'],
            detail=f['detail'],
            tip=f['tip'],
            matched=f['matched']
        )
        for f in raw_flags
    ]

    # Combine risk level if high-severity heuristic flags exist
    has_critical_flag = any(f.severity in ["HIGH", "CRITICAL"] for f in heuristic_flags)
    final_is_fraud = pred['is_fraud'] or has_critical_flag
    
    if final_is_fraud:
        final_risk = "HIGH RISK (SUSPECTED SCAM)"
    elif pred['fraud_probability'] >= 0.25 or len(heuristic_flags) > 0:
        final_risk = "MODERATE RISK (PROCEED WITH CAUTION)"
    else:
        final_risk = "LOW RISK (LIKELY LEGITIMATE)"

    # 3. Token-Level Explainability
    fraud_tokens = []
    legit_tokens = []
    if payload.include_explanation and payload.model_type in ['primary', 'standard']:
        expl = pipeline.explain(payload.text, top_k=6)
        if expl:
            if not expl['fraud_leaning'].empty:
                for _, r in expl['fraud_leaning'].iterrows():
                    fraud_tokens.append(TokenContribution(
                        term=r['term'],
                        contribution=round(float(r['contribution']), 4),
                        weight=round(float(r['weight']), 4)
                    ))
            if not expl['legit_leaning'].empty:
                for _, r in expl['legit_leaning'].iterrows():
                    legit_tokens.append(TokenContribution(
                        term=r['term'],
                        contribution=round(float(r['contribution']), 4),
                        weight=round(float(r['weight']), 4)
                    ))

    return JobPredictionResponse(
        is_fraud=final_is_fraud,
        risk_level=final_risk,
        fraud_probability=pred['fraud_probability'],
        legitimate_probability=pred['legitimate_probability'],
        decision_threshold=payload.threshold,
        model_used=pred['model_used'],
        heuristic_flags=heuristic_flags,
        flag_count=len(heuristic_flags),
        top_fraud_tokens=fraud_tokens if payload.include_explanation else None,
        top_legit_tokens=legit_tokens if payload.include_explanation else None
    )

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(payload: BatchPredictionRequest):
    """
    Batch evaluates up to 100 job listings in a single HTTP call.
    Ideal for ATS platforms, job scrapers, and automated moderation queues.
    """
    if pipeline.tfidf is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready yet.")

    results = []
    flagged_count = 0

    for idx, text in enumerate(payload.jobs):
        pred = pipeline.predict(text, threshold=payload.threshold)
        flags = scan_risk_indicators(text)
        has_critical = any(f['severity'] in ['HIGH', 'CRITICAL'] for f in flags)
        is_fraud = pred['is_fraud'] or has_critical
        
        if is_fraud:
            risk = "HIGH RISK"
            flagged_count += 1
        elif pred['fraud_probability'] >= 0.25 or len(flags) > 0:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

        results.append(BatchPredictionItem(
            index=idx,
            is_fraud=is_fraud,
            risk_level=risk,
            fraud_probability=pred['fraud_probability'],
            flag_count=len(flags)
        ))

    return BatchPredictionResponse(
        total_processed=len(payload.jobs),
        flagged_fraud_count=flagged_count,
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Job Fraud Shield REST API on http://0.0.0.0:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
