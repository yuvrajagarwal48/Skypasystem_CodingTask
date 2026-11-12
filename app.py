"""
Claim Outcome Prediction Serving Service (app.py)
-------------------------------------------------
Production FastAPI service with strict Pydantic v2 400 validation,
real-time imbalance-aware inference, local SHAP explainability,
and automated appeal brief generation.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.appeal_helper import generate_appeal_brief

app = FastAPI(
    title="Claim Outcome Prediction API",
    description="Imbalance-Aware Insurance Claim Adjudication & Explainable AI Serving Service",
    version="1.0.0"
)

# Strict 400 Validation Error exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field_loc,
            "message": err.get("msg"),
            "type": err.get("type")
        })
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid Claim Payload - Validation Failed", "errors": errors}
    )

# Pydantic Schemas with strict field validation
class ClaimInput(BaseModel):
    id: Optional[str] = Field("CLM-DEMO", description="Unique claim identifier")
    payer: str = Field(..., min_length=2, max_length=50, description="Health insurer name")
    provider: str = Field(..., min_length=2, max_length=50, description="Provider NPI or ID")
    cpt_code: str = Field(..., min_length=4, max_length=10, description="Procedure CPT code")
    diagnosis_code: str = Field(..., min_length=2, max_length=10, description="ICD-10 Diagnosis code")
    billed_amount: float = Field(..., gt=0.0, le=100000.0, description="Dollar amount billed (must be > 0)")
    patient_age: int = Field(..., ge=0, le=125, description="Patient age between 0 and 125")

    @field_validator("billed_amount")
    def validate_positive_billing(cls, v):
        if v <= 0.0:
            raise ValueError("Billed amount must be strictly greater than $0.00")
        return round(v, 2)

    @field_validator("diagnosis_code")
    def validate_icd10_format(cls, v):
        v = v.strip().upper()
        if not re.match(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$", v):
            raise ValueError(f"Invalid ICD-10 format: '{v}'. Expected format like 'M54.5', 'I10', etc.")
        return v

class RiskFactor(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    directional_impact: str
    clinical_rationale: str

class PredictionResponse(BaseModel):
    claim_id: str
    predicted_status: str
    probabilities: Dict[str, float]
    confidence: float
    risk_tier: str
    top_risk_factors: List[RiskFactor]
    suggested_action: str

# Lazy loader for pipeline artifacts
_pipeline = None
_explainer = None

def get_artifacts():
    global _pipeline, _explainer
    if _pipeline is None:
        p_path = os.path.join(ROOT_DIR, "model", "model_pipeline.joblib")
        e_path = os.path.join(ROOT_DIR, "model", "shap_explainer.joblib")
        if os.path.exists(p_path) and os.path.exists(e_path):
            _pipeline = joblib.load(p_path)
            _explainer = joblib.load(e_path)
    return _pipeline, _explainer

@app.get("/", tags=["System"])
def root(request: Request):
    """
    Root endpoint: serves an interactive browser UI dashboard,
    or returns API metadata JSON when requested programmatically.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claim Outcome Predictor API</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: #f8fafc; color: #1e293b; padding: 2rem 1rem; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
        .header h1 { font-size: 1.75rem; color: #0f172a; margin-bottom: 0.5rem; }
        .header p { color: #64748b; font-size: 0.95rem; }
        .badge-row { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .badge { background: #e2e8f0; color: #334155; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
        .badge.online { background: #dcfce7; color: #166534; }
        .links-bar { display: flex; gap: 1rem; margin-top: 1.25rem; }
        .btn-link { display: inline-flex; align-items: center; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.9rem; font-weight: 500; text-decoration: none; transition: 0.2s; }
        .btn-primary { background: #2563eb; color: white; }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-secondary { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
        .btn-secondary:hover { background: #e2e8f0; }
        .card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
        .card h2 { font-size: 1.25rem; margin-bottom: 1rem; color: #0f172a; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
        .form-group { display: flex; flex-direction: column; gap: 0.35rem; }
        label { font-size: 0.85rem; font-weight: 600; color: #475569; }
        input, select { padding: 0.6rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.9rem; }
        input:focus, select:focus { outline: none; border-color: #2563eb; ring: 2px #bfdbfe; }
        .submit-btn { margin-top: 1.25rem; padding: 0.75rem 1.5rem; background: #0f172a; color: white; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .submit-btn:hover { background: #1e293b; }
        .result-box { margin-top: 1.5rem; padding: 1.25rem; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; display: none; }
        .status-badge { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 6px; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; margin-bottom: 0.75rem; }
        .status-Paid { background: #dcfce7; color: #166534; }
        .status-Partially_Paid { background: #fef3c7; color: #92400e; }
        .status-Denied { background: #fee2e2; color: #991b1b; }
        .factors-list { margin-top: 0.75rem; list-style: none; display: flex; flex-direction: column; gap: 0.5rem; }
        .factor-item { background: white; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #cbd5e1; font-size: 0.85rem; }
        .factor-item.pos { border-left-color: #ef4444; }
        .factor-item.neg { border-left-color: #22c55e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Claim Outcome Prediction Service</h1>
            <p>Leak-Free Feature Engineering &bull; Imbalance-Aware Boosting &bull; Tree SHAP Local Explainability</p>
            <div class="badge-row">
                <span class="badge online">&bull; Service Online</span>
                <span class="badge">14 Features</span>
                <span class="badge">70/20/10 Imbalance Calibrated</span>
            </div>
            <div class="links-bar">
                <a href="/docs" class="btn-link btn-primary">Open Interactive Swagger Docs (/docs)</a>
                <a href="/redoc" class="btn-link btn-secondary">ReDoc API Spec (/redoc)</a>
                <a href="/health" class="btn-link btn-secondary">Health Status (/health)</a>
            </div>
        </div>

        <div class="card">
            <h2>Interactive Claim Outcome Tester</h2>
            <form id="claimForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="payer">Health Insurer (Payer)</label>
                        <select id="payer" required>
                            <option value="Medicare">Medicare</option>
                            <option value="UnitedHealth">UnitedHealth</option>
                            <option value="BlueCross" selected>BlueCross</option>
                            <option value="Medicaid">Medicaid</option>
                            <option value="Aetna">Aetna</option>
                            <option value="Cigna">Cigna</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="provider">Provider ID</label>
                        <input type="text" id="provider" value="PRV_002" required>
                    </div>
                    <div class="form-group">
                        <label for="cpt_code">CPT Procedure Code</label>
                        <input type="text" id="cpt_code" value="99213" required>
                    </div>
                    <div class="form-group">
                        <label for="diagnosis_code">ICD-10 Diagnosis Code</label>
                        <input type="text" id="diagnosis_code" value="M54.5" required>
                    </div>
                    <div class="form-group">
                        <label for="billed_amount">Billed Dollar Amount ($)</label>
                        <input type="number" id="billed_amount" step="0.01" value="115.00" required>
                    </div>
                    <div class="form-group">
                        <label for="patient_age">Patient Age</label>
                        <input type="number" id="patient_age" value="45" min="0" max="125" required>
                    </div>
                </div>
                <button type="submit" class="submit-btn">Predict Claim Outcome &amp; Run SHAP</button>
            </form>

            <div id="resultBox" class="result-box">
                <span id="statusBadge" class="status-badge"></span>
                <p><strong>Risk Tier:</strong> <span id="riskTier"></span> | <strong>Confidence:</strong> <span id="confidence"></span></p>
                <p style="margin-top: 0.35rem;"><strong>Probabilities:</strong> <span id="probs"></span></p>
                <p style="margin-top: 0.35rem; color: #475569;"><strong>Recommendation:</strong> <span id="action"></span></p>
                <div style="margin-top: 1rem;">
                    <strong>Top SHAP Risk Factors:</strong>
                    <ul id="factorsList" class="factors-list"></ul>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('claimForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const payload = {
                id: 'CLM-WEB-TEST',
                payer: document.getElementById('payer').value,
                provider: document.getElementById('provider').value,
                cpt_code: document.getElementById('cpt_code').value,
                diagnosis_code: document.getElementById('diagnosis_code').value,
                billed_amount: parseFloat(document.getElementById('billed_amount').value),
                patient_age: parseInt(document.getElementById('patient_age').value)
            };

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            const box = document.getElementById('resultBox');
            box.style.display = 'block';
            
            const badge = document.getElementById('statusBadge');
            badge.className = 'status-badge status-' + data.predicted_status;
            badge.innerText = data.predicted_status.replace('_', ' ');

            document.getElementById('riskTier').innerText = data.risk_tier;
            document.getElementById('confidence').innerText = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('probs').innerText = JSON.stringify(data.probabilities);
            document.getElementById('action').innerText = data.suggested_action;

            const list = document.getElementById('factorsList');
            list.innerHTML = '';
            (data.top_risk_factors || []).forEach(f => {
                const li = document.createElement('li');
                const isPos = f.directional_impact.includes('Increases');
                li.className = 'factor-item ' + (isPos ? 'pos' : 'neg');
                li.innerHTML = `<strong>${f.feature} (${f.feature_value})</strong>: ${f.directional_impact}<br><span style="color:#64748b;">${f.clinical_rationale}</span>`;
                list.appendChild(li);
            });
        });
    </script>
</body>
</html>
"""
        return HTMLResponse(content=html_content)

    return {
        "message": "Claim Outcome Prediction API",
        "status": "online",
        "documentation_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "appeal_helper": "POST /appeal-helper",
            "docs": "GET /docs"
        }
    }

@app.get("/health", tags=["System"])
def health_check():
    pipeline, explainer = get_artifacts()
    return {
        "status": "healthy",
        "service": "Claim Outcome Predictor",
        "model_loaded": pipeline is not None,
        "features_count": 14,
        "class_labels": ["Paid", "Partially_Paid", "Denied"]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_claim(claim: ClaimInput):
    pipeline, explainer = get_artifacts()
    if pipeline is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded")

    fe = pipeline["feature_engineer"]
    model = pipeline["model"]
    classes = pipeline["class_labels"]
    threshold_multipliers = np.array(pipeline["threshold_multipliers"], dtype=np.float32)

    # 1. Leak-free feature transform
    raw_dict = claim.model_dump()
    engineered = fe.transform([raw_dict])
    x_vec = fe.transform_to_numeric(engineered)[0]

    # 2. Probability inference & calibrated thresholding
    probs = model.predict_proba(x_vec.reshape(1, -1))[0]
    scaled_probs = probs * threshold_multipliers
    pred_idx = int(np.argmax(scaled_probs))
    pred_status = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # 3. Fast local SHAP attribution
    shapper = explainer["shap_explainer"]
    explanation = shapper.explain_instance(x_vec, top_k=3)

    # 4. Action recommendation & Risk tiering
    denied_prob = float(probs[2])
    if denied_prob >= 0.40 or pred_status == "Denied":
        risk_tier = "HIGH"
        action = "High denial risk detected. Review fee schedule and attach prior-authorization operative notes before filing."
    elif float(probs[1]) >= 0.30 or pred_status == "Partially_Paid":
        risk_tier = "MEDIUM"
        action = "Moderate partial reduction risk. Verify payer allowable fee schedule cap."
    else:
        risk_tier = "LOW"
        action = "Clean claim profile. Ready for automated primary EDI 837 submission."

    prob_dict = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}

    return PredictionResponse(
        claim_id=claim.id or "CLM-DEMO",
        predicted_status=pred_status,
        probabilities=prob_dict,
        confidence=round(confidence, 4),
        risk_tier=risk_tier,
        top_risk_factors=explanation["top_risk_factors"],
        suggested_action=action
    )

class AppealRequest(BaseModel):
    claim_id: str
    payer: str
    cpt_code: str
    diagnosis_code: str
    billed_amount: float
    patient_age: Optional[int] = 52
    notes: Optional[str] = ""

@app.post("/appeal-helper", tags=["Revenue Cycle Bonus"])
def generate_appeal(req: AppealRequest):
    brief = generate_appeal_brief(
        claim_id=req.claim_id,
        payer=req.payer,
        cpt_code=req.cpt_code,
        diagnosis_code=req.diagnosis_code,
        billed_amount=req.billed_amount,
        top_risk_factors=[{'feature': 'billed_to_cpt_mean_ratio'}],
        patient_age=req.patient_age or 50,
        additional_notes=req.notes or ""
    )
    return brief

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

