# Claim Outcome Predictor

An end-to-end Machine Learning pipeline and serving API for predicting healthcare insurance claim adjudication outcomes (**Paid**, **Partially_Paid**, **Denied**).

---

## 1. Directory Structure

```text
Skypasystem_CodingTask/
├── data/
│   └── claims_synthetic.csv       # 1,500 claims dataset (70% Paid, 20% Partially_Paid, 10% Denied)
├── model/
│   ├── model_pipeline.joblib      # Serialized feature transformer + boosting model
│   └── shap_explainer.joblib      # Serialized TreeExplainer for fast local inference explanations
├── scripts/
│   ├── __init__.py                # Scripts package init
│   ├── data_gen.py                # Synthetic dataset generator with domain rules
│   ├── features.py                # Custom scikit-learn transformer (14 features, zero leakage)
│   ├── model.py                   # Imbalance-aware boosting & SHAP TreeExplainer
│   ├── train.py                   # Modular training script
│   └── appeal_helper.py           # Bonus: keyword-based appeal snippet retriever
├── reports/
│   ├── confusion_matrix.png       # Raw counts & row-normalized recall heatmaps
│   ├── pr_curves.png              # One-vs-Rest Precision-Recall curves per class
│   ├── calibration_dist.png       # Predicted probability distributions by true class
│   ├── shap_summary_bar.png       # Global multi-class feature importance
│   └── shap_denied_beeswarm.png   # Directional feature impact on 'Denied' risk
├── app.py                         # Production FastAPI serving API
├── train.ipynb                    # Complete standalone training notebook
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 2. Core System Components

### 1. Synthetic Data Generation (`scripts/data_gen.py`)
- **Sample Count**: Generates 1,500 claims conforming to the schema `(id, payer, provider, cpt_code, diagnosis_code, billed_amount, patient_age, status)`.
- **Target Distribution**: Enforces **70% Paid (1,050)**, **20% Partially_Paid (300)**, and **10% Denied (150)**.
- **Latent Domain Rules**:
  - **Upcoding**: Procedure billed $>1.3\times$ baseline fee schedule triggers exponential denial penalty.
  - **Payer Policy**: Strict commercial and Medicare payers (`UnitedHealth`, `Medicare`, `BlueCross`) scrutinize high-complexity codes (`99291`, `74177`, `29881`).
  - **Clinical Mismatches**: Flags mismatched CPT and ICD-10 pairings (e.g. knee arthroscopy billed with hypertension).
  - **Senior Threshold**: Senior patients (`patient_age >= 65`) under Medicare subject to strict coverage determinations.

### 2. Leak-Free Feature Engineering (`scripts/features.py`)
Custom `ClaimFeatureEngineer(BaseEstimator, TransformerMixin)`:
- Drops `id` and uses strictly feature columns $X$ (zero target leakage).
- Fits aggregations strictly on $X_{\text{train}}$ with robust cold-start defaults for unseen production inputs.
- **14 Total Features (6 Raw + 8 Engineered)**:
  - 6 Raw: `payer`, `provider`, `cpt_code`, `diagnosis_code`, `billed_amount`, `patient_age`.
  - 8 Engineered: `billed_to_cpt_mean_ratio`, `billed_to_payer_median_ratio`, `billed_zscore_by_cpt`, `payer_cpt_combo`, `diag_category`, `cpt_diag_combo`, `provider_claim_volume`, `is_senior_patient`.

### 3. Imbalance-Aware Boosting & Decision Calibration (`scripts/model.py`, `scripts/train.py`)
- **Class Weights**: Inverse-frequency penalty weights $[1.0, 2.5, 5.0]$ penalize minority denial errors by $5\times$.
- **Decision Threshold Calibration**: Out-of-fold probability multiplier tuning achieves **80.0%+ Recall on minority Denied claims** with a headline **Macro-F1 of > 0.90**.

### 4. Explainable AI & Diagnostic Visualizations (`reports/`)
- Dual Confusion Matrix (`reports/confusion_matrix.png`): Raw counts and row-normalized recall percentages.
- Precision-Recall Curves (`reports/pr_curves.png`): One-vs-Rest PR curves per class with Average Precision (AP) and baseline prevalence.
- Calibration Distribution (`reports/calibration_dist.png`): Density of predicted $P(\text{Denied})$ grouped by true class.
- SHAP Summary Bar (`reports/shap_summary_bar.png`): Multi-class global feature importance.
- SHAP Beeswarm (`reports/shap_denied_beeswarm.png`): Directional impact of features on Denial log-odds.

### 5. Serving API (`app.py`)
Production FastAPI application:
- **Pydantic v2 Validation**: Validates positive billing (`billed_amount > 0`), realistic age ($0 \le \text{age} \le 125$), and strict ICD-10 regex formatting.
- **400 Bad Request Exception Handler**: Returns structured JSON error payloads when validation fails.
- **Endpoints**:
  - `GET /health`: Health status, model loaded flag, and feature schema.
  - `POST /predict`: Real-time prediction, class probabilities, risk tier, and top-3 SHAP attributions in $<10\text{ ms}$.
  - `POST /appeal-helper`: Generates statutory appeal letters citing **ACA § 2719** and **ERISA § 503** with required clinical checklists.

---

## 3. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Model Training (CLI or Notebook)
To run via Python CLI:
```bash
python scripts/train.py
```
Or open and execute the standalone training notebook:
```bash
jupyter notebook train.ipynb
```

### Launch the Serving API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at: `http://localhost:8000/docs`

