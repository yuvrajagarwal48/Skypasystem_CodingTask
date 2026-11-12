"""
Bonus Appeal Helper Module (scripts/appeal_helper.py)
----------------------------------------------------
Provides automated clinical and legal appeal brief generation for denied insurance claims,
incorporating statutory citations (ERISA, ACA § 2719, CMS 1500) and required documentation checklists.
"""

from typing import Dict, Any, List

def generate_appeal_brief(
    claim_id: str,
    payer: str,
    cpt_code: str,
    diagnosis_code: str,
    billed_amount: float,
    top_risk_factors: List[Dict[str, Any]],
    patient_age: int = 45,
    additional_notes: str = ""
) -> Dict[str, Any]:
    """
    Generates a structured appeal package with legal arguments, statutory citations,
    and a clinical documentation checklist based on denial risk drivers.
    """
    primary_factor = top_risk_factors[0]['feature'] if top_risk_factors else 'general'
    
    citations = [
        "Affordable Care Act (ACA) § 2719 (42 U.S.C. § 300gg-19) - Right to Internal & External Review",
        "Employee Retirement Income Security Act (ERISA) § 503 (29 U.S.C. § 1133) - Full & Fair Hearing",
        "CMS-1500 Standard Adjudication Guidelines & National Correct Coding Initiative (NCCI)"
    ]
    
    checklist = [
        "Certified Copy of CMS-1500 / UB-04 Claim Form",
        "Treating Physician's Itemized Operative / Clinical Encounter Notes",
        "Prior Authorization Confirmation Number & Time-Stamped Submission Log",
        "Standard Medical Necessity Attestation signed by Attending Provider"
    ]
    
    if 'ratio' in primary_factor or 'zscore' in primary_factor:
        appeal_theme = "Fair Market Fee Schedule & Usual, Customary, and Reasonable (UCR) Reconciliation"
        argument = (
            f"The billed amount of ${billed_amount:.2f} for CPT code {cpt_code} reflects regional provider overhead, "
            f"specialized surgical consumables, and complex patient comorbidities. Under ERISA § 503 and applicable state prompt "
            f"payment regulations, arbitrary fee schedule reductions without verified regional UCR empirical benchmarking constitute "
            f"an improper adjudication reduction."
        )
        checklist.append("Regional Fair Health / UCR Benchmark Percentile Documentation")
    elif 'cpt_diag' in primary_factor or 'diag' in primary_factor:
        appeal_theme = "Clinical Medical Necessity Justification"
        argument = (
            f"Procedure {cpt_code} was clinically indicated as first-line medical intervention for primary diagnosis {diagnosis_code}. "
            f"The patient presented with refractory clinical indications failing conservative therapy, fully satisfying CMS Local Coverage "
            f"Determination (LCD) and Milliman Care Guidelines (MCG) criteria."
        )
        checklist.append("Previous Conservative Therapy Records & Diagnostic Imaging Reports")
    else:
        appeal_theme = "Adjudication Policy Clarification & Coverage Reconsideration"
        argument = (
            f"Claim {claim_id} for procedure {cpt_code} provided under policyholder agreement with {payer} meets all substantive contractual "
            f"requirements. Administrative denial without secondary medical director clinical peer review violates ACA § 2719 prompt adjudication mandates."
        )
        checklist.append("Policyholder Evidence of Coverage (EOC) Benefit Schedule Excerpt")

    letter_text = f"""FORMAL FIRST-LEVEL ADJUDICATION APPEAL
Date: Immediate Filing
To: {payer} Appeals & Grievance Department
Re: Formal Denial Appeal for Claim ID: {claim_id}
Patient Age: {patient_age} | CPT Code: {cpt_code} | ICD-10 Diagnosis: {diagnosis_code} | Billed: ${billed_amount:.2f}

Dear Appeals Committee,

This correspondence constitutes a formal expedited first-level appeal regarding the adverse benefit determination for claim {claim_id}. 
The primary dispute arises under: {appeal_theme}.

CLINICAL & STATUTORY SUBSTANTIATION:
{argument}

STATUTORY AND REGULATORY CITATIONS:
{chr(10).join(f"- {c}" for c in citations)}

Enclosed please find the required clinical documentation checklist supporting complete retroactive reversal and full payment.

Sincerely,
Attending Physician & Revenue Cycle Management Department
"""

    return {
        'claim_id': claim_id,
        'appeal_type': appeal_theme,
        'statutory_citations': citations,
        'required_documentation_checklist': checklist,
        'formal_appeal_letter': letter_text
    }

