"""
Synthetic Claims Generator (scripts/data_gen.py)
------------------------------------------------
Generates 1,500 claims matching schema (id, payer, provider, cpt_code, diagnosis_code,
billed_amount, patient_age, status) with exact 70/20/10 target imbalance and latent domain signals.
"""

import os
import csv
import random
import numpy as np
from typing import List, Dict, Any

CPT_CATALOG = {
    '99213': {'desc': 'Office Outpatient Visit 15m (Low)', 'base_fee': 110.0, 'valid_diag': ['M54.5', 'I10', 'E11.9', 'J06.9', 'K21.9', 'Z00.00'], 'complexity': 'low'},
    '99214': {'desc': 'Office Outpatient Visit 25m (Moderate)', 'base_fee': 165.0, 'valid_diag': ['M54.5', 'I10', 'E11.9', 'R07.9', 'K21.9', 'M17.11'], 'complexity': 'medium'},
    '99215': {'desc': 'Office Outpatient Visit 40m (High)', 'base_fee': 240.0, 'valid_diag': ['E11.9', 'R07.9', 'J18.9', 'I10'], 'complexity': 'high'},
    '99285': {'desc': 'Emergency Dept High Severity', 'base_fee': 380.0, 'valid_diag': ['R07.9', 'J18.9', 'S83.2'], 'complexity': 'high'},
    '99291': {'desc': 'Critical Care First 30-74m', 'base_fee': 520.0, 'valid_diag': ['R07.9', 'J18.9'], 'complexity': 'critical'},
    '71046': {'desc': 'Chest X-Ray 2 Views', 'base_fee': 95.0,  'valid_diag': ['R07.9', 'J18.9', 'J06.9'], 'complexity': 'low'},
    '74177': {'desc': 'CT Abdomen & Pelvis with Contrast', 'base_fee': 650.0, 'valid_diag': ['R07.9', 'K21.9'], 'complexity': 'high'},
    '93000': {'desc': 'Routine Electrocardiogram (ECG)', 'base_fee': 60.0,  'valid_diag': ['I10', 'R07.9'], 'complexity': 'low'},
    '29881': {'desc': 'Knee Arthroscopy Meniscectomy', 'base_fee': 1950.0, 'valid_diag': ['S83.2', 'M17.11'], 'complexity': 'surgical'},
    '43239': {'desc': 'Upper GI Endoscopy Biopsy', 'base_fee': 850.0, 'valid_diag': ['K21.9'], 'complexity': 'surgical'},
}

PAYERS = ['Medicare', 'Medicaid', 'BlueCross', 'UnitedHealth', 'Aetna', 'Cigna']
STRICT_PAYERS = {'Medicare', 'UnitedHealth', 'BlueCross'}
PROVIDERS = [f"PRV_{i:03d}" for i in range(1, 26)]
HIGH_VOLUME_PROVIDERS = {'PRV_001', 'PRV_002', 'PRV_003', 'PRV_004', 'PRV_005'}
ALL_DIAGNOSES = ['M54.5', 'I10', 'E11.9', 'J06.9', 'R07.9', 'S83.2', 'K21.9', 'Z00.00', 'M17.11', 'J18.9']

def generate_claims(num_samples: int = 1500, seed: int = 42, output_csv: str = "data/claims_synthetic.csv") -> List[Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    
    target_denied = int(round(num_samples * 0.10))
    target_partial = int(round(num_samples * 0.20))
    
    raw_claims = []
    cpt_keys = list(CPT_CATALOG.keys())
    
    for i in range(1, num_samples + 1):
        claim_id = f"CLM-{i:05d}"
        payer = random.choice(PAYERS)
        provider = random.choice(list(HIGH_VOLUME_PROVIDERS)) if random.random() < 0.65 else random.choice(PROVIDERS)
        cpt = random.choice(cpt_keys)
        meta = CPT_CATALOG[cpt]
        base_fee = meta['base_fee']
        
        # Clinical compatibility check (80% matching, 20% mismatch)
        is_mismatch = (random.random() >= 0.80)
        if is_mismatch:
            invalid_d = [d for d in ALL_DIAGNOSES if d not in meta['valid_diag']]
            diag = random.choice(invalid_d) if invalid_d else 'Z00.00'
        else:
            diag = random.choice(meta['valid_diag'])
            
        # Age distribution
        if payer == 'Medicare':
            patient_age = int(np.clip(np.random.normal(71, 7), 65, 96))
        else:
            patient_age = int(np.clip(np.random.normal(68, 6), 65, 92)) if random.random() < 0.25 else int(np.clip(np.random.normal(42, 14), 18, 64))
            
        # Billed amount & upcoding
        overbilled = (random.random() < 0.28)
        multiplier = np.random.uniform(1.35, 3.20) if overbilled else np.random.uniform(0.88, 1.25)
        billed_amount = round(max(base_fee * 0.75, base_fee * multiplier + np.random.normal(0, 5)), 2)
        
        ratio = billed_amount / base_fee
        is_senior = 1 if patient_age >= 65 else 0
        is_strict = 1 if payer in STRICT_PAYERS else 0
        is_high = 1 if meta['complexity'] in {'high', 'critical', 'surgical'} else 0
        
        # Latent risk signals
        denied_score = (
            2.5 * max(0.0, ratio - 1.30) ** 1.8 +
            3.0 * (1.0 if is_mismatch and is_high else 0.0) +
            2.0 * (1.0 if is_strict and is_high and ratio > 1.2 else 0.0) +
            1.2 * (1.0 if is_mismatch else 0.0) +
            1.0 * (1.0 if is_senior and payer == 'Medicare' and is_high else 0.0) +
            0.8 * (1.0 if provider not in HIGH_VOLUME_PROVIDERS and ratio > 1.4 else 0.0) +
            np.random.normal(0, 0.45)
        )
        partial_score = (
            2.2 * max(0.0, ratio - 1.15) +
            1.5 * (1.0 if is_strict and not is_high and ratio > 1.2 else 0.0) +
            1.2 * (1.0 if ratio > 1.28 else 0.0) +
            0.7 * (1.0 if provider in HIGH_VOLUME_PROVIDERS and ratio > 1.25 else 0.0) +
            np.random.normal(0, 0.40)
        )
        
        raw_claims.append({
            'id': claim_id, 'payer': payer, 'provider': provider, 'cpt_code': cpt,
            'diagnosis_code': diag, 'billed_amount': billed_amount, 'patient_age': patient_age,
            'denied_score': denied_score, 'partial_score': partial_score
        })
        
    # Enforce exact 70/20/10 target imbalance
    raw_claims.sort(key=lambda x: x['denied_score'], reverse=True)
    for c in raw_claims[:target_denied]:
        c['status'] = 'Denied'
    rest = raw_claims[target_denied:]
    rest.sort(key=lambda x: x['partial_score'], reverse=True)
    for c in rest[:target_partial]:
        c['status'] = 'Partially_Paid'
    for c in rest[target_partial:]:
        c['status'] = 'Paid'
        
    cleaned = []
    for c in raw_claims:
        cleaned.append({
            'id': c['id'], 'payer': c['payer'], 'provider': c['provider'],
            'cpt_code': c['cpt_code'], 'diagnosis_code': c['diagnosis_code'],
            'billed_amount': c['billed_amount'], 'patient_age': c['patient_age'],
            'status': c['status']
        })
    random.shuffle(cleaned)
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'payer', 'provider', 'cpt_code', 'diagnosis_code', 'billed_amount', 'patient_age', 'status'])
            writer.writeheader()
            writer.writerows(cleaned)
    return cleaned

if __name__ == '__main__':
    data = generate_claims()
    print(f"Generated {len(data)} claims into data/claims_synthetic.csv")

