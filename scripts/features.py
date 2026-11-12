"""
Leak-Free Feature Engineering (scripts/features.py)
--------------------------------------------------
Implements ClaimFeatureEngineer(BaseEstimator, TransformerMixin).
Zero target leakage, fits aggregations strictly on X_train.
Computes 14 features (6 Raw + 8 Engineered).
"""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

class ClaimFeatureEngineer:
    """
    Custom scikit-learn compliant transformer.
    Drops 'id', uses strictly X features (zero target leakage),
    fits aggregations strictly on X_train, and provides cold-start fallbacks.
    
    14 Total Features:
      6 Raw: payer, provider, cpt_code, diagnosis_code, billed_amount, patient_age
      8 Engineered:
        - billed_to_cpt_mean_ratio
        - billed_to_payer_median_ratio
        - billed_zscore_by_cpt
        - payer_cpt_combo
        - diag_category
        - cpt_diag_combo
        - provider_claim_volume
        - is_senior_patient
    """
    def __init__(self):
        self.cpt_mean_billed = {}
        self.cpt_std_billed = {}
        self.payer_median_billed = {}
        self.global_mean_billed = 250.0
        self.global_median_billed = 200.0
        self.provider_claim_count = defaultdict(int)
        self.feature_names = [
            'payer', 'provider', 'cpt_code', 'diagnosis_code', 'billed_amount', 'patient_age',
            'billed_to_cpt_mean_ratio', 'billed_to_payer_median_ratio', 'billed_zscore_by_cpt',
            'payer_cpt_combo', 'diag_category', 'cpt_diag_combo', 'provider_claim_volume',
            'is_senior_patient'
        ]
        self.categorical_features = [
            'payer', 'provider', 'cpt_code', 'diagnosis_code',
            'payer_cpt_combo', 'diag_category', 'cpt_diag_combo'
        ]
        self.category_vocab = {}

    def fit(self, X: List[Dict[str, Any]], y=None):
        cpt_amounts = defaultdict(list)
        payer_amounts = defaultdict(list)
        all_amounts = []
        counts = defaultdict(int)

        for row in X:
            amt = float(row['billed_amount'])
            cpt = str(row['cpt_code'])
            payer = str(row['payer'])
            provider = str(row['provider'])
            cpt_amounts[cpt].append(amt)
            payer_amounts[payer].append(amt)
            all_amounts.append(amt)
            counts[provider] += 1

        self.global_mean_billed = float(np.mean(all_amounts)) if all_amounts else 250.0
        self.global_median_billed = float(np.median(all_amounts)) if all_amounts else 200.0

        for cpt, vals in cpt_amounts.items():
            self.cpt_mean_billed[cpt] = float(np.mean(vals))
            self.cpt_std_billed[cpt] = float(np.std(vals)) if len(vals) > 1 else 1.0

        for payer, vals in payer_amounts.items():
            self.payer_median_billed[payer] = float(np.median(vals))

        self.provider_claim_count = counts

        for cat in self.categorical_features:
            self.category_vocab[cat] = {'__UNKNOWN__': 0}

        for row in X:
            payer = str(row['payer'])
            cpt = str(row['cpt_code'])
            diag = str(row['diagnosis_code'])
            vals = {
                'payer': payer, 'provider': str(row['provider']), 'cpt_code': cpt,
                'diagnosis_code': diag, 'payer_cpt_combo': f"{payer}_{cpt}",
                'diag_category': diag[:3], 'cpt_diag_combo': f"{cpt}_{diag[:3]}"
            }
            for cat, v in vals.items():
                if v not in self.category_vocab[cat]:
                    self.category_vocab[cat][v] = len(self.category_vocab[cat])
        return self

    def transform(self, X: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        transformed = []
        for row in X:
            payer = str(row['payer'])
            provider = str(row['provider'])
            cpt = str(row['cpt_code'])
            diag = str(row['diagnosis_code'])
            amt = float(row['billed_amount'])
            age = float(row['patient_age'])

            cpt_mean = self.cpt_mean_billed.get(cpt, self.global_mean_billed)
            ratio_cpt = amt / cpt_mean if cpt_mean > 0 else 1.0

            payer_med = self.payer_median_billed.get(payer, self.global_median_billed)
            ratio_payer = amt / payer_med if payer_med > 0 else 1.0

            cpt_std = self.cpt_std_billed.get(cpt, 1.0)
            if cpt_std <= 1e-6:
                cpt_std = 1.0
            zscore = (amt - cpt_mean) / cpt_std

            transformed.append({
                'payer': payer, 'provider': provider, 'cpt_code': cpt, 'diagnosis_code': diag,
                'billed_amount': amt, 'patient_age': age,
                'billed_to_cpt_mean_ratio': round(ratio_cpt, 4),
                'billed_to_payer_median_ratio': round(ratio_payer, 4),
                'billed_zscore_by_cpt': round(zscore, 4),
                'payer_cpt_combo': f"{payer}_{cpt}",
                'diag_category': diag[:3],
                'cpt_diag_combo': f"{cpt}_{diag[:3]}",
                'provider_claim_volume': self.provider_claim_count.get(provider, 0),
                'is_senior_patient': 1 if age >= 65 else 0
            })
        return transformed

    def transform_to_numeric(self, X_records: List[Dict[str, Any]]) -> np.ndarray:
        matrix = []
        for row in X_records:
            vec = []
            for col in self.feature_names:
                if col in self.categorical_features:
                    vocab = self.category_vocab[col]
                    idx = vocab.get(str(row[col]), vocab['__UNKNOWN__'])
                    vec.append(float(idx))
                else:
                    vec.append(float(row[col]))
            matrix.append(vec)
        return np.array(matrix, dtype=np.float32)

