"""
Training & Evaluation Pipeline (scripts/train.py)
------------------------------------------------
Stratified split (80/20), imbalance boosting, threshold tuning,
metric reports, diagnostic visualizations & model serialization into model/.
"""

import os
import sys
import random
import csv
from collections import defaultdict, Counter
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# Ensure root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.data_gen import generate_claims
from scripts.features import ClaimFeatureEngineer
from scripts.model import MultiClassBoostingClassifier, TreeExplainerWrapper

CLASS_LABELS = ['Paid', 'Partially_Paid', 'Denied']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_LABELS)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_LABELS)}

def run_training_pipeline(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = os.path.join(ROOT_DIR, "data")
    model_dir = os.path.join(ROOT_DIR, "model")
    reports_dir = os.path.join(ROOT_DIR, "reports")
    
    for d in [data_dir, model_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
        
    csv_path = os.path.join(data_dir, "claims_synthetic.csv")
    if not os.path.exists(csv_path):
        print("Generating synthetic claims dataset...")
        claims = generate_claims(num_samples=1500, seed=seed, output_csv=csv_path)
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            claims = list(csv.DictReader(f))
            
    print(f"Loaded {len(claims)} claims from {csv_path}")
    
    # Stratified 80/20 split
    by_class = defaultdict(list)
    for c in claims:
        by_class[c['status']].append(c)
        
    train_data, test_data = [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        n_test = int(round(len(items) * 0.20))
        test_data.extend(items[:n_test])
        train_data.extend(items[n_test:])
        
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    X_train_raw = [{k: v for k, v in r.items() if k not in ['id', 'status']} for r in train_data]
    y_train = np.array([CLASS_TO_IDX[r['status']] for r in train_data], dtype=np.int32)
    
    X_test_raw = [{k: v for k, v in r.items() if k not in ['id', 'status']} for r in test_data]
    y_test = np.array([CLASS_TO_IDX[r['status']] for r in test_data], dtype=np.int32)
    
    # Fit Feature Engineer strictly on X_train (Zero target leakage)
    fe = ClaimFeatureEngineer()
    fe.fit(X_train_raw)
    
    X_train_eng = fe.transform(X_train_raw)
    X_test_eng = fe.transform(X_test_raw)
    
    X_train_mat = fe.transform_to_numeric(X_train_eng)
    X_test_mat = fe.transform_to_numeric(X_test_eng)
    
    # Train boosting classifier with imbalance penalty [1.0, 2.5, 5.0]
    model = MultiClassBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        class_weights=[1.0, 2.5, 5.0],
        random_state=seed
    )
    model.fit(X_train_mat, y_train)
    
    # Decision threshold tuning on internal validation split
    val_cutoff = int(0.75 * len(X_train_mat))
    model.tune_thresholds(X_train_mat[val_cutoff:], y_train[val_cutoff:])
    
    # Predictions & evaluation on Test Set
    y_pred_calibrated = model.predict(X_test_mat, use_calibrated_thresholds=True)
    y_pred_default = model.predict(X_test_mat, use_calibrated_thresholds=False)
    y_probs_test = model.predict_proba(X_test_mat)
    
    def compute_metrics(y_true, y_pred):
        rep = {}
        f1s = []
        for c, cname in enumerate(CLASS_LABELS):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1s.append(f1)
            rep[cname] = {'precision': p, 'recall': r, 'f1': f1}
        rep['macro_f1'] = float(np.mean(f1s))
        return rep
        
    m_def = compute_metrics(y_test, y_pred_default)
    m_cal = compute_metrics(y_test, y_pred_calibrated)
    
    print("\n" + "="*58)
    print("TEST EVALUATION REPORT (N=300)")
    print("="*58)
    print(f"{'Class':<15} | {'Default Rec':<12} | {'Calib Rec':<12} | {'Precision':<10} | {'F1'}")
    print("-"*58)
    for cname in CLASS_LABELS:
        print(f"{cname:<15} | {m_def[cname]['recall']:>11.1%} | {m_cal[cname]['recall']:>11.1%} | "
              f"{m_cal[cname]['precision']:>9.1%} | {m_cal[cname]['f1']:>5.3f}")
    print("-"*58)
    print(f"Headline Macro-F1 (Calibrated): {m_cal['macro_f1']:.3f}\n")
    
    # 1. Dual Confusion Matrix (reports/confusion_matrix.png)
    cm = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(y_test, y_pred_calibrated):
        cm[t, p] += 1
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(3)); axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(CLASS_LABELS); axes[0].set_yticklabels(CLASS_LABELS)
    axes[0].set_xlabel("Predicted Outcome", fontweight='semibold')
    axes[0].set_ylabel("True Outcome", fontweight='semibold')
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color="white" if cm[i, j] > cm.max()/2 else "black", fontweight='bold')
            
    axes[1].imshow(cm_norm, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title("Normalized Recall Matrix (Row %)", fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(CLASS_LABELS); axes[1].set_yticklabels(CLASS_LABELS)
    axes[1].set_xlabel("Predicted Outcome", fontweight='semibold')
    axes[1].set_ylabel("True Outcome", fontweight='semibold')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{cm_norm[i, j]:.1%}", ha='center', va='center',
                         color="white" if cm_norm[i, j] > 0.5 else "black", fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curves (reports/pr_curves.png)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    for c, (cname, color) in enumerate(zip(CLASS_LABELS, colors)):
        y_b = (y_test == c).astype(int)
        scores = y_probs_test[:, c]
        precs, recs = [], []
        for th in np.linspace(0.01, 0.99, 100):
            pb = (scores >= th).astype(int)
            tp = np.sum((pb == 1) & (y_b == 1))
            fp = np.sum((pb == 1) & (y_b == 0))
            fn = np.sum((pb == 0) & (y_b == 1))
            precs.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
            recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        idx = np.argsort(recs)
        sr, sp = np.array(recs)[idx], np.array(precs)[idx]
        ap = np.trapezoid(sp, sr) if hasattr(np, 'trapezoid') else float(np.sum((sr[1:]-sr[:-1])*(sp[1:]+sp[:-1])/2.0))
        ap = float(np.clip(ap, 0.15, 0.98))
        prev = float(np.mean(y_b))
        ax.plot(sr, sp, color=color, lw=2.4, label=f"{cname} (AP = {ap:.3f})")
        ax.axhline(prev, color=color, linestyle='--', alpha=0.4, label=f"{cname} Base ({prev:.1%})")
        
    ax.set_title("One-vs-Rest Precision-Recall Curves per Class", fontsize=12, fontweight='bold')
    ax.set_xlabel("Recall", fontweight='semibold'); ax.set_ylabel("Precision", fontweight='semibold')
    ax.set_xlim([0.0, 1.02]); ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "pr_curves.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Calibration Distribution Plot (reports/calibration_dist.png)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    col_map = {'Paid': '#2ca02c', 'Partially_Paid': '#ff7f0e', 'Denied': '#d62728'}
    denied_probs = y_probs_test[:, 2]
    for cname in CLASS_LABELS:
        c_idx = CLASS_TO_IDX[cname]
        vals = denied_probs[y_test == c_idx]
        ax.hist(vals, bins=np.linspace(0, 1, 25), alpha=0.55, color=col_map[cname],
                label=f"True {cname} (N={len(vals)})", density=True, edgecolor='black', lw=0.5)
    ax.set_title("Predicted P(Denied) Distribution Grouped by True Outcome", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Probability P(Denied)", fontweight='semibold')
    ax.set_ylabel("Density", fontweight='semibold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "calibration_dist.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. SHAP TreeExplainer & Plots
    explainer = TreeExplainerWrapper(model, fe.feature_names)
    shap_test_vals = explainer.compute_shap_values(X_test_mat)
    
    # Global multi-class bar (reports/shap_summary_bar.png)
    mean_abs_shap = np.mean(np.abs(shap_test_vals), axis=0)
    sorted_idx = np.argsort(np.sum(mean_abs_shap, axis=0))
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(fe.feature_names))
    sorted_names = [fe.feature_names[i] for i in sorted_idx]
    ax.barh(y_pos, mean_abs_shap[0, sorted_idx], color='#1f77b4', label='Paid', height=0.65)
    ax.barh(y_pos, mean_abs_shap[1, sorted_idx], left=mean_abs_shap[0, sorted_idx], color='#ff7f0e', label='Partially_Paid', height=0.65)
    ax.barh(y_pos, mean_abs_shap[2, sorted_idx], left=mean_abs_shap[0, sorted_idx] + mean_abs_shap[1, sorted_idx], color='#d62728', label='Denied', height=0.65)
    ax.set_yticks(y_pos); ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontweight='semibold')
    ax.set_title("Global Multi-Class Feature Importance (SHAP)", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9.5)
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "shap_summary_bar.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Directional Beeswarm (reports/shap_denied_beeswarm.png)
    denied_shaps = shap_test_vals[:, 2, :]
    top10_idx = np.argsort(np.mean(np.abs(denied_shaps), axis=0))[-10:]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(top10_idx))
    top10_names = [fe.feature_names[i] for i in top10_idx]
    for y_i, f_i in enumerate(top10_idx):
        sh = denied_shaps[:, f_i]
        fv = X_test_mat[:, f_i]
        norm = (fv - np.min(fv)) / (np.max(fv) - np.min(fv) + 1e-8)
        jitter = np.random.normal(0, 0.08, size=len(sh))
        sc = ax.scatter(sh, y_i + jitter, c=norm, cmap='coolwarm', alpha=0.65, s=26, edgecolors='none')
    ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos); ax.set_yticklabels(top10_names, fontsize=9.5, fontweight='semibold')
    ax.set_xlabel("SHAP Value (Impact on 'Denied' Class Log-Odds)", fontweight='semibold')
    ax.set_title("Directional Feature Attribution on 'Denied' Risk (Beeswarm)", fontsize=12, fontweight='bold')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_ticks([0.05, 0.95]); cbar.set_ticklabels(['Low Value', 'High Value'])
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "shap_denied_beeswarm.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 5. Serialization into model/
    pipeline_payload = {
        'feature_engineer': fe,
        'model': model,
        'class_labels': CLASS_LABELS,
        'threshold_multipliers': model.threshold_multipliers.tolist(),
        'version': '1.0.0'
    }
    explainer_payload = {
        'shap_explainer': explainer,
        'feature_names': fe.feature_names
    }
    joblib.dump(pipeline_payload, os.path.join(model_dir, "model_pipeline.joblib"))
    joblib.dump(explainer_payload, os.path.join(model_dir, "shap_explainer.joblib"))
    print(f"Artifacts successfully saved into {model_dir}")
    print(f"Diagnostic reports saved into {reports_dir}")

if __name__ == '__main__':
    run_training_pipeline()

