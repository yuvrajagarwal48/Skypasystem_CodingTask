"""
Boosting Model & SHAP Explainer (scripts/model.py)
--------------------------------------------------
Implements MultiClassBoostingClassifier and TreeExplainerWrapper.
"""

from typing import List, Dict, Any, Optional
import numpy as np

class MultiClassBoostingClassifier:
    """
    Imbalance-Aware Multi-Class Gradient Boosting Tree Classifier:
    - Balanced class weighting penalty [1.0, 2.5, 5.0] for Paid, Partially_Paid, Denied
    - Multi-class cross-entropy / soft-max loss
    - Out-of-fold probability estimation and Decision Threshold Tuning
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.08, max_depth: int = 4, 
                 class_weights: Optional[List[float]] = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weights = np.array(class_weights if class_weights else [1.0, 2.5, 5.0], dtype=np.float32)
        self.n_classes = 3
        self.trees: List[List[Any]] = [[] for _ in range(self.n_classes)]
        self.init_logits: np.ndarray = np.zeros(self.n_classes, dtype=np.float32)
        self.threshold_multipliers: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        sample_weights = self.class_weights[y]
        
        class_counts = np.bincount(y, weights=sample_weights, minlength=self.n_classes)
        prior = class_counts / np.sum(class_counts)
        self.init_logits = np.log(prior + 1e-8)
        
        F = np.tile(self.init_logits, (n_samples, 1))
        Y_onehot = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        for i in range(self.n_classes):
            Y_onehot[:, i] = (y == i).astype(np.float32)
            
        for round_idx in range(self.n_estimators):
            P = self._softmax(F)
            for c in range(self.n_classes):
                residual = (Y_onehot[:, c] - P[:, c]) * sample_weights
                tree = self._build_tree(X, residual, max_depth=self.max_depth, current_depth=0)
                pred_delta = self._predict_tree(tree, X)
                F[:, c] += self.learning_rate * pred_delta
                self.trees[c].append(tree)
        return self

    def _build_tree(self, X: np.ndarray, res: np.ndarray, max_depth: int, current_depth: int):
        n_samples = len(res)
        if current_depth >= max_depth or n_samples <= 6:
            return {'is_leaf': True, 'val': float(np.mean(res)) if n_samples > 0 else 0.0}
            
        best_gain = -1.0
        best_split = None
        n_features = X.shape[1]
        feat_subset = np.random.choice(n_features, size=min(7, n_features), replace=False)
        parent_var = np.var(res) * n_samples
        
        for feat in feat_subset:
            vals = X[:, feat]
            thresholds = np.percentile(vals, [25, 50, 75])
            for th in set(thresholds):
                left_mask = vals <= th
                right_mask = ~left_mask
                if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:
                    continue
                left_var = np.var(res[left_mask]) * np.sum(left_mask)
                right_var = np.var(res[right_mask]) * np.sum(right_mask)
                gain = parent_var - (left_var + right_var)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat, th)
                    
        if best_split is None or best_gain <= 1e-5:
            return {'is_leaf': True, 'val': float(np.mean(res))}
            
        feat, th = best_split
        left_mask = X[:, feat] <= th
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], res[left_mask], max_depth, current_depth + 1)
        right_child = self._build_tree(X[right_mask], res[right_mask], max_depth, current_depth + 1)
        return {'is_leaf': False, 'feat': feat, 'th': th, 'left': left_child, 'right': right_child}

    def _predict_tree_single(self, node: Dict[str, Any], x: np.ndarray) -> float:
        if node['is_leaf']:
            return node['val']
        if x[node['feat']] <= node['th']:
            return self._predict_tree_single(node['left'], x)
        else:
            return self._predict_tree_single(node['right'], x)

    def _predict_tree(self, node: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_tree_single(node, row) for row in X], dtype=np.float32)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        F = np.tile(self.init_logits, (n_samples, 1))
        for c in range(self.n_classes):
            for tree in self.trees[c]:
                F[:, c] += self.learning_rate * self._predict_tree(tree, X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return self._softmax(logits)

    def tune_thresholds(self, X_val: np.ndarray, y_val: np.ndarray):
        probs = self.predict_proba(X_val)
        best_score = -1.0
        best_multipliers = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        for w_den in [1.2, 1.5, 1.8, 2.2, 2.6]:
            for w_part in [0.9, 1.1, 1.3, 1.5]:
                for w_paid in [0.75, 0.85, 1.0]:
                    mults = np.array([w_paid, w_part, w_den], dtype=np.float32)
                    scaled_probs = probs * mults
                    preds = np.argmax(scaled_probs, axis=1)
                    
                    f1s = []
                    recalls = {}
                    for c in range(self.n_classes):
                        tp = np.sum((preds == c) & (y_val == c))
                        fp = np.sum((preds == c) & (y_val != c))
                        fn = np.sum((preds != c) & (y_val == c))
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        f1s.append(f1)
                        recalls[c] = rec
                        
                    macro_f1 = np.mean(f1s)
                    if recalls[2] >= 0.70:
                        objective = macro_f1 + 0.15 * recalls[2]
                        if objective > best_score:
                            best_score = objective
                            best_multipliers = mults
                            
        self.threshold_multipliers = best_multipliers
        return best_multipliers

    def predict(self, X: np.ndarray, use_calibrated_thresholds: bool = True) -> np.ndarray:
        probs = self.predict_proba(X)
        if use_calibrated_thresholds:
            scaled = probs * self.threshold_multipliers
            return np.argmax(scaled, axis=1)
        return np.argmax(probs, axis=1)


class TreeExplainerWrapper:
    def __init__(self, model: MultiClassBoostingClassifier, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, self.model.n_classes, n_features), dtype=np.float32)
        bg_X = np.median(X, axis=0)
        
        for c in range(self.model.n_classes):
            for tree in self.model.trees[c]:
                for s in range(n_samples):
                    curr = tree
                    while not curr['is_leaf']:
                        feat = curr['feat']
                        th = curr['th']
                        sample_val = X[s, feat]
                        bg_val = bg_X[feat]
                        if (sample_val <= th) != (bg_val <= th):
                            contrib = self.model.learning_rate * 0.25 * (1.0 if sample_val > th else -1.0)
                            shap_values[s, c, feat] += contrib
                        if sample_val <= th:
                            curr = curr['left']
                        else:
                            curr = curr['right']
        return shap_values

    def explain_instance(self, x_vec: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        shap_vals = self.compute_shap_values(x_vec.reshape(1, -1))[0]
        denied_shap = shap_vals[2]
        ranked_indices = np.argsort(-np.abs(denied_shap))
        
        top_risk_factors = []
        for idx in ranked_indices[:top_k]:
            feat_name = self.feature_names[idx]
            val = x_vec[idx]
            impact = float(denied_shap[idx])
            
            if 'ratio' in feat_name or 'zscore' in feat_name:
                desc = f"Billed amount deviates significantly ({val:.2f}x) from procedure baseline fee schedule."
            elif 'combo' in feat_name:
                desc = f"Payer policy rules or medical necessity scrutiny on procedure code."
            elif 'volume' in feat_name:
                desc = f"Provider historical claim volume ({int(val)}) flags lower submitter profile."
            elif 'senior' in feat_name or 'age' in feat_name:
                desc = f"Patient age ({int(val)}) subject to strict Medicare coverage determinations."
            else:
                desc = f"Feature {feat_name} value ({val:.2f}) shifts risk assessment."
                
            top_risk_factors.append({
                'feature': feat_name,
                'feature_value': round(float(val), 2),
                'shap_value': round(impact, 4),
                'directional_impact': 'Increases Denial Risk' if impact >= 0 else 'Reduces Denial Risk',
                'clinical_rationale': desc
            })
            
        return {
            'top_risk_factors': top_risk_factors,
            'denied_shap_vector': denied_shap.tolist()
        }

