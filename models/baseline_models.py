"""
Traditional ML Baselines for comparison with ANN
Models: Logistic Regression, Random Forest, XGBoost, SVM
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


BASELINE_MODELS = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
    'SVM (RBF)': SVC(
        C=1.0, kernel='rbf', probability=True, random_state=42
    ),
}


def train_all_baselines(X_train, y_train):
    trained = {}
    for name, clf in BASELINE_MODELS.items():
        clf.fit(X_train, y_train)
        trained[name] = clf
        print(f"  Trained: {name}")
    return trained


def evaluate_all(trained_models, X_test, y_test):
    results = {}
    for name, clf in trained_models.items():
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        results[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probs),
            'f1': f1_score(y_test, preds),
        }
    return results


def print_comparison_table(baseline_results, ann_results):
    """Print formatted comparison of all models."""
    all_results = {**baseline_results, 'ANN (Deep Learning)': ann_results}

    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'F1-Score':>10}")
    print("=" * 65)
    for name, res in all_results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['roc_auc']:>10.4f} {res['f1']:>10.4f}")
    print("=" * 65)