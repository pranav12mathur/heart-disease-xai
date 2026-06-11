"""
Main Pipeline: Interpretable Deep Learning for Heart Disease Risk Assessment

Usage:
    python main.py                          # Use synthetic data
    python main.py --data heart.csv         # Use your CSV file
    python main.py --data heart.csv --epochs 300 --shap-samples 50
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_data, preprocess, FEATURE_DESCRIPTIONS
from models.ann_model import HeartDiseaseANN, ANNTrainer
from models.baseline_models import train_all_baselines, evaluate_all, print_comparison_table
from utils.explainability import (
    IntegratedGradients, SHAPExplainer,
    plot_ig_attribution, plot_global_importance,
    plot_training_history, plot_model_comparison
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Heart Disease XAI Pipeline")
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--shap-samples', type=int, default=30)
    parser.add_argument('--ig-steps', type=int, default=50)
    parser.add_argument('--no-shap', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n[1/5] Loading & Preprocessing Data...")
    df = load_data(args.data)
    print(f"  Dataset shape: {df.shape} | Positive rate: {df['target'].mean():.1%}")

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, feature_names) = preprocess(df)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[2/5] Training Baseline Models...")
    baselines = train_all_baselines(X_train, y_train)
    baseline_results = evaluate_all(baselines, X_test, y_test)

    print("\n[3/5] Training ANN...")
    model = HeartDiseaseANN(input_dim=len(feature_names), dropout=args.dropout)
    trainer = ANNTrainer(model, lr=args.lr)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size
    )

    ann_results = trainer.evaluate(X_test, y_test)
    print(f"\n  ANN Test Results:")
    print(f"  Accuracy : {ann_results['accuracy']:.4f}")
    print(f"  ROC-AUC  : {ann_results['roc_auc']:.4f}")
    print(f"  F1-Score : {ann_results['f1']:.4f}")
    print(f"\n{ann_results['classification_report']}")

    print_comparison_table(
        baseline_results,
        {'accuracy': ann_results['accuracy'],
         'roc_auc': ann_results['roc_auc'],
         'f1': ann_results['f1']}
    )

    trainer.save(os.path.join(MODELS_DIR, "ann_best.pt"))

    print("\n[4/5] Generating Visualizations...")
    plot_training_history(history, save_path=os.path.join(RESULTS_DIR, "training_history.png"))

    all_results = {
        **baseline_results,
        'ANN': {'accuracy': ann_results['accuracy'],
                'roc_auc': ann_results['roc_auc'],
                'f1': ann_results['f1']}
    }
    plot_model_comparison(all_results, save_path=os.path.join(RESULTS_DIR, "model_comparison.png"))

    print("\n[5/5] Explainability Analysis...")
    print("  Computing Integrated Gradients...")
    ig = IntegratedGradients(model)
    high_risk_idx = np.where(y_test == 1)[0]
    sample = X_test[high_risk_idx[0]]
    ig_attr = ig.attribute(sample, steps=args.ig_steps)

    plot_ig_attribution(
        ig_attr, feature_names,
        title="Integrated Gradients - High-Risk Patient (ANN)",
        save_path=os.path.join(RESULTS_DIR, "ig_attribution_sample.png")
    )

    print("  Computing batch IG attributions...")
    ig_all = ig.batch_attribute(X_test, steps=args.ig_steps)
    mean_ig = np.abs(ig_all).mean(axis=0)
    plot_global_importance(
        mean_ig, feature_names,
        title="Global Feature Importance - Mean |IG| (ANN)",
        save_path=os.path.join(RESULTS_DIR, "ig_global_importance.png")
    )

    if not args.no_shap:
        try:
            print("  Computing SHAP values (KernelSHAP)...")
            shap_explainer = SHAPExplainer(
                model, model_type='ann',
                feature_names=feature_names,
                X_background=X_train
            )
            shap_vals = shap_explainer.compute(X_test, n_samples=args.shap_samples)
            mean_shap = shap_explainer.mean_abs_shap()
            plot_global_importance(
                mean_shap, feature_names,
                title="Global Feature Importance - Mean |SHAP| (ANN)",
                save_path=os.path.join(RESULTS_DIR, "shap_global_importance.png")
            )
            plot_ig_attribution(
                shap_vals[0], feature_names,
                title="SHAP Values - High-Risk Patient (ANN)",
                save_path=os.path.join(RESULTS_DIR, "shap_local_sample.png")
            )
            print("  SHAP analysis complete.")
        except ImportError:
            print("  SHAP not installed. Run: pip install shap")
    else:
        print("  SHAP skipped (--no-shap).")

    print(f"\nPipeline complete. Results saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()