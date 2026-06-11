"""
Explainable AI Module
- SHAP (SHapley Additive exPlanations) for global & local interpretability
- Integrated Gradients for neural network attribution
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  INTEGRATED GRADIENTS (for ANN)
# ─────────────────────────────────────────────

class IntegratedGradients:
    """
    Computes Integrated Gradients attribution for a PyTorch model.
    Reference: Sundararajan et al. (2017) - "Axiomatic Attribution for DNNs"
    """

    def __init__(self, model):
        self.model = model

    def _interpolate(self, baseline, input_sample, alpha):
        return baseline + alpha * (input_sample - baseline)

    def attribute(self, input_sample: np.ndarray, baseline: np.ndarray = None, steps: int = 50):
        """
        Args:
            input_sample: (n_features,) or (n_samples, n_features) numpy array
            baseline: same shape as input_sample (default: zeros)
            steps: number of integral approximation steps
        Returns:
            attributions: same shape as input_sample
        """
        if baseline is None:
            baseline = np.zeros_like(input_sample)

        input_t = torch.FloatTensor(input_sample)
        baseline_t = torch.FloatTensor(baseline)

        if input_t.dim() == 1:
            input_t = input_t.unsqueeze(0)
            baseline_t = baseline_t.unsqueeze(0)

        alphas = torch.linspace(0, 1, steps)
        integrated_grads = torch.zeros_like(input_t)

        for alpha in alphas:
            interp = self._interpolate(baseline_t, input_t, alpha.item())
            interp.requires_grad_(True)
            self.model.eval()
            with torch.enable_grad():
                output = torch.sigmoid(self.model(interp))
                output.sum().backward()
            integrated_grads += interp.grad.detach()

        self.model.eval()
        attributions = (input_t - baseline_t) * (integrated_grads / steps)
        return attributions.squeeze().numpy()

    def batch_attribute(self, X: np.ndarray, baseline: np.ndarray = None, steps: int = 50):
        """Compute IG attributions for all samples."""
        attrs = []
        for i, x in enumerate(X):
            attrs.append(self.attribute(x, baseline, steps))
            if (i + 1) % 20 == 0:
                print(f"  IG: {i+1}/{len(X)} samples processed")
        return np.array(attrs)


# ─────────────────────────────────────────────
#  SHAP WRAPPER (for all models)
# ─────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP-based explainer supporting both ML and ANN models.
    Uses KernelSHAP (model-agnostic) and DeepSHAP (for PyTorch ANNs).
    """

    def __init__(self, model, model_type='sklearn', feature_names=None, X_background=None):
        """
        Args:
            model: trained model (sklearn estimator or PyTorch nn.Module)
            model_type: 'sklearn' | 'ann'
            feature_names: list of feature names
            X_background: background dataset for SHAP (subset of training data)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install SHAP: pip install shap")

        self.shap = shap
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

        if X_background is None:
            raise ValueError("X_background is required for SHAP initialization.")

        # Sample background for efficiency
        bg = shap.sample(X_background, min(100, len(X_background)))

        if model_type == 'ann':
            def predict_fn(X_np):
                t = torch.FloatTensor(X_np)
                self.model.eval()
                with torch.no_grad():
                    out = torch.sigmoid(self.model(t))
                    return out.reshape(-1).numpy()
            self.explainer = shap.KernelExplainer(predict_fn, bg)
        else:
            self.explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], bg
            )

    def compute(self, X_test, n_samples=None):
        """Compute SHAP values for X_test."""
        X = X_test[:n_samples] if n_samples else X_test
        print(f"  Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X, silent=True)
        return self.shap_values

    def mean_abs_shap(self):
        """Global feature importance (mean |SHAP|)."""
        if self.shap_values is None:
            raise ValueError("Run compute() first.")
        return np.abs(self.shap_values).mean(axis=0)


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

def plot_ig_attribution(attribution, feature_names, title="Integrated Gradients Attribution",
                        save_path=None):
    """Bar chart of IG attributions for a single sample."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c' if v > 0 else '#2980b9' for v in attribution]
    bars = ax.barh(feature_names, attribution, color=colors, edgecolor='white', linewidth=0.5)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel("Attribution Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=10)

    pos_patch = mpatches.Patch(color='#e74c3c', label='Increases Risk')
    neg_patch = mpatches.Patch(color='#2980b9', label='Decreases Risk')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_global_importance(mean_shap, feature_names, title="Global Feature Importance (SHAP)",
                           save_path=None):
    """Sorted global feature importance bar chart."""
    sorted_idx = np.argsort(mean_shap)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_shap[sorted_idx],
        color='#2ecc71', edgecolor='white', linewidth=0.5
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_training_history(history, save_path=None):
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train', color='#e74c3c')
    ax1.plot(history['val_loss'], label='Validation', color='#3498db')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history['train_acc'], label='Train', color='#e74c3c')
    ax2.plot(history['val_acc'], label='Validation', color='#3498db')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_model_comparison(all_results, save_path=None):
    """Grouped bar chart comparing all models."""
    models = list(all_results.keys())
    metrics = ['accuracy', 'roc_auc', 'f1']
    metric_labels = ['Accuracy', 'ROC-AUC', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [all_results[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig