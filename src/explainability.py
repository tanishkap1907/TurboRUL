"""
explainability.py - SHAP analysis for interpreting RUL predictions.

Uses TreeExplainer to show which sensors drive the model's
predictions. Generates importance, summary, and waterfall plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def compute_shap_values(model, X_train, X_test, feature_names,
                        n_background: int = 100):
    """
    Compute SHAP values for the XGBoost model using TreeExplainer.

    Parameters
    ----------
    model : trained XGBoost model
    X_train : training features (for background)
    X_test : test features to explain
    feature_names : list of feature column names
    n_background : number of background samples

    Returns
    -------
    shap_values : SHAP values array
    explainer : fitted SHAP explainer
    """
    print("Computing SHAP values...")

    # Use TreeExplainer for tree-based models (fast, exact)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP values shape: {shap_values.shape}")
    return shap_values, explainer


def plot_feature_importance(shap_values, feature_names,
                            save_path="results/figures/shap_importance.png"):
    """Global feature importance: mean |SHAP| per feature."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[-15:]  # Top 15

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)),
            mean_abs_shap[sorted_idx],
            color="#4A90D9", edgecolor="white")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top 15 Features — Global Importance for RUL Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary(shap_values, X_test, feature_names,
                 save_path="results/figures/shap_summary.png"):
    """SHAP beeswarm summary plot — shows direction of feature effects."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      max_display=15, show=False)
    plt.title("SHAP Summary — Feature Impact on RUL Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_waterfall(shap_values, explainer, X_test, feature_names,
                   sample_idx: int = 0,
                   save_path="results/figures/shap_waterfall.png"):
    """
    Waterfall plot for a single prediction — shows how each feature
    pushed the prediction up or down from the base value.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test[sample_idx],
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — Engine Sample #{sample_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_explainability(model, X_train, X_test, feature_names,
                       save_dir="results/figures"):
    """Run the full SHAP analysis pipeline."""
    shap_values, explainer = compute_shap_values(model, X_train, X_test, feature_names)

    plot_feature_importance(shap_values, feature_names,
                            os.path.join(save_dir, "shap_importance.png"))
    plot_summary(shap_values, X_test, feature_names,
                 os.path.join(save_dir, "shap_summary.png"))
    plot_waterfall(shap_values, explainer, X_test, feature_names,
                   sample_idx=0,
                   save_path=os.path.join(save_dir, "shap_waterfall.png"))

    return shap_values, explainer
