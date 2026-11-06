"""
TurboRUL — Streamlit Dashboard for Remaining Useful Life Prediction.

Run:  streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from src.data_loader import prepare_data
from src.feature_engineering import (
    normalize_features, build_flat_features, build_sequences, FEATURE_COLS,
)
from src.train_classical import phm_score


# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="TurboRUL — Engine RUL Predictor",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ TurboRUL — Turbofan Engine RUL Prediction")
st.markdown(
    "Predict **Remaining Useful Life** of turbofan engines using "
    "NASA's C-MAPSS FD001 dataset. Compare XGBoost, Random Forest, "
    "and LSTM models with SHAP explainability."
)


# ── Helper: load cached data ──────────────────────────────────
@st.cache_data
def load_data():
    train_df, test_df = prepare_data(data_dir="data/raw", subset="FD001", rul_cap=125)
    train_norm, test_norm, scaler = normalize_features(train_df, test_df)
    X_train, y_train, feat_names = build_flat_features(train_norm, window=5)
    X_test, y_test, _ = build_flat_features(test_norm, window=5)
    return train_df, test_df, X_train, y_train, X_test, y_test, feat_names


@st.cache_resource
def load_models():
    models = {}
    xgb_path = "models/xgboost_rul.pkl"
    rf_path = "models/random_forest_rul.pkl"
    if os.path.exists(xgb_path):
        models["XGBoost"] = joblib.load(xgb_path)
    if os.path.exists(rf_path):
        models["Random Forest"] = joblib.load(rf_path)
    return models


# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Comparison", "Engine Explorer", "SHAP Analysis"],
)


# ── Page: Overview ─────────────────────────────────────────────
if page == "Overview":
    st.header("Dataset Overview — NASA C-MAPSS FD001")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Engines", "100")
    col2.metric("Test Engines", "100")
    col3.metric("Sensors", "21")
    col4.metric("RUL Cap", "125 cycles")

    st.markdown("---")
    st.subheader("What is Remaining Useful Life?")
    st.markdown(
        """
        **RUL** is the number of operational cycles an engine has left before
        failure. Predicting RUL enables **condition-based maintenance** —
        replacing parts only when needed, instead of on a fixed schedule.

        This project uses the **FD001** subset (single fault mode, single
        operating condition) from NASA's Commercial Modular Aero-Propulsion
        System Simulation (C-MAPSS).

        **Pipeline:** Raw Sensors → Feature Engineering → Model Training →
        SHAP Explainability → This Dashboard
        """
    )

    # Show sample data
    try:
        train_df, test_df, *_ = load_data()
        st.subheader("Sample Training Data")
        st.dataframe(train_df.head(20), use_container_width=True)

        st.subheader("RUL Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(train_df.groupby("engine_id")["cycle"].max(), bins=30,
                     color="#4A90D9", edgecolor="white")
        axes[0].set_xlabel("Engine Lifetime (cycles)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Engine Lifetime Distribution")

        axes[1].hist(train_df["rul"], bins=50, color="#E8724A", edgecolor="white")
        axes[1].set_xlabel("RUL (capped at 125)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("RUL Label Distribution")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Run the pipeline first: `python run_pipeline.py`\n\n{e}")


# ── Page: Model Comparison ─────────────────────────────────────
elif page == "Model Comparison":
    st.header("Model Performance Comparison")

    results_path = "results/model_comparison.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        # Metrics table
        metrics_df = pd.DataFrame(results).T
        metrics_df.index.name = "Model"
        st.dataframe(metrics_df.style.highlight_min(subset=["rmse", "mae", "phm_score"],
                                                     color="#c6efce")
                                      .highlight_max(subset=["r2"], color="#c6efce"),
                      use_container_width=True)

        # Bar charts
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        models_list = list(results.keys())
        colors = ["#4A90D9", "#50C878", "#E8724A"]

        for ax, metric, title in zip(axes,
                                      ["rmse", "mae", "r2"],
                                      ["RMSE ↓", "MAE ↓", "R² ↑"]):
            vals = [results[m][metric] for m in models_list]
            ax.bar(models_list, vals, color=colors, edgecolor="white")
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(metric.upper())
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01 * max(vals), f"{v:.2f}", ha="center", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("PHM08 Scoring")
        st.markdown(
            "The PHM08 score uses an **asymmetric penalty**: late predictions "
            "(missing a failure) are penalised more than early ones. Lower is better."
        )
        phm_vals = [results[m]["phm_score"] for m in models_list]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(models_list, phm_vals, color=colors, edgecolor="white")
        ax2.set_xlabel("PHM Score (lower = better)")
        ax2.set_title("PHM08 Asymmetric Score Comparison")
        for i, v in enumerate(phm_vals):
            ax2.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("No results found. Run `python run_pipeline.py` first.")


# ── Page: Engine Explorer ──────────────────────────────────────
elif page == "Engine Explorer":
    st.header("Engine-Level Predictions")

    try:
        train_df, test_df, X_train, y_train, X_test, y_test, feat_names = load_data()
        models = load_models()

        if not models:
            st.warning("No trained models found. Run the pipeline first.")
        else:
            model_name = st.selectbox("Select Model", list(models.keys()))
            model = models[model_name]

            y_pred = model.predict(X_test)

            # Scatter: predicted vs actual
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#4A90D9")
            lims = [0, max(y_test.max(), y_pred.max()) + 5]
            ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
            ax.set_xlabel("Actual RUL", fontsize=12)
            ax.set_ylabel("Predicted RUL", fontsize=12)
            ax.set_title(f"{model_name} — Predicted vs Actual RUL", fontsize=14)
            ax.legend()
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            plt.tight_layout()
            st.pyplot(fig)

            # Error distribution
            errors = y_pred - y_test
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.hist(errors, bins=60, color="#50C878", edgecolor="white", alpha=0.8)
            ax2.axvline(0, color="red", linestyle="--", linewidth=2)
            ax2.set_xlabel("Prediction Error (positive = early, negative = late)")
            ax2.set_title("Error Distribution")
            plt.tight_layout()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")


# ── Page: SHAP Analysis ───────────────────────────────────────
elif page == "SHAP Analysis":
    st.header("SHAP Explainability Analysis")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) decomposes each prediction "
        "into per-feature contributions, revealing **which sensors drive "
        "RUL predictions** and how."
    )

    fig_dir = "results/figures"
    shap_files = {
        "Global Feature Importance": "shap_importance.png",
        "SHAP Summary (Beeswarm)": "shap_summary.png",
        "Waterfall — Single Prediction": "shap_waterfall.png",
    }

    for title, fname in shap_files.items():
        path = os.path.join(fig_dir, fname)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)
        else:
            st.info(f"{title}: not generated yet. Run the pipeline first.")


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "TurboRUL — Built with Python, XGBoost, PyTorch, SHAP & Streamlit | "
    "Data: NASA C-MAPSS FD001"
)
