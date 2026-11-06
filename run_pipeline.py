"""
run_pipeline.py — End-to-end TurboRUL pipeline.

Usage:
    python run_pipeline.py

Runs: data loading → feature engineering → model training → SHAP analysis
Saves trained models to models/ and results to results/
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import prepare_data
from src.feature_engineering import (
    normalize_features, build_flat_features, build_sequences, FEATURE_COLS,
)
from src.train_classical import train_xgboost, train_random_forest, save_results
from src.train_lstm import train_lstm
from src.explainability import run_explainability


def main():
    print("=" * 60)
    print("  TurboRUL — Predictive Remaining Useful Life Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/6] Loading and labeling data...")
    train_df, test_df = prepare_data(data_dir="data/raw", subset="FD001", rul_cap=125)

    # ── Step 2: Normalize ──────────────────────────────────────
    print("\n[2/6] Normalizing features...")
    train_norm, test_norm, scaler = normalize_features(train_df, test_df)

    # ── Step 3: Feature engineering ────────────────────────────
    print("\n[3/6] Engineering features...")
    X_train, y_train, feature_names = build_flat_features(train_norm, window=5)
    X_test, y_test, _ = build_flat_features(test_norm, window=5)
    print(f"Flat features: {X_train.shape[1]} columns")

    # ── Step 4: Classical ML models ────────────────────────────
    print("\n[4/6] Training classical models...")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    # ── Step 5: LSTM ───────────────────────────────────────────
    print("\n[5/6] Training LSTM model...")
    X_train_seq, y_train_seq = build_sequences(train_norm, seq_length=30)
    X_test_seq, y_test_seq = build_sequences(test_norm, seq_length=30)
    print(f"Sequence shape: {X_train_seq.shape}")

    lstm_model, lstm_metrics, lstm_history = train_lstm(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        epochs=50, batch_size=256,
    )

    # ── Step 6: SHAP explainability ────────────────────────────
    print("\n[6/6] Running SHAP explainability analysis...")
    shap_values, explainer = run_explainability(
        xgb_model, X_train, X_test, feature_names
    )

    # ── Save all results ───────────────────────────────────────
    all_results = {
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics,
        "lstm": lstm_metrics,
    }
    save_results(all_results, "results/model_comparison.json")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    print("\nModel comparison:")
    for name, m in all_results.items():
        print(f"  {name:>15}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}")
    print(f"\nArtifacts saved:")
    print(f"  models/       — trained model files")
    print(f"  results/      — metrics JSON + SHAP figures")


if __name__ == "__main__":
    main()
