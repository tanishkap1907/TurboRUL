"""
train_classical.py - XGBoost and Random Forest for RUL prediction.

Uses the PHM08 asymmetric scoring function alongside standard
metrics (RMSE, MAE, R2).
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 asymmetric scoring - penalizes late predictions more
    because missing a failure is worse than early maintenance."""
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def evaluate(y_true, y_pred, name="Model"):
    """Compute and print evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = phm_score(y_true, y_pred)

    metrics = {"rmse": round(rmse, 2), "mae": round(mae, 2),
               "r2": round(r2, 4), "phm_score": round(score, 2)}
    print(f"\n{name} Results:")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")
    return metrics


def train_xgboost(X_train, y_train, X_test, y_test, save_dir="models"):
    """Train XGBoost regressor with tuned hyperparameters."""
    print("\n--- Training XGBoost ---")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, "XGBoost")

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "xgboost_rul.pkl"))

    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test, save_dir="models"):
    """Train Random Forest regressor."""
    print("\n--- Training Random Forest ---")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, "Random Forest")

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "random_forest_rul.pkl"))

    return model, metrics


def save_results(results: dict, path: str = "results/classical_results.json"):
    """Save metrics to JSON for comparison."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
