"""
feature_engineering.py - Build features for RUL prediction.

Drops constant sensors, normalizes, and adds rolling window
statistics to capture degradation trends. Outputs flat features
for tree models and sequences for LSTM.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# these sensors actually change over time - sensors 1,5,6,10,16,18,19
# are basically constant so we drop them
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
FEATURE_COLS = USEFUL_SENSORS + SETTING_COLS


def drop_constant_sensors(df: pd.DataFrame) -> list:
    """Identify sensors with near-zero variance (useless for prediction)."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    variances = df[sensor_cols].var()
    constant = variances[variances < 1e-6].index.tolist()
    useful = [c for c in sensor_cols if c not in constant]
    print(f"Dropped {len(constant)} constant sensors: {constant}")
    print(f"Retained {len(useful)} useful sensors")
    return useful


def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       feature_cols: list = FEATURE_COLS):
    """
    Fit MinMaxScaler on training data, transform both sets.
    Returns copies with scaled feature columns.
    """
    scaler = MinMaxScaler()
    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_out[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_out, test_out, scaler


def add_rolling_features(df: pd.DataFrame, window: int = 5,
                         sensor_cols: list = USEFUL_SENSORS) -> pd.DataFrame:
    """Add rolling mean and std for each sensor to capture trends."""
    df = df.copy()
    for col in sensor_cols:
        grouped = df.groupby("engine_id")[col]

        # Rolling mean
        df[f"{col}_rmean_{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Rolling std (volatility)
        df[f"{col}_rstd_{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    return df


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear slope using least-squares over a window."""
    slopes = []
    values = series.values
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start:i + 1]
        if len(segment) < 2:
            slopes.append(0.0)
        else:
            x = np.arange(len(segment))
            slope = np.polyfit(x, segment, 1)[0]
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def build_flat_features(df: pd.DataFrame, window: int = 5) -> tuple:
    """
    Build a flat feature matrix for tree-based models.

    Returns
    -------
    X : np.ndarray — feature matrix
    y : np.ndarray — RUL targets
    feature_names : list — column names for SHAP
    """
    df_feat = add_rolling_features(df, window=window)

    feature_names = FEATURE_COLS.copy()
    for col in USEFUL_SENSORS:
        feature_names.append(f"{col}_rmean_{window}")
        feature_names.append(f"{col}_rstd_{window}")

    X = df_feat[feature_names].values
    y = df_feat["rul"].values

    return X, y, feature_names


def build_sequences(df: pd.DataFrame, seq_length: int = 30,
                    feature_cols: list = FEATURE_COLS) -> tuple:
    """
    Build sequence arrays for LSTM: for each engine, create overlapping
    windows of `seq_length` consecutive cycles.

    Returns
    -------
    X_seq : np.ndarray of shape (n_samples, seq_length, n_features)
    y_seq : np.ndarray of shape (n_samples,) — RUL at the last timestep
    """
    X_list, y_list = [], []

    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")
        features = engine_data[feature_cols].values
        targets = engine_data["rul"].values

        # Pad short sequences with the first row
        if len(features) < seq_length:
            pad_len = seq_length - len(features)
            features = np.vstack([np.tile(features[0], (pad_len, 1)), features])
            targets = np.concatenate([np.full(pad_len, targets[0]), targets])

        for i in range(len(features) - seq_length + 1):
            X_list.append(features[i:i + seq_length])
            y_list.append(targets[i + seq_length - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
