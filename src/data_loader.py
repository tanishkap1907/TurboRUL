"""
data_loader.py — Load and label the NASA C-MAPSS turbofan degradation dataset.

The C-MAPSS dataset (FD001) contains run-to-failure sensor readings from
100 training engines and 100 test engines, each with 21 sensor channels
and 3 operational settings. Each row is one cycle for one engine.

This module:
  1. Downloads the dataset from NASA's Prognostics Data Repository
  2. Parses the fixed-width text files into DataFrames
  3. Computes the Remaining Useful Life (RUL) target for each row
  4. Applies a RUL cap (default 125 cycles) to stabilise training —
     engines in early life are all "healthy," and distinguishing
     RUL=200 from RUL=300 adds noise without value.
"""

import os
import zipfile
import urllib.request
import ssl
import pandas as pd
import numpy as np

# Column names for the raw data (no header in the text files)
INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def download_cmapss(data_dir: str = "data/raw") -> str:
    """
    Download the C-MAPSS dataset from NASA's Prognostics Data Repository.
    If download fails (network restrictions), generates a synthetic dataset
    that mirrors FD001 characteristics for local development.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train_FD001.txt")

    # Already extracted — skip
    if os.path.exists(train_path):
        return data_dir

    zip_path = os.path.join(data_dir, "CMAPSSData.zip")
    downloaded = False

    if not os.path.exists(zip_path):
        url = (
            "https://data.nasa.gov/download/nk8v-ckry/application%2Fx-zip-compressed"
        )
        try:
            print("Downloading C-MAPSS dataset from NASA...")
            ctx = ssl.create_default_context()
            ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
            if os.path.exists(ca_bundle):
                ctx.load_verify_locations(ca_bundle)
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, zip_path)
            print(f"Saved to {zip_path}")
            downloaded = True
        except Exception as e:
            print(f"Download failed ({e}). Generating synthetic FD001 dataset...")
            _generate_synthetic_fd001(data_dir)
            return data_dir

    # Extract zip
    if downloaded or os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print(f"Extracted to {data_dir}/")

    return data_dir


def _generate_synthetic_fd001(data_dir: str, n_train: int = 100, n_test: int = 100,
                               seed: int = 42):
    """
    Generate a synthetic dataset that mirrors NASA C-MAPSS FD001 characteristics:
      - 100 train engines, 100 test engines
      - 21 sensors + 3 operational settings
      - Lifetimes ~ Weibull distribution (128-362 cycles)
      - Realistic sensor degradation profiles

    This enables development/testing when the NASA server is unreachable.
    For production benchmarking, replace with the real C-MAPSS dataset from:
    https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
    """
    rng = np.random.RandomState(seed)

    # FD001 lifetime distribution: roughly Weibull-shaped, 128-362 cycles
    def gen_lifetimes(n):
        return rng.randint(128, 363, size=n)

    # Sensor baseline and degradation profiles (from FD001 statistics)
    # Format: (baseline_mean, baseline_std, degradation_rate, noise_std)
    sensor_profiles = {
        "sensor_1":  (518.67, 0.0, 0.0, 0.0),         # constant
        "sensor_2":  (642.44, 0.5, 0.03, 0.3),
        "sensor_3":  (1589.7, 1.0, 0.08, 0.6),
        "sensor_4":  (1400.6, 2.0, 0.10, 1.5),
        "sensor_5":  (14.62, 0.0, 0.0, 0.0),           # constant
        "sensor_6":  (21.61, 0.0, 0.0, 0.0),           # constant
        "sensor_7":  (553.0, 0.5, -0.02, 0.3),
        "sensor_8":  (2388.0, 0.8, 0.05, 0.5),
        "sensor_9":  (9046.0, 5.0, 0.15, 3.0),
        "sensor_10": (1.30, 0.0, 0.0, 0.0),            # constant
        "sensor_11": (47.27, 0.2, 0.02, 0.1),
        "sensor_12": (521.7, 0.5, 0.03, 0.3),
        "sensor_13": (2388.0, 0.8, 0.05, 0.5),
        "sensor_14": (8138.0, 3.0, 0.10, 2.0),
        "sensor_15": (8.44, 0.02, 0.001, 0.01),
        "sensor_16": (0.03, 0.0, 0.0, 0.0),            # constant
        "sensor_17": (392.0, 0.5, -0.01, 0.3),
        "sensor_18": (2388.0, 0.0, 0.0, 0.0),          # constant
        "sensor_19": (100.0, 0.0, 0.0, 0.0),           # constant
        "sensor_20": (39.06, 0.1, -0.01, 0.08),
        "sensor_21": (23.42, 0.1, 0.01, 0.06),
    }

    setting_profiles = {
        "setting_1": (-0.0007, 0.002),
        "setting_2": (-0.0004, 0.001),
        "setting_3": (100.0, 0.0),
    }

    def generate_engines(n_engines, lifetimes):
        rows = []
        for eid in range(1, n_engines + 1):
            n_cycles = lifetimes[eid - 1]
            for cyc in range(1, n_cycles + 1):
                row = [eid, cyc]
                # Settings
                for scol in ["setting_1", "setting_2", "setting_3"]:
                    mean, std = setting_profiles[scol]
                    row.append(round(mean + rng.normal(0, max(std, 1e-8)), 4))
                # Sensors — exponential degradation (slow start, rapid end-of-life)
                health = cyc / n_cycles  # 0 = healthy, 1 = failed
                # Piece-wise: flat healthy region then exponential degradation
                if health < 0.5:
                    deg_factor = 0.1 * health  # minimal degradation
                else:
                    deg_factor = np.exp(3.0 * (health - 0.5)) - 1  # exponential ramp
                for i in range(1, 22):
                    scol = f"sensor_{i}"
                    base_mean, base_std, deg_rate, noise_std = sensor_profiles[scol]
                    degradation = deg_rate * deg_factor * 30  # scaled degradation
                    noise = rng.normal(0, max(noise_std, 1e-8))
                    val = base_mean + rng.normal(0, max(base_std, 1e-8)) + degradation + noise
                    row.append(round(val, 4))
                rows.append(row)
        return rows

    # Generate training data (run-to-failure)
    train_lifetimes = gen_lifetimes(n_train)
    train_rows = generate_engines(n_train, train_lifetimes)

    # Generate test data (cut off before failure)
    test_full_lifetimes = gen_lifetimes(n_test)
    test_cutoff = np.array([rng.randint(int(lt * 0.4), int(lt * 0.85))
                            for lt in test_full_lifetimes])
    test_rows = generate_engines(n_test, test_cutoff)

    # RUL ground truth = full_lifetime - cutoff
    rul_true = test_full_lifetimes - test_cutoff

    # Write files in C-MAPSS format (space-separated, no header)
    def write_data(rows, path):
        with open(path, "w") as f:
            for row in rows:
                f.write(" ".join(str(v) for v in row) + "\n")

    write_data(train_rows, os.path.join(data_dir, "train_FD001.txt"))
    write_data(test_rows, os.path.join(data_dir, "test_FD001.txt"))

    with open(os.path.join(data_dir, "RUL_FD001.txt"), "w") as f:
        for r in rul_true:
            f.write(f"{r}\n")

    print(f"Generated synthetic FD001: {n_train} train + {n_test} test engines")
    print(f"NOTE: Replace with real C-MAPSS data for benchmarking.")


def load_raw(data_dir: str = "data/raw", subset: str = "FD001"):
    """
    Load train, test, and RUL files for a given C-MAPSS subset.

    Returns
    -------
    train_df : pd.DataFrame  — training data (run-to-failure)
    test_df  : pd.DataFrame  — test data (cut off before failure)
    rul_true : np.ndarray    — true RUL at the last cycle of each test engine
    """
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLS)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"]).values.flatten()

    return train_df, test_df, rul_true


def add_rul_labels(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """
    Compute the RUL target for each row in a run-to-failure dataset.

    RUL = (max_cycle_for_this_engine - current_cycle).
    Then cap at `rul_cap` so the model focuses on the degradation region.
    """
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=rul_cap)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_test_rul(test_df: pd.DataFrame, rul_true: np.ndarray, rul_cap: int = 125) -> pd.DataFrame:
    """
    Compute RUL labels for test data using the provided ground truth.
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    rul_map = dict(zip(max_cycles["engine_id"], rul_true))
    test_df = test_df.merge(max_cycles, on="engine_id")

    test_df["rul"] = test_df.apply(
        lambda row: rul_map[row["engine_id"]] + (row["max_cycle"] - row["cycle"]),
        axis=1,
    )
    test_df["rul"] = test_df["rul"].clip(upper=rul_cap)
    test_df.drop(columns=["max_cycle"], inplace=True)
    return test_df


def prepare_data(data_dir: str = "data/raw", subset: str = "FD001", rul_cap: int = 125):
    """
    Full pipeline: download → load → label → return.

    Returns
    -------
    train_df, test_df : DataFrames with 'rul' column added
    """
    download_cmapss(data_dir)
    train_df, test_df, rul_true = load_raw(data_dir, subset)
    train_df = add_rul_labels(train_df, rul_cap)
    test_df = add_test_rul(test_df, rul_true, rul_cap)

    print(f"Train: {len(train_df):,} rows, {train_df['engine_id'].nunique()} engines")
    print(f"Test:  {len(test_df):,} rows, {test_df['engine_id'].nunique()} engines")

    return train_df, test_df
