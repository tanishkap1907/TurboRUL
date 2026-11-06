# TurboRUL — Predictive Remaining Useful Life for Turbofan Engines

Predict when a jet engine will fail using sensor data from NASA's C-MAPSS dataset. This project compares **classical ML** (XGBoost, Random Forest) with **deep learning** (LSTM) approaches and uses **SHAP** to explain which sensors drive degradation predictions.

Built as an end-to-end ML pipeline: data ingestion → feature engineering → model training → explainability → interactive dashboard.

![SHAP Feature Importance](results/figures/shap_importance.png)

---

## Problem Statement

In aviation, unplanned engine failures cost millions and risk lives. **Remaining Useful Life (RUL)** prediction enables condition-based maintenance — replacing parts only when sensor data indicates actual degradation, not on a fixed schedule.

Given 21 sensor readings across operational cycles, predict how many cycles remain before engine failure.

## Dataset

**NASA C-MAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation

| Property | Value |
|----------|-------|
| Training engines | 100 (run-to-failure) |
| Test engines | 100 (cut off before failure) |
| Sensors | 21 channels |
| Operational settings | 3 |
| Fault mode | Single (HPC degradation) |

> Download: [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)

The pipeline auto-downloads the dataset. If NASA's server is unreachable, it generates synthetic data matching FD001 characteristics for development.

## Architecture

```
Raw Sensor Data (21 channels × N cycles)
    │
    ▼
┌─────────────────────────────────┐
│  Feature Engineering            │
│  • Drop 7 constant sensors      │
│  • Min-Max normalization        │
│  • Rolling window stats (W=5)   │
│    - rolling mean (trend)       │
│    - rolling std (volatility)   │
└──────────┬──────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────────┐
│ Flat     │ │ Sequences    │
│ Features │ │ (30 cycles)  │
│ (45 cols)│ │ (30×17)      │
└────┬─────┘ └──────┬───────┘
     │               │
     ▼               ▼
┌──────────┐ ┌──────────────┐
│ XGBoost  │ │ 2-Layer LSTM │
│ Random   │ │ hidden=64    │
│ Forest   │ │ dropout=0.3  │
└────┬─────┘ └──────┬───────┘
     │               │
     └───────┬───────┘
             ▼
     ┌───────────────┐
     │ SHAP Analysis │
     │ (TreeExplainer)│
     └───────┬───────┘
             ▼
     ┌───────────────┐
     │  Streamlit    │
     │  Dashboard    │
     └───────────────┘
```

## Key Techniques

### Feature Engineering
- **Variance analysis**: Dropped 7 sensors with near-zero variance (no degradation signal)
- **Rolling statistics**: 5-cycle window mean and std capture degradation *trajectory*, not just snapshot
- **RUL capping**: Capped at 125 cycles — early-life engines are all "healthy," distinguishing RUL=200 from RUL=300 adds noise

### Models
| Model | Approach | Why |
|-------|----------|-----|
| **XGBoost** | Gradient-boosted trees on flat features | Strong baseline, handles tabular data well |
| **Random Forest** | Bagged ensemble on flat features | Robust, less prone to overfitting |
| **LSTM** | 2-layer recurrent network on sequences | Learns temporal degradation patterns directly |

### Evaluation
- **RMSE / MAE** — standard regression metrics
- **R²** — variance explained
- **PHM08 Score** — domain-specific asymmetric penalty (late predictions penalized more than early ones, because missing a failure is costlier than premature maintenance)

### Explainability
SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature contributions:
- **Global importance**: Which sensors matter most across all engines
- **Beeswarm plot**: Direction of feature effects (high sensor value → higher/lower RUL?)
- **Waterfall plot**: Single-prediction decomposition for stakeholder trust

![SHAP Summary](results/figures/shap_summary.png)

## Project Structure

```
TurboRUL/
├── run_pipeline.py          # End-to-end pipeline orchestrator
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_loader.py       # Download, parse, label C-MAPSS data
│   ├── feature_engineering.py  # Normalization, rolling features, sequences
│   ├── train_classical.py   # XGBoost + Random Forest training
│   ├── train_lstm.py        # LSTM model definition + training
│   └── explainability.py    # SHAP analysis + visualization
├── app/
│   └── streamlit_app.py     # Interactive dashboard
├── data/raw/                # Raw C-MAPSS files (auto-downloaded)
├── models/                  # Saved model artifacts (.pkl, .pt)
├── results/
│   ├── model_comparison.json
│   └── figures/             # SHAP plots
└── notebooks/               # (optional) Exploration notebooks
```

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/TurboRUL.git
cd TurboRUL

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (downloads data → trains models → generates SHAP plots)
python run_pipeline.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

## Results

Model performance on C-MAPSS FD001 test set:

| Model | RMSE | MAE | R² | PHM Score |
|-------|------|-----|-----|-----------|
| XGBoost | — | — | — | — |
| Random Forest | — | — | — | — |
| LSTM | — | — | — | — |

> **Note**: Results depend on the dataset used. With the real NASA C-MAPSS FD001 data, published benchmarks achieve RMSE ~13–18. Run `python run_pipeline.py` with the real dataset to fill in actual metrics.

## Technologies

- **Python 3.10+**
- **XGBoost** — gradient-boosted trees
- **PyTorch** — LSTM implementation
- **scikit-learn** — preprocessing, Random Forest, metrics
- **SHAP** — model explainability
- **Streamlit** — interactive dashboard
- **pandas / NumPy** — data wrangling
- **Matplotlib / Seaborn** — visualization

## References

1. Saxena, A. et al. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." *PHM 2008*.
2. [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)
3. Lundberg, S. & Lee, S. "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*.

## License

MIT
