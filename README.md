# 🌡️ AI/ML Urban Heat Stress Hotspot Prediction System

> **An end-to-end geospatial ML pipeline for predicting, clustering, simulating, and optimizing urban heat stress hotspots across India using Landsat 8, Google Earth Engine, XGBoost, and SHAP.**

---

## 📁 Project Structure

```
Isro/
├── config.py                    ← Central config, GEE settings, synthetic data generator
├── module1_data_ingestion.py    ← GEE data pull + Landsat LST/NDVI/NDBI computation
├── module2_hotspot_clustering.py← Getis-Ord Gi* spatial statistics
├── module3_predictive_modeling.py ← XGBoost LST regressor + evaluation
├── module4_shap_interpretation.py ← SHAP TreeExplainer driver analysis
├── module5_scenario_simulation.py ← Intervention scenario simulation
├── module6_optimization.py      ← Multi-objective priority optimization
├── module7_dashboard.py         ← Streamlit interactive dashboard
├── pipeline_runner.py           ← End-to-end orchestrator (CLI)
└── README.md                    ← This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install earthengine-api geemap pandas numpy scikit-learn xgboost shap \
            streamlit streamlit-folium folium plotly matplotlib scipy
```

### 2. Run in Demo Mode (No GEE account needed)

```bash
# Run the full pipeline with synthetic data
python pipeline_runner.py

# Launch the Streamlit dashboard
streamlit run module7_dashboard.py
```

The dashboard will open at **http://localhost:8501**. Select **"🧪 Synthetic Demo"** in the sidebar and click **"🚀 Run Pipeline"**.

---

## 🌍 Using Live Google Earth Engine Data

### Step 1 — Create a GEE Project
1. Go to [https://code.earthengine.google.com/](https://code.earthengine.google.com/)
2. Create a Cloud Project at [https://console.cloud.google.com/](https://console.cloud.google.com/)
3. Enable the **Earth Engine API** for your project

### Step 2 — Authenticate
```bash
earthengine authenticate
```
Follow the browser prompts. This stores credentials locally.

### Step 3 — Configure Project ID
Open `config.py` and set:
```python
GEE_PROJECT_ID = "your-gcp-project-id"
```

### Step 4 — Run with GEE
```bash
python pipeline_runner.py --gee
```

Or select **"🌐 Google Earth Engine (Live)"** in the dashboard sidebar.

---

## 🧩 Module Reference

### Module 1 — Data Ingestion (`module1_data_ingestion.py`)
| | |
|---|---|
| **Input**  | Google Earth Engine / synthetic fallback |
| **Output** | `heat_grid.csv` |
| **Key function** | `run_ingestion(use_gee, roi_bbox)` |

Pulls Landsat 8 C2 L2 imagery (March–June 2024), masks clouds, computes:
- **LST** = `ST_B10 × 0.00341802 + 149.0 − 273.15` (°C)
- **NDVI** = `(B5 − B4) / (B5 + B4)`
- **NDBI** = `(B6 − B5) / (B6 + B5)`

Integrates: WorldPop, JRC/GHSL, ESA WorldCover. Samples at 100 m grid.

---

### Module 2 — Hotspot Clustering (`module2_hotspot_clustering.py`)
| | |
|---|---|
| **Input**  | `heat_grid.csv` or DataFrame |
| **Output** | DataFrame + `gi_zscore` + `hotspot_category` columns |
| **Key function** | `run_hotspot_analysis(df, bandwidth)` |

Implements Getis-Ord **Gi*** statistic using `scipy.spatial.distance.cdist`.

| Category | Z-score Threshold |
|---|---|
| Extreme Hotspot (99%) | z > 2.576 |
| Hotspot (95%) | z > 1.960 |
| Neutral | −1.96 ≤ z ≤ 1.96 |
| Coldspot | z < −1.960 |

---

### Module 3 — Predictive Modeling (`module3_predictive_modeling.py`)
| | |
|---|---|
| **Input**  | DataFrame |
| **Output** | `xgb_lst_model.json`, metrics dict |
| **Key function** | `run_modeling(df)` |

- **Features**: NDVI, NDBI, Pop_Density, Built_Fraction, LULC
- **Target**: LST_Celsius
- **Split**: 80% train / 20% test
- **Model**: XGBoost Regressor (hist tree method)
- **Metrics**: MAE, RMSE, R²

---

### Module 4 — SHAP Analysis (`module4_shap_interpretation.py`)
| | |
|---|---|
| **Input**  | Trained model + test features |
| **Output** | `shap_summary_beeswarm.png`, `shap_feature_importance.png` |
| **Key function** | `run_shap_analysis(modeling_results)` |

Uses `shap.TreeExplainer` with interventional feature perturbation. Generates:
1. **Beeswarm plot** — global feature impact distribution
2. **Bar chart** — mean |SHAP| feature ranking

---

### Module 5 — Scenario Simulation (`module5_scenario_simulation.py`)
| | |
|---|---|
| **Input**  | Trained model + test features |
| **Output** | `scenario_results.csv`, `scenario_results_summary.csv` |
| **Key function** | `run_simulation(modeling_results)` |

| Scenario | NDVI Delta | NDBI Delta |
|---|---|---|
| Urban Greening | +0.20 | −0.05 |
| Cool Roofs/Albedo | 0.00 | −0.25 |
| Mixed Strategy | +0.15 | −0.15 |

Computes per-cell **ΔT** (°C) = LST_baseline − LST_scenario.

---

### Module 6 — Optimization (`module6_optimization.py`)
| | |
|---|---|
| **Input**  | Scenario results + test DataFrame |
| **Output** | `optimization_results.csv` |
| **Key function** | `run_full_optimization(sim_results, budget_n, alpha, beta)` |

Priority Score: **P(i) = α × S(i) + β × C(i)**

- **S(i)** = min-max normalized LST severity
- **C(i)** = normalized max cooling sensitivity
- **α** = 0.6 (severity weight), **β** = 0.4 (sensitivity weight)

Selects top-N cells and recommends the best single intervention per cell.

---

### Module 7 — Streamlit Dashboard (`module7_dashboard.py`)
```bash
streamlit run module7_dashboard.py
```

**5 interactive tabs:**
| Tab | Content |
|---|---|
| 🗺️ Heat Map | Folium choropleth + LST heatmap + target markers |
| 📈 Model Performance | Metric cards + scatter + residual histogram |
| 🔬 SHAP Analysis | Interactive bar chart + SHAP PNG plots |
| 🔄 Scenarios | Cooling comparison bars + violin plots |
| 🎯 Optimization | Strategy donut + priority scatter + target table |

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `GEE_PROJECT_ID` | `"your-gee-project-id"` | Your GCP project ID |
| `DEFAULT_ROI_BBOX` | Delhi NCR | `[west, south, east, north]` |
| `START_DATE` | `2024-03-01` | Imagery start date |
| `END_DATE` | `2024-06-30` | Imagery end date |
| `CLOUD_COVER_PCT` | `15` | Max cloud cover (%) |
| `SAMPLE_SCALE_M` | `100` | Grid resolution (metres) |
| `DEFAULT_BUDGET_N` | `50` | Default optimization budget |
| `ALPHA_SEVERITY` | `0.6` | Severity weight (α) |
| `BETA_SENSITIVITY` | `0.4` | Sensitivity weight (β) |

---

## 📦 Full Requirements

```
earthengine-api>=0.1.380
geemap>=0.30.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.43.0
matplotlib>=3.7.0
streamlit>=1.35.0
streamlit-folium>=0.20.0
folium>=0.16.0
plotly>=5.20.0
```

Install all at once:
```bash
pip install earthengine-api geemap pandas numpy scipy scikit-learn xgboost shap \
            matplotlib streamlit streamlit-folium folium plotly
```

---

## 🏃 CLI Pipeline Runner

```
usage: python pipeline_runner.py [options]

Options:
  --gee          Use live GEE data (requires earthengine authenticate)
  --budget N     Number of target cells to optimize (default: 50)
  --alpha A      Severity weight α ∈ [0,1] (default: 0.6)
  --skip-shap    Skip SHAP analysis (faster)
```

---

## 🗺️ Custom Region of Interest

To change the study area, edit `config.py`:

```python
DEFAULT_ROI_BBOX = [72.77, 18.85, 73.10, 19.30]  # Mumbai
# or
DEFAULT_ROI_BBOX = [80.18, 12.90, 80.32, 13.20]  # Chennai
# or
DEFAULT_ROI_BBOX = [88.20, 22.45, 88.48, 22.70]  # Kolkata
```

Or draw a custom ROI in a Jupyter notebook:
```python
from module1_data_ingestion import get_roi_from_geemap
m = get_roi_from_geemap()
m  # display map; use drawing tools
```

---

## 🔬 Research Notes

- **LST Validation**: Compare with MODIS LST (MOD11A1) for cross-sensor validation
- **Bandwidth Selection**: Default Gi* bandwidth of 0.009° ≈ 1 km at Delhi latitude. Adjust for different city scales.
- **Temporal Coverage**: The system uses March–June (pre-monsoon) dry season imagery when urban heat stress is most severe in India.
- **LULC Classes** (ESA WorldCover): 10=Tree, 20=Shrub, 30=Grass, 40=Crop, 50=Built, 60=Bare, 80=Water

---

*Built for ISRO Urban Climate Research | 2025–2026*
