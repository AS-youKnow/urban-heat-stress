"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the AI/ML Urban Heat Stress Hotspot Prediction System.

Contains:
  - Google Earth Engine project / auth settings
  - Default Region-of-Interest (Delhi NCR bounding box)
  - Band scaling constants
  - ML hyperparameters
  - Synthetic data generator (offline / demo mode fallback)

Usage:
    from config import CFG, get_synthetic_dataframe
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd

# ─── 1. GOOGLE EARTH ENGINE SETTINGS ──────────────────────────────────────────
GEE_PROJECT_ID = "your-gee-project-id"   # ← Replace with your GCP project ID
GEE_SERVICE_ACCOUNT_KEY = None            # Path to JSON key file (optional)

# ─── 2. DEFAULT REGION OF INTEREST (Delhi NCR bounding box) ───────────────────
# Coordinates: [west, south, east, north] (WGS-84 longitude/latitude)
DEFAULT_ROI_BBOX = [76.84, 28.40, 77.35, 28.88]   # Delhi NCR

# ─── 3. IMAGERY SETTINGS ──────────────────────────────────────────────────────
LANDSAT_COLLECTION   = "LANDSAT/LC08/C02/T1_L2"    # Landsat 8 C2 L2
START_DATE           = "2024-03-01"
END_DATE             = "2024-06-30"
CLOUD_COVER_PCT      = 15                           # Max cloud cover (%)

# WorldPop, GHSL, WorldCover dataset IDs
WORLDPOP_DATASET     = "WorldPop/GP/100m/pop"
GHSL_DATASET         = "JRC/GHSL/P2023A/GHS_BUILT_S"
ESA_WORLDCOVER       = "ESA/WorldCover/v200"

# ─── 4. BAND / INDEX CONSTANTS ────────────────────────────────────────────────
# LST from Landsat ST_B10:  DN * scale_factor + offset − 273.15  → Celsius
LST_SCALE_FACTOR     = 0.00341802
LST_OFFSET           = 149.0
LST_KELVIN_OFFSET    = 273.15

# NDVI = (NIR - Red) / (NIR + Red)   [B5, B4 for Landsat 8]
# NDBI = (SWIR - NIR) / (SWIR + NIR) [B6, B5 for Landsat 8]

# ─── 5. SAMPLING / GRID SETTINGS ──────────────────────────────────────────────
SAMPLE_SCALE_M       = 100          # Grid resolution in metres
MAX_SAMPLE_POINTS    = 5000         # Cap on GEE sample count

# ─── 6. COLUMN NAMES (schema shared across all modules) ───────────────────────
# Raw GEE band outputs → rename map
BAND_RENAME_MAP = {
    "ST_B10"          : "LST_raw",
    "SR_B4"           : "SR_B4",
    "SR_B5"           : "SR_B5",
    "SR_B6"           : "SR_B6",
    "population"      : "Pop_Density",
    "built_surface"   : "Built_Fraction",
    "Map"             : "LULC",
}

FEATURE_COLS = ["NDVI", "NDBI", "Pop_Density", "Built_Fraction", "LULC"]
TARGET_COL   = "LST_Celsius"
GEO_COLS     = ["longitude", "latitude"]

# ─── 7. XGBOOST HYPERPARAMETERS ───────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"     : 400,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "tree_method"      : "hist",
}

# ─── 8. HOTSPOT Z-SCORE THRESHOLDS ────────────────────────────────────────────
HOTSPOT_THRESHOLDS = {
    "extreme" : 2.576,    # 99% confidence
    "high"    : 1.960,    # 95% confidence
    "cold"    : -1.960,   # Coldspot boundary
}

# ─── 9. OUTPUT FILE PATHS ─────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
OUTPUT_GRID_CSV    = os.path.join(BASE_DIR, "heat_grid.csv")
MODEL_PATH         = os.path.join(BASE_DIR, "xgb_lst_model.json")
SCENARIO_CSV       = os.path.join(BASE_DIR, "scenario_results.csv")
OPTIM_CSV          = os.path.join(BASE_DIR, "optimization_results.csv")
SHAP_BEESWARM_PNG  = os.path.join(BASE_DIR, "shap_summary_beeswarm.png")
SHAP_BAR_PNG       = os.path.join(BASE_DIR, "shap_feature_importance.png")

# ─── 10. INTERVENTION SCENARIO DEFINITIONS ────────────────────────────────────
SCENARIOS = {
    "Urban Greening"   : {"NDVI": +0.20, "NDBI": -0.05},
    "Cool Roofs"       : {"NDVI":  0.00, "NDBI": -0.25},
    "Mixed Strategy"   : {"NDVI": +0.15, "NDBI": -0.15},
}

# ─── 11. OPTIMIZATION WEIGHTS ─────────────────────────────────────────────────
# Priority score = ALPHA * (LST severity) + BETA * (cooling sensitivity)
ALPHA_SEVERITY   = 0.6
BETA_SENSITIVITY = 0.4
DEFAULT_BUDGET_N = 50    # Default number of target cells to optimise

# ─── 12. SYNTHETIC DATA GENERATOR (offline / demo mode) ───────────────────────
def get_synthetic_dataframe(n_points: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset that mirrors the schema produced
    by Module 1 (GEE ingestion).  Used when GEE credentials are unavailable
    or for rapid prototyping / CI testing.

    Parameters
    ----------
    n_points : int
        Number of grid-cell rows to synthesise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        longitude, latitude, LST_Celsius, NDVI, NDBI,
        Pop_Density, Built_Fraction, LULC
    """
    rng = np.random.default_rng(seed)

    # Spatial grid within Delhi NCR bounding box
    lon = rng.uniform(DEFAULT_ROI_BBOX[0], DEFAULT_ROI_BBOX[2], n_points)
    lat = rng.uniform(DEFAULT_ROI_BBOX[1], DEFAULT_ROI_BBOX[3], n_points)

    # ── Feature synthesis with realistic urban correlations ──────────────────
    # NDVI: vegetation index [−1, 1]; lower in dense urban cores
    ndvi = rng.beta(2, 5, n_points) * 0.8 - 0.1          # skewed toward lower values

    # NDBI: built-up index [−1, 1]; higher in commercial / industrial zones
    ndbi = rng.beta(3, 4, n_points) * 0.9 - 0.3

    # Population density (persons / km²): log-normal distribution
    pop_density = rng.lognormal(mean=7.5, sigma=1.2, size=n_points)
    pop_density = np.clip(pop_density, 100, 80_000)

    # Built Surface Fraction [0, 1]
    built_fraction = np.clip(0.6 - 0.5 * ndvi + 0.3 * ndbi
                             + rng.normal(0, 0.08, n_points), 0.0, 1.0)

    # LULC classes (ESA WorldCover codes simplified):
    # 10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland,
    # 50=Built-up, 60=Bare/sparse, 80=Water
    lulc_choices = [10, 20, 30, 40, 50, 60, 80]
    lulc_weights = [0.05, 0.03, 0.05, 0.15, 0.55, 0.10, 0.07]
    lulc = rng.choice(lulc_choices, size=n_points, p=lulc_weights)

    # LST (Celsius): physically motivated formula + noise
    # Higher LST where: high NDBI, low NDVI, high built fraction
    lst_base = (
        38.0
        - 12.0 * ndvi       # vegetated areas are cooler
        + 8.0  * ndbi       # built surfaces are hotter
        + 0.00003 * pop_density
        + 5.0  * built_fraction
        + rng.normal(0, 1.5, n_points)   # measurement noise
    )
    # Urban heat island: slight spatial clustering effect
    lst_celsius = np.clip(lst_base, 28.0, 62.0)

    df = pd.DataFrame({
        "longitude"     : lon,
        "latitude"      : lat,
        TARGET_COL      : lst_celsius,          # "LST_Celsius"
        "NDVI"          : ndvi,
        "NDBI"          : ndbi,
        "Pop_Density"   : pop_density,
        "Built_Fraction": built_fraction,
        "LULC"          : lulc.astype(float),
    })

    # Flag as synthetic so downstream modules can log a warning
    df.attrs["source"] = "synthetic"
    return df


# ─── 13. CONSOLIDATED SETTINGS OBJECT ─────────────────────────────────────────
class CFG:
    """Namespace-style config object for convenient dot-access."""
    gee_project        = GEE_PROJECT_ID
    roi_bbox           = DEFAULT_ROI_BBOX
    landsat            = LANDSAT_COLLECTION
    start_date         = START_DATE
    end_date           = END_DATE
    cloud_pct          = CLOUD_COVER_PCT
    lst_scale          = LST_SCALE_FACTOR
    lst_offset         = LST_OFFSET
    lst_k_offset       = LST_KELVIN_OFFSET
    sample_scale       = SAMPLE_SCALE_M
    max_points         = MAX_SAMPLE_POINTS
    feature_cols       = FEATURE_COLS
    target_col         = TARGET_COL
    geo_cols           = GEO_COLS
    xgb_params         = XGB_PARAMS
    hotspot_thresholds = HOTSPOT_THRESHOLDS
    scenarios          = SCENARIOS
    alpha              = ALPHA_SEVERITY
    beta               = BETA_SENSITIVITY
    budget_n           = DEFAULT_BUDGET_N
    # output paths
    grid_csv           = OUTPUT_GRID_CSV
    model_path         = MODEL_PATH
    scenario_csv       = SCENARIO_CSV
    optim_csv          = OPTIM_CSV
    shap_beeswarm      = SHAP_BEESWARM_PNG
    shap_bar           = SHAP_BAR_PNG
