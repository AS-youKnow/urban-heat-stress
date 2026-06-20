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
# Step 1: Create a GCP project at https://console.cloud.google.com
# Step 2: Enable the Earth Engine API for your project
# Step 3: Replace the project ID below with your actual GCP project ID
GEE_PROJECT_ID = "your-gee-project-id"   # ← REPLACE THIS

# For local use:  run  `earthengine authenticate`  in terminal (one-time setup)
# For Streamlit Cloud: set up a Service Account (see README for instructions)
# Path to local service account JSON key (leave as None to use earthengine authenticate)
GEE_SERVICE_ACCOUNT_KEY = None           # e.g. r"C:\keys\gee-key.json"


# ─── 2. WORLD REGIONS — Major cities across all continents ────────────────────
# Format: "City, Country": [west, south, east, north]  (WGS-84 lon/lat)
WORLD_REGIONS = {
    # ── ASIA ────────────────────────────────────────────────────────────────────
    "Delhi, India"           : [76.84, 28.40, 77.35, 28.88],
    "Mumbai, India"          : [72.77, 18.85, 73.10, 19.30],
    "Kolkata, India"         : [88.20, 22.45, 88.48, 22.72],
    "Chennai, India"         : [80.18, 12.90, 80.35, 13.18],
    "Bangalore, India"       : [77.45, 12.85, 77.75, 13.10],
    "Hyderabad, India"       : [78.35, 17.30, 78.60, 17.55],
    "Ahmedabad, India"       : [72.48, 22.95, 72.70, 23.15],
    "Jaipur, India"          : [75.72, 26.82, 75.95, 27.02],
    "Tokyo, Japan"           : [139.60, 35.55, 139.90, 35.80],
    "Beijing, China"         : [116.20, 39.80, 116.60, 40.10],
    "Shanghai, China"        : [121.35, 31.10, 121.65, 31.40],
    "Karachi, Pakistan"      : [66.90, 24.78, 67.20, 25.05],
    "Lahore, Pakistan"       : [74.25, 31.40, 74.50, 31.65],
    "Dhaka, Bangladesh"      : [90.30, 23.68, 90.50, 23.88],
    "Bangkok, Thailand"      : [100.45, 13.65, 100.70, 13.90],
    "Jakarta, Indonesia"     : [106.72, -6.30, 107.00, -6.05],
    "Manila, Philippines"    : [120.95, 14.50, 121.15, 14.70],
    "Singapore"              : [103.65, 1.22,  104.00, 1.48],
    "Seoul, South Korea"     : [126.85, 37.45, 127.15, 37.65],
    "Riyadh, Saudi Arabia"   : [46.55, 24.55, 46.85, 24.80],
    "Dubai, UAE"             : [55.15, 25.05, 55.45, 25.30],
    "Tehran, Iran"           : [51.25, 35.60, 51.55, 35.80],
    "Kabul, Afghanistan"     : [69.08, 34.45, 69.30, 34.65],
    "Colombo, Sri Lanka"     : [79.82, 6.82,  80.00, 7.00],
    # ── AFRICA ──────────────────────────────────────────────────────────────────
    "Cairo, Egypt"           : [31.15, 29.95, 31.45, 30.15],
    "Lagos, Nigeria"         : [3.30,  6.40,  3.55,  6.60],
    "Nairobi, Kenya"         : [36.70, -1.40, 36.92, -1.20],
    "Johannesburg, S.Africa" : [27.95, -26.30,28.15, -26.10],
    "Khartoum, Sudan"        : [32.45, 15.50, 32.65, 15.65],
    "Kinshasa, DR Congo"     : [15.25, -4.45, 15.45, -4.25],
    "Addis Ababa, Ethiopia"  : [38.68, 8.95,  38.88, 9.10],
    "Accra, Ghana"           : [-0.30, 5.50,  -0.10, 5.65],
    # ── EUROPE ──────────────────────────────────────────────────────────────────
    "London, UK"             : [-0.25, 51.40, 0.05,  51.60],
    "Paris, France"          : [2.25,  48.80, 2.45,  48.95],
    "Madrid, Spain"          : [-3.80, 40.35, -3.60, 40.50],
    "Rome, Italy"            : [12.40, 41.80, 12.60, 41.95],
    "Athens, Greece"         : [23.65, 37.90, 23.85, 38.05],
    "Istanbul, Turkey"       : [28.85, 40.95, 29.15, 41.15],
    "Moscow, Russia"         : [37.50, 55.65, 37.80, 55.85],
    # ── AMERICAS ────────────────────────────────────────────────────────────────
    "New York, USA"          : [-74.10, 40.60, -73.85, 40.80],
    "Los Angeles, USA"       : [-118.45,33.95, -118.15,34.15],
    "Chicago, USA"           : [-87.75, 41.75, -87.55, 41.95],
    "Houston, USA"           : [-95.55, 29.65, -95.25, 29.85],
    "Mexico City, Mexico"    : [-99.25, 19.35, -99.05, 19.50],
    "Sao Paulo, Brazil"      : [-46.75, -23.65,-46.55,-23.45],
    "Rio de Janeiro, Brazil" : [-43.30, -23.00,-43.10,-22.85],
    "Buenos Aires, Argentina": [-58.55, -34.70,-58.35,-34.55],
    "Bogota, Colombia"       : [-74.20,  4.55, -74.00,  4.75],
    "Lima, Peru"             : [-77.15, -12.15,-76.95,-11.95],
    # ── AUSTRALIA / OCEANIA ─────────────────────────────────────────────────────
    "Sydney, Australia"      : [150.95,-33.95, 151.25,-33.75],
    "Melbourne, Australia"   : [144.85,-37.90, 145.10,-37.70],
}

# Default ROI — Delhi NCR (used as fallback)
DEFAULT_ROI_BBOX = WORLD_REGIONS["Delhi, India"]
DEFAULT_REGION   = "Delhi, India"


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
def get_synthetic_dataframe(n_points: int = 2000,
                             seed: int = 42,
                             bbox: list = None,
                             region_name: str = None) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset that mirrors the schema produced
    by Module 1 (GEE ingestion). Used when GEE credentials are unavailable
    or for rapid prototyping / CI testing.

    Parameters
    ----------
    n_points    : int  — Number of grid-cell rows to synthesise.
    seed        : int  — Random seed for reproducibility.
    bbox        : list — [west, south, east, north] bounding box.
                         Defaults to DEFAULT_ROI_BBOX (Delhi NCR) when None.
    region_name : str  — Human-readable city name (used to adjust climate).

    Returns
    -------
    pd.DataFrame with columns:
        longitude, latitude, LST_Celsius, NDVI, NDBI,
        Pop_Density, Built_Fraction, LULC
    """
    rng = np.random.default_rng(seed)

    # Use provided bbox or fall back to default
    active_bbox = bbox if bbox is not None else DEFAULT_ROI_BBOX

    # Spatial grid within selected bounding box
    lon = rng.uniform(active_bbox[0], active_bbox[2], n_points)
    lat = rng.uniform(active_bbox[1], active_bbox[3], n_points)

    # ── Climate-aware LST baseline ────────────────────────────────────────────
    # Centre latitude of the selected city
    centre_lat = (active_bbox[1] + active_bbox[3]) / 2.0
    abs_lat    = abs(centre_lat)

    # Base temperature by latitude zone (realistic climatology):
    #   Equatorial (0-15°)  → 32-40°C  e.g. Lagos, Jakarta, Singapore
    #   Tropical   (15-30°) → 35-45°C  e.g. Delhi, Cairo, Riyadh (peak summer)
    #   Sub-tropical(30-40°)→ 28-40°C  e.g. Tokyo, Beijing, Los Angeles
    #   Temperate  (40-55°) → 20-32°C  e.g. London, Paris, New York
    #   Cold       (55°+)   → 10-22°C  e.g. Moscow
    if abs_lat < 15:
        lst_base_val = 36.0   # Equatorial — hot & humid
    elif abs_lat < 30:
        lst_base_val = 40.0   # Tropical / arid — hottest zone
    elif abs_lat < 40:
        lst_base_val = 34.0   # Sub-tropical
    elif abs_lat < 55:
        lst_base_val = 26.0   # Temperate
    else:
        lst_base_val = 18.0   # Cold / subarctic

    # Southern Hemisphere cities tend to have lower peak summer temps
    if centre_lat < 0:
        lst_base_val -= 3.0

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

    # LST (Celsius): physically motivated formula + city-climate baseline
    lst_base = (
        lst_base_val
        - 12.0 * ndvi       # vegetated areas are cooler
        + 8.0  * ndbi       # built surfaces are hotter
        + 0.00003 * pop_density
        + 5.0  * built_fraction
        + rng.normal(0, 1.5, n_points)   # measurement noise
    )
    # Urban heat island: slight spatial clustering effect
    # Clip range varies by climate zone
    lst_min_clip = max(5.0,  lst_base_val - 20.0)
    lst_max_clip = min(70.0, lst_base_val + 22.0)
    lst_celsius = np.clip(lst_base, lst_min_clip, lst_max_clip)

    df = pd.DataFrame({
        "longitude"     : lon,
        "latitude"      : lat,
        TARGET_COL      : lst_celsius,
        "NDVI"          : ndvi,
        "NDBI"          : ndbi,
        "Pop_Density"   : pop_density,
        "Built_Fraction": built_fraction,
        "LULC"          : lulc.astype(float),
    })

    # Store metadata
    df.attrs["source"]      = "synthetic"
    df.attrs["region"]      = region_name or "Unknown"
    df.attrs["bbox"]        = active_bbox
    df.attrs["lst_baseline"]= round(lst_base_val, 1)
    return df


# ─── 13. CONSOLIDATED SETTINGS OBJECT ─────────────────────────────────────────
class CFG:
    """Namespace-style config object for convenient dot-access."""
    gee_project            = GEE_PROJECT_ID
    gee_service_account_key= str(GEE_SERVICE_ACCOUNT_KEY)
    default_region         = DEFAULT_REGION
    world_regions          = WORLD_REGIONS
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
