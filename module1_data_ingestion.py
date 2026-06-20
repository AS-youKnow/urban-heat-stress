"""
module1_data_ingestion.py
─────────────────────────────────────────────────────────────────────────────
MODULE 1 : DATA INGESTION & GRID EXTRACTION
─────────────────────────────────────────────────────────────────────────────

Workflow:
  1. Authenticate and initialise Google Earth Engine.
  2. Accept a user-drawn ROI from geemap or fall back to the default Delhi NCR
     bounding box defined in config.py.
  3. Pull Landsat 8 Collection 2 Level-2 imagery for March–June (dry summer),
     filtered for < 15 % cloud cover, and create a median composite.
  4. Compute spectral indices:
       • LST  = ST_B10 × 0.00341802 + 149.0 − 273.15  (°C)
       • NDVI = (B5 − B4) / (B5 + B4)
       • NDBI = (B6 − B5) / (B6 + B5)
  5. Integrate ancillary layers:
       • WorldPop  → Population Density  (persons / 100 m²)
       • JRC/GHSL  → Built Surface Fraction [0–1]
       • ESA WorldCover → LULC class codes
  6. Stack all bands, sample at 100 m, convert to a clean Pandas DataFrame,
     apply quality filters, and persist as heat_grid.csv.

Run standalone:
    python module1_data_ingestion.py

Dependencies: earthengine-api, geemap, pandas, numpy
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd

# Suppress noisy deprecation warnings from third-party libs
warnings.filterwarnings("ignore")

from config import CFG, get_synthetic_dataframe

# Set up a module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Module1")

# ─── GEE INITIALISATION ───────────────────────────────────────────────────────

def init_gee(project: str = CFG.gee_project) -> bool:
    """
    Attempt to authenticate and initialise Google Earth Engine.

    Returns True if successful, False if GEE is unavailable (triggers
    synthetic-data fallback in the caller).

    Notes
    -----
    • On first run in a new environment call: ``earthengine authenticate``
      in the terminal before executing this script.
    • If a service-account JSON key is present at CFG.service_key_path,
      that is used instead of interactive authentication.
    """
    try:
        import ee  # noqa: PLC0415

        try:
            ee.Initialize(project=project)
            log.info("GEE initialised with project '%s'.", project)
            return True
        except ee.EEException:
            log.warning("GEE init failed — attempting re-auth...")
            ee.Authenticate()
            ee.Initialize(project=project)
            log.info("GEE re-authenticated and initialised.")
            return True

    except Exception as exc:  # catches ImportError, auth failures, network errors
        log.warning("GEE unavailable (%s). Switching to synthetic data.", exc)
        return False


# ─── LANDSAT COLLECTION BUILDER ───────────────────────────────────────────────

def build_landsat_composite(roi, start: str, end: str, cloud_pct: float):
    """
    Filter Landsat 8 C2 L2, mask clouds, apply SR scaling, and return a
    single median composite image clipped to *roi*.

    Parameters
    ----------
    roi         : ee.Geometry  — region of interest
    start / end : str          — 'YYYY-MM-DD' date strings
    cloud_pct   : float        — maximum allowed cloud cover percentage

    Returns
    -------
    ee.Image — median composite with bands: SR_B4, SR_B5, SR_B6, ST_B10
    """
    import ee  # noqa: PLC0415

    def mask_clouds(image):
        """Apply the QA_PIXEL cloud mask supplied with C2 L2 products."""
        qa = image.select("QA_PIXEL")
        # Bits 3 (cloud shadow) and 4 (cloud) must be 0
        cloud_shadow_bit = 1 << 3
        cloud_bit        = 1 << 4
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit).eq(0)
            .And(qa.bitwiseAnd(cloud_bit).eq(0))
        )
        return image.updateMask(mask)

    def apply_scale_factors(image):
        """
        Apply Landsat C2 L2 official scale factors:
          • Optical bands (SR_B*): multiply × 0.0000275, add −0.2
          • Thermal band (ST_B10): multiply × 0.00341802, add 149.0
        """
        optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        thermal = image.select("ST_B10").multiply(0.00341802).add(149.0)
        return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

    collection = (
        ee.ImageCollection(CFG.landsat)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
        .map(mask_clouds)
        .map(apply_scale_factors)
        .select(["SR_B4", "SR_B5", "SR_B6", "ST_B10"])
    )

    count = collection.size().getInfo()
    log.info("Landsat collection: %d images found (%s → %s).", count, start, end)
    if count == 0:
        raise ValueError(
            "No Landsat images found for the given ROI and date range. "
            "Try relaxing the cloud cover threshold or extending the date range."
        )

    composite = collection.median().clip(roi)
    return composite


# ─── SPECTRAL INDEX COMPUTATION ───────────────────────────────────────────────

def compute_indices(image):
    """
    Derive LST (°C), NDVI, and NDBI from a scaled Landsat composite.

    Band assignments (Landsat 8):
      B4  = Red (0.64–0.67 µm)
      B5  = Near-Infrared / NIR (0.85–0.88 µm)
      B6  = Short-Wave Infrared / SWIR-1 (1.57–1.65 µm)
      B10 = Thermal Infrared (already in Kelvin after scaling)
    """
    import ee  # noqa: PLC0415

    # LST: thermal band is already in Kelvin after scale-factor application;
    # subtract 273.15 to convert to Celsius.
    lst = image.select("ST_B10").subtract(273.15).rename("LST_Celsius")

    # NDVI — Normalised Difference Vegetation Index
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

    # NDBI — Normalised Difference Built-up Index
    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")

    return image.addBands([lst, ndvi, ndbi])


# ─── ANCILLARY LAYER INTEGRATION ──────────────────────────────────────────────

def get_ancillary_layers(roi):
    """
    Retrieve and prepare ancillary human-infrastructure datasets:
      • WorldPop   → population density (persons / 100 m pixel)
      • JRC GHSL   → built surface fraction (m² / 100 m²), normalised to [0,1]
      • ESA WorldCover → LULC classification codes

    All layers are clipped to *roi*.
    """
    import ee  # noqa: PLC0415

    # ── WorldPop: most recent annual composite available ─────────────────────
    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterBounds(roi)
        .sort("system:time_start", False)   # descending → latest first
        .first()
        .select("population")
        .rename("Pop_Density")
        .clip(roi)
    )

    # ── GHSL Built Surface: 2020 epoch, expressed as m² of built area per cell
    #    Divide by 10 000 to normalise into a [0,1] fraction (100 m × 100 m cell)
    ghsl = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
        .filterBounds(roi)
        .filter(ee.Filter.eq("epoch", 2020))
        .first()
        .select("built_surface")
        .divide(10_000)                     # m² → fraction
        .rename("Built_Fraction")
        .clip(roi)
    )

    # ── ESA WorldCover 2021 (10 m resolution; will be resampled at sample time)
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .rename("LULC")
        .clip(roi)
    )

    return worldpop, ghsl, worldcover


# ─── MULTI-BAND STACK & SAMPLING ──────────────────────────────────────────────

def build_and_sample_stack(roi) -> pd.DataFrame:
    """
    Combine the Landsat composite and ancillary layers into a single
    multi-band image, sample it at 100 m, and return a tidy DataFrame.
    """
    import ee  # noqa: PLC0415

    log.info("Building Landsat composite…")
    composite = build_landsat_composite(roi, CFG.start_date, CFG.end_date, CFG.cloud_pct)
    composite = compute_indices(composite)

    log.info("Fetching ancillary layers…")
    worldpop, ghsl, worldcover = get_ancillary_layers(roi)

    # Stack into a single image — GEE will resample ancillary layers to
    # match the projection/scale requested during .sample()
    stack = (
        composite
        .select(["LST_Celsius", "NDVI", "NDBI"])
        .addBands(worldpop)
        .addBands(ghsl)
        .addBands(worldcover)
    )

    log.info("Sampling stack at %d m scale (max %d points)…",
             CFG.sample_scale, CFG.max_points)

    sample = stack.sample(
        region       = roi,
        scale        = CFG.sample_scale,       # 100 m grid
        numPixels    = CFG.max_points,
        seed         = 42,
        geometries   = True,                   # include lon/lat
        dropNulls    = True,                   # skip masked pixels
    )

    # Convert to a list of dicts and then to a DataFrame
    features = sample.getInfo()["features"]
    log.info("GEE returned %d sample points.", len(features))

    rows = []
    for feat in features:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]  # [lon, lat]
        props["longitude"] = coords[0]
        props["latitude"]  = coords[1]
        rows.append(props)

    df = pd.DataFrame(rows)
    return df


# ─── DATA CLEANING & VALIDATION ───────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive cleaning of the raw sampled DataFrame:
      • Rename / standardise column names.
      • Drop rows with any NaN in critical columns.
      • Clip values to physically meaningful ranges.
      • Reset index.

    Parameters
    ----------
    df : pd.DataFrame — raw output from build_and_sample_stack or synthetic gen

    Returns
    -------
    pd.DataFrame — cleaned, validated dataframe
    """
    required_cols = ["LST_Celsius", "NDVI", "NDBI",
                     "Pop_Density", "Built_Fraction", "LULC",
                     "longitude", "latitude"]

    # Drop columns that are not in our schema (e.g. GEE metadata columns)
    extra = [c for c in df.columns if c not in required_cols]
    if extra:
        log.debug("Dropping unexpected columns: %s", extra)
        df = df.drop(columns=extra, errors="ignore")

    # Check all required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    initial_len = len(df)

    # Drop rows with NaN in any required column
    df = df.dropna(subset=required_cols)
    log.info("Dropped %d rows with NaN values (%d remain).",
             initial_len - len(df), len(df))

    # ── Physical range clipping ───────────────────────────────────────────────
    df["LST_Celsius"]   = df["LST_Celsius"].clip(lower=15.0, upper=70.0)
    df["NDVI"]          = df["NDVI"].clip(lower=-1.0, upper=1.0)
    df["NDBI"]          = df["NDBI"].clip(lower=-1.0, upper=1.0)
    df["Pop_Density"]   = df["Pop_Density"].clip(lower=0.0)
    df["Built_Fraction"]= df["Built_Fraction"].clip(lower=0.0, upper=1.0)

    # LULC must be an integer class code
    df["LULC"] = df["LULC"].round(0).astype(int)

    df = df.reset_index(drop=True)

    # Summary statistics
    log.info("Dataset summary:\n%s", df[["LST_Celsius", "NDVI", "NDBI"]].describe().to_string())

    return df


# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def run_ingestion(use_gee: bool = True,
                  roi_bbox: list = None,
                  region_name: str = None) -> pd.DataFrame:
    """
    End-to-end data ingestion pipeline.

    Parameters
    ----------
    use_gee     : bool  — If False, skip GEE and return synthetic data.
    roi_bbox    : list  — [west, south, east, north] in WGS-84.
                          Defaults to CFG.roi_bbox (Delhi NCR) when None.
    region_name : str   — Human-readable city name for climate adjustment.

    Returns
    -------
    pd.DataFrame — cleaned, analysis-ready grid dataframe
    """
    bbox   = roi_bbox or CFG.roi_bbox
    region = region_name or CFG.default_region

    if use_gee and init_gee():
        try:
            import ee  # noqa: PLC0415
            roi = ee.Geometry.Rectangle(bbox)
            df  = build_and_sample_stack(roi)
            df  = clean_dataframe(df)
            log.info("Source: Google Earth Engine (live data).")
            df.attrs["source"] = "gee"
            df.attrs["region"] = region
        except Exception as exc:
            log.error("GEE pipeline failed: %s — falling back to synthetic data.", exc)
            df = get_synthetic_dataframe(bbox=bbox, region_name=region)
    else:
        log.info("GEE disabled — generating synthetic data for '%s'.", region)
        df = get_synthetic_dataframe(bbox=bbox, region_name=region)

    # Persist to disk for downstream modules
    df.to_csv(CFG.grid_csv, index=False)
    log.info("Grid data saved -> %s  (%d rows x %d cols)",
             CFG.grid_csv, len(df), df.shape[1])

    return df



# ─── INTERACTIVE ROI HELPER ───────────────────────────────────────────────────

def get_roi_from_geemap():
    """
    Launch an interactive geemap panel for drawing a custom ROI.
    Returns the drawn geometry as an ee.Geometry, or None if drawing
    is unavailable (e.g. non-Jupyter environment).

    Usage (in a Jupyter cell):
        roi = get_roi_from_geemap()
        df  = run_ingestion(roi_bbox=roi.bounds().getInfo()['coordinates'][0])
    """
    try:
        import geemap  # noqa: PLC0415
        import ee      # noqa: PLC0415
        init_gee()

        m = geemap.Map(center=[28.65, 77.10], zoom=10)
        m.add_basemap("HYBRID")
        print("➤ Use the drawing tools on the map to define your ROI.")
        print("  After drawing, call: roi = m.draw_last_feature.geometry()")
        return m
    except ImportError:
        log.warning("geemap not installed. Using default ROI.")
        return None


# ─── STANDALONE EXECUTION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 1 : DATA INGESTION & GRID EXTRACTION")
    print("=" * 70)

    # Detect whether to attempt live GEE (pass --synthetic flag to force demo)
    force_synthetic = "--synthetic" in sys.argv

    df = run_ingestion(use_gee=not force_synthetic)

    print(f"\n✓ DataFrame shape  : {df.shape}")
    print(f"✓ Source           : {df.attrs.get('source', 'unknown')}")
    print(f"✓ Saved to         : {CFG.grid_csv}")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
