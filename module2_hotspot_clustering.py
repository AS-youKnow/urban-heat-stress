"""
module2_hotspot_clustering.py
─────────────────────────────────────────────────────────────────────────────
MODULE 2 : HOTSPOT CLUSTERING (SPATIAL STATISTICS)
─────────────────────────────────────────────────────────────────────────────

Implements the Getis-Ord Gi* local spatial statistic to identify statistically
significant urban heat clusters.

Background
──────────
The Getis-Ord Gi* statistic measures the degree to which high (or low) values
cluster spatially.  For each cell *i*:

         Σ_j w_ij · x_j  −  x̄ · Σ_j w_ij
  Gi* = ─────────────────────────────────────────────────────────────────────
          s · √[ (n·Σ_j w²_ij − (Σ_j w_ij)²) / (n−1) ]

where:
  x_j   = attribute value of neighbour j (here: LST_Celsius)
  w_ij  = spatial weight (1 if within bandwidth, 0 otherwise)
  x̄     = global mean
  s     = global standard deviation
  n     = total number of cells

Positive Gi* → high-value cluster (hotspot)
Negative Gi* → low-value cluster  (coldspot)

Z-score thresholds used:
  |z| > 2.576  → 99 % confidence
  |z| > 1.960  → 95 % confidence

Workflow:
  1. Load heat_grid.csv (or accept a DataFrame directly).
  2. Build a spatial weight matrix using a fixed-bandwidth distance threshold.
  3. Compute Gi* z-scores for every grid cell.
  4. Classify into categories and append columns to the DataFrame.
  5. Persist enriched DataFrame and print a cluster summary.

Run standalone:
    python module2_hotspot_clustering.py

Dependencies: pandas, numpy, scipy
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from config import CFG

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Module2")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
# Distance bandwidth (degrees) for defining spatial neighbours.
# ~1 km at Delhi's latitude ≈ 0.009°.  Adjust for other study areas.
DEFAULT_BANDWIDTH_DEG = 0.009

# Hard cap on matrix size to prevent memory exhaustion on large datasets
MAX_MATRIX_CELLS = 3000   # if n > this, use chunked sampling


# ─── SPATIAL WEIGHT MATRIX ────────────────────────────────────────────────────

def build_weight_matrix(coords: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Construct a binary spatial weight matrix W (n × n).

    W[i, j] = 1  if Euclidean distance(i, j) ≤ bandwidth
    W[i, j] = 0  otherwise
    Diagonal (self-weight) is kept as 1, consistent with the Gi*
    formulation that includes the focal cell in its own neighbourhood sum.

    Parameters
    ----------
    coords    : np.ndarray, shape (n, 2) — [longitude, latitude] columns
    bandwidth : float — distance threshold in the same units as coords (degrees)

    Returns
    -------
    W : np.ndarray, shape (n, n), dtype float32 (memory efficient)
    """
    log.info("Computing pairwise distances for %d cells (bandwidth=%.4f°)…",
             len(coords), bandwidth)

    # cdist computes the full pairwise distance matrix
    dist_matrix = cdist(coords, coords, metric="euclidean")

    # Binary weight: 1 within bandwidth (includes self → diagonal = 1)
    W = (dist_matrix <= bandwidth).astype(np.float32)

    log.info("Weight matrix built: %d × %d  (%.1f %% non-zero).",
             W.shape[0], W.shape[1],
             100.0 * W.sum() / W.size)
    return W


# ─── GETIS-ORD GI* COMPUTATION ────────────────────────────────────────────────

def compute_gi_star(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the Getis-Ord Gi* z-score for each cell.

    Parameters
    ----------
    values : np.ndarray, shape (n,) — attribute values (LST_Celsius)
    W      : np.ndarray, shape (n, n) — spatial weight matrix

    Returns
    -------
    gi_z : np.ndarray, shape (n,) — Gi* z-scores

    Formula
    -------
        Gi*(i) = ( Σ_j w_ij·x_j  −  x̄·Σ_j w_ij )
                 / ( s · √[ (n·Σ_j w²_ij − (Σ_j w_ij)²) / (n−1) ] )

    where x̄ and s are the global mean and standard deviation.
    """
    n    = len(values)
    xbar = np.mean(values)
    s    = np.std(values, ddof=0)   # population std (consistent with Gi* formula)

    if s == 0:
        log.warning("All LST values are identical — Gi* z-scores are undefined (0).")
        return np.zeros(n)

    # Row-wise weighted sum:  (n,) vector  ← W (n×n) · values (n,)
    W_x  = W @ values               # Σ_j w_ij · x_j  for all i simultaneously
    W_1  = W.sum(axis=1)            # Σ_j w_ij  (row sums)
    W2_1 = (W ** 2).sum(axis=1)     # Σ_j w²_ij (row sums of squared weights)

    numerator   = W_x - xbar * W_1
    denominator = s * np.sqrt(
        (n * W2_1 - W_1 ** 2) / (n - 1)
    )

    # Guard against near-zero denominator (isolated cells with no neighbours)
    with np.errstate(divide="ignore", invalid="ignore"):
        gi_z = np.where(denominator > 1e-10, numerator / denominator, 0.0)

    return gi_z.astype(np.float32)


# ─── CHUNKED VARIANT FOR LARGE DATASETS ───────────────────────────────────────

def compute_gi_star_chunked(values: np.ndarray,
                             coords: np.ndarray,
                             bandwidth: float,
                             chunk_size: int = 500) -> np.ndarray:
    """
    Memory-efficient Gi* computation for datasets with n > MAX_MATRIX_CELLS.
    Processes rows of the weight matrix in chunks to avoid allocating an
    n×n float32 array all at once.

    Parameters
    ----------
    values     : np.ndarray (n,)
    coords     : np.ndarray (n, 2)
    bandwidth  : float
    chunk_size : int — number of rows per chunk

    Returns
    -------
    gi_z : np.ndarray (n,)
    """
    n    = len(values)
    xbar = np.mean(values)
    s    = np.std(values, ddof=0)

    if s == 0:
        return np.zeros(n)

    gi_z     = np.zeros(n, dtype=np.float32)
    log.info("Chunked Gi* computation: %d cells, chunk_size=%d.", n, chunk_size)

    for start in range(0, n, chunk_size):
        end    = min(start + chunk_size, n)
        chunk  = coords[start:end]                          # (chunk, 2)
        dists  = cdist(chunk, coords, metric="euclidean")   # (chunk, n)
        W_sub  = (dists <= bandwidth).astype(np.float32)

        W_x    = W_sub @ values
        W_1    = W_sub.sum(axis=1)
        W2_1   = (W_sub ** 2).sum(axis=1)

        num    = W_x - xbar * W_1
        denom  = s * np.sqrt((n * W2_1 - W_1 ** 2) / (n - 1))

        with np.errstate(divide="ignore", invalid="ignore"):
            gi_z[start:end] = np.where(denom > 1e-10, num / denom, 0.0)

    return gi_z


# ─── CATEGORY ASSIGNMENT ──────────────────────────────────────────────────────

def categorise_hotspots(gi_z: np.ndarray) -> pd.Series:
    """
    Convert Gi* z-scores to labelled hotspot categories.

    Categories
    ----------
    "Extreme Hotspot (99%)" — z  >  2.576
    "Hotspot (95%)"         — z  >  1.960
    "Neutral"               — |z| ≤ 1.960
    "Coldspot"              — z  < -1.960

    Parameters
    ----------
    gi_z : np.ndarray — Gi* z-scores

    Returns
    -------
    pd.Series of string category labels
    """
    thresholds = CFG.hotspot_thresholds

    conditions = [
        gi_z >  thresholds["extreme"],
        gi_z >  thresholds["high"],
        gi_z < -thresholds["high"],
    ]
    choices = [
        "Extreme Hotspot (99%)",
        "Hotspot (95%)",
        "Coldspot",
    ]

    categories = np.select(conditions, choices, default="Neutral")
    return pd.Series(categories, name="hotspot_category")


# ─── MAIN CLUSTERING FUNCTION ─────────────────────────────────────────────────

def run_hotspot_analysis(df: pd.DataFrame,
                          bandwidth: float = DEFAULT_BANDWIDTH_DEG) -> pd.DataFrame:
    """
    Full Getis-Ord Gi* analysis pipeline.

    Parameters
    ----------
    df        : pd.DataFrame — must contain columns: longitude, latitude, LST_Celsius
    bandwidth : float        — spatial bandwidth in degrees

    Returns
    -------
    pd.DataFrame — input DataFrame enriched with:
        • gi_zscore        (float)  — raw Gi* z-score
        • hotspot_category (str)    — classified hotspot label
    """
    # ── Validate input ────────────────────────────────────────────────────────
    required = ["longitude", "latitude", "LST_Celsius"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Extract coordinates and LST values as numpy arrays
    coords = df[["longitude", "latitude"]].values.astype(np.float64)
    values = df["LST_Celsius"].values.astype(np.float64)

    n = len(df)
    log.info("Starting Gi* analysis on %d cells…", n)

    # ── Choose full-matrix or chunked method based on dataset size ────────────
    if n <= MAX_MATRIX_CELLS:
        W    = build_weight_matrix(coords, bandwidth)
        gi_z = compute_gi_star(values, W)
    else:
        log.info("Large dataset (%d cells) → using chunked method.", n)
        gi_z = compute_gi_star_chunked(values, coords, bandwidth)

    # Attach results to DataFrame
    df["gi_zscore"]        = gi_z
    df["hotspot_category"] = categorise_hotspots(gi_z).values

    # ── Summary report ────────────────────────────────────────────────────────
    cat_counts = df["hotspot_category"].value_counts()
    log.info("Hotspot classification summary:\n%s", cat_counts.to_string())

    return df


# ─── HEAT VULNERABILITY PROFILE ───────────────────────────────────────────────

def heat_vulnerability_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean feature values per hotspot category to build an
    urban heat vulnerability profile.

    Parameters
    ----------
    df : pd.DataFrame — output of run_hotspot_analysis()

    Returns
    -------
    pd.DataFrame — pivot table of mean feature values per category
    """
    feature_cols = ["LST_Celsius", "NDVI", "NDBI", "Pop_Density", "Built_Fraction"]
    existing     = [c for c in feature_cols if c in df.columns]

    profile = df.groupby("hotspot_category")[existing].mean().round(3)
    # Reorder rows for intuitive display
    order = ["Extreme Hotspot (99%)", "Hotspot (95%)", "Neutral", "Coldspot"]
    profile = profile.reindex([r for r in order if r in profile.index])
    return profile


# ─── STANDALONE EXECUTION ─────────────────────────────────────────────────────

def run_from_csv(csv_path: str = CFG.grid_csv) -> pd.DataFrame:
    """Load heat_grid.csv, run clustering, save enriched CSV."""
    try:
        df = pd.read_csv(csv_path)
        log.info("Loaded grid from %s (%d rows).", csv_path, len(df))
    except FileNotFoundError:
        log.warning("%s not found — generating synthetic data.", csv_path)
        from config import get_synthetic_dataframe
        df = get_synthetic_dataframe()

    df = run_hotspot_analysis(df)

    enriched_path = csv_path.replace(".csv", "_clustered.csv")
    df.to_csv(enriched_path, index=False)
    log.info("Enriched grid saved → %s", enriched_path)

    profile = heat_vulnerability_profile(df)
    print("\n── Urban Heat Vulnerability Profile ──────────────────────────")
    print(profile.to_string())
    print("──────────────────────────────────────────────────────────────\n")

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 2 : HOTSPOT CLUSTERING (Getis-Ord Gi*)")
    print("=" * 70)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else CFG.grid_csv
    df_out   = run_from_csv(csv_path)

    print(f"\n✓ Classified {len(df_out):,} grid cells.")
    print(f"✓ Category counts:\n{df_out['hotspot_category'].value_counts().to_string()}")
