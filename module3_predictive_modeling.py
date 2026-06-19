"""
module3_predictive_modeling.py
─────────────────────────────────────────────────────────────────────────────
MODULE 3 : PREDICTIVE MODELING (XGBoost LST Regressor)
─────────────────────────────────────────────────────────────────────────────

Workflow:
  1. Load heat_grid.csv (or accept a DataFrame directly).
  2. Prepare feature matrix X and target vector y.
  3. Split into 80 % train / 20 % test with a reproducible random seed.
  4. Train an XGBoost Regressor with the hyperparameters in config.py.
  5. Evaluate on the held-out test set, printing MAE, RMSE, and R² Score.
  6. Persist the trained model to xgb_lst_model.json.
  7. Return the trained model, feature sets, and predictions for use by
     Module 4 (SHAP) and Modules 5–6 (simulation & optimisation).

Run standalone:
    python module3_predictive_modeling.py

Dependencies: pandas, numpy, scikit-learn, xgboost
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import CFG, get_synthetic_dataframe

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Module3")


# ─── DATA PREPARATION ─────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """
    Extract and validate feature matrix and target vector.

    Parameters
    ----------
    df : pd.DataFrame — cleaned grid data (output of Module 1)

    Returns
    -------
    X : pd.DataFrame — feature matrix (NDVI, NDBI, Pop_Density,
                                        Built_Fraction, LULC)
    y : pd.Series    — target vector  (LST_Celsius)
    """
    # Validate all required columns are present
    all_required = CFG.feature_cols + [CFG.target_col]
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    X = df[CFG.feature_cols].copy()
    y = df[CFG.target_col].copy()

    # Defensively handle any remaining NaN values (should be 0 after Module 1 cleaning)
    initial_len = len(X)
    valid_mask  = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < initial_len:
        log.warning("Removed %d rows with NaN in features/target.",
                    initial_len - len(X))

    log.info("Feature matrix: %d rows × %d features.", len(X), len(CFG.feature_cols))
    log.info("Target range: %.2f – %.2f °C (mean %.2f °C).",
             y.min(), y.max(), y.mean())

    return X, y


# ─── TRAIN/TEST SPLIT ─────────────────────────────────────────────────────────

def spatial_train_test_split(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
                              test_size: float = 0.2):
    """
    80/20 split that preserves the shared index between X, y, and the full df
    so that downstream modules can recover geographic coordinates.

    Parameters
    ----------
    X         : feature DataFrame
    y         : target Series
    df        : full DataFrame (for preserving geo coordinates)
    test_size : fraction for test set

    Returns
    -------
    X_train, X_test, y_train, y_test, df_train, df_test
    """
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)
    df_train= df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    log.info("Train set: %d rows | Test set: %d rows.", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, df_train, df_test


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Train an XGBoost Regressor on the training set.

    Hyperparameters are defined in config.CFG.xgb_params for easy tuning.
    Early stopping is disabled here to keep the interface simple; for
    production use, pass eval_set and early_stopping_rounds.

    Parameters
    ----------
    X_train : pd.DataFrame — training feature matrix
    y_train : pd.Series    — training target values

    Returns
    -------
    XGBRegressor — fitted model
    """
    log.info("Training XGBoost Regressor …")
    log.info("Hyperparameters: %s", CFG.xgb_params)

    model = XGBRegressor(**CFG.xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    log.info("Training complete. Best iteration: %s",
             getattr(model, "best_iteration", "N/A"))
    return model


# ─── MODEL EVALUATION ─────────────────────────────────────────────────────────

def evaluate_model(model: XGBRegressor,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    Compute and display MAE, RMSE, and R² on the test set.

    Parameters
    ----------
    model  : trained XGBRegressor
    X_test : pd.DataFrame — test feature matrix
    y_test : pd.Series    — true LST values

    Returns
    -------
    dict with keys: 'MAE', 'RMSE', 'R2', 'y_pred'
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "y_pred": y_pred}

    print("\n" + "=" * 50)
    print("  MODEL PERFORMANCE ON TEST SET")
    print("=" * 50)
    print(f"  MAE   (Mean Absolute Error)   : {mae:.4f} °C")
    print(f"  RMSE  (Root Mean Square Error) : {rmse:.4f} °C")
    print(f"  R²    (Coefficient of Det.)   : {r2:.4f}")
    print("=" * 50 + "\n")

    log.info("Evaluation complete: MAE=%.4f, RMSE=%.4f, R²=%.4f", mae, rmse, r2)
    return metrics


# ─── RESIDUAL ANALYSIS ────────────────────────────────────────────────────────

def compute_residuals(y_test: pd.Series, y_pred: np.ndarray) -> pd.Series:
    """
    Compute prediction residuals (actual − predicted).

    Useful for spatial plotting of under/over-prediction zones.
    """
    residuals = pd.Series(y_test.values - y_pred, name="residual")
    log.info("Residuals — mean: %.4f, std: %.4f, max: %.4f",
             residuals.mean(), residuals.std(), residuals.abs().max())
    return residuals


# ─── MODEL PERSISTENCE ────────────────────────────────────────────────────────

def save_model(model: XGBRegressor, path: str = CFG.model_path) -> None:
    """Save trained XGBoost model in the portable JSON format."""
    model.save_model(path)
    log.info("Model saved → %s", path)


def load_model(path: str = CFG.model_path) -> XGBRegressor:
    """Load a previously saved XGBoost model from disk."""
    model = XGBRegressor()
    model.load_model(path)
    log.info("Model loaded from %s", path)
    return model


# ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def run_modeling(df: pd.DataFrame = None) -> dict:
    """
    End-to-end modeling pipeline.

    Parameters
    ----------
    df : pd.DataFrame | None
        If None, attempts to load heat_grid.csv; falls back to synthetic data.

    Returns
    -------
    dict containing:
        model     : XGBRegressor      — trained model
        X_train   : pd.DataFrame
        X_test    : pd.DataFrame
        y_train   : pd.Series
        y_test    : pd.Series
        df_train  : pd.DataFrame      — full rows for train set (with coords)
        df_test   : pd.DataFrame      — full rows for test set  (with coords)
        metrics   : dict              — MAE, RMSE, R2, y_pred
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    if df is None:
        try:
            df = pd.read_csv(CFG.grid_csv)
            log.info("Loaded grid from %s (%d rows).", CFG.grid_csv, len(df))
        except FileNotFoundError:
            log.warning("heat_grid.csv not found — using synthetic data.")
            df = get_synthetic_dataframe()

    # ── Prepare, split, train, evaluate ───────────────────────────────────────
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test, df_train, df_test = spatial_train_test_split(
        X, y, df
    )

    model   = train_xgboost(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # ── Attach predictions and residuals to test DataFrame ────────────────────
    df_test = df_test.copy()
    df_test["LST_Predicted"]  = metrics["y_pred"]
    df_test["Residual"]       = df_test["LST_Celsius"] - df_test["LST_Predicted"]

    # ── Save model ────────────────────────────────────────────────────────────
    save_model(model)

    return {
        "model"   : model,
        "X_train" : X_train,
        "X_test"  : X_test,
        "y_train" : y_train,
        "y_test"  : y_test,
        "df_train": df_train,
        "df_test" : df_test,
        "metrics" : metrics,
    }


# ─── STANDALONE EXECUTION ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 3 : PREDICTIVE MODELING (XGBoost)")
    print("=" * 70)

    results = run_modeling()
    model   = results["model"]
    metrics = results["metrics"]

    print(f"\n✓ Model saved to : {CFG.model_path}")
    print(f"✓ Test set size  : {len(results['X_test'])} rows")
    print(f"✓ Feature importances (gain):")
    fi = dict(zip(CFG.feature_cols, model.feature_importances_))
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {feat:<20s} : {imp:.4f}")
