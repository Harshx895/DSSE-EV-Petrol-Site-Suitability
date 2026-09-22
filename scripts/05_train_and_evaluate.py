"""
scripts/05_train_and_evaluate.py
----------------------------------
DSSE Phase 2 — Model Training & Evaluation

Orchestrates the full ML pipeline for one city:
  1. Load stations_master.csv + extract 16 spatial features
  2. Generate background/unlabeled points (H3 hex centroids)
  3. Compute ACTI regression targets
  4. Train PU classifiers + conformal regressors (Spatial Block CV k=5)
  5. Extract SHAP feature importances
  6. Compute SFDS + 95% bootstrap CI
  7. Generate UMAP scatter + weights comparison figure
  8. Log everything to MLflow

Outputs:
  models/{city}/{city}_pu_petrol.pkl
  models/{city}/{city}_pu_ev.pkl
  models/{city}/{city}_reg_petrol.pkl
  models/{city}/{city}_reg_ev.pkl
  models/{city}/shap_importances.json
  figures/sfds_weights_{city}.png
  figures/umap_feature_space_{city}.png
  data/processed/{city}/sfds_results_{city}.json

Usage:
  python -X utf8 scripts/05_train_and_evaluate.py --city pune
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import mlflow
import shap

from src.models.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.models.target_engineering import (
    compute_acti,
    generate_background_points,
    AHP_WEIGHTS_PETROL,
    AHP_WEIGHTS_EV,
)
from src.models.trainer import train_city
from src.models.sfds_engine import (
    compute_sfds,
    plot_sfds_comparison,
    plot_umap_scatter,
    save_sfds_results,
)


def extract_features_for_df(df: pd.DataFrame, extractor: FeatureExtractor) -> np.ndarray:
    """Batch-extract 16 features for all rows in df."""
    coords = df[["lat", "lon"]].values
    print(f"  Extracting features for {len(coords)} points...")
    X = extractor.extract_batch(coords)
    return X


def get_shap_importances(model, X_pos: np.ndarray) -> np.ndarray:
    """
    Compute mean |SHAP| values for the positive class from a PUClassifier.
    Returns array of shape (16,) normalized to sum=1.
    """
    try:
        explainer = shap.TreeExplainer(model.clf)
        X_scaled  = model.scaler_.transform(X_pos)
        sv = explainer.shap_values(X_scaled)
        # XGBoost binary: sv is (N, F) for positive class
        vals = np.abs(sv).mean(axis=0)
        return vals / (vals.sum() + 1e-12)
    except Exception as e:
        print(f"    SHAP extraction failed: {e}")
        return model.feature_importances_


def main(city: str) -> None:
    print(f"\n{'='*60}")
    print(f"  DSSE Phase 2 — Model Training: {city.upper()}")
    print(f"{'='*60}\n")

    processed_dir = Path("data/processed") / city
    models_dir    = Path("models") / city
    figures_dir   = Path("figures")
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load station data ─────────────────────────────────────────────────
    print("[Step 1] Loading station data...")
    master = pd.read_csv(processed_dir / "stations_master.csv")

    df_petrol = master[master["ev_label"] == "petrol"].reset_index(drop=True)
    df_ev_pos = master[master["ev_label"] == "commercial"].reset_index(drop=True)
    # For EV regressor, include candidates with ratings
    df_ev_all = master[master["ev_label"].isin(["commercial","candidate"])].reset_index(drop=True)

    print(f"  Petrol stations : {len(df_petrol)}")
    print(f"  EV commercial   : {len(df_ev_pos)}  (PU positive class)")
    print(f"  EV candidates   : {len(df_ev_all)}  (regressor pool)")

    # ── 2. Feature extraction ────────────────────────────────────────────────
    print("\n[Step 2] Extracting spatial features...")
    extractor = FeatureExtractor(city, Path("models"), processed_dir)

    X_petrol      = extract_features_for_df(df_petrol, extractor)
    X_ev_pos      = extract_features_for_df(df_ev_pos, extractor)
    X_ev_all      = extract_features_for_df(df_ev_all, extractor)
    coords_petrol = df_petrol[["lat","lon"]].values
    coords_ev_pos = df_ev_pos[["lat","lon"]].values
    coords_ev_all = df_ev_all[["lat","lon"]].values

    # ── 3. Background points ─────────────────────────────────────────────────
    print("\n[Step 3] Generating background/unlabeled points (H3 res-9)...")
    bg_coords = generate_background_points(city, n_points=2000)
    X_bg      = extractor.extract_batch(bg_coords)
    print(f"  Background points: {len(X_bg)}")

    # ── 4. ACTI targets ──────────────────────────────────────────────────────
    print("\n[Step 4] Computing ACTI regression targets...")
    # Note: opening_date_est from Google review timestamps (oldest of 5 recent reviews)
    # is an unreliable age proxy. ACTI = Y_raw = log10(1+reviews)*(rating/5) only.
    # Age correction documented as a limitation requiring operator-reported opening dates.
    acti_petrol = compute_acti(df_petrol).values
    acti_ev     = compute_acti(df_ev_all).values
    n_rated_p   = np.sum(~np.isnan(acti_petrol))
    n_rated_e   = np.sum(~np.isnan(acti_ev))
    print(f"  Petrol rated: {n_rated_p}/{len(df_petrol)}")
    print(f"  EV rated    : {n_rated_e}/{len(df_ev_all)}")

    # ── 5. MLflow run ────────────────────────────────────────────────────────
    mlflow.set_experiment(f"dsse_{city}")
    with mlflow.start_run(run_name=f"{city}_phase2"):
        mlflow.log_params({
            "city": city,
            "n_petrol_pos": len(X_petrol),
            "n_ev_pos":     len(X_ev_pos),
            "n_background": len(X_bg),
            "n_features":   len(FEATURE_NAMES),
            "spatial_cv_k": 5,
            "pu_calibration": "elkan_noto",
            "conformal_alpha": 0.10,
        })

        # ── 6. Train all 4 models ─────────────────────────────────────────────
        print("\n[Step 5] Training models (Spatial Block CV k=5)...")
        report = train_city(
            city=city,
            X_petrol=X_petrol,
            X_ev=X_ev_pos,
            X_ev_reg=X_ev_all,
            X_bg=X_bg,
            coords_petrol=coords_petrol,
            coords_ev=coords_ev_pos,
            coords_ev_reg=coords_ev_all,
            acti_petrol=acti_petrol,
            acti_ev=acti_ev,                  # all rated EV candidates
            out_dir=models_dir,
        )

        # Log CV metrics
        for key, val in report.items():
            if isinstance(val, dict):
                for mk, mv in val.items():
                    if isinstance(mv, float):
                        mlflow.log_metric(f"{key}/{mk}", mv)

        # ── 7. SHAP importances ───────────────────────────────────────────────
        print("\n[Step 6] Computing SHAP feature importances...")
        import pickle
        with open(models_dir / f"{city}_pu_petrol.pkl", "rb") as f:
            pu_petrol = pickle.load(f)
        with open(models_dir / f"{city}_pu_ev.pkl", "rb") as f:
            pu_ev = pickle.load(f)

        w_petrol = get_shap_importances(pu_petrol, X_petrol)
        w_ev     = get_shap_importances(pu_ev, X_ev_pos)

        shap_out = {
            "feature_names":    FEATURE_NAMES,
            "weights_petrol":   w_petrol.tolist(),
            "weights_ev":       w_ev.tolist(),
            "ahp_weights_petrol": list(AHP_WEIGHTS_PETROL.values()),
            "ahp_weights_ev":     list(AHP_WEIGHTS_EV.values()),
        }
        with open(models_dir / "shap_importances.json", "w") as f:
            json.dump(shap_out, f, indent=2)
        print("  Saved → models/{city}/shap_importances.json")

        # ── 8. SFDS ──────────────────────────────────────────────────────────
        print("\n[Step 7] Computing SFDS...")
        sfds_result = compute_sfds(w_petrol, w_ev, n_bootstrap=1000)
        save_sfds_results(sfds_result, city, processed_dir)

        mlflow.log_metrics({
            "sfds_point": sfds_result["sfds"],
            "sfds_ci_low":  sfds_result["ci_low"],
            "sfds_ci_high": sfds_result["ci_high"],
        })
        print(f"\n  >>> {sfds_result['report_str']}")
        print(f"  >>> {sfds_result['interpretation']}")

        # ── 9. Figures ────────────────────────────────────────────────────────
        print("\n[Step 8] Generating figures...")
        plot_sfds_comparison(sfds_result, city, figures_dir)
        plot_umap_scatter(X_petrol, X_ev_pos, city, figures_dir)

        mlflow.log_artifacts(str(figures_dir), artifact_path="figures")
        mlflow.log_artifact(str(models_dir / "train_report.json"))

    print(f"\n{'='*60}")
    print(f"  Phase 2 complete for {city.upper()}!")
    print(f"  SFDS = {sfds_result['sfds']:.4f} "
          f"[95% CI: {sfds_result['ci_low']:.4f}–{sfds_result['ci_high']:.4f}]")
    print(f"\n  Models saved to: models/{city}/")
    print(f"  Figures  saved to: figures/")
    print(f"\n  Next: python -X utf8 scripts/06_cache_heatmap.py --city {city}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    args = parser.parse_args()
    main(args.city)
