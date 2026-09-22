"""
src/models/trainer.py
-----------------------
Trains city-specific expert models for Formulations 1 & 2:

  Formulation 1 — PU Viability Classifier
    Model:    XGBoostClassifier wrapped with Elkan-Noto PU calibration
    Protocol: Spatial Block CV (k=5)
    Output:   {city}_pu_{type}.pkl

  Formulation 2 — Footfall Intensity Regressor
    Model:    XGBoostRegressor wrapped with MAPIE conformal prediction
    Protocol: Spatial Block CV (k=5)
    Metrics:  R², MAE, RMSE
    Output:   {city}_reg_{type}.pkl
    
Saves:
  models/{city}/pune_pu_petrol.pkl
  models/{city}/pune_pu_ev.pkl
  models/{city}/pune_reg_petrol.pkl
  models/{city}/pune_reg_ev.pkl
  models/{city}/train_report.json
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from mapie.regression import MapieRegressor

from src.models.feature_extractor import FEATURE_NAMES
from src.models.spatial_cv import SpatialBlockCV

warnings.filterwarnings("ignore", category=UserWarning)

InfraType = Literal["petrol", "ev"]


# ──────────────────────────────────────────────────────────────────────────────
# Elkan-Noto PU Calibration
# ──────────────────────────────────────────────────────────────────────────────

class PUClassifier:
    """
    XGBoost classifier with Elkan-Noto PU calibration.

    Training:  P (label=1) vs U (label=0, unlabeled)
    Inference: s(x) / c  where c = P(s=1 | y=1) estimated on hold-out P set.

    References:
      Elkan & Noto (2008) "Learning Classifiers from Only Positive and Unlabeled Data"
    """

    def __init__(self, random_state: int = 42):
        self.clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3.0,   # up-weight the sparse positive class
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
        self.c_: float | None = None
        self.scaler_ = StandardScaler()

    def fit(self, X_p: np.ndarray, X_u: np.ndarray) -> "PUClassifier":
        """
        X_p: feature matrix for POSITIVE set (label=1)
        X_u: feature matrix for UNLABELED/BACKGROUND set (label=0)
        """
        X = np.vstack([X_p, X_u])
        y = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_u))])

        X_scaled = self.scaler_.fit_transform(X)
        self.clf.fit(X_scaled, y)

        # Estimate c = P(s=1 | y=1) from positive set scores
        p_scores = self.clf.predict_proba(self.scaler_.transform(X_p))[:, 1]
        self.c_ = float(np.mean(p_scores))
        self.c_ = max(self.c_, 1e-6)   # guard against zero
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated P(y=1 | x) = P(s=1 | x) / c, clipped to [0, 1].
        """
        X_scaled = self.scaler_.transform(X)
        raw = self.clf.predict_proba(X_scaled)[:, 1]
        calibrated = np.clip(raw / self.c_, 0.0, 1.0)
        return calibrated

    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_


# ──────────────────────────────────────────────────────────────────────────────
# Conformal Regressor
# ──────────────────────────────────────────────────────────────────────────────

class ConformalRegressor:
    """
    XGBoost regressor wrapped with MAPIE conformal prediction.
    Produces calibrated prediction intervals at 90% coverage.
    """

    def __init__(self, random_state: int = 42):
        base = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        )
        self.mapie = MapieRegressor(base, method="base", cv=5)
        self.scaler_ = StandardScaler()
        self._base_xgb = base
        self._fitted_xgb: xgb.XGBRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalRegressor":
        X_scaled = self.scaler_.fit_transform(X)
        # Fit base for feature importances
        self._fitted_xgb = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=0
        )
        self._fitted_xgb.fit(X_scaled, y)
        # Fit MAPIE for conformal intervals
        self.mapie.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray, alpha: float = 0.10):
        """
        Returns (y_pred, y_lower, y_upper) at (1-alpha)=90% coverage.
        """
        X_scaled = self.scaler_.transform(X)
        y_pred, intervals = self.mapie.predict(X_scaled, alpha=alpha)
        return y_pred, intervals[:, 0, 0], intervals[:, 1, 0]

    def feature_importances(self) -> np.ndarray:
        if self._fitted_xgb:
            return self._fitted_xgb.feature_importances_
        return np.zeros(len(FEATURE_NAMES))


# ──────────────────────────────────────────────────────────────────────────────
# Training orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _spatial_block_cv_pu(
    X_p: np.ndarray,
    X_u: np.ndarray,
    coords_p: np.ndarray,
    k: int = 5,
) -> dict:
    """Run Spatial Block CV on PU classifier. Returns fold metrics."""
    cv = SpatialBlockCV(k=k)
    aucs, loglosses = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(coords_p)):
        if len(test_idx) == 0:
            continue
        X_p_train = X_p[train_idx]
        X_p_test  = X_p[test_idx]

        clf = PUClassifier()
        clf.fit(X_p_train, X_u)

        # Score test positives vs sample of background
        bg_sample = X_u[np.random.choice(len(X_u),
                        min(len(X_u), len(X_p_test)*3), replace=False)]
        X_test = np.vstack([X_p_test, bg_sample])
        y_test = np.concatenate([np.ones(len(X_p_test)), np.zeros(len(bg_sample))])
        probs  = clf.predict_proba(X_test)

        try:
            aucs.append(roc_auc_score(y_test, probs))
            loglosses.append(log_loss(y_test, probs))
        except Exception:
            pass

    return {
        "auc_mean":     float(np.mean(aucs))     if aucs else 0.0,
        "auc_std":      float(np.std(aucs))      if aucs else 0.0,
        "logloss_mean": float(np.mean(loglosses)) if loglosses else 0.0,
    }


def _spatial_block_cv_reg(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    k: int = 5,
) -> dict:
    """Run Spatial Block CV on regressor. Returns fold metrics."""
    cv = SpatialBlockCV(k=k)
    r2s, maes = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(coords)):
        if len(test_idx) < 2 or len(train_idx) < 4:
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        reg = xgb.XGBRegressor(n_estimators=300, max_depth=4,
                                learning_rate=0.05, verbosity=0)
        reg.fit(X_tr_s, y_tr)
        preds = reg.predict(X_te_s)

        r2s.append(r2_score(y_te, preds))
        maes.append(mean_absolute_error(y_te, preds))

    return {
        "r2_mean":  float(np.mean(r2s))  if r2s else 0.0,
        "r2_std":   float(np.std(r2s))   if r2s else 0.0,
        "mae_mean": float(np.mean(maes)) if maes else 0.0,
    }


def train_city(
    city: str,
    X_petrol: np.ndarray,
    X_ev: np.ndarray,
    X_ev_reg: np.ndarray,
    X_bg: np.ndarray,
    coords_petrol: np.ndarray,
    coords_ev: np.ndarray,
    coords_ev_reg: np.ndarray,
    acti_petrol: np.ndarray,
    acti_ev: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    Train all 4 models for one city. Returns metrics dict.

    Parameters:
      X_petrol / X_ev     : (N, 16) feature matrices for PU positive sets
      X_ev_reg            : (M, 16) features for ALL rated EV candidates (regressor)
      X_bg                : (K, 16) features for background/unlabeled set
      coords_*            : (N, 2) [lat, lon] for spatial CV block assignment
      acti_*              : (N,) ACTI target (NaN rows excluded from regressor)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"city": city}

    # ── 1. PU Classifier — Petrol ─────────────────────────────────────────────
    print(f"  [Petrol PU]  Positives={len(X_petrol)}, Background={len(X_bg)}")
    cv_pu_p = _spatial_block_cv_pu(X_petrol, X_bg, coords_petrol)
    report["petrol_pu_cv"] = cv_pu_p

    pu_petrol = PUClassifier()
    pu_petrol.fit(X_petrol, X_bg)
    with open(out_dir / f"{city}_pu_petrol.pkl", "wb") as f:
        pickle.dump(pu_petrol, f)
    print(f"    CV AUC: {cv_pu_p['auc_mean']:.3f} ± {cv_pu_p['auc_std']:.3f}")

    # ── 2. PU Classifier — EV ────────────────────────────────────────────────
    print(f"  [EV PU]      Positives={len(X_ev)}, Background={len(X_bg)}")
    cv_pu_e = _spatial_block_cv_pu(X_ev, X_bg, coords_ev)
    report["ev_pu_cv"] = cv_pu_e

    pu_ev = PUClassifier()
    pu_ev.fit(X_ev, X_bg)
    with open(out_dir / f"{city}_pu_ev.pkl", "wb") as f:
        pickle.dump(pu_ev, f)
    print(f"    CV AUC: {cv_pu_e['auc_mean']:.3f} ± {cv_pu_e['auc_std']:.3f}")

    # ── 3. Regressor — Petrol ────────────────────────────────────────────────
    mask_p = ~np.isnan(acti_petrol)
    if mask_p.sum() >= 10:
        X_p_reg = X_petrol[mask_p]
        y_p_reg = acti_petrol[mask_p]
        cv_reg_p = _spatial_block_cv_reg(X_p_reg, y_p_reg, coords_petrol[mask_p])
        report["petrol_reg_cv"] = cv_reg_p

        reg_petrol = ConformalRegressor()
        reg_petrol.fit(X_p_reg, y_p_reg)
        with open(out_dir / f"{city}_reg_petrol.pkl", "wb") as f:
            pickle.dump(reg_petrol, f)
        print(f"  [Petrol Reg] n={mask_p.sum()}  R²: {cv_reg_p['r2_mean']:.3f} ± {cv_reg_p['r2_std']:.3f}")
    else:
        print(f"  [Petrol Reg] Insufficient rated stations ({mask_p.sum()}), skipping.")
        report["petrol_reg_cv"] = {"note": "insufficient_data"}

    # ── 4. Regressor — EV ────────────────────────────────────────────
    mask_e = ~np.isnan(acti_ev)
    if mask_e.sum() >= 5:
        X_e_reg = X_ev_reg[mask_e]
        y_e_reg = acti_ev[mask_e]
        coords_e_reg_masked = coords_ev_reg[mask_e]
        cv_reg_e = _spatial_block_cv_reg(X_e_reg, y_e_reg, coords_e_reg_masked)
        report["ev_reg_cv"] = cv_reg_e

        reg_ev = ConformalRegressor()
        reg_ev.fit(X_e_reg, y_e_reg)
        with open(out_dir / f"{city}_reg_ev.pkl", "wb") as f:
            pickle.dump(reg_ev, f)
        print(f"  [EV Reg]     n={mask_e.sum()}  R²: {cv_reg_e['r2_mean']:.3f} ± {cv_reg_e['r2_std']:.3f}")
    else:
        print(f"  [EV Reg]     Insufficient rated EV stations ({mask_e.sum()}), skipping.")
        report["ev_reg_cv"] = {"note": "insufficient_data"}

    # Save report
    with open(out_dir / "train_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
