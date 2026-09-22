"""
src/models/target_engineering.py
----------------------------------
Computes training targets and label vectors:

  1. ACTI (Age-Corrected Throughput Index) — Formulation 2 regression target
     Y_raw = log10(1 + ratings_total) * (rating / 5.0)
     Y     = Y_raw / log10(1 + months_open)   [if opening_date known]
     Fallback: Y = Y_raw                       [if opening_date missing]

  2. PU label vector — Formulation 1 classification labels
     label=1  → confirmed positives (petrol or commercial EV)
     label=0  → unlabeled background (H3 hex centroids in city bbox)

  3. AHP baseline weight vector — expert-assigned weights for §7 comparison
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from src.models.feature_extractor import FEATURE_NAMES

# ── AHP baseline (expert-assigned weights, sum = 1.0) ─────────────────────────
# Derived from literature review: Funke et al. (2021), Namdeo & Tiwary (2019)
# + engineering domain knowledge. Used as comparison baseline for SFDS.
AHP_WEIGHTS_PETROL = {
    "dist_primary_highway_m":  0.22,
    "dist_secondary_road_m":   0.12,
    "cafe_count_300m":         0.03,
    "restaurant_count_300m":   0.04,
    "retail_count_500m":       0.07,
    "supermarket_count_500m":  0.04,
    "office_count_500m":       0.05,
    "bus_stop_count_500m":     0.08,
    "metro_station_dist_m":    0.05,
    "residential_count_500m":  0.06,
    "power_grid_dist_m":       0.03,
    "competitor_petrol_1km":   0.07,
    "competitor_ev_1km":       0.01,
    "dwell_index":             0.04,
    "transit_accessibility":   0.05,
    "parking_count_300m":      0.04,
}

AHP_WEIGHTS_EV = {
    "dist_primary_highway_m":  0.07,
    "dist_secondary_road_m":   0.06,
    "cafe_count_300m":         0.12,
    "restaurant_count_300m":   0.09,
    "retail_count_500m":       0.08,
    "supermarket_count_500m":  0.05,
    "office_count_500m":       0.10,
    "bus_stop_count_500m":     0.05,
    "metro_station_dist_m":    0.06,
    "residential_count_500m":  0.06,
    "power_grid_dist_m":       0.14,
    "competitor_petrol_1km":   0.01,
    "competitor_ev_1km":       0.03,
    "dwell_index":             0.03,
    "transit_accessibility":   0.03,
    "parking_count_300m":      0.02,
}


def _months_since(date_str: str | None) -> float | None:
    """Parse ISO date string → months elapsed since then."""
    if not date_str or pd.isna(date_str):
        return None
    try:
        opened = pd.to_datetime(date_str)
        now    = pd.Timestamp.now(tz=opened.tz)
        delta  = now - opened
        return max(1.0, delta.days / 30.44)
    except Exception:
        return None


def compute_acti(df: pd.DataFrame) -> pd.Series:
    """
    Compute Age-Corrected Throughput Index for stations with ratings.
    Rows missing ratings get NaN (excluded from regressor training).
    """
    acti = []
    for _, row in df.iterrows():
        ratings_total = row.get("user_ratings_total")
        rating        = row.get("rating")
        date_str      = row.get("opening_date_est")

        if pd.isna(ratings_total) or pd.isna(rating) or ratings_total <= 0:
            acti.append(np.nan)
            continue

        y_raw = np.log10(1 + float(ratings_total)) * (float(rating) / 5.0)
        # Age correction requires operator-reported opening date.
        # Google review timestamps (oldest of 5 recent reviews) are unreliable
        # as an opening date proxy — applying correction adds noise. Disabled.
        acti.append(float(y_raw))

    return pd.Series(acti, index=df.index, name="acti")


def generate_background_points(
    city: str,
    n_points: int = 2000,
    h3_res: int = 9,
) -> np.ndarray:
    """
    Generate background (unlabeled) candidate points using H3 hexagon centroids.
    Falls back to uniform random grid if H3 polyfill is empty.

    Returns (N, 2) array of [lat, lon].
    """
    BBOXES = {
        "pune":      (18.40, 18.63, 73.75, 73.98),
        "bengaluru": (12.83, 13.14, 77.48, 77.78),
        "mumbai":    (18.89, 19.27, 72.77, 73.01),
        "delhi":     (28.40, 28.88, 76.84, 77.35),
    }
    south, north, west, east = BBOXES[city]

    # H3 polyfill the bounding-box polygon
    np.random.seed(42)   # reproducible background points across runs
    geojson_poly = {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south],
            [east, north], [west, north], [west, south],
        ]]
    }
    try:
        hexes = h3.polyfill_geojson(geojson_poly, h3_res)
        if hexes:
            pts = np.array([h3.h3_to_geo(hx) for hx in hexes])  # (lat, lon)
            if len(pts) > n_points:
                idx = np.random.choice(len(pts), n_points, replace=False)
                pts = pts[idx]
            return pts
    except Exception:
        pass

    # Fallback: uniform random grid
    lats = np.random.uniform(south, north, n_points)
    lons = np.random.uniform(west,  east,  n_points)
    return np.column_stack([lats, lons])


def build_pu_dataset(
    df_positive: pd.DataFrame,
    X_positive: np.ndarray,
    X_background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble PU Learning dataset:
      X = [X_positive; X_background]   shape (N_pos + N_bg, 16)
      y = [1, 1, …, 0, 0, …]

    Background points are treated as unlabeled (label=0 in standard PU-Bagging).
    """
    X = np.vstack([X_positive, X_background])
    y = np.concatenate([
        np.ones(len(X_positive),  dtype=np.int32),
        np.zeros(len(X_background), dtype=np.int32),
    ])
    return X, y
