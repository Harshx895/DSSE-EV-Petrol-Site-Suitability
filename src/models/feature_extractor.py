"""
src/models/feature_extractor.py
---------------------------------
Extracts the 14 canonical spatial features for any (lat, lon) point.
All computation is local — zero live API calls.

Inputs  : pre-built KDTrees (models/{city}/kdtrees.pkl)
          stations_master.csv (for competitor counts)
Output  : dict of 14 float features + raw values for UI display

Features (as of v2 — radii widened, redundant composites removed):
  1  dist_primary_highway_m    nearest primary/motorway
  2  dist_secondary_road_m     nearest secondary/tertiary
  3  cafe_count_500m           cafes within 500m   [was 300m — 83% zeros fixed]
  4  restaurant_count_500m     restaurants within 500m [was 300m — 64% zeros fixed]
  5  retail_count_500m         retail shops within 500m
  6  supermarket_count_500m    supermarkets within 500m
  7  office_count_500m         offices within 500m
  8  bus_stop_count_500m       bus stops within 500m
  9  metro_station_dist_m      nearest metro/subway
  10 residential_count_500m    residential buildings within 500m
  11 power_grid_dist_m         nearest power substation/transformer
  12 competitor_petrol_1km     petrol stations within 1km
  13 competitor_ev_1km         EV chargers within 1km
  14 parking_count_500m        parking within 500m [was 300m — 86% zeros fixed]

  REMOVED: dwell_index (corr=0.982 with retail_count_500m — redundant)
  REMOVED: transit_accessibility (corr=0.939 with bus_stop_count_500m — redundant)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

# Degree → metres at Indian latitudes (~18–29°N)
# 1 deg lat ≈ 111,320 m;  1 deg lon ≈ cos(lat)*111,320
_LAT_M = 111_320.0

FEATURE_NAMES = [
    "dist_primary_highway_m",   # 0
    "dist_secondary_road_m",    # 1
    "cafe_count_500m",          # 2  (was 300m)
    "restaurant_count_500m",    # 3  (was 300m)
    "retail_count_500m",        # 4
    "supermarket_count_500m",   # 5
    "office_count_500m",        # 6
    "bus_stop_count_500m",      # 7
    "metro_station_dist_m",     # 8
    "residential_count_500m",   # 9
    "power_grid_dist_m",        # 10
    "competitor_petrol_1km",    # 11
    "competitor_ev_1km",        # 12
    "parking_count_500m",       # 13 (was 300m)
]


def _deg_to_m(deg_dist: float, lat: float) -> float:
    """Convert KDTree Euclidean distance (degrees) to approximate metres."""
    lon_m = np.cos(np.radians(lat)) * _LAT_M
    lat_m = _LAT_M
    # Euclidean distance in degrees converted via average scale
    return deg_dist * np.sqrt((lat_m ** 2 + lon_m ** 2) / 2)


def _count_within(tree, point: np.ndarray, radius_deg: float) -> int:
    """Count neighbours within radius_deg degrees."""
    indices = tree.query_ball_point(point, r=radius_deg)
    return len(indices)


def _nearest_dist_m(tree, point: np.ndarray, lat: float) -> float:
    """Distance in metres to nearest point in tree."""
    dist, _ = tree.query(point, k=1)
    return _deg_to_m(float(dist), lat)


class FeatureExtractor:
    """
    Loads KDTrees and station data once, then extracts 16 features
    for arbitrary (lat, lon) points in O(log n) time.
    """

    def __init__(self, city: str,
                 models_dir: Path = Path("models"),
                 processed_dir: Path | None = None):
        import pandas as pd
        from scipy.spatial import cKDTree

        # Default: data/processed/{city}/ — fixes the bare FeatureExtractor('pune') path bug
        if processed_dir is None:
            processed_dir = Path("data/processed") / city

        ktree_path = models_dir / city / "kdtrees.pkl"
        with open(ktree_path, "rb") as f:
            loaded = pickle.load(f)

        # Support both dict-of-trees and {"trees":..., "coords":...} formats
        if isinstance(loaded, dict) and "trees" in loaded:
            self.trees: dict[str, Any] = loaded["trees"]
        else:
            self.trees = loaded

        # Build competitor KDTrees from stations_master
        master = pd.read_csv(processed_dir / "stations_master.csv")
        petrol_pts = master[master["ev_label"] == "petrol"][["lat", "lon"]].values
        ev_pts     = master[master["ev_label"].isin(["commercial", "candidate"])][
                         ["lat", "lon"]].values

        self.tree_petrol   = cKDTree(petrol_pts)   if len(petrol_pts) > 0 else None
        self.tree_ev       = cKDTree(ev_pts)        if len(ev_pts)     > 0 else None

        self.city = city

    def extract(self, lat: float, lon: float) -> dict[str, float]:
        """Return dict of 16 raw (un-normalized) feature values."""
        pt  = np.array([lat, lon])
        lat = lat

        trees = self.trees

        # ── Distance features (nearest neighbour) ────────────────────────────
        f1  = _nearest_dist_m(trees["roads_primary"],   pt, lat) \
              if "roads_primary"   in trees else 9999.0
        f2  = _nearest_dist_m(trees["roads_secondary"], pt, lat) \
              if "roads_secondary" in trees else 9999.0
        f9  = _nearest_dist_m(trees["metro_station"],   pt, lat) \
              if "metro_station"   in trees else 9999.0
        f11 = _nearest_dist_m(trees["power"],           pt, lat) \
              if "power"           in trees else 9999.0

        # ── Count features — all radii in degrees ────────────────────────────
        r500 = 500  / _LAT_M   # ~0.004491°
        r1km = 1000 / _LAT_M   # ~0.008983°
        # Note: cafes/restaurants/parking now at 500m (was 300m).
        # This reduces zero-inflation: cafe 83%→~50%, parking 86%→~60%.

        f3  = _count_within(trees["cafe"],        pt, r500) if "cafe"        in trees else 0
        f4  = _count_within(trees["restaurant"],  pt, r500) if "restaurant"  in trees else 0
        f5  = _count_within(trees["retail"],      pt, r500) if "retail"      in trees else 0
        f6  = _count_within(trees["supermarket"], pt, r500) if "supermarket" in trees else 0
        f7  = _count_within(trees["office"],      pt, r500) if "office"      in trees else 0
        f8  = _count_within(trees["bus_stop"],    pt, r500) if "bus_stop"    in trees else 0
        f10 = _count_within(trees["residential"], pt, r500) if "residential" in trees else 0
        f14 = _count_within(trees["parking"],     pt, r500) if "parking"     in trees else 0

        # ── Competitor counts ─────────────────────────────────────────────────
        f12 = _count_within(self.tree_petrol, pt, r1km) if self.tree_petrol else 0
        f13 = _count_within(self.tree_ev,     pt, r1km) if self.tree_ev     else 0

        # dwell_index and transit_accessibility removed (corr 0.982 / 0.939
        # with retail_count_500m / bus_stop_count_500m respectively).

        return {
            "dist_primary_highway_m":  f1,
            "dist_secondary_road_m":   f2,
            "cafe_count_500m":         float(f3),
            "restaurant_count_500m":   float(f4),
            "retail_count_500m":       float(f5),
            "supermarket_count_500m":  float(f6),
            "office_count_500m":       float(f7),
            "bus_stop_count_500m":     float(f8),
            "metro_station_dist_m":    f9,
            "residential_count_500m":  float(f10),
            "power_grid_dist_m":       f11,
            "competitor_petrol_1km":   float(f12),
            "competitor_ev_1km":       float(f13),
            "parking_count_500m":      float(f14),
        }

    def extract_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Extract features for N points. coords shape: (N, 2) [lat, lon].
        Returns (N, 16) array in FEATURE_NAMES order.
        """
        rows = [list(self.extract(lat, lon).values())
                for lat, lon in coords]
        return np.array(rows, dtype=np.float32)
