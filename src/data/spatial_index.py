"""
src/data/spatial_index.py
──────────────────────────
Builds and serialises the local cKDTree spatial indices from:
  - poi_layers_osm.parquet  (OSM POI data)
  - DISCOM substation data   (merged via discom_ingest.py)

Produces data/processed/{city}/poi_layers.parquet — the unified spatial
layer file used by the feature extractor at training AND inference time.

Also pickles layer-specific cKDTrees to models/{city}/kdtrees.pkl
for sub-millisecond lookup during FastAPI inference (<15ms SLA).

Feature extraction at inference time uses ONLY local data — zero live
API calls (CONTEXT.md §12, Rule 1).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LAYER_TO_INDEX = {
    "cafes":           "cafe",
    "restaurants":     "restaurant",
    "retail":          "retail",
    "supermarkets":    "supermarket",
    "offices":         "office",
    "bus_stops":       "bus_stop",
    "metro_stations":  "metro_station",
    "residential":     "residential",
    "parking":         "parking",
    "power_osm":       "power",
    "power_discom":    "power",
    "roads_primary":   "roads_primary",
    "roads_secondary": "roads_secondary",
    "water_exclusion": "water",
}


def build_indices(city: str, processed_dir: Path, models_dir: Path) -> dict[str, cKDTree]:
    """
    Loads poi_layers.parquet (merged OSM + DISCOM) and builds one cKDTree
    per spatial layer. Returns the dict of trees and saves to kdtrees.pkl.
    """
    parquet_path = processed_dir / city / "poi_layers.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"poi_layers.parquet not found at {parquet_path}. "
            "Run scripts/02_build_spatial_indices.py first."
        )

    df = pd.read_parquet(parquet_path)
    trees: dict[str, cKDTree] = {}
    coords_store: dict[str, np.ndarray] = {}


    layer_priority: dict[str, pd.DataFrame] = {}
    for layer_name in df["layer"].unique():
        mapped_key = LAYER_TO_INDEX.get(layer_name)
        if mapped_key is None:
            continue
        layer_df = df[df["layer"] == layer_name][["lat", "lon"]].dropna()
        if layer_df.empty:
            continue
        # power_discom overrides power_osm
        if mapped_key == "power":
            if "power_discom" in df["layer"].values and layer_name == "power_osm":
                print(f"  [power] Skipping power_osm (power_discom available)")
                continue
        layer_priority[mapped_key] = layer_df

    for key, layer_df in layer_priority.items():
        coords = layer_df.values
        trees[key] = cKDTree(coords)
        coords_store[key] = coords
        print(f"  [{key}] {len(coords)} points")

    # ── Serialise ─────────────────────────────────────────────────────────────
    models_city_dir = models_dir / city
    models_city_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = models_city_dir / "kdtrees.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"trees": trees, "coords": coords_store}, f, protocol=5)
    print(f"\n[spatial_index] Saved {len(trees)} KDTrees → {pkl_path}")
    return trees


def load_indices(city: str, models_dir: Path) -> dict[str, cKDTree]:
    """Load pre-built KDTrees from disk (used by FastAPI at inference time)."""
    pkl_path = models_dir / city / "kdtrees.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"KDTree index not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["trees"]


if __name__ == "__main__":
    trees = build_indices(
        city="pune",
        processed_dir=Path("data/processed"),
        models_dir=Path("models"),
    )
    print(f"\nAvailable tree keys: {list(trees.keys())}")
