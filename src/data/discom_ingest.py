"""
src/data/discom_ingest.py
──────────────────────────
Loads DISCOM (state electricity board) substation shapefiles as the
PRIMARY source for power_grid_dist_m (Feature #11).

Supported DISCOMs:
  - MSEDCL  (Pune, Mumbai)  — Maharashtra GIS Portal
  - BESCOM  (Bengaluru)     — Karnataka Open Data Portal
  - BRPL/BYPL (Delhi)       — Delhi Data Portal

Expected shapefile location:
  data/external/{city}/discom_substations.shp  (or .geojson)

If no DISCOM file is found for a city, returns None and the
spatial_index.py builder falls back to OSM power_osm layer.

Returns a DataFrame with columns: lat, lon, layer='power_discom', name, tags
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

# Where to look for DISCOM shapefiles
DISCOM_SEARCH_PATTERNS = [
    "discom_substations.shp",
    "discom_substations.geojson",
    "discom_substations.gpkg",
    "substations.shp",
    "substations.geojson",
    "msedcl_substations.shp",
    "bescom_substations.shp",
    "brpl_substations.shp",
]

CITY_DISCOM_NOTES = {
    "pune":      "MSEDCL — download from https://mahagis.maharashtra.gov.in",
    "mumbai":    "MSEDCL — download from https://mahagis.maharashtra.gov.in",
    "bengaluru": "BESCOM — download from https://data.karnataka.gov.in",
    "delhi":     "BRPL/BYPL/TPDDL — download from https://data.delhi.gov.in",
}


def load_city(city: str, external_dir: Path) -> pd.DataFrame | None:
    """
    Attempts to load a DISCOM substation shapefile for the city.
    Returns a standardised DataFrame or None if no file is found.

    DataFrame columns: lat, lon, layer, name, tags
    """
    city_dir = external_dir / city
    city_dir.mkdir(parents=True, exist_ok=True)

    found_path: Path | None = None
    for pattern in DISCOM_SEARCH_PATTERNS:
        candidate = city_dir / pattern
        if candidate.exists():
            found_path = candidate
            break

    if found_path is None:
        print(f"[DISCOM] No shapefile found for {city} in {city_dir}.")
        print(f"  ⚠  Please download from: {CITY_DISCOM_NOTES.get(city, 'your DISCOM portal')}")
        print(f"  ⚠  Expected filename: {DISCOM_SEARCH_PATTERNS[0]}")
        print(f"  Falling back to OSM power layer for {city}.")
        return None

    print(f"[DISCOM] Loading {found_path.name} for {city}...")
    gdf = gpd.read_file(found_path)

    # Reproject to WGS84 (EPSG:4326) if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        # Use centroid for polygon geometries (transformer areas)
        pt = geom if geom.geom_type == "Point" else geom.centroid
        name = row.get("name") or row.get("Name") or row.get("SUBSTATION") or ""
        rows.append({
            "layer": "power_discom",
            "lat":   pt.y,
            "lon":   pt.x,
            "name":  str(name),
            "tags":  '{"source": "discom"}',
        })

    df = pd.DataFrame(rows)
    print(f"[DISCOM] Loaded {len(df)} substation points for {city}.")
    return df


def merge_with_osm_fallback(
    discom_df: pd.DataFrame | None,
    osm_power_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge DISCOM substations with OSM power layer.
    DISCOM data is preferred; OSM records not within 200m of any DISCOM
    substation are appended as supplementary coverage.

    Both DataFrames must have columns: lat, lon, layer, name, tags
    """
    if discom_df is None or discom_df.empty:
        print("[DISCOM merge] Using OSM power data only (no DISCOM file).")
        return osm_power_df.copy()

    from scipy.spatial import cKDTree
    import numpy as np

    discom_coords = discom_df[["lat", "lon"]].values
    osm_coords = osm_power_df[["lat", "lon"]].values

    # Build KDTree from DISCOM points
    tree = cKDTree(discom_coords)
    # For each OSM point, find nearest DISCOM point (in degrees ~approx km)
    dist, _ = tree.query(osm_coords)
    # ~200m ≈ 0.0018 degrees latitude
    DUPLICATE_THRESHOLD_DEG = 0.0018
    osm_unique_mask = dist > DUPLICATE_THRESHOLD_DEG

    osm_supplement = osm_power_df[osm_unique_mask].copy()
    merged = pd.concat([discom_df, osm_supplement], ignore_index=True)
    print(f"[DISCOM merge] DISCOM: {len(discom_df)} | "
          f"OSM supplement (non-duplicate): {osm_unique_mask.sum()} | "
          f"Total power grid points: {len(merged)}")
    return merged


if __name__ == "__main__":
    # Test load for Pune
    df = load_city("pune", Path("data/external"))
    print(df)
