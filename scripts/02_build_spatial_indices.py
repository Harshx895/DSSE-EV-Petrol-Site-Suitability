"""
scripts/02_build_spatial_indices.py
────────────────────────────────────
PUNE PHASE 1 — Step 2 of 2
Merges OSM POI layers with DISCOM substation data, then builds and
serialises the cKDTree spatial indices used at inference time.

Input:
  data/processed/{city}/poi_layers_osm.parquet  (from script 01)
  data/external/{city}/discom_substations.*      (optional)

Output:
  data/processed/{city}/poi_layers.parquet  (unified, final layer file)
  models/{city}/kdtrees.pkl                 (pre-built KDTrees)

Also runs a quick coverage validation of OSM vs DISCOM power points.

Usage:
  python scripts/02_build_spatial_indices.py --city pune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.discom_ingest import load_city as load_discom, merge_with_osm_fallback
from src.data.spatial_index import build_indices


def validate_power_coverage(osm_power_rows: int, discom_rows: int | None) -> None:
    """Print a coverage report for the power grid layer."""
    print("\n  ── Power Grid Coverage Report ────────────────────────────")
    print(f"  OSM power nodes:    {osm_power_rows}")
    if discom_rows is not None:
        print(f"  DISCOM substations: {discom_rows}")
        ratio = discom_rows / max(osm_power_rows, 1)
        if ratio < 0.3:
            print("  ⚠  DISCOM has far fewer points than OSM — check shapefile quality.")
        elif ratio > 3:
            print("  ✓  DISCOM significantly supplements OSM power coverage.")
        else:
            print("  ✓  DISCOM and OSM power coverage are roughly consistent.")
    else:
        print("  ⚠  No DISCOM file found — using OSM power only.")
        print("     Feature #11 (power_grid_dist_m) may be unreliable for EV model.")
    print("  ────────────────────────────────────────────────────────\n")


def main(city: str) -> None:
    print(f"\n{'='*60}")
    print(f"  DSSE Phase 1 Step 2 — Build Spatial Indices: {city.upper()}")
    print(f"{'='*60}\n")

    processed_dir = Path("data/processed") / city
    external_dir  = Path("data/external")
    models_dir    = Path("models")

    # ── Load OSM POI layers ───────────────────────────────────────────────────
    osm_parquet = processed_dir / "poi_layers_osm.parquet"
    if not osm_parquet.exists():
        raise FileNotFoundError(
            f"{osm_parquet} not found. Run script 01 first:\n"
            f"  python scripts/01_download_raw_data.py --city {city}"
        )
    osm_df = pd.read_parquet(osm_parquet)
    print(f"[Step 1] Loaded OSM POI layers: {len(osm_df)} rows")

    # ── Load & merge DISCOM ───────────────────────────────────────────────────
    print("\n[Step 2] Checking DISCOM substation data...")
    osm_power_df = osm_df[osm_df["layer"].isin(["power_osm"])].copy()
    discom_df    = load_discom(city, external_dir)

    validate_power_coverage(
        osm_power_rows=len(osm_power_df),
        discom_rows=len(discom_df) if discom_df is not None else None,
    )

    merged_power = merge_with_osm_fallback(discom_df, osm_power_df)

    # ── Also merge scraped power grid data (Overpass targeted + Wikidata) ─────
    scraped_power_path = Path("data/external") / city / "power_grid_scraped.csv"
    if scraped_power_path.exists():
        scraped_df = pd.read_csv(scraped_power_path)
        print(f"  Loading scraped power grid: {len(scraped_df)} points")
        # Standardise to poi_layers schema
        scraped_df = scraped_df.rename(columns={"layer": "layer"})[
            ["layer", "lat", "lon", "name", "tags"]
        ].copy()
        scraped_df["layer"] = "power_discom"
        merged_power = pd.concat([merged_power, scraped_df], ignore_index=True)
        # Spatial deduplicate merged power (within ~100m)
        from scipy.spatial import cKDTree as _KDT
        if len(merged_power) > 1:
            coords = merged_power[["lat", "lon"]].values
            tree_p = _KDT(coords)
            pairs  = tree_p.query_pairs(r=0.0009)
            to_drop = {max(i, j) for i, j in pairs}
            merged_power = merged_power.drop(index=list(to_drop)).reset_index(drop=True)
        print(f"  Power layer after merge+dedup: {len(merged_power)} unique nodes")
    else:
        print("  [Power] No scraped power file found — using OSM/DISCOM only.")

    # Rebuild unified poi_layers.parquet (replace power_osm with merged power)
    non_power_df = osm_df[~osm_df["layer"].isin(["power_osm", "power_discom"])].copy()
    final_df = pd.concat([non_power_df, merged_power], ignore_index=True)

    final_parquet = processed_dir / "poi_layers.parquet"
    final_df.to_parquet(final_parquet, index=False)
    print(f"[Step 2] Saved unified POI layers → {final_parquet}")
    print(f"         Total rows: {len(final_df)}")

    # ── Layer summary ─────────────────────────────────────────────────────────
    print("\n  Layer summary:")
    for layer, grp in final_df.groupby("layer"):
        print(f"    {layer:<25} {len(grp):>6} features")

    # ── Build KDTrees ─────────────────────────────────────────────────────────
    print("\n[Step 3] Building cKDTree spatial indices...")
    trees = build_indices(city, Path("data/processed"), models_dir)

    print(f"\n{'='*60}")
    print(f"  Spatial indices complete for {city.upper()}.")
    print(f"  Available trees: {sorted(trees.keys())}")
    print(f"\n  Next: python scripts/03_train_and_evaluate.py --city {city}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSSE Phase 1 Step 2: Build KDTree indices")
    parser.add_argument("--city", type=str, default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    args = parser.parse_args()
    main(args.city)
