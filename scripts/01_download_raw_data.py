"""
scripts/01_download_raw_data.py
--------------------------------
DSSE Phase 1 — Step 1 of 2
Downloads all raw data for a given city:
  1. Google Places API  → Petrol + EV stations (5×5 grid tiling)
  2. OpenChargeMap API  → EV commercial chargers (≥7kW, labeled at ≥22kW)
  3. Overpass API       → All 13 OSM POI layers (single combined query)
  4. Deduplication      → stations_master.csv with ev_label column

Outputs:
  data/raw/{city}/google_places.csv
  data/raw/{city}/ocm_stations.csv
  data/processed/{city}/poi_layers_osm.parquet
  data/processed/{city}/stations_master.csv

Usage:
  python -X utf8 scripts/01_download_raw_data.py --city pune
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import google_places_client, ocm_client, osm_ingest
from src.data.station_deduplicator import deduplicate, save_master


def main(city: str) -> None:
    print(f"\n{'='*60}")
    print(f"  DSSE Phase 1 — Data Ingestion: {city.upper()}")
    print(f"{'='*60}\n")

    raw_dir       = Path("data/raw") / city
    processed_dir = Path("data/processed") / city

    # ── 1. Google Places (5×5 grid + branded EV searches) ────────────────────
    print(">> Step 1/4: Google Places API (grid-tiled)")
    google_df = google_places_client.fetch_city(city, raw_dir)
    google_df.to_csv(raw_dir / "google_places.csv", index=False)
    print(f"  Saved → {raw_dir}/google_places.csv\n")

    # ── 2. OpenChargeMap (≥7kW, labels assigned at ≥22kW) ────────────────────
    print(">> Step 2/4: OpenChargeMap API")
    ocm_df = ocm_client.fetch_city(city, raw_dir)
    ocm_df.to_csv(raw_dir / "ocm_stations.csv", index=False)
    print(f"  Saved → {raw_dir}/ocm_stations.csv\n")

    # ── 3. Deduplicate EV stations ────────────────────────────────────────────
    print(">> Step 3/4: Deduplicating stations across sources")
    master_df = deduplicate(google_df, ocm_df, plugshare_df=None)
    save_master(master_df, processed_dir)
    print(f"  Saved → {processed_dir}/stations_master.csv\n")

    # ── 4. OSM POI Layers (single Overpass query via mail.ru mirror) ──────────
    print(">> Step 4/4: OSM POI Layers (Overpass API)")
    osm_df = osm_ingest.fetch_city(city, processed_dir)
    if osm_df.empty:
        print("  WARNING: OSM returned no data. Check network connectivity.")
    print()

    print(f"{'='*60}")
    print(f"  Phase 1 complete for {city.upper()}.")
    print(f"  Next: python -X utf8 scripts/02_build_spatial_indices.py --city {city}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSSE Phase 1: Download raw data")
    parser.add_argument("--city", type=str, default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    args = parser.parse_args()
    main(args.city)
