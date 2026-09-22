"""
scripts/04_scrape_ev_networks.py
----------------------------------
Scrapes EV stations from Ather Grid, Tata Power EZ, ChargeZone, Statiq.
Merges results into stations_master.csv and rebuilds KDTrees.

Also attempts to get better power grid data from data.gov.in as
a substitute for the unreachable mahagis.maharashtra.gov.in portal.

Usage:
  python -X utf8 scripts/04_scrape_ev_networks.py --city pune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ev_networks_scraper import fetch_all as scrape_ev_networks

HEADERS = {"User-Agent": "Mozilla/5.0 (research; academic use)"}

CITY_BBOXES = {
    "pune":      {"south": 18.40, "north": 18.63, "west": 73.75, "east": 73.98},
    "bengaluru": {"south": 12.83, "north": 13.14, "west": 77.48, "east": 77.78},
}


def fetch_power_datagov(city: str, out_dir: Path) -> pd.DataFrame:
    """
    Attempt to fetch substation/power infrastructure data from data.gov.in
    (India Open Government Data portal) as a substitute for mahagis.

    Also tries POSOCO (Power System Operation Corporation) GIS data.
    """
    print("\n[PowerGrid] Trying data.gov.in for MSEDCL substation data...")

    # data.gov.in CKAN API search
    search_terms = ["MSEDCL substation", "Maharashtra substation", "power substation pune"]
    found_resources: list[str] = []

    for term in search_terms:
        try:
            url = "https://data.gov.in/api/datastore/resource.json"
            # Try the CKAN search API
            r = requests.get(
                "https://data.gov.in/api/3/action/package_search",
                params={"q": term, "rows": 5},
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code == 200:
                results = r.json().get("result", {}).get("results", [])
                for pkg in results:
                    for res in pkg.get("resources", []):
                        fmt = res.get("format", "").lower()
                        if fmt in ("csv", "json", "geojson", "xls", "xlsx"):
                            found_resources.append(res.get("url", ""))
        except Exception as e:
            print(f"  data.gov.in search '{term}': {type(e).__name__}")

    # Try to download any CSV resources found
    rows = []
    for url in found_resources[:5]:
        if not url:
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and "latitude" in r.text.lower():
                import io
                df_raw = pd.read_csv(io.StringIO(r.text))
                lat_col = next((c for c in df_raw.columns if "lat" in c.lower()), None)
                lon_col = next((c for c in df_raw.columns if "lon" in c.lower() or "lng" in c.lower()), None)
                if lat_col and lon_col:
                    bbox = CITY_BBOXES.get(city, {})
                    for _, row in df_raw.iterrows():
                        try:
                            lat, lon = float(row[lat_col]), float(row[lon_col])
                            if (bbox.get("south", 0) <= lat <= bbox.get("north", 90) and
                                    bbox.get("west", 0) <= lon <= bbox.get("east", 180)):
                                rows.append({
                                    "lat": lat, "lon": lon,
                                    "name": str(row.get("name", row.get("station_name", ""))),
                                    "layer": "power_discom",
                                    "tags": json.dumps({"source": "data.gov.in"}),
                                })
                        except (ValueError, TypeError):
                            pass
                if rows:
                    print(f"  Found {len(rows)} power points from data.gov.in resource!")
                    break
        except Exception as e:
            print(f"  Resource fetch error: {type(e).__name__}")

    if not rows:
        print("  data.gov.in: No usable power data found for this area.")
        print("  Recommendation: Manually download from one of these alternatives:")
        print("    1. https://mahadiscom.in/en/ (MSEDCL consumer portal)")
        print("    2. https://cea.nic.in/gis/ (Central Electricity Authority GIS)")
        print("    3. https://powermin.gov.in/ (Ministry of Power India)")
        print("    4. Request data under RTI (Right to Information Act)")

    df = pd.DataFrame(rows)
    if not df.empty:
        path = out_dir / "power_datagov.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} power points -> {path}")
    return df


def merge_ev_into_master(city: str, ev_df: pd.DataFrame, processed_dir: Path) -> None:
    """Merge newly scraped EV stations into stations_master.csv."""
    if ev_df.empty:
        print("\n[Merge] No new EV stations to merge.")
        return

    master_path = processed_dir / "stations_master.csv"
    if not master_path.exists():
        print(f"  ERROR: {master_path} not found. Run script 01 first.")
        return

    master = pd.read_csv(master_path)

    # Assign consistent columns
    ev_df = ev_df.copy()
    ev_df["place_id"]           = ev_df["source"] + "_" + ev_df.index.astype(str)
    ev_df["ocm_id"]             = None
    ev_df["rating"]             = None
    ev_df["user_ratings_total"] = None
    ev_df["opening_date_est"]   = None
    ev_df["connector_types"]    = None
    ev_df["is_fast_charge"]     = ev_df["ev_label"] == "commercial"
    ev_df["sources"]            = ev_df["source"]

    # Spatial dedup against existing master (50m threshold)
    from scipy.spatial import cKDTree
    import numpy as np

    existing_coords = master[["lat", "lon"]].values
    tree = cKDTree(existing_coords)
    new_rows = []
    for _, row in ev_df.iterrows():
        coord = np.array([row["lat"], row["lon"]])
        dist, _ = tree.query(coord, k=1)
        if dist > 0.00045:   # ~50m
            new_rows.append(row.to_dict())

    if not new_rows:
        print("\n[Merge] All scraped EV stations already exist in master (within 50m).")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([master, new_df], ignore_index=True)
    combined["canonical_id"] = [f"DSSE_{i:05d}" for i in range(len(combined))]

    combined.to_csv(master_path, index=False)

    n_comm = (combined["ev_label"] == "commercial").sum()
    n_cand = (combined["ev_label"] == "candidate").sum()
    n_petr = (combined["ev_label"] == "petrol").sum()

    print(f"\n[Merge] Updated stations_master.csv: {len(combined)} total stations")
    print(f"  commercial EV (verified >=22kW): {n_comm}")
    print(f"  candidate EV  (kW unknown):      {n_cand}")
    print(f"  petrol:                          {n_petr}")
    print(f"  New stations added from scrapers: {len(new_rows)}")


def main(city: str) -> None:
    print(f"\n{'='*60}")
    print(f"  DSSE Phase 1 Step 4 — EV Network Scraping: {city.upper()}")
    print(f"{'='*60}")

    external_dir  = Path("data/external") / city
    processed_dir = Path("data/processed") / city

    # Step 1: Scrape EV networks
    ev_df = scrape_ev_networks(city, external_dir)

    # Step 2: Merge into master
    merge_ev_into_master(city, ev_df, processed_dir)

    # Step 3: Try to get better power grid data
    fetch_power_datagov(city, external_dir)

    # Step 4: Rebuild KDTrees with updated data
    print("\n[Step 4] Rebuilding spatial indices with updated data...")
    import subprocess
    result = subprocess.run(
        ["python", "-X", "utf8",
         "scripts/02_build_spatial_indices.py", "--city", city],
        capture_output=False,
    )

    print(f"\n{'='*60}")
    print(f"  EV scraping complete for {city.upper()}")
    print(f"  Next: python scripts/05_train_and_evaluate.py --city {city}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    args = parser.parse_args()
    main(args.city)
