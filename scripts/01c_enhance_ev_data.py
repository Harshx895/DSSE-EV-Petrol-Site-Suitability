"""
scripts/01c_enhance_ev_data.py
---------------------------------
DSSE Phase 1.5 — EV Data Enhancement

Dramatically increases the number of confirmed commercial EV positives by:
  1. BRAND PROMOTION — Known EV networks (Ather, Tata Power, ChargeZone,
     Bolt.Earth, etc.) are promoted from 'candidate' to 'commercial' based
     on name matching alone. These are all ≥22kW by network specification.
  2. GOOGLE NEARBY SEARCH — Uses `includedTypes: ['electric_vehicle_charging_station']`
     (the dedicated Google Maps type, much more precise than text search)
     across a 6×6 grid to discover stations missed by the original text search.
  3. PLACE DETAILS — For remaining unknowns, fetches Google Place Details
     with evChargeOptions to determine kW class.
  4. FALSE POSITIVE REMOVAL — Filters out EV showrooms, repair shops,
     and non-charging locations that appeared in text search results.
  5. MASTER REBUILD — Rebuilds stations_master.csv with updated labels.

API cost estimate:
  Nearby Search: 36 tiles × 3 pages = ~$3.46
  Place Details:  ~100 unknowns × $0.017 = ~$1.70
  Total: ~$5 (well within $200/month free credit)

Usage:
  python -X utf8 scripts/01c_enhance_ev_data.py --city pune
  python -X utf8 scripts/01c_enhance_ev_data.py --city pune --dry-run   # preview only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.spatial import cKDTree

load_dotenv()

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# ── Known commercial EV networks in India (all ≥22kW) ──────────────────────────
# Sources: official network websites, MNRE registry
COMMERCIAL_EV_NETWORKS = [
    # Major national networks
    "ather grid",
    "tata power",
    "tata ev",
    "chargezone",
    "charge+zone",
    "chargegrid",
    "charge grid",
    "magenta chargegrid",
    "magenta charge grid",
    "statiq",
    "bolt.earth",
    "bolt earth",
    "fortum",
    "volttic",
    "zeon",
    "glida",
    "kazam",
    # Oil company networks (all DC fast)
    "bpcl pulse",
    "hpcl ev",
    "indian oil ev",
    "iocl ev",
    "shell recharge",
    # OEM networks
    "mg motor",  # MG has ≥50kW DC chargers
    "hyundai charging",
    # Government
    "eesl",
    "electric fuel",
    "electricfuel",
    # Regional
    "charge point",
    "chargepoint",
    "goec",
    "jio-bp",
    "jio bp",
    "ultraviolette",
]

# Patterns that indicate FALSE POSITIVES (not actual chargers)
FALSE_POSITIVE_PATTERNS = [
    r"showroom",
    r"dealership",
    r"service center",
    r"service centre",
    r"repair shop",
    r"repair centre",
    r"accessories",
    r"spare parts",
    r"battery shop",
    r"mechanic",
    r"insurance",
    r"driving school",
    r"auto parts",
    r"car wash(?!.*charg)",
    r"garage(?!.*charg)",
    r"body shop",
    r"rental(?!.*charg)",
    r"parking(?!.*charg)",
    r"workshop(?!.*charg)",
    r"towing",
]

CITY_BBOXES = {
    "pune":      {"south": 18.40, "west": 73.75, "north": 18.63, "east": 73.98},
    "bengaluru": {"south": 12.83, "west": 77.48, "north": 13.14, "east": 77.78},
    "mumbai":    {"south": 18.89, "west": 72.77, "north": 19.27, "east": 73.01},
    "delhi":     {"south": 28.40, "west": 76.84, "north": 28.88, "east": 77.35},
}

DEDUP_THRESHOLD_DEG = 0.00045  # ~50m

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Brand-based Promotion (ZERO API cost)
# ──────────────────────────────────────────────────────────────────────────────

def is_known_ev_network(name: str) -> str | None:
    """Check if a station name matches a known commercial EV network.
    Returns the matched network name, or None."""
    if not name:
        return None
    name_lower = name.lower().strip()
    for network in COMMERCIAL_EV_NETWORKS:
        if network in name_lower:
            return network
    return None


def is_false_positive(name: str) -> bool:
    """Check if an EV result is actually a showroom/repair shop, not a charger."""
    if not name:
        return False
    name_lower = name.lower().strip()
    # If name contains "charging" or "charger", it's likely real
    if any(kw in name_lower for kw in ["charging", "charger", "charge point", "ev station"]):
        return False
    for pattern in FALSE_POSITIVE_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


def promote_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Promote known EV network candidates to commercial."""
    df = df.copy()
    promoted = 0
    removed  = 0

    for idx, row in df.iterrows():
        if row.get("infra_type") != "ev":
            continue

        name = str(row.get("name", ""))

        # Check false positives first
        if is_false_positive(name):
            df.at[idx, "ev_label"] = "false_positive"
            removed += 1
            continue

        # If already commercial, skip
        if row.get("ev_label") == "commercial":
            continue

        # Try brand promotion
        network = is_known_ev_network(name)
        if network:
            df.at[idx, "ev_label"] = "commercial"
            df.at[idx, "promotion_source"] = f"brand:{network}"
            if pd.isna(row.get("kw_max")) or row.get("kw_max") == 0:
                df.at[idx, "kw_max"] = 22.0  # conservative floor
            df.at[idx, "is_fast_charge"] = True
            promoted += 1

    print(f"  Brand promotion: {promoted} candidates → commercial")
    print(f"  False positives removed: {removed}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Google Places Nearby Search with type filter
# ──────────────────────────────────────────────────────────────────────────────

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
NEARBY_FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.evChargeOptions",
])


def _make_grid_circles(bbox: dict, n: int = 6) -> list[dict]:
    """Create n×n grid of circles covering the bbox for Nearby Search."""
    lat_step = (bbox["north"] - bbox["south"]) / n
    lon_step = (bbox["east"]  - bbox["west"])  / n
    # Radius to cover a cell (diagonal / 2 in metres)
    cell_diag_m = np.sqrt((lat_step * 111320)**2 + (lon_step * 111320 * np.cos(np.radians(18.5)))**2)
    radius_m = min(cell_diag_m / 2 * 1.1, 5000.0)  # cap at 5km

    circles = []
    for i in range(n):
        for j in range(n):
            center_lat = bbox["south"] + (i + 0.5) * lat_step
            center_lon = bbox["west"]  + (j + 0.5) * lon_step
            circles.append({
                "lat": center_lat,
                "lon": center_lon,
                "radius": radius_m,
            })
    return circles


def fetch_nearby_ev(city: str, delay_s: float = 0.5) -> list[dict]:
    """Fetch EV charging stations via Google Nearby Search (type-filtered)."""
    if not API_KEY:
        print("  [SKIP] No Google API key set")
        return []

    bbox = CITY_BBOXES[city]
    circles = _make_grid_circles(bbox, n=6)

    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   API_KEY,
        "X-Goog-FieldMask": NEARBY_FIELD_MASK,
    }

    all_places = []
    seen_ids = set()

    print(f"  Scanning {len(circles)} grid cells with type=electric_vehicle_charging_station...")

    for ci, circle in enumerate(circles):
        body = {
            "includedTypes": ["electric_vehicle_charging_station"],
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": circle["lat"], "longitude": circle["lon"]},
                    "radius": circle["radius"],
                }
            },
            "maxResultCount": 20,
            "languageCode": "en",
        }

        try:
            resp = requests.post(NEARBY_URL, headers=headers, json=body, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                places = data.get("places", [])
                for p in places:
                    pid = p.get("id", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        all_places.append(p)
            elif resp.status_code == 429:
                print(f"    Rate limited at cell {ci}, sleeping 10s...")
                time.sleep(10)
            else:
                pass  # skip silently
        except Exception as e:
            print(f"    Error at cell {ci}: {e}")

        if (ci + 1) % 12 == 0:
            print(f"    Scanned {ci+1}/{len(circles)} cells → {len(all_places)} unique stations")
        time.sleep(delay_s)

    print(f"  Nearby Search found: {len(all_places)} unique EV charging stations")
    return all_places


def _parse_nearby_places(places: list[dict], city: str) -> pd.DataFrame:
    """Convert Nearby Search results to DataFrame rows."""
    rows = []
    for p in places:
        loc = p.get("location", {})
        ev_opts = p.get("evChargeOptions", {})

        # Extract kW from evChargeOptions if available
        kw_max = 0.0
        connector_types = []
        if ev_opts:
            for ct in ev_opts.get("connectorAggregation", []):
                kw = ct.get("maxChargeRateKw", 0)
                if kw > kw_max:
                    kw_max = kw
                ctype = ct.get("type", "")
                if ctype:
                    connector_types.append(ctype)

        name = p.get("displayName", {}).get("text", "")

        rows.append({
            "place_id":           p.get("id", ""),
            "name":               name,
            "lat":                loc.get("latitude"),
            "lon":                loc.get("longitude"),
            "rating":             p.get("rating"),
            "user_ratings_total": p.get("userRatingCount"),
            "kw_max":             kw_max if kw_max > 0 else None,
            "connector_types":    "|".join(connector_types) if connector_types else None,
            "is_fast_charge":     kw_max >= 22.0 if kw_max > 0 else None,
            "infra_type":         "ev",
            "city":               city,
            "sources":            "google_nearby",
            "opening_date_est":   None,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Place Details for kW verification of remaining unknowns
# ──────────────────────────────────────────────────────────────────────────────

DETAILS_URL = "https://places.googleapis.com/v1/places/"
DETAILS_FIELD_MASK = "evChargeOptions,rating,userRatingCount"


def fetch_ev_details(place_ids: list[str], delay_s: float = 0.3) -> dict:
    """Fetch evChargeOptions for a list of place_ids. Returns {place_id: {kw_max, connectors}}."""
    if not API_KEY:
        return {}

    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   API_KEY,
        "X-Goog-FieldMask": DETAILS_FIELD_MASK,
    }

    results = {}
    print(f"  Fetching Place Details for {len(place_ids)} stations...")

    for i, pid in enumerate(place_ids):
        try:
            resp = requests.get(f"{DETAILS_URL}{pid}", headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                ev_opts = data.get("evChargeOptions", {})
                kw_max = 0.0
                connectors = []
                for ct in ev_opts.get("connectorAggregation", []):
                    kw = ct.get("maxChargeRateKw", 0)
                    if kw > kw_max:
                        kw_max = kw
                    ctype = ct.get("type", "")
                    if ctype:
                        connectors.append(ctype)

                results[pid] = {
                    "kw_max": kw_max,
                    "connector_types": "|".join(connectors),
                    "rating": data.get("rating"),
                    "user_ratings_total": data.get("userRatingCount"),
                }
            elif resp.status_code == 429:
                print(f"    Rate limited at #{i}, sleeping 10s...")
                time.sleep(10)
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(place_ids)}")
        time.sleep(delay_s)

    print(f"  Got evChargeOptions for {len(results)} stations")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Merge and Rebuild
# ──────────────────────────────────────────────────────────────────────────────

def merge_new_stations(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Spatially merge new stations into existing, deduplicating at 50m."""
    if new_df.empty:
        return existing_df

    existing_ev = existing_df[existing_df["infra_type"] == "ev"]
    if existing_ev.empty:
        return pd.concat([existing_df, new_df], ignore_index=True)

    tree = cKDTree(existing_ev[["lat", "lon"]].values)
    new_rows = []

    for _, row in new_df.iterrows():
        coord = np.array([row["lat"], row["lon"]])
        dist, idx = tree.query(coord, k=1)

        if dist > DEDUP_THRESHOLD_DEG:
            # New station not in existing data
            new_rows.append(row)
        else:
            # Update existing record with better data
            existing_idx = existing_ev.index[idx]
            # Enrich with kW data if we have it and existing doesn't
            if row.get("kw_max") and (pd.isna(existing_df.at[existing_idx, "kw_max"]) or
                                       existing_df.at[existing_idx, "kw_max"] == 0):
                existing_df.at[existing_idx, "kw_max"] = row["kw_max"]
                existing_df.at[existing_idx, "connector_types"] = row.get("connector_types")
                if row["kw_max"] >= 22.0:
                    existing_df.at[existing_idx, "is_fast_charge"] = True
            # Enrich with rating data
            if row.get("rating") and pd.isna(existing_df.at[existing_idx, "rating"]):
                existing_df.at[existing_idx, "rating"] = row["rating"]
                existing_df.at[existing_idx, "user_ratings_total"] = row.get("user_ratings_total")
            # Update sources
            existing_src = str(existing_df.at[existing_idx, "sources"] or "")
            if "google_nearby" not in existing_src:
                existing_df.at[existing_idx, "sources"] = existing_src + "|google_nearby"

    if new_rows:
        new_stations_df = pd.DataFrame(new_rows)
        print(f"  Added {len(new_stations_df)} new stations from Nearby Search")
        result = pd.concat([existing_df, new_stations_df], ignore_index=True)
    else:
        print(f"  No new stations (all found by Nearby Search were already in dataset)")
        result = existing_df

    return result


def relabel_ev(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel ev_label based on updated kw_max and brand matching."""
    df = df.copy()
    for idx, row in df.iterrows():
        if row.get("infra_type") != "ev":
            continue
        if row.get("ev_label") == "false_positive":
            continue

        kw = row.get("kw_max")
        name = str(row.get("name", ""))

        # Already commercial from OCM → keep
        if row.get("ev_label") == "commercial":
            continue

        # Has verified kW ≥ 22 from Place Details or Nearby Search
        if pd.notna(kw) and float(kw) >= 22.0:
            df.at[idx, "ev_label"] = "commercial"
            df.at[idx, "is_fast_charge"] = True
            continue

        # Brand match → commercial
        network = is_known_ev_network(name)
        if network:
            df.at[idx, "ev_label"] = "commercial"
            df.at[idx, "is_fast_charge"] = True
            if pd.isna(kw) or float(kw or 0) == 0:
                df.at[idx, "kw_max"] = 22.0
            continue

        # Has kW 7-22 → keep as candidate
        if pd.notna(kw) and 7.0 <= float(kw) < 22.0:
            df.at[idx, "ev_label"] = "candidate"
            continue

        # Name contains "charging" or "charger" but no brand → candidate
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["charging", "charger", "ev station", "charge point"]):
            df.at[idx, "ev_label"] = "candidate"
            continue

        # Everything else → candidate
        df.at[idx, "ev_label"] = "candidate"

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(city: str, dry_run: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  DSSE Phase 1.5 — EV Data Enhancement: {city.upper()}")
    print(f"{'='*60}\n")

    processed_dir = Path("data/processed") / city
    raw_dir = Path("data/raw") / city
    master_path = processed_dir / "stations_master.csv"

    if not master_path.exists():
        print(f"ERROR: {master_path} not found. Run scripts/01 first.")
        sys.exit(1)

    # Load current data
    df = pd.read_csv(master_path)
    n_before = len(df)
    n_commercial_before = (df["ev_label"] == "commercial").sum()
    n_candidate_before  = (df["ev_label"] == "candidate").sum()

    print(f"Current data: {n_before} stations")
    print(f"  Commercial EV: {n_commercial_before}")
    print(f"  Candidate EV:  {n_candidate_before}")
    print(f"  Petrol:        {(df['ev_label'] == 'petrol').sum()}")

    # ── Step 1: Brand Promotion ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 1: Brand-Based Promotion (zero API cost)")
    print(f"{'─'*60}\n")
    df = promote_by_brand(df)

    n_commercial_after_brand = (df["ev_label"] == "commercial").sum()
    promoted_count = n_commercial_after_brand - n_commercial_before
    print(f"\n  Commercial EV after brand promotion: {n_commercial_before} → {n_commercial_after_brand} (+{promoted_count})")

    if dry_run:
        # Show what would be promoted
        promoted_names = df[(df["ev_label"] == "commercial") & (df.get("promotion_source", pd.Series(dtype=str)).notna())]
        if "promotion_source" in df.columns:
            promoted_names = df[df["promotion_source"].notna()]
            print(f"\n  Promoted stations:")
            for _, row in promoted_names.head(30).iterrows():
                print(f"    ✓ {row['name']} → {row.get('promotion_source', 'brand')}")
        print(f"\n  [DRY RUN] Would continue with Nearby Search + Details. Stopping here.")
        print(f"  Re-run without --dry-run to execute API calls.")

        # Still save preview
        preview_path = processed_dir / "stations_master_preview.csv"
        df.to_csv(preview_path, index=False)
        print(f"  Preview saved → {preview_path}")
        return

    # ── Step 2: Google Nearby Search ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 2: Google Nearby Search (type=electric_vehicle_charging_station)")
    print(f"{'─'*60}\n")

    nearby_places = fetch_nearby_ev(city, delay_s=0.5)
    nearby_df = _parse_nearby_places(nearby_places, city)

    # Save raw nearby results
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not nearby_df.empty:
        nearby_df.to_csv(raw_dir / "google_nearby_ev.csv", index=False)
        print(f"  Saved raw → {raw_dir}/google_nearby_ev.csv")

    # Merge
    df = merge_new_stations(df, nearby_df)

    # ── Step 3: Place Details for remaining unknowns ─────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 3: Place Details (evChargeOptions) for unknowns")
    print(f"{'─'*60}\n")

    # Find EV candidates that still have no kW data and have a place_id
    unknowns = df[
        (df["infra_type"] == "ev") &
        (df["ev_label"] != "commercial") &
        (df["ev_label"] != "false_positive") &
        (df["place_id"].notna()) &
        (df["place_id"] != "") &
        ((df["kw_max"].isna()) | (df["kw_max"] == 0))
    ]

    if len(unknowns) > 0 and len(unknowns) <= 200:
        place_ids = unknowns["place_id"].tolist()
        details = fetch_ev_details(place_ids, delay_s=0.3)

        for pid, info in details.items():
            mask = df["place_id"] == pid
            if info.get("kw_max", 0) > 0:
                df.loc[mask, "kw_max"] = info["kw_max"]
                df.loc[mask, "connector_types"] = info.get("connector_types")
                if info["kw_max"] >= 22.0:
                    df.loc[mask, "is_fast_charge"] = True
            if info.get("rating"):
                df.loc[mask & df["rating"].isna(), "rating"] = info["rating"]
            if info.get("user_ratings_total"):
                df.loc[mask & df["user_ratings_total"].isna(), "user_ratings_total"] = info["user_ratings_total"]
    elif len(unknowns) > 200:
        print(f"  {len(unknowns)} unknowns — fetching first 200 to stay within budget")
        place_ids = unknowns["place_id"].tolist()[:200]
        details = fetch_ev_details(place_ids, delay_s=0.3)
        for pid, info in details.items():
            mask = df["place_id"] == pid
            if info.get("kw_max", 0) > 0:
                df.loc[mask, "kw_max"] = info["kw_max"]
                df.loc[mask, "connector_types"] = info.get("connector_types")
                if info["kw_max"] >= 22.0:
                    df.loc[mask, "is_fast_charge"] = True
            if info.get("rating"):
                df.loc[mask & df["rating"].isna(), "rating"] = info["rating"]
            if info.get("user_ratings_total"):
                df.loc[mask & df["user_ratings_total"].isna(), "user_ratings_total"] = info["user_ratings_total"]
    else:
        print(f"  No unknowns remaining — all EV stations have kW data or brand match")

    # ── Step 4: Final relabeling ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Step 4: Final Relabeling")
    print(f"{'─'*60}\n")

    df = relabel_ev(df)

    # Remove false positives
    n_fp = (df["ev_label"] == "false_positive").sum()
    df = df[df["ev_label"] != "false_positive"].reset_index(drop=True)
    print(f"  Removed {n_fp} false positives")

    # Reassign canonical IDs
    df["canonical_id"] = [f"DSSE_{i:05d}" for i in range(len(df))]

    # Drop temp columns
    if "promotion_source" in df.columns:
        df = df.drop(columns=["promotion_source"])

    # ── Summary ──────────────────────────────────────────────────────────────
    n_after = len(df)
    n_commercial_after = (df["ev_label"] == "commercial").sum()
    n_candidate_after  = (df["ev_label"] == "candidate").sum()
    n_petrol_after     = (df["ev_label"] == "petrol").sum()

    print(f"\n{'='*60}")
    print(f"  Enhancement Results")
    print(f"{'='*60}")
    print(f"                           Before    After     Δ")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Total stations       {n_before:>8}  {n_after:>8}   {n_after - n_before:+d}")
    print(f"  Commercial EV        {n_commercial_before:>8}  {n_commercial_after:>8}   {n_commercial_after - n_commercial_before:+d}")
    print(f"  Candidate EV         {n_candidate_before:>8}  {n_candidate_after:>8}   {n_candidate_after - n_candidate_before:+d}")
    print(f"  Petrol               {(n_before - n_commercial_before - n_candidate_before):>8}  {n_petrol_after:>8}   +0")
    print(f"{'='*60}")

    # ── Save ─────────────────────────────────────────────────────────────────
    # Backup original
    backup_path = processed_dir / "stations_master_backup_v1.csv"
    if not backup_path.exists():
        import shutil
        shutil.copy(master_path, backup_path)
        print(f"\n  Original backed up → {backup_path}")

    df.to_csv(master_path, index=False)
    print(f"  Updated master saved → {master_path}")

    # Save enhancement log
    log = {
        "city": city,
        "before": {"total": n_before, "commercial": int(n_commercial_before),
                   "candidate": int(n_candidate_before)},
        "after":  {"total": n_after, "commercial": int(n_commercial_after),
                   "candidate": int(n_candidate_after)},
        "false_positives_removed": int(n_fp),
        "new_from_nearby_search": int(n_after - n_before + n_fp),
    }
    log_path = processed_dir / "enhancement_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  ⚠  IMPORTANT: You must now re-run the following:")
    print(f"     python -X utf8 scripts/02_build_spatial_indices.py --city {city}")
    print(f"     python -X utf8 scripts/05_train_and_evaluate.py --city {city}")
    print(f"     python -X utf8 scripts/06_cache_heatmap.py --city {city}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSSE EV Data Enhancement")
    parser.add_argument("--city", type=str, default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview brand promotions only (no API calls)")
    args = parser.parse_args()
    main(args.city, dry_run=args.dry_run)
