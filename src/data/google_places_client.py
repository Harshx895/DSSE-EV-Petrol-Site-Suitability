"""
src/data/google_places_client.py  (v2 — Grid-Tiled)
------------------------------------------------------
Divides the city bounding box into a grid of smaller tiles and queries
Google Places for each tile, giving up to 20x more results than a single
full-city query.

Grid: 5x5 = 25 tiles for Pune. Each tile queries 3 pages max = 60 results.
Total ceiling: 25 * 60 = 1,500 raw results -> ~400-700 after deduplication.

Also runs BRANDED EV SEARCHES to find charger networks that don't appear
under generic "EV charging" queries (Ather Grid, Tata Power, ChargeZone, etc.)

Returns a unified DataFrame with columns:
  place_id, name, lat, lon, rating, user_ratings_total,
  opening_date_est, infra_type ('petrol' | 'ev'), city
"""

from __future__ import annotations

import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY  = os.getenv("GOOGLE_PLACES_API_KEY")
BASE_URL = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.location",
    "places.rating",
    "places.userRatingCount",
])

CITY_BBOXES = {
    "pune":      {"south": 18.40, "west": 73.75, "north": 18.63, "east": 73.98},
    "bengaluru": {"south": 12.83, "west": 77.48, "north": 13.14, "east": 77.78},
    "mumbai":    {"south": 18.89, "west": 72.77, "north": 19.27, "east": 73.01},
    "delhi":     {"south": 28.40, "west": 76.84, "north": 28.88, "east": 77.35},
}

# Generic queries
GENERIC_QUERIES: dict[str, str] = {
    "petrol": "petrol pump OR fuel station OR petrol station",
    "ev":     "EV charging station OR electric vehicle charger OR EV charger",
}

# Branded EV network queries — finds stations that generic search misses
BRANDED_EV_QUERIES = [
    "Ather Grid charging station",
    "Tata Power EV charging",
    "ChargeZone charging station",
    "BPCL Pulse EV charging",
    "Fortum charging",
    "Charge+Zone",
    "Volttic charging station",
    "Magenta ChargeGrid",
    "Indian Oil EV charging",
    "HPCL EV charging",
    "ZEON Charging",
    "Statiq charging station",
]

GRID_DIVISIONS = 5   # 5x5 = 25 tiles


def _make_tiles(bbox: dict, n: int) -> list[dict]:
    """Divide bbox into n×n sub-tiles."""
    lat_step = (bbox["north"] - bbox["south"]) / n
    lon_step = (bbox["east"]  - bbox["west"])  / n
    tiles = []
    for i, j in product(range(n), range(n)):
        tiles.append({
            "south": bbox["south"] + i * lat_step,
            "north": bbox["south"] + (i + 1) * lat_step,
            "west":  bbox["west"]  + j * lon_step,
            "east":  bbox["west"]  + (j + 1) * lon_step,
        })
    return tiles


def _build_body(query: str, tile: dict, page_token: str | None = None) -> dict:
    body: dict = {
        "textQuery": query,
        "locationRestriction": {
            "rectangle": {
                "low":  {"latitude": tile["south"], "longitude": tile["west"]},
                "high": {"latitude": tile["north"], "longitude": tile["east"]},
            }
        },
        "maxResultCount": 20,
        "languageCode": "en",
    }
    if page_token:
        body["pageToken"] = page_token
    return body


def _fetch_page(query: str, tile: dict, page_token: str | None = None,
                retries: int = 3) -> tuple[list[dict], str | None]:
    headers = {
        "Content-Type":    "application/json",
        "X-Goog-Api-Key":  API_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    body = _build_body(query, tile, page_token)
    for attempt in range(retries):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=body, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("places", []), data.get("nextPageToken")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(5 * (attempt + 1))   # backoff on rate limit
            else:
                return [], None
        except Exception:
            return [], None
    return [], None


def _places_to_rows(places: list[dict], infra_type: str, city: str) -> list[dict]:
    rows = []
    for p in places:
        loc = p.get("location", {})
        rows.append({
            "place_id":           p.get("id", ""),
            "name":               p.get("displayName", {}).get("text", ""),
            "lat":                loc.get("latitude"),
            "lon":                loc.get("longitude"),
            "rating":             p.get("rating"),
            "user_ratings_total": p.get("userRatingCount"),
            "opening_date_est":   None,
            "infra_type":         infra_type,
            "city":               city,
        })
    return rows


def _query_tiles(query: str, tiles: list[dict], infra_type: str,
                 city: str, delay: float, max_pages: int = 3) -> list[dict]:
    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for tile in tiles:
        page_token = None
        for page in range(max_pages):
            places, page_token = _fetch_page(query, tile, page_token)
            for row in _places_to_rows(places, infra_type, city):
                if row["place_id"] and row["place_id"] not in seen_ids:
                    seen_ids.add(row["place_id"])
                    all_rows.append(row)
            time.sleep(delay)
            if not page_token:
                break

    return all_rows


def fetch_city(city: str, out_dir: Path, delay_s: float = 0.8) -> pd.DataFrame:
    if not API_KEY:
        raise EnvironmentError("GOOGLE_PLACES_API_KEY not set in .env")

    bbox  = CITY_BBOXES[city]
    tiles = _make_tiles(bbox, GRID_DIVISIONS)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Google Places v2] City: {city.upper()} | "
          f"Grid: {GRID_DIVISIONS}x{GRID_DIVISIONS}={len(tiles)} tiles\n")

    all_rows: list[dict] = []

    # ── 1. Generic tiled queries ──────────────────────────────────────────────
    for infra_type, query in GENERIC_QUERIES.items():
        print(f"  [{infra_type.upper()}] Generic query across {len(tiles)} tiles...")
        rows = _query_tiles(query, tiles, infra_type, city, delay_s)
        print(f"    -> {len(rows)} unique results")
        all_rows.extend(rows)

    # ── 2. Branded EV network queries (full-city bbox, not tiled) ─────────────
    print(f"\n  [EV] Branded network queries ({len(BRANDED_EV_QUERIES)} brands)...")
    ev_ids = {r["place_id"] for r in all_rows if r["infra_type"] == "ev"}
    branded_rows: list[dict] = []

    for brand_query in tqdm(BRANDED_EV_QUERIES, desc="    Brands", leave=False):
        # Use full city bbox for branded searches (they're sparse)
        places, _ = _fetch_page(brand_query, bbox)
        for row in _places_to_rows(places, "ev", city):
            if row["place_id"] and row["place_id"] not in ev_ids:
                ev_ids.add(row["place_id"])
                branded_rows.append(row)
        time.sleep(delay_s)

    print(f"    -> {len(branded_rows)} additional EV stations from branded search")
    all_rows.extend(branded_rows)

    # ── Finalise ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows).dropna(subset=["lat", "lon"]).reset_index(drop=True)

    n_petrol = (df["infra_type"] == "petrol").sum()
    n_ev     = (df["infra_type"] == "ev").sum()
    print(f"\n[Google Places v2] Total: {len(df)} places "
          f"({n_petrol} petrol, {n_ev} EV)")

    # Save raw
    raw_path = out_dir / "google_places_v2.csv"
    df.to_csv(raw_path, index=False)
    print(f"  Saved -> {raw_path}")

    return df


if __name__ == "__main__":
    df = fetch_city("pune", Path("data/raw/pune"))
    print(df[["name", "infra_type", "lat", "lon"]].head(20))
