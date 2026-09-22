"""
src/data/osm_ingest.py
───────────────────────
Downloads OSM POI layers for a city via a single combined Overpass API query.
Uses the maps.mail.ru Overpass mirror (confirmed reachable).

One request replaces 13 separate queries — typical response time: 15–30 seconds.

Output: data/processed/{city}/poi_layers_osm.parquet
Columns: layer, lat, lon, name (str), tags (JSON str)

Layers extracted:
  cafes, restaurants, supermarkets, retail, offices,
  bus_stops, metro_stations, residential, parking,
  power_osm, roads_primary, roads_secondary, water_exclusion
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

OVERPASS_URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # (south, north, west, east)
    "pune":      (18.40, 18.63, 73.75, 73.98),
    "bengaluru": (12.83, 13.14, 77.48, 77.78),
    "mumbai":    (18.89, 19.27, 72.77, 73.01),
    "delhi":     (28.40, 28.88, 76.84, 77.35),
}

# Tag → layer classification rules
TAG_TO_LAYER = [
    ("amenity", "cafe",            "cafes"),
    ("amenity", "restaurant",      "restaurants"),
    ("shop",    "supermarket",     "supermarkets"),
    ("highway", "bus_stop",        "bus_stops"),
    ("railway", "station",         "metro_stations"),
    ("railway", "subway_entrance", "metro_stations"),
    ("power",   "substation",      "power_osm"),
    ("power",   "transformer",     "power_osm"),
    ("natural", "water",           "water_exclusion"),
    ("amenity", "parking",         "parking"),
    ("amenity", "parking_space",   "parking"),
    ("highway", "primary",         "roads_primary"),
    ("highway", "trunk",           "roads_primary"),
    ("highway", "motorway",        "roads_primary"),
    ("highway", "primary_link",    "roads_primary"),
    ("highway", "trunk_link",      "roads_primary"),
    ("highway", "motorway_link",   "roads_primary"),
    ("highway", "secondary",       "roads_secondary"),
    ("highway", "tertiary",        "roads_secondary"),
    ("highway", "secondary_link",  "roads_secondary"),
    ("highway", "tertiary_link",   "roads_secondary"),
    ("building","residential",     "residential"),
    ("building","apartments",      "residential"),
]


def _classify(tags: dict) -> list[str]:
    """Return all matching layer names for a set of OSM tags."""
    layers: list[str] = []
    for key, val, layer in TAG_TO_LAYER:
        if tags.get(key) == val and layer not in layers:
            layers.append(layer)
    # Wildcard layers
    if "shop" in tags and "retail" not in layers:
        layers.append("retail")
    if "office" in tags and "offices" not in layers:
        layers.append("offices")
    # Drop private parking
    if "parking" in layers and tags.get("access") in ("private", "no"):
        layers.remove("parking")
    return layers


def _build_query(south: float, north: float, west: float, east: float) -> str:
    bbox = f"{south},{west},{north},{east}"
    return f"""
[out:json][timeout:300];
(
  node["amenity"~"cafe|restaurant|parking|parking_space"]({bbox});
  node["shop"]({bbox});
  node["office"]({bbox});
  node["highway"="bus_stop"]({bbox});
  node["railway"~"station|subway_entrance"]({bbox});
  node["power"~"substation|transformer"]({bbox});
  node["natural"="water"]({bbox});
  node["building"~"residential|apartments"]({bbox});
  way["highway"~"primary|trunk|motorway|secondary|tertiary|primary_link|trunk_link|motorway_link|secondary_link|tertiary_link"]({bbox});
  way["amenity"~"parking|parking_space"]["access"!~"private|no"]({bbox});
  way["natural"="water"]({bbox});
  way["building"~"residential|apartments"]({bbox});
);
out center tags;
"""


def fetch_city(city: str, out_dir: Path, retries: int = 3) -> pd.DataFrame:
    """
    Fetch all OSM POI layers for a city in a single Overpass query.
    Saves poi_layers_osm.parquet to out_dir. Returns the DataFrame.
    """
    if city not in CITY_BBOXES:
        raise ValueError(f"Unknown city '{city}'. Add its bbox to CITY_BBOXES.")

    south, north, west, east = CITY_BBOXES[city]
    out_dir.mkdir(parents=True, exist_ok=True)
    query = _build_query(south, north, west, east)

    print(f"\n[OSM] Single combined Overpass query for {city.upper()}...")
    print(f"      Mirror: {OVERPASS_URL}")
    print(f"      Bbox: S={south} N={north} W={west} E={east}")
    print("      Timeout: 300s — please wait...\n")

    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=320,
                headers={"Accept": "application/json"},
            )
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            print(f"  Response: {len(data['elements'])} elements in {elapsed:.1f}s")
            break
        except requests.exceptions.Timeout:
            print(f"  Attempt {attempt}/{retries}: Timeout. Retrying in 15s...")
            time.sleep(15)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  Attempt {attempt}/{retries}: {e}. Retrying...")
            time.sleep(15)

    # Parse elements into rows
    rows: list[dict] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if not tags:
            continue
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        elif el["type"] == "way" and "center" in el:
            lat, lon = el["center"].get("lat"), el["center"].get("lon")
        else:
            continue
        if lat is None or lon is None:
            continue
        for layer in _classify(tags):
            rows.append({
                "layer": layer,
                "lat":   lat,
                "lon":   lon,
                "name":  tags.get("name", ""),
                "tags":  json.dumps(tags),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[OSM] WARNING: No features extracted.")
        return df

    out_path = out_dir / "poi_layers_osm.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[OSM] Saved {len(df):,} rows → {out_path}")
    print("\n  Layer summary:")
    for layer, grp in df.groupby("layer"):
        print(f"    {layer:<25} {len(grp):>7}")

    return df


if __name__ == "__main__":
    fetch_city("pune", Path("data/processed/pune"))
