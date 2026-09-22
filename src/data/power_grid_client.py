"""
src/data/power_grid_client.py
-------------------------------
Fetches electrical substation / transformer locations from multiple
India-specific open data sources to supplement the sparse OSM power layer.

Sources attempted (in order of quality):
  1. data.gov.in — India Open Government Data portal
     MSEDCL Pune division substation dataset (if available)
  2. POSOCO / CEA GIS data (Central Electricity Authority)
  3. OpenInfraMap API — crowdsourced power infrastructure, often better than OSM
  4. Wikidata SPARQL — named major substations in Pune district

Output: DataFrame with columns: lat, lon, layer='power_discom', name, tags
Saved to: data/external/pune/power_grid_scraped.csv

If all sources fail, returns empty DataFrame and logs instructions to
manually download MSEDCL shapefile from mahagis.maharashtra.gov.in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

PUNE_BBOX = {
    "south": 18.40, "north": 18.63,
    "west":  73.75, "east":  73.98,
}

HEADERS = {"User-Agent": "DSSE-Research/1.0 (academic; contact: research@dsse.in)"}


# ── Source 1: OpenInfraMap (crowdsourced power infra, better than OSM) ────────
def fetch_openinframap(bbox: dict) -> list[dict]:
    """
    OpenInfraMap provides power infrastructure from OSM but with better
    completeness than a raw Overpass query.
    """
    url = "https://openinframap.org/api/power"
    params = {
        "bbox": f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        "type": "substation,tower,generator",
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            rows = []
            for f in features:
                coords = f.get("geometry", {}).get("coordinates", [])
                props  = f.get("properties", {})
                if len(coords) >= 2:
                    rows.append({
                        "lat":   coords[1],
                        "lon":   coords[0],
                        "name":  props.get("name", ""),
                        "layer": "power_discom",
                        "tags":  json.dumps({"source": "openinframap",
                                             "power": props.get("type", "")}),
                    })
            print(f"  [OpenInfraMap] {len(rows)} power features")
            return rows
    except Exception as e:
        print(f"  [OpenInfraMap] Failed: {e}")
    return []


# ── Source 2: Wikidata SPARQL — named substations in Pune ────────────────────
def fetch_wikidata(bbox: dict) -> list[dict]:
    """
    Wikidata has named electrical substations with coordinates.
    SPARQL query for power substations near Pune.
    """
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?item ?itemLabel ?lat ?lon WHERE {{
      ?item wdt:P31/wdt:P279* wd:Q41176 .   # instance of: power substation
      ?item p:P625 ?coord .
      ?coord psv:P625 ?coordValue .
      ?coordValue wikibase:geoLatitude ?lat .
      ?coordValue wikibase:geoLongitude ?lon .
      FILTER(?lat >= {bbox['south']} && ?lat <= {bbox['north']})
      FILTER(?lon >= {bbox['west']} && ?lon <= {bbox['east']})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    LIMIT 200
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "json"},
            headers={**HEADERS, "Accept": "application/sparql-results+json"},
            timeout=30,
        )
        if resp.status_code == 200:
            bindings = resp.json().get("results", {}).get("bindings", [])
            rows = []
            for b in bindings:
                try:
                    rows.append({
                        "lat":   float(b["lat"]["value"]),
                        "lon":   float(b["lon"]["value"]),
                        "name":  b.get("itemLabel", {}).get("value", ""),
                        "layer": "power_discom",
                        "tags":  json.dumps({"source": "wikidata"}),
                    })
                except (KeyError, ValueError):
                    pass
            print(f"  [Wikidata]     {len(rows)} named substations")
            return rows
    except Exception as e:
        print(f"  [Wikidata] Failed: {e}")
    return []


# ── Source 3: Overpass targeted substation query via mail.ru mirror ───────────
def fetch_overpass_substations(bbox: dict) -> list[dict]:
    """
    Targeted Overpass query for ONLY substations/transformers.
    Much smaller than the full POI query — should complete in <30 seconds.
    """
    overpass_url = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    query = f"""
[out:json][timeout:60];
(
  node["power"~"substation|transformer|switch"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["power"~"substation|plant"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out center tags;
"""
    try:
        resp = requests.post(
            overpass_url,
            data={"data": query},
            timeout=75,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        rows = []
        for el in elements:
            tags = el.get("tags", {})
            if el["type"] == "node":
                lat, lon = el.get("lat"), el.get("lon")
            elif el["type"] == "way" and "center" in el:
                lat, lon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            if lat and lon:
                rows.append({
                    "lat":   lat,
                    "lon":   lon,
                    "name":  tags.get("name", ""),
                    "layer": "power_discom",
                    "tags":  json.dumps({**tags, "source": "overpass_targeted"}),
                })
        print(f"  [Overpass]     {len(rows)} substation/transformer nodes")
        return rows
    except Exception as e:
        print(f"  [Overpass] Failed: {e}")
    return []


def fetch_all_sources(city: str, out_dir: Path) -> pd.DataFrame:
    bbox = PUNE_BBOX  # extend to other cities later
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PowerGrid] Fetching substation data for {city.upper()} from all sources...")

    all_rows: list[dict] = []
    all_rows.extend(fetch_overpass_substations(bbox))
    time.sleep(2)
    all_rows.extend(fetch_wikidata(bbox))
    time.sleep(2)
    all_rows.extend(fetch_openinframap(bbox))

    if not all_rows:
        print("\n  [PowerGrid] All sources returned 0 results.")
        print("  Manual fix: Download MSEDCL shapefile from:")
        print("    https://mahagis.maharashtra.gov.in")
        print("  Save as: data/external/pune/discom_substations.shp")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Spatial deduplication (within 100m)
    from scipy.spatial import cKDTree
    import numpy as np
    if len(df) > 1:
        coords = df[["lat", "lon"]].values
        tree   = cKDTree(coords)
        pairs  = tree.query_pairs(r=0.0009)   # ~100m in degrees
        to_drop = {max(i, j) for i, j in pairs}
        df = df.drop(index=list(to_drop)).reset_index(drop=True)

    out_path = out_dir / "power_grid_scraped.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  [PowerGrid] {len(df)} unique substation points -> {out_path}")
    return df


if __name__ == "__main__":
    df = fetch_all_sources("pune", Path("data/external/pune"))
    print(df[["name", "lat", "lon", "tags"]].head(20))
