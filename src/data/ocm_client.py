"""
src/data/ocm_client.py  (v2)
------------------------------
Fetches EV charging stations from OpenChargeMap (OCM) REST API.

Change from v1:
  - Collects ALL chargers >= 7kW (not just >= 22kW)
  - kw_max is stored as a feature; the 22kW threshold is applied LATER
    in target_engineering.py to define the "commercial positive" label.
  - This gives more data for the PU Learning background set.

Returns DataFrame with columns:
  ocm_id, name, lat, lon, kw_max, connector_types, is_fast_charge,
  infra_type ('ev'), city
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OCM_API_KEY = os.getenv("OCM_API_KEY")
BASE_URL    = "https://api.openchargemap.io/v3/poi/"

# Collect everything >= 7kW (AC Type 2 and above)
# 22kW+ = commercial DC/fast AC -> positive label for EV classifier
# 7-22kW = slow commercial AC   -> used in PU background / regressor
MIN_KW_COLLECT = 7.0
MIN_KW_POSITIVE = 22.0   # used downstream in target_engineering.py

CITY_BBOXES = {
    "pune":      {"south": 18.40, "west": 73.75, "north": 18.63, "east": 73.98},
    "bengaluru": {"south": 12.83, "west": 77.48, "north": 13.14, "east": 77.78},
    "mumbai":    {"south": 18.89, "west": 72.77, "north": 19.27, "east": 73.01},
    "delhi":     {"south": 28.40, "west": 76.84, "north": 28.88, "east": 77.35},
}


def _get_max_kw(connections: list[dict]) -> float:
    kw_values = []
    for conn in connections:
        level = conn.get("Level", {}) or {}
        kw = level.get("KilowattsPerHour") or conn.get("PowerKW")
        if kw:
            try:
                kw_values.append(float(kw))
            except (ValueError, TypeError):
                pass
    return max(kw_values, default=0.0)


def _get_connector_types(connections: list[dict]) -> str:
    types = set()
    for conn in connections:
        ct = conn.get("ConnectionType", {}) or {}
        name = ct.get("Title") or ct.get("FormalName")
        if name:
            types.add(name)
    return "|".join(sorted(types))


def _parse_station(poi: dict, city: str) -> dict | None:
    addr_info  = poi.get("AddressInfo", {}) or {}
    connections = poi.get("Connections", []) or []

    lat = addr_info.get("Latitude")
    lon = addr_info.get("Longitude")
    if lat is None or lon is None:
        return None

    kw_max = _get_max_kw(connections)
    if kw_max < MIN_KW_COLLECT:
        return None

    return {
        "ocm_id":            str(poi.get("ID", "")),
        "name":              addr_info.get("Title", ""),
        "lat":               float(lat),
        "lon":               float(lon),
        "kw_max":            kw_max,
        "connector_types":   _get_connector_types(connections),
        "is_fast_charge":    kw_max >= MIN_KW_POSITIVE,
        "infra_type":        "ev",
        "city":              city,
    }


def fetch_city(city: str, out_dir: Path, delay_s: float = 1.0) -> pd.DataFrame:
    if not OCM_API_KEY:
        raise EnvironmentError("OCM_API_KEY not set in .env")

    bbox = CITY_BBOXES[city]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OCM v2] Fetching EV stations for {city} (>={MIN_KW_COLLECT}kW)...")

    params = {
        "key":         OCM_API_KEY,
        "boundingbox": (f"({bbox['south']},{bbox['west']}),"
                        f"({bbox['north']},{bbox['east']})"),
        "maxresults":  500,
        "compact":     False,
        "verbose":     False,
        "output":      "json",
    }

    all_pois: list[dict] = []
    offset = 0

    while True:
        params["offset"] = offset
        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"  OCM error at offset={offset}: {e}")
            break

        if not batch:
            break

        all_pois.extend(batch)
        print(f"  offset={offset}: {len(batch)} POIs | total: {len(all_pois)}")

        if len(batch) < 500:
            break
        offset += 500
        time.sleep(delay_s)

    # Save raw
    raw_path = out_dir / "raw_ocm.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_pois, f, indent=2)

    # Parse
    rows = [r for poi in all_pois if (r := _parse_station(poi, city)) is not None]
    df = pd.DataFrame(rows)

    n_fast = (df["is_fast_charge"] == True).sum() if not df.empty else 0
    n_slow = len(df) - n_fast
    print(f"  Total: {len(df)} stations "
          f"(>={MIN_KW_POSITIVE}kW fast: {n_fast}, {MIN_KW_COLLECT}-{MIN_KW_POSITIVE}kW slow: {n_slow})")

    return df


if __name__ == "__main__":
    df = fetch_city("pune", Path("data/raw/pune"))
    print(df[["name", "kw_max", "is_fast_charge"]].to_string())
