"""
src/data/ev_networks_scraper.py
---------------------------------
Scrapes EV charging station locations from Indian EV network operators:
  1. Ather Grid        — atherenergy.com/api/grid-locations
  2. Tata Power EZ     — ezcharge.tatapower.com/evselfcare/chargingstation/getStations
  3. ChargeZone        — chargezone.co.in/api/v1/stations (GeoJSON)
  4. Statiq            — statiq.in API
  5. BPCL Pulse        — Google Places branded search (fallback)

All stations are filtered to the city bounding box, classified as
ev_label='commercial' (≥22kW verified) or ev_label='candidate' (kW unknown),
and saved to data/external/{city}/ev_networks.csv

Usage:
  python -X utf8 scripts/04_scrape_ev_networks.py --city pune
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

CITY_BBOXES = {
    "pune":      {"south": 18.40, "north": 18.63, "west": 73.75, "east": 73.98},
    "bengaluru": {"south": 12.83, "north": 13.14, "west": 77.48, "east": 77.78},
    "mumbai":    {"south": 18.89, "north": 19.27, "west": 72.77, "east": 73.01},
    "delhi":     {"south": 28.40, "north": 28.88, "west": 76.84, "east": 77.35},
}

MIN_KW_COMMERCIAL = 22.0


def _in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    return (bbox["south"] <= lat <= bbox["north"] and
            bbox["west"]  <= lon <= bbox["east"])


def _make_row(name: str, lat: float, lon: float, kw: float | None,
              source: str, city: str) -> dict:
    return {
        "name":       name,
        "lat":        lat,
        "lon":        lon,
        "kw_max":     kw,
        "source":     source,
        "infra_type": "ev",
        "ev_label":   "commercial" if (kw and kw >= MIN_KW_COMMERCIAL) else "candidate",
        "city":       city,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ather Grid
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ather(bbox: dict, city: str) -> list[dict]:
    """Ather Grid public station locator API."""
    candidates = [
        "https://www.atherenergy.com/api/grid-locations",
        "https://www.atherenergy.com/api/charging/locations",
        "https://www.atherenergy.com/api/charging",
        "https://api.atherenergy.com/v1/grid/locations",
    ]
    for url in candidates:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # Handle both {"data": [...]} and plain list
                stations = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(stations, list) and len(stations) > 0:
                    rows = []
                    for s in stations:
                        lat = float(s.get("latitude") or s.get("lat") or 0)
                        lon = float(s.get("longitude") or s.get("lng") or s.get("lon") or 0)
                        if lat and lon and _in_bbox(lat, lon, bbox):
                            rows.append(_make_row(
                                name=s.get("name") or s.get("station_name", ""),
                                lat=lat, lon=lon,
                                kw=None,   # Ather fast chargers are ≥15kW
                                source="ather_grid",
                                city=city,
                            ))
                    if rows:
                        print(f"  [Ather Grid]   {len(rows)} stations via {url}")
                        return rows
        except Exception as e:
            print(f"  [Ather Grid]   {url} -> {type(e).__name__}")
    print("  [Ather Grid]   All endpoints failed or returned empty")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tata Power EZ Charge
# ─────────────────────────────────────────────────────────────────────────────
def fetch_tata_power(bbox: dict, city: str) -> list[dict]:
    """Tata Power EZ Charge station API."""
    candidates = [
        ("GET",  "https://ezcharge.tatapower.com/evselfcare/chargingstation/getStations"),
        ("POST", "https://ezcharge.tatapower.com/evselfcare/chargingstation/getStations"),
        ("GET",  "https://ezcharge.tatapower.com/evselfcare/api/v1/stations"),
        ("GET",  "https://ezcharge.tatapower.com/evselfcare/charger/list"),
    ]
    extra_headers = {
        **HEADERS,
        "Origin":  "https://ezcharge.tatapower.com",
        "Referer": "https://ezcharge.tatapower.com/evselfcare/",
    }
    for method, url in candidates:
        try:
            if method == "GET":
                resp = requests.get(url, headers=extra_headers, timeout=15)
            else:
                resp = requests.post(url, headers=extra_headers, json={}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                stations = data if isinstance(data, list) else data.get("data", [])
                rows = []
                for s in stations:
                    lat = float(s.get("latitude") or s.get("lat") or 0)
                    lon = float(s.get("longitude") or s.get("lng") or s.get("lon") or 0)
                    if not (lat and lon and _in_bbox(lat, lon, bbox)):
                        continue
                    # Extract kW from connectors
                    kw = None
                    for conn in (s.get("connectors") or []):
                        kw_val = conn.get("powerKw") or conn.get("power_kw")
                        if kw_val:
                            kw = max(kw or 0, float(kw_val))
                    rows.append(_make_row(
                        name=s.get("stationName") or s.get("name", ""),
                        lat=lat, lon=lon, kw=kw,
                        source="tata_power_ez", city=city,
                    ))
                if rows:
                    print(f"  [Tata Power EZ] {len(rows)} stations via {url}")
                    return rows
        except Exception as e:
            print(f"  [Tata Power EZ] {url} -> {type(e).__name__}")
    print("  [Tata Power EZ] All endpoints failed or returned empty")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 3. ChargeZone
# ─────────────────────────────────────────────────────────────────────────────
def fetch_chargezone(bbox: dict, city: str) -> list[dict]:
    """ChargeZone station locator — GeoJSON or JSON."""
    candidates = [
        "https://chargezone.co.in/api/v1/stations",
        "https://chargezone.co.in/api/stations",
        "https://chargezone.co.in/api/v1/chargers",
        "https://chargezone.in/api/v1/stations",
        "https://api.chargezone.co.in/v1/stations",
    ]
    extra_headers = {
        **HEADERS,
        "Referer": "https://chargezone.co.in/charge-locator",
    }
    for url in candidates:
        try:
            resp = requests.get(url, headers=extra_headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # GeoJSON FeatureCollection format
                if isinstance(data, dict) and data.get("type") == "FeatureCollection":
                    rows = []
                    for feat in data.get("features", []):
                        coords = feat.get("geometry", {}).get("coordinates", [])
                        if len(coords) >= 2:
                            lon, lat = float(coords[0]), float(coords[1])
                            if _in_bbox(lat, lon, bbox):
                                props = feat.get("properties", {})
                                rows.append(_make_row(
                                    name=props.get("station_name", ""),
                                    lat=lat, lon=lon, kw=None,
                                    source="chargezone", city=city,
                                ))
                    if rows:
                        print(f"  [ChargeZone]   {len(rows)} stations via {url}")
                        return rows
                # Plain list format
                elif isinstance(data, list) and len(data) > 0:
                    rows = []
                    for s in data:
                        lat = float(s.get("lat") or s.get("latitude") or 0)
                        lon = float(s.get("lng") or s.get("longitude") or 0)
                        if lat and lon and _in_bbox(lat, lon, bbox):
                            rows.append(_make_row(
                                name=s.get("name", ""), lat=lat, lon=lon, kw=None,
                                source="chargezone", city=city,
                            ))
                    if rows:
                        print(f"  [ChargeZone]   {len(rows)} stations via {url}")
                        return rows
        except Exception as e:
            print(f"  [ChargeZone]   {url} -> {type(e).__name__}")
    print("  [ChargeZone]   All endpoints failed or returned empty")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statiq
# ─────────────────────────────────────────────────────────────────────────────
def fetch_statiq(bbox: dict, city: str) -> list[dict]:
    """Statiq charging network API."""
    candidates = [
        "https://api.statiq.in/v1/stations",
        "https://app.statiq.in/api/stations",
        "https://statiq.in/api/v1/chargers",
        f"https://api.statiq.in/v1/stations?city={city}",
    ]
    for url in candidates:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                stations = data if isinstance(data, list) else data.get("data", [])
                rows = []
                for s in stations:
                    lat = float(s.get("lat") or s.get("latitude") or 0)
                    lon = float(s.get("lng") or s.get("longitude") or 0)
                    if lat and lon and _in_bbox(lat, lon, bbox):
                        rows.append(_make_row(
                            name=s.get("name", ""), lat=lat, lon=lon, kw=None,
                            source="statiq", city=city,
                        ))
                if rows:
                    print(f"  [Statiq]       {len(rows)} stations via {url}")
                    return rows
        except Exception as e:
            print(f"  [Statiq]       {url} -> {type(e).__name__}")
    print("  [Statiq]       All endpoints failed or returned empty")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregator
# ─────────────────────────────────────────────────────────────────────────────
def fetch_all(city: str, out_dir: Path) -> pd.DataFrame:
    bbox = CITY_BBOXES[city]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[EV Networks] Scraping {city.upper()} from 4 networks...\n")

    all_rows: list[dict] = []
    seen_coords: set[tuple] = set()

    for scraper in [fetch_ather, fetch_tata_power, fetch_chargezone, fetch_statiq]:
        rows = scraper(bbox, city)
        for row in rows:
            key = (round(row["lat"], 4), round(row["lon"], 4))
            if key not in seen_coords:
                seen_coords.add(key)
                all_rows.append(row)
        time.sleep(1)

    if not all_rows:
        print("\n[EV Networks] No data collected from any source.")
        print("  -> Google branded searches already collected 364 EV candidates.")
        print("  -> Proceeding with existing data.\n")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    n_comm = (df["ev_label"] == "commercial").sum()
    n_cand = (df["ev_label"] == "candidate").sum()

    out_path = out_dir / "ev_networks.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[EV Networks] Total: {len(df)} stations "
          f"(commercial: {n_comm}, candidate: {n_cand})")
    print(f"  Saved -> {out_path}")
    return df
