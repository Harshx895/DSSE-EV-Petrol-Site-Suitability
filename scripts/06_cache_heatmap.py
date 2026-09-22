"""
scripts/06_cache_heatmap.py
-----------------------------
DSSE Phase 3 — H3 Hexagonal Heatmap Pre-computation

Scores every H3 resolution-8 hexagon centroid in the city bounding box
using the trained PU viability models. Saves a GeoJSON layer for the
frontend Leaflet map overlay.

H3 Resolution 8:
  - Cell area: ~0.46 km²
  - Pune bbox (~230 km²) → ~500 hexagons
  - Scoring time: <30s on single CPU

Outputs:
  data/processed/heatmaps/heatmap_{city}.geojson
  data/processed/heatmaps/heatmap_{city}_meta.json

Usage:
  python -X utf8 scripts/06_cache_heatmap.py --city pune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import h3
import numpy as np
from tqdm import tqdm

from src.models.predictor import Predictor

CITY_BBOXES = {
    "pune":      (18.40, 18.63, 73.75, 73.98),
    "bengaluru": (12.83, 13.14, 77.48, 77.78),
    "mumbai":    (18.89, 19.27, 72.77, 73.01),
    "delhi":     (28.40, 28.88, 76.84, 77.35),
}

H3_RES = 8   # ~0.46 km² per cell — optimal for Pune per CONTEXT §3


def get_hex_centroids(city: str) -> list[tuple[str, float, float]]:
    """Return list of (hex_id, lat, lon) for all H3 res-8 cells in city bbox."""
    south, north, west, east = CITY_BBOXES[city]
    geojson_poly = {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south],
            [east, north], [west, north], [west, south],
        ]]
    }
    hexes = h3.polyfill_geojson(geojson_poly, H3_RES)
    result = []
    for hx in hexes:
        lat, lon = h3.h3_to_geo(hx)
        result.append((hx, lat, lon))
    return result


def build_heatmap(city: str) -> None:
    print(f"\n{'='*60}")
    print(f"  DSSE Phase 3 — H3 Heatmap: {city.upper()}")
    print(f"{'='*60}\n")

    out_dir = Path("data/processed/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor (KDTrees + models)
    print("[Step 1] Loading models...")
    predictor = Predictor(
        city=city,
        models_dir=Path("models"),
        processed_dir=Path("data/processed") / city,
    )

    # Get hex centroids
    print("[Step 2] Generating H3 grid...")
    hexes = get_hex_centroids(city)
    print(f"  {len(hexes)} hexagons at resolution {H3_RES}")

    # Score each hexagon centroid
    print("[Step 3] Scoring hexagons...")
    features = []
    for hx, lat, lon in tqdm(hexes, desc="Scoring"):
        try:
            result = predictor.score(lat, lon)
            features.append({
                "type": "Feature",
                "properties": {
                    "hex_id":       hx,
                    "petrol_score": round(result["petrol"]["composite"], 1),
                    "ev_score":     round(result["ev"]["composite"], 1),
                    "quadrant":     result["quadrant"],
                    "warnings":     result["warnings"],
                },
                "geometry": {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(hx, geo_json=True)]},
            })
        except Exception:
            continue

    geojson = {
        "type":     "FeatureCollection",
        "features": features,
    }

    # Save GeoJSON
    out_path = out_dir / f"heatmap_{city}.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))

    # Save metadata
    ev_scores     = [f["properties"]["ev_score"]     for f in features]
    petrol_scores = [f["properties"]["petrol_score"] for f in features]
    quadrant_counts = {}
    for f in features:
        q = f["properties"]["quadrant"]
        quadrant_counts[q] = quadrant_counts.get(q, 0) + 1

    meta = {
        "city":           city,
        "h3_resolution":  H3_RES,
        "n_hexagons":     len(features),
        "ev_score_mean":     round(float(np.mean(ev_scores)), 2),
        "ev_score_p75":      round(float(np.percentile(ev_scores, 75)), 2),
        "petrol_score_mean": round(float(np.mean(petrol_scores)), 2),
        "quadrant_counts":   quadrant_counts,
    }
    meta_path = out_dir / f"heatmap_{city}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Hexagons scored: {len(features)}")
    print(f"  EV score mean:   {meta['ev_score_mean']}")
    print(f"  Quadrant breakdown: {quadrant_counts}")
    print(f"\n  GeoJSON → {out_path}  ({out_path.stat().st_size//1024} KB)")
    print(f"  Meta    → {meta_path}")
    print(f"\n  Next: python -X utf8 src/api/main.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    args = parser.parse_args()
    build_heatmap(args.city)
