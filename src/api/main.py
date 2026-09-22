"""
src/api/main.py
-----------------
DSSE FastAPI Application — Real-Time Inference Engine

Endpoints (per CONTEXT §3):
  GET /api/health          → system status & loaded models
  GET /api/cities          → available cities & bounding boxes
  GET /api/score           → score a (lat, lon) in <15ms
  GET /api/heatmap/{city}  → pre-cached H3 GeoJSON layer
  GET /api/divergence/{city} → SFDS + feature weight comparison

Usage:
  uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models.predictor import Predictor
from src.models.recommender import QUADRANT_LABELS

# ── City configuration ─────────────────────────────────────────────────────────
CITIES = {
    "pune": {
        "name":  "Pune",
        "bbox":  {"south": 18.40, "west": 73.75, "north": 18.63, "east": 73.98},
        "state": "Maharashtra",
    },
    "bengaluru": {
        "name":  "Bengaluru",
        "bbox":  {"south": 12.83, "west": 77.48, "north": 13.14, "east": 77.78},
        "state": "Karnataka",
    },
    "mumbai": {
        "name":  "Mumbai",
        "bbox":  {"south": 18.89, "west": 72.77, "north": 19.27, "east": 73.01},
        "state": "Maharashtra",
    },
    "delhi": {
        "name":  "Delhi",
        "bbox":  {"south": 28.40, "west": 76.84, "north": 28.88, "east": 77.35},
        "state": "NCT",
    },
}

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
HEATMAP_DIR   = PROCESSED_DIR / "heatmaps"

# ── Predictor cache (loaded once at startup) ───────────────────────────────────
_predictors: dict[str, Predictor] = {}
_heatmap_cache: dict[str, dict]   = {}
_sfds_cache: dict[str, dict]      = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all available city models at startup."""
    for city in CITIES:
        model_path = MODELS_DIR / city / f"{city}_pu_petrol.pkl"
        if model_path.exists():
            try:
                _predictors[city] = Predictor(
                    city=city,
                    models_dir=MODELS_DIR,
                    processed_dir=PROCESSED_DIR / city,
                )
                print(f"  [startup] Loaded models: {city}")
            except Exception as e:
                print(f"  [startup] Failed to load {city}: {e}")

        # Cache SFDS results
        sfds_path = PROCESSED_DIR / city / f"sfds_results_{city}.json"
        if sfds_path.exists():
            with open(sfds_path) as f:
                _sfds_cache[city] = json.load(f)

    print(f"  [startup] Ready. Loaded cities: {list(_predictors.keys())}")
    yield
    # Cleanup
    _predictors.clear()
    _heatmap_cache.clear()


app = FastAPI(
    title="DSSE — Dynamic Site Suitability Engine",
    description="Real-time EV & petrol station siting intelligence for Indian cities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":         "ok",
        "loaded_cities":  list(_predictors.keys()),
        "total_cities":   len(CITIES),
        "model_version":  "1.0.0",
    }


@app.get("/api/cities")
def get_cities():
    result = []
    for city_id, info in CITIES.items():
        result.append({
            "id":       city_id,
            "name":     info["name"],
            "state":    info["state"],
            "bbox":     info["bbox"],
            "ready":    city_id in _predictors,
            "heatmap":  (HEATMAP_DIR / f"heatmap_{city_id}.geojson").exists(),
        })
    return {"cities": result}


@app.get("/api/score")
def score_location(
    lat:  float = Query(..., ge=6.0,  le=38.0,  description="Latitude (India range)"),
    lon:  float = Query(..., ge=68.0, le=97.0,  description="Longitude (India range)"),
    city: str   = Query("pune", description="Target city"),
):
    if city not in _predictors:
        raise HTTPException(
            status_code=503,
            detail=f"Models for '{city}' not loaded. Available: {list(_predictors.keys())}"
        )

    bbox = CITIES[city]["bbox"]
    if not (bbox["south"] <= lat <= bbox["north"] and
            bbox["west"]  <= lon <= bbox["east"]):
        raise HTTPException(
            status_code=400,
            detail=f"Coordinates ({lat}, {lon}) are outside {city} bounding box"
        )

    result = _predictors[city].score(lat, lon)
    result["quadrant_label"] = QUADRANT_LABELS.get(result["quadrant"], "")
    return result


@app.get("/api/heatmap/{city}")
def get_heatmap(city: str):
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    # Serve from in-memory cache if available
    if city in _heatmap_cache:
        return JSONResponse(content=_heatmap_cache[city])

    geojson_path = HEATMAP_DIR / f"heatmap_{city}.geojson"
    if not geojson_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Heatmap not yet generated for {city}. "
                   f"Run: python scripts/06_cache_heatmap.py --city {city}"
        )

    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    _heatmap_cache[city] = data    # cache in memory
    return JSONResponse(content=data)


@app.get("/api/divergence/{city}")
def get_divergence(city: str):
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")
    if city not in _sfds_cache:
        raise HTTPException(
            status_code=503,
            detail=f"SFDS not yet computed for {city}. Run training first."
        )
    return _sfds_cache[city]


@app.get("/api/compare")
def compare_locations(
    lat1: float = Query(...), lon1: float = Query(...),
    lat2: float = Query(...), lon2: float = Query(...),
    city: str   = Query("pune"),
):
    """Score two locations and return side-by-side comparison."""
    r1 = score_location(lat1, lon1, city)
    r2 = score_location(lat2, lon2, city)
    return {
        "location_a": r1,
        "location_b": r2,
        "winner_petrol": "A" if r1["petrol"]["composite"] >= r2["petrol"]["composite"] else "B",
        "winner_ev":     "A" if r1["ev"]["composite"]     >= r2["ev"]["composite"]     else "B",
    }
