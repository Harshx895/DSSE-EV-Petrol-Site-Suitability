# DSSE Project Context — Auto-Loaded Agent Reference

> **IMPORTANT:** This file is automatically loaded by every agent session working in this workspace.
> After making significant changes (new features, model retraining, data changes, bug fixes),
> UPDATE the relevant sections below so the next session doesn't have to re-analyze the codebase.
>
> Last updated: 2026-10-22 by Opus 4.6

---

## Project Identity

- **Name:** DSSE — Dynamic Site Suitability Engine
- **Purpose:** AI-driven site suitability analysis for Petrol Stations vs EV Charging Stations
- **City:** Pune only (Bengaluru ruled out)
- **Course:** EDI-5 (Entrepreneurship Development Initiative, Semester 5)
- **Team:** 4 members (2 Data, 2 ML)
- **GitHub:** https://github.com/Harshx895/DSSE-EV-Petrol-Site-Suitability (user: Harshx895, email: harshx895@gmail.com, PAT stored in .env)

---

## Architecture Overview

```
User inputs (lat, lon) → FeatureExtractor (13 KDTrees) → 16 spatial features
  → PU Classifier (viability) + Conformal Regressor (throughput)
  → Composite Score 0-100 → Four-Quadrant Decision → SHAP explanation
```

### Key Design Decisions
- **PU Learning** (not standard classification) because we don't have negative labels — a location without a station might still be a great location
- **Elkan-Noto calibration**: P(y=1|x) = P(s=1|x) / c, where c = E[s|y=1]
- **Spatial Block CV** (not random CV) to prevent geographic leakage
- **SFDS metric** (novel): Jensen-Shannon Divergence on normalized SHAP weight vectors to quantify spatial divergence between petrol and EV siting patterns

---

## Current Data State (as of 2026-10-22)

### stations_master.csv (832 rows)
| Type | Count | Source |
|---|---|---|
| Petrol | 363 | Google Places Text Search |
| EV Commercial | 328 | OCM (13 original) + Brand promotion (231) + Google Nearby Search (92) + Place Details kW verification |
| EV Candidate | 141 | Remaining after promotion (unknown kW) |

### Rating Coverage
- Petrol: 311/363 (86%) have ratings, median 296 reviews
- EV: 321/469 (68%) have ratings, **median 4 reviews** ← major issue for ACTI regressor

### POI Layers (17,412 features across 14 layers)
- residential: 6073, roads_secondary: 3467, retail: 2538, roads_primary: 2501
- restaurants: 839, bus_stops: 581, offices: 336, cafes: 321
- parking: 237, water: 205, supermarkets: 144, metro: 112
- power_discom: 45, power_osm: 13

### 13 KDTrees stored in `models/pune/kdtrees.pkl`
bus_stop, cafe, metro_station, office, parking, power, residential, restaurant, retail, roads_primary, roads_secondary, supermarket, water

---

## Current Model Performance (as of 2026-10-22)

| Model | Metric | Value | Notes |
|---|---|---|
|---|
| Petrol PU | AUC | 0.648 ± 0.034 | Moderate |
| EV PU | AUC | **0.668 ± 0.044** | Improved from 0.646 after Phase 1 |
| Petrol Regressor | R² | **−0.135** | Still worse than mean, but reduced weight |
| EV Regressor | R² | **−0.211** | Improved from -0.277 after Phase 1 |
| SFDS | | 0.3355 | Partial Decoupling (stable) |

### Composite Score Formula
```
Score = 0.85 × PU_viability + 0.15 × ACTI_regressor
```
Viability threshold: **45** (recalibrated from 60 for 0.85/0.15 formula)

### Heatmap (896 H3 hexagons) — as of Phase 1
- BEST_PETROL: 94, BOTH_VIABLE: 79, BEST_EV: 57, POOR_BOTH: 666
- EV score mean: 25.88 (honest, not inflated by regressor)

---

## 14 Features (FEATURE_NAMES order) — v2 after Phase 1

| # | Feature | Type | Source | Notes |
|---|---|---|---|---|
| 0 | dist_primary_highway_m | distance | roads_primary KDTree | Top feature for both models |
| 1 | dist_secondary_road_m | distance | roads_secondary KDTree | |
| 2 | cafe_count_500m | count@500m | cafe KDTree | Was 300m (83% zeros fixed) |
| 3 | restaurant_count_500m | count@500m | restaurant KDTree | Was 300m (64% zeros fixed) |
| 4 | retail_count_500m | count@500m | retail KDTree | |
| 5 | supermarket_count_500m | count@500m | supermarket KDTree | 78% zeros — near-dead |
| 6 | office_count_500m | count@500m | office KDTree | |
| 7 | bus_stop_count_500m | count@500m | bus_stop KDTree | |
| 8 | metro_station_dist_m | distance | metro_station KDTree | |
| 9 | residential_count_500m | count@500m | residential KDTree | |
| 10 | power_grid_dist_m | distance | power KDTree | |
| 11 | competitor_petrol_1km | count@1km | stations_master petrol | CV leakage corrected |
| 12 | competitor_ev_1km | count@1km | stations_master EV | CV leakage corrected |
| 13 | parking_count_500m | count@500m | parking KDTree | Was 300m (86% zeros fixed) |

**Removed from v1:** `dwell_index` (corr=0.982 with retail), `transit_accessibility` (corr=0.939 with bus_stops)

---

## ACTI (Age-Corrected Throughput Index)

### Formula
```python
ACTI = log10(1 + user_ratings_total) × (rating / 5.0)
# Age correction disabled — opening dates unreliable
```

### Known Problems
1. EV stations have median 4 reviews vs petrol median 296 → ACTI measures station age, not throughput
2. R² < 0 for both regressors → regressor is noise, not signal
3. The regressor gets 60% composite weight despite being harmful

File: `src/models/target_engineering.py`

---

## SFDS (Spatial Factor Divergence Score)

### Formula
```
SFDS = JSD(normalize(SHAP_petrol), normalize(SHAP_ev))
```
where JSD = Jensen-Shannon Divergence (base 2), range [0, 1]

### Interpretation Thresholds
- < 0.20: Co-Location Archetype
- 0.20–0.45: Partial Decoupling
- > 0.45: Decoupled Archetype

### Known Problems
1. Feature correlation inflates JSD (dwell_index ≈ retail, transit ≈ bus_stops)
2. Bootstrap resamples weight vector elements, not training data → measures sensitivity, not CI
3. Thresholds (0.20, 0.45) are arbitrary — no null distribution test

File: `src/models/sfds_engine.py`

---

## File Structure

```
5_EDI/
├── .env                          # GOOGLE_PLACES_API_KEY, OCM_API_KEY
├── data/
│   ├── raw/pune/                 # Raw API responses (Google, OCM)
│   ├── processed/pune/           # stations_master.csv, poi_layers.parquet, sfds_results
│   ├── processed/heatmaps/       # heatmap_pune.geojson (896 hexagons)
│   └── external/pune/            # power_grid_scraped.csv, ev_browser_extracted.csv
├── models/pune/                  # kdtrees.pkl, *_pu_*.pkl, *_reg_*.pkl, shap_importances.json
├── figures/                      # sfds_weights_pune.png, umap_feature_space_pune.png
├── scripts/
│   ├── 01_collect_stations.py    # Google Places + OCM data collection
│   ├── 01b_collect_osm_poi.py    # Overpass API → POI layers
│   ├── 01c_enhance_ev_data.py    # Brand promotion + Nearby Search + Place Details
│   ├── 02_build_spatial_indices.py # Build 13 KDTrees
│   ├── 05_train_and_evaluate.py  # Train PU + Regressor + SHAP + SFDS
│   ├── 06_cache_heatmap.py       # Score 896 H3 hexagons → GeoJSON
│   ├── 07_demo.py                # CLI demo (single point or interactive)
│   └── 08_visualize_heatmap.py   # Folium interactive map
├── src/
│   ├── api/main.py               # FastAPI backend (5 endpoints, tested working)
│   ├── data/                     # google_places_client, ocm_client, station_deduplicator
│   ├── models/                   # trainer, predictor, feature_extractor, sfds_engine, recommender, target_engineering
│   └── spatial/                  # constraints_filter.py (road/water/power hard constraints)
├── PROJECT_DOCUMENTATION.md      # Full A-to-Z explanation
├── DATA_TEAM_DOC.md              # Data team reference + Q&A
├── ML_TEAM_DOC.md                # ML team reference + Q&A
└── AGENTS.md                     # THIS FILE — auto-loaded agent context
```

---

## API Endpoints (FastAPI on port 8000)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Status check |
| `/api/cities` | GET | List available cities + readiness |
| `/api/score?lat=&lon=&city=` | GET | Full scoring with SHAP explanation |
| `/api/divergence/{city}` | GET | SFDS results |
| `/api/compare?lat1=&lon1=&lat2=&lon2=&city=` | GET | Compare two locations |

---

## Known Bugs & Issues (as of 2026-10-22)

### Fixed (2026-10-22 — Phase 1)
- ✅ scale_pos_weight hardcoded 3.0 → dynamic len(bg)/len(pos)
- ✅ CV competitor leakage → `_fix_competitor_leakage()` added
- ✅ Water body exclusion always False → now queries water KDTree
- ✅ constraints_filter.py never called → predictor now uses it
- ✅ SFDS "95% CI" mislabelled → "95% sensitivity range"
- ✅ Null bytes in `src/api/__init__.py` → cleaned
- ✅ Regressor 60% weight with R²<0 → now 15% weight (0.85/0.15 formula)
- ✅ Count features 83-86% zeros → radii widened 300m→500m
- ✅ dwell_index corr=0.982 with retail → removed
- ✅ transit_accessibility corr=0.939 with bus_stops → removed
- ✅ Threshold 60 miscalibrated for new formula → now 45
- ✅ FeatureExtractor default path bug (data/processed vs data/processed/city) → fixed

### Open Issues
- ⚠️ No OSM landuse feature (commercial/residential zone) → Phase 2
- ⚠️ SFDS bootstrap tests sensitivity not true CI → Phase 2
- ⚠️ SFDS thresholds arbitrary → needs null distribution test (Phase 2)
- ⚠️ Supermarket 78% zeros at 500m → consider widening to 1km or dropping

---

## Improvement Plan (prioritized)

### Phase 1: Quick Wins ✅ COMPLETE (2026-10-22)
1. ✅ Change composite weights to 0.85×PU + 0.15×Reg
2. ✅ Widen count radii: 300m → 500m for cafes, restaurants, parking
3. ✅ Drop dwell_index and transit_accessibility (redundant composites)
4. ✅ Fix FeatureExtractor path bug
5. ✅ Recalibrate threshold 60→45

### Phase 2: Paper Quality (NEXT)
1. Add OSM landuse zone feature (commercial/IT park/residential polygon)
2. SFDS with permutation importance (more robust than SHAP under correlation)
3. SFDS null distribution test (label-shuffle → publishable significance)
4. Feature ablation study (train without each feature group)
5. Proper data-resample bootstrap for SFDS CI

### Phase 3: Advanced (optional)
1. Road network betweenness centrality (OSMnx)
2. NASA VIIRS night light intensity
3. AHP baseline comparison table
4. Frontend dashboard

---

## Running the Pipeline

```bash
# Full pipeline (in order):
python -X utf8 scripts/01_collect_stations.py --city pune
python -X utf8 scripts/01b_collect_osm_poi.py --city pune
python -X utf8 scripts/01c_enhance_ev_data.py --city pune
python -X utf8 scripts/02_build_spatial_indices.py --city pune
python -X utf8 scripts/05_train_and_evaluate.py --city pune
python -X utf8 scripts/06_cache_heatmap.py --city pune
python -X utf8 scripts/07_demo.py --lat 18.5312 --lon 73.8474 --city pune

# API server:
python -m uvicorn src.api.main:app --port 8000
```

---

## Agent Instructions

1. **After every significant code change**, update this file's relevant section (data counts, model metrics, feature list, known issues).
2. **Don't re-analyze** the codebase from scratch — trust this file as the starting point, verify only what you need.
3. **All paths are relative** to `c:\Users\Harsh Gupta\5_EDI\` (the workspace root).
4. **PowerShell quirks**: use semicolons not `&&`, quote paths with spaces, `python -X utf8` for encoding.
5. **Google API**: Key in `.env`, ~$195 of $200 free credit remaining.
