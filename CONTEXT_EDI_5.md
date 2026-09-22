# CONTEXT.md — Dynamic Site Suitability Engine (DSSE)
> **Master Project Specification & Agent Instruction Manual**  
> Feed this file to any coding agent (Claude Code, Cursor, Windsurf, Antigravity, Copilot, etc.) at the start of every session.  
> **Last updated:** September 2026 | **Version:** 3.4 (Pune-First Scope + SFDS Statistical Hardening + UQ Scores + Site Comparison Mode)

---

## 1. PROJECT IDENTITY

- **Project Name:** Dynamic Site Suitability Engine (DSSE)
- **Academic Context:** Engineering Design and Innovation (EDI 5) Semester Capstone Project
- **Target Output:** High-impact Research Paper (IEEE Access / CEUS / ACM SIGSPATIAL) + Provisional Method Patent Application + Production-Grade Interactive Web Application
- **Domain:** Spatial Data Science, Urban Computing, Machine Learning, Clean Energy Infrastructure
- **Geographic Scope (Quad-City Study — Pune-First Development Strategy):**

  > ⚠️ **IMPLEMENTATION ORDER:** Build and validate the complete end-to-end pipeline for **Pune only** first. Every phase (data ingestion → model training → heatmap caching → API → frontend) must be verified and paper-ready for Pune before replicating to other cities. This prevents compounding data errors across cities and ensures a working demo exists at all times.

  1. **Pune, Maharashtra (Phase 1 — Active Development):** `south=18.40, west=73.75, north=18.63, east=73.98` (Polycentric automotive hub, high 2-wheeler EV adoption; MSEDCL power grid data available)
  2. **Bengaluru, Karnataka (Phase 2 — After Pune validated):** `south=12.83, west=77.48, north=13.14, east=77.78` (Sprawling tech corridors, highest private EV density; best OCM coverage in India)
  3. **Mumbai, Maharashtra (Phase 3):** `south=18.89, west=72.77, north=19.27, east=73.01` (Extreme land scarcity, linear transit bottlenecks; MSEDCL shared with Pune)
  4. **Delhi (NCT / NCR) (Phase 4):** `south=28.40, west=76.84, north=28.88, east=77.35` (Radial ring roads, aggressive EV fleet transition mandates; BRPL/BYPL data needed)

### Core Mission & The Two Dual Objectives
1. **The Product Objective (Real-Time Decision Support):**  
   Given any arbitrary geographic coordinate `(lat, lon)` or map pin in any target city, the engine instantly computes:
   - **Physical Siting Viability ($P_{\text{viability}}$):** Does this location have the spatial fabric required to sustain operational infrastructure? (0–100%)
   - **Footfall & Throughput Potential:** If built here, what is the expected customer activity and utilization volume? (0–100%)
   - **Dual Infrastructure Scoring:** Independent scores for both **Petrol Pump (ICE)** and **EV Fast Charging Station**.
   - **Four-Quadrant Decision State:** `BEST_PETROL` | `BEST_EV` | `BOTH_VIABLE (EV Preferred)` | `POOR_BOTH`.
   - **SHAP Local Explanation:** Granular waterfall attribution explaining exactly *why* the site scored high or low (e.g., `+18 pts cafe dwell proximity`, `-12 pts electrical transformer distance`).

2. **The Research & Patent Objective (Empirical Spatial Decoupling):**  
   Empirically discover how spatial success drivers differ between fossil-fuel stations (high-flow, 3-minute turnaround) and EV charging hubs (extended dwell times, 30–60 minutes, electrical grid constraints).  
   - Replace subjective, manual Analytic Hierarchy Process (AHP) weights with **Inverse MCDA Weight Learning** via Tree Ensembles.
   - Quantify infrastructure divergence using the **Spatial Factor Divergence Score (SFDS)** via Jensen-Shannon Divergence.
   - Compare SFDS across Pune, Bengaluru, Mumbai, and Delhi to prove how distinct urban topologies shape infrastructure decoupling.

---

## 2. THE DUAL-FORMULATION METHODOLOGY (SCIENTIFIC CORE)

To guarantee academic rigor and prevent the pitfalls of noisy star ratings, DSSE implements two complementary modeling formulations:

```
Candidate Location (lat, lon)
        │
        ├──▶ FORMULATION 1: Positive-Unlabeled (PU) Viability Classifier
        │      • Question: "Is this spot spatially suited to host infrastructure at all?"
        │      • Output: Viability Probability P(Viable | Spatial Features) ∈ [0, 100]
        │      • Purpose: Eliminates dead-ends, non-drivable alleys, and unsuitable parcels.
        │
        └──▶ FORMULATION 2: Customer Activity & Footfall Regressor
               • Question: "If operational, what is the expected customer throughput yield?"
               • Target Label: Y = log10(1 + user_ratings_total) * (rating / 5.0)
               • Output: Potential Footfall Intensity Index ∈ [0, 100]
               • Purpose: Distinguishes sleepy suburban sites from prime, high-yield retail hubs.
```

### 2a. Formulation 1: Positive-Unlabeled (PU) Spatial Viability
- **Ground Truth Problem:** In urban planning, an unbuilt plot is not a "failed" station; it is merely **unlabeled**. Traditional binary classification (treating non-stations as negative) causes massive sampling bias.
- **Positive Set ($P$):** All existing, operational petrol pumps and EV charging stations verified via Google Places, OpenChargeMap, and PlugShare API (triple-sourced to maximize EV station coverage). For EV stations, only those with `kw_rating >= 22kW` (semi-fast and DC fast chargers) are included — slow AC home chargers are excluded as they do not represent commercial infrastructure. (Capital expenditure spent = proven operational feasibility).
- **Unlabeled Background ($U$):** Uniformly sampled random background points / H3 hexagon centroids across the urban bounding box.
- **Model:** PU-Bagging with XGBoost / Random Forest Classifier (Elkan-Noto calibration). Prior $c = P(s=1 \mid y=1)$ estimated via iterative EM on the hold-out unlabeled set.
- **Mathematical Form:** Predicts $s(x) = P(y=1 \mid x)$, calibrated by the estimated prior $c = P(s=1 \mid y=1)$ to yield the true marginal suitability $P(y=1 \mid x) = \frac{P(s=1 \mid x)}{c}$.

### 2b. Formulation 2: Customer Activity / Footfall Intensity Index
- **Why Not Star Ratings (1.0–5.0)?** Raw ratings reflect washroom cleanliness, staff courtesy, or billing app bugs, and suffer from variance compression (almost all stations sit between 3.8 and 4.3 stars).
- **The Solution:** We derive an empirical **Age-Corrected Throughput Index**:
  $$Y_{\text{raw}} = \log_{10}(1 + \text{user\_ratings\_total}) \times \left(\frac{\text{rating}}{5.0}\right)$$
  $$Y = \frac{Y_{\text{raw}}}{\log_{10}(1 + M_{\text{open}})}$$
  where $M_{\text{open}}$ is the number of months since the listing's opening date (sourced from Google Places `opening_hours` editorial timestamp or `current_opening_hours`). This age-correction prevents a 10-year-old low-traffic station from outscoring a 1-year-old high-traffic station purely due to review accumulation.
  - `user_ratings_total` (Google review count) directly scales with cumulative customer transactions and vehicle volume (proven power-law proxy in urban computing literature).
  - The $\log_{10}$ scale stabilizes high-volume outliers (e.g., 4,000 reviews vs. 50 reviews).
  - Scaling by $(\text{rating} / 5.0)$ penalizes stations with persistent technical faults or severe dissatisfaction.
  - **Cross-Type Note:** $Y$ is type-normalized and used only within each infrastructure type's model. Absolute $Y$ values are **not** compared across petrol vs. EV types.
- **Model:** XGBoostRegressor / GradientBoostingRegressor predicting continuous $Y$.

---

## 3. SYSTEM ARCHITECTURE & DATA FLOW

```
[DATA INGESTION PHASE — Runs Once Offline per City]
  ├── Google Places API (New)   ──▶ Lat/Lon, Place ID, Rating, Review Count, Opening Date (Petrol & EV)
  ├── OpenChargeMap (OCM) API   ──▶ EV Charger Specs (kW rating, DC Fast vs AC Slow, Connectors)
  ├── PlugShare API (Free)      ──▶ Supplementary EV charger locations (catches OCM gaps in Pune/Delhi)
  ├── OSM / Geofabrik Extracts  ──▶ Road Network, Commercial POIs, Transit, Parking
  └── DISCOM Shapefiles         ──▶ Verified electrical substation/transformer locations (Primary power grid source)
                                      │   (MSEDCL for Pune/Mumbai · BESCOM for Bengaluru · BRPL/BYPL for Delhi)
                                      │   Fallback: OSM power=substation|transformer|line
                                      ▼
                        Saved locally in data/processed/{city}/:
                        • stations_master.csv  (Deduplicated, kW-filtered ground truth EV + petrol points)
                        • poi_layers.parquet   (Roads, POIs, Parking, Power Grid — merged OSM + DISCOM)

[MODEL TRAINING & RESEARCH PIPELINE — Runs Offline]
  ├── Local KDTree Spatial Feature Extractor (Extracts 16 spatial features per station)
  ├── City-Relative Z-Score Standardization (Solves cross-city density disparity)
  ├── Spatial Block Cross-Validation (Split by administrative wards/K-Means spatial clusters)
  ├── Train City-Specific Expert Models:
  │     • {city}_pu_petrol.pkl & {city}_pu_ev.pkl
  │     • {city}_reg_petrol.pkl & {city}_reg_ev.pkl
  ├── Tree SHAP Fit ({city}_shap_petrol.pkl, {city}_shap_ev.pkl)
  ├── Compute SFDS Metric (Jensen-Shannon Divergence across Pune, Bengaluru, Mumbai, Delhi)
  └── MLflow Experiment Tracking (Logs R², F1, Log-Loss, SFDS, Feature Importances)

[H3 HEXAGONAL HEATMAP CACHING — Runs Once Offline per City]
  ├── Uber H3 Grid (Resolution 8: ~0.46 km² per hexagon) covering city bounding boxes
  │     Resolution 8 selected rationale: finest resolution where Pune's full bbox (~230 km²) scores
  │     in <10 min on a single CPU; Res 9 (~0.1 km²) would require ~2,300 hexagons × 4 cities = too slow.
  │     Res 7 (~5.16 km²) is too coarse to distinguish intra-neighbourhood siting decisions.
  ├── Batch score all hex centroids using city-specific models
  └── Pre-render GeoJSON layers: heatmap_{city}.geojson

[FASTAPI BACKEND — Real-Time Inference Engine (<15ms)]
  ├── GET /api/cities           ──▶ Available cities, bounding boxes & active archetypes
  ├── GET /api/score            ──▶ Scores (lat, lon, city) via local KDTree + XGBoost + SHAP (<15ms)
  ├── GET /api/heatmap          ──▶ Serves pre-cached H3 GeoJSON layers
  ├── GET /api/divergence       ──▶ Feature weights & SFDS comparison for research view
  └── GET /api/health           ──▶ System status & model metadata

[REACT FRONTEND (Vite + Leaflet + Recharts + Tailwind)]
  ├── Interactive Map: Leaflet map with toggleable H3 heatmap overlay & City Selector dropdown
  ├── Score Card: Real-time dual gauge (Petrol vs EV) + 4-Quadrant Badge + Conformal Prediction Interval band
  ├── Site Comparison Mode: Pin two locations simultaneously; side-by-side score + SHAP diff panel
  ├── Explainability Panel: Dynamic SHAP waterfall / horizontal bar charts
  └── Research Dashboard: Side-by-side MCDA weight vector comparison, SFDS cross-city graph & UMAP scatter
```

---

## 4. SPATIAL FEATURE MATRIX & NORMALIZATION

Every candidate point `(lat, lon)` is profiled across 16 spatial features computed strictly from **local cached spatial layers** via `scipy.spatial.cKDTree` / GeoPandas (Zero live API calls at inference time):

| # | Feature Name | Raw Unit | Search Radius | Source Tag / Logic | Theoretical / Urban Justification |
|---|---|---|---|---|---|
| 1 | `dist_primary_highway_m` | Meters (float) | Nearest Neighbor | `highway=primary\|trunk\|motorway` | Arterial vehicle throughput & accessibility |
| 2 | `dist_secondary_road_m` | Meters (float) | Nearest Neighbor | `highway=secondary\|tertiary` | Feeder street ingress/egress |
| 3 | `cafe_count_300m` | Count (int) | Radius: 300m | `amenity=cafe` | Dwell-time hospitality for EV charging (30–45 min) |
| 4 | `restaurant_count_300m` | Count (int) | Radius: 300m | `amenity=restaurant` | Food & dining dwell opportunity |
| 5 | `retail_count_500m` | Count (int) | Radius: 500m | `shop=*` | Errand / shopping dwell compatibility |
| 6 | `supermarket_count_500m`| Count (int) | Radius: 500m | `shop=supermarket` | High-frequency weekly destination |
| 7 | `office_count_500m` | Count (int) | Radius: 500m | `office=*` | Daytime commuter parking & charging demand |
| 8 | `bus_stop_count_500m` | Count (int) | Radius: 500m | `highway=bus_stop` | Multimodal transit node footfall |
| 9 | `metro_station_dist_m` | Meters (float) | Nearest Neighbor | `railway=station\|subway_entrance` | Rapid transit hub proximity |
| 10| `residential_count_500m`| Count (int) | Radius: 500m | `building=residential\|apartments` | Overnight residential charging / neighborhood fuel |
| 11| `power_grid_dist_m` | Meters (float) | Nearest Neighbor | **Primary:** DISCOM shapefiles (MSEDCL/BESCOM/BRPL) · **Fallback:** OSM `power=substation\|transformer\|line` | **Critical Real-World EV Bottleneck:** Grid connection cost; DISCOM data used preferentially as OSM power tagging in India is sparse (~25–55% coverage) |
| 12| `competitor_petrol_1km` | Count (int) | Radius: 1km | Existing Petrol Pumps | Cluster agglomeration vs. market cannibalization |
| 13| `competitor_ev_1km` | Count (int) | Radius: 1km | Existing EV Chargers | Charger density & saturation |
| 14| `dwell_index` | Composite | Radius: 500m | `(cafe_300m * 2) + restaurant_300m + retail_500m + (parking_300m * 1.5)` | Synthesized human dwell-time opportunity; parking weighted at 1.5× as a hard operational prerequisite for EV charging |
| 15| `transit_accessibility` | Composite | Radius: 500m | `bus_stops_500m + (1000 / (metro_dist_m + 100))` | Overall transit corridor connectivity |
| 16| `parking_count_300m` | Count (int) | Radius: 300m | OSM: `amenity=parking OR amenity=parking_space OR parking=surface\|multi-storey\|underground` | **Critical EV Operational Constraint:** EV charging requires dwell-capable parking bays; petrol stations self-contain parking; high count signals dwell-compatible forecourt. Broad OSM query used to compensate for inconsistent Indian tagging. |

### City-Relative Spatial Normalization (Handling Density Disparity)
To prevent extreme baseline differences between dense island cities (Mumbai) and polycentric/sprawling regions (Pune, Bengaluru) from distorting multi-city evaluations, all count-based POI features are transformed using **City-Relative Z-Scores**:

$$Z_{\text{feature}, i} = \frac{X_{\text{feature}, i} - \mu_{\text{feature, city}}}{\sigma_{\text{feature, city}}}$$

- A score of $+1.5\sigma$ in Pune (15 cafes) and $+1.5\sigma$ in Mumbai (75 cafes) both represent the **top 10% highest dwell density** in their respective urban systems.
- Raw values are preserved in the UI response for human readability, while normalized values feed the statistical comparisons.
- `parking_count_300m` is normalized city-relative like all count features; the 1.5× weight in `dwell_index` applies to the **raw count** before normalization to preserve the operational primacy signal.

---

## 5. HARD SPATIAL CONSTRAINTS (PRE-ML FILTER)

Before evaluating machine learning models, any candidate location must pass a deterministic feasibility filter:
1. **Setback Rule:** Must be within $\le 150\text{ meters}$ of a motorized road (`highway!=footway|cycleway|steps`).
2. **Exclusion Mask:** Cannot fall inside water bodies (`natural=water`), military/defense reservations, or public parks/sanctuaries.
3. **Power Grid Hard Ceiling (EV DC Fast Charge):** If `power_grid_dist_m > 2500\text{ meters}`, trigger a warning badge: *"High Transformer Substation Interconnection Capex"*.

---

## 6. FOUR-QUADRANT RECOMMENDATION LOGIC

The final decision state integrates the normalized composite scores ($0–100$) for both infrastructure types:

$$\text{Score}_{\text{final}} = 0.4 \times \text{Viability}_{\text{PU}} + 0.6 \times \text{Throughput}_{\text{Regression}}$$

| State Code | Condition ($\text{Threshold} = 60/100$) | Semantic Meaning & Strategic Action |
|---|---|---|
| `BEST_PETROL` | $\text{Petrol} \ge 60 \text{ and } \text{EV} < 60$ | **High-Throughput Transit Node:** Ideal for rapid fuel turnover; low dwell amenities for EV. |
| `BEST_EV` | $\text{EV} \ge 60 \text{ and } \text{Petrol} < 60$ | **High-Dwell Destination Hub:** Commercial/cafe district; unsuitable for heavy fuel tanker ingress. |
| `BOTH_VIABLE` | $\text{Petrol} \ge 60 \text{ and } \text{EV} \ge 60$ | **Prime Multi-Modal Interchange:** Highly viable for both; **Recommend EV** or Hybrid forecourt (future-proofing). |
| `POOR_BOTH` | $\text{Petrol} < 60 \text{ and } \text{EV} < 60$ | **Infrastructural Dead Zone:** Unviable for either; residential interior, remote, or inaccessible. |

---

## 7. SPATIAL FACTOR DIVERGENCE SCORE (SFDS) — THE RESEARCH CORE

SFDS mathematically quantifies how fundamentally different the spatial environments for petrol and EV stations are:

```python
from scipy.spatial.distance import jensenshannon
from sklearn.utils import resample
import numpy as np
import umap

# --- Point Estimate ---
# Feature importance vectors from fitted models (sum normalized to 1.0)
p = weights_petrol / np.sum(weights_petrol)
q = weights_ev / np.sum(weights_ev)
sfds_point = jensenshannon(p, q, base=2)  # bounded [0, 1]

# --- Bootstrapped 95% Confidence Interval (n=1000 resamples) ---
# Required for claiming cities differ significantly in the paper.
sfds_samples = []
for _ in range(1000):
    p_boot = resample(p, replace=True)
    q_boot = resample(q, replace=True)
    p_boot /= p_boot.sum()
    q_boot /= q_boot.sum()
    sfds_samples.append(jensenshannon(p_boot, q_boot, base=2))
sfds_ci_low, sfds_ci_high = np.percentile(sfds_samples, [2.5, 97.5])
# Report as: SFDS = 0.41 [95% CI: 0.38–0.44]

# --- UMAP Feature Space Scatter (Paper Figure) ---
# Visualises the decoupling of petrol vs EV feature environments directly.
# Run once after training; save scatter plot as figures/umap_feature_space.png
embedder = umap.UMAP(n_components=2, random_state=42)
X_all = np.vstack([X_petrol_stations, X_ev_stations])  # raw 16-feature matrix
labels = ['Petrol'] * len(X_petrol_stations) + ['EV'] * len(X_ev_stations)
embedding = embedder.fit_transform(X_all)  # colour-code by label in scatter
```

- $\text{SFDS} \to 0$: **Co-Location Archetype.** Both infrastructure types succeed in the exact same spatial environments (e.g., highway corridors).
- $\text{SFDS} \to 1$: **Decoupled Archetype.** EV infrastructure has completely detached from transit highways, migrating into commercial, retail, and dwell zones.
- **Statistical Reporting (mandatory for paper):** Always report SFDS as `point estimate [95% bootstrap CI]`. Cross-city differences may only be claimed significant if CIs do not overlap.
- **The Quad-City Comparison (built city-by-city):**
  - Start with $\text{SFDS}_{\text{Pune}}$ (Phase 1). Add cities as data is validated.
  - Final paper: $\text{SFDS}_{\text{Pune}}$ vs. $\text{SFDS}_{\text{Bengaluru}}$ vs. $\text{SFDS}_{\text{Mumbai}}$ vs. $\text{SFDS}_{\text{Delhi}}$.
  - Proves how city topography (radial vs. linear vs. sprawling) and urban density shape the EV transition.

---

## 8. SPATIAL HETEROGENEITY & CROSS-CITY DOMAIN ADAPTATION

To address the reality that different cities have different spatial dynamics without blurring predictive accuracy, the architecture adopts a **City-Specific Expert Model** strategy coupled with **Transferability Benchmarks**:

```
                       [City Selector Dropdown in UI]
                                     │
           ┌────────────────┬────────┴───────┬────────────────┐
           ▼                ▼                ▼                ▼
     [Pune Models]   [Bengaluru Models] [Mumbai Models]  [Delhi Models]
      Tuned to 2W      Tuned to Tech    Tuned to Land    Tuned to Ring
      Automotive Hub   Park Corridors   Scarcity/Transit Roads & Fleets
```

### The Three-Way Experimental Paradigm (Research Paper Contribution)
In the research paper, we benchmark three distinct modeling paradigms across all four cities:
1. **Intra-City Expert (Baseline):** Trained on City $A$ $\to$ Evaluated on City $A$ (via Spatial Block CV).
2. **Cross-City Zero-Shot Transfer:** Trained on City $A+B+C$ $\to$ Evaluated on unseen City $D$ (evaluates model generalization across India).
3. **Unified Standardized Model:** Trained on all four cities combined using City-Relative Z-Score feature standardization.

*Key Empirical Finding to Report:* Identify which spatial features are **universally invariant** across Indian cities (e.g., primary highway distance and electrical transformer proximity) vs. which features are **locally idiosyncratic** (e.g., cafe density thresholds).

---

## 9. SPATIAL VALIDATION PROTOCOL (ZERO DATA LEAKAGE)

Standard random $K$-fold cross-validation is **strictly prohibited** because spatial points within 300 meters share near-identical neighborhood features (spatial autocorrelation). Using standard CV inflates $R^2$ and guarantees academic rejection.

- **Mandatory Protocol:** **Spatial Block Cross-Validation (Spatial CV)**
  - Group locations into $K=5$ disjoint geographic spatial blocks (using administrative wards or $K$-Means clustering on coordinates `lat, lon`).
  - Iteratively train on 4 blocks and test on the unseen 5th geographic block.
  - Ensures the model generalizes to completely unobserved neighborhoods of the city.

---

## 10. TECH STACK & COMPUTATIONAL SPECIFICATIONS

### Backend (Python 3.11+)
- **API Framework:** FastAPI (Asynchronous endpoints, CORS-enabled, auto OpenAPI docs)
- **Spatial Processing:** GeoPandas, Shapely, PySCD/DuckDB, H3-py (Uber H3 hex indexing)
- **Machine Learning:** Scikit-learn, XGBoost, SHAP (`TreeExplainer`), MAPIE (`ConformalizedQuantileRegressor` for UQ)
- **Uncertainty Quantification:** MAPIE library — wraps XGBoost regressor with conformal prediction to produce calibrated score intervals (e.g., `EV Score: 74 ± 8`) at inference time with <1ms overhead.
- **Dimensionality Analysis:** UMAP (`umap-learn`) — produces 2D feature space scatter plot (petrol vs EV stations) used as a paper figure and a diagnostic for SFDS.
- **Spatial Indexing:** `scipy.spatial.cKDTree` (Sub-millisecond nearest neighbor & radius queries)
- **Experiment Tracking:** MLflow (Logs parameters, metrics, SFDS point + CI, UMAP embeddings, model artifacts locally)

### Frontend (Modern React SPA)
- **Framework:** React 18+ (Vite build tool)
- **Styling:** Vanilla CSS & Tailwind CSS (Dark/Light mode, glassmorphism, responsive)
- **Map Engine:** Leaflet.js with `react-leaflet` (Custom H3 polygon GeoJSON renderer)
- **Data Visualizations:** Recharts (SHAP waterfall bars, feature importance radars, dual gauges)

### Data Sources (100% Free / Zero Out-of-Pocket Cost)
1. **Google Places API (New):** Ground-truth coordinates, star ratings, review counts, and opening date. (Covered completely by Google Cloud's $200/month recurring free credit; 1,000 places cost ~$1.60). Also the primary source for EV station discovery via `type=electric_vehicle_charging_station`.
2. **OpenChargeMap (OCM) API:** Open global EV charger registry with technical charger specs (Free REST API key). Filtered to `kw_rating >= 22kW` to retain only commercial-grade fast chargers.
3. **PlugShare API (Free Tier):** Supplementary EV charger registry — largest EV community database in India; captures mall, corporate campus, and destination chargers missed by OCM. Used for Pune and Delhi where OCM coverage is sparse.
4. **OpenStreetMap (OSM) / Geofabrik:** Pune, Bengaluru, Mumbai, and Delhi `.osm.pbf` and Overpass extracts for roads, POIs, parking, and electrical infrastructure (100% free open data).
5. **DISCOM Electrical Substation Shapefiles (Free Government Open Data):**
   - **Pune & Mumbai (MSEDCL):** Maharashtra GIS Portal / data.maharashtra.gov.in
   - **Bengaluru (BESCOM):** Karnataka Open Data Portal / data.karnataka.gov.in
   - **Delhi (BRPL/BYPL/TPDDL):** Delhi Data Portal / data.delhi.gov.in
   Used as the primary source for `power_grid_dist_m` (Feature #11). OSM power data is a fallback only.


---

## 11. REPOSITORY DIRECTORY STRUCTURE

```
5_EDI/
├── CONTEXT.md                         ← Master Project Specification (This File)
├── README.md                          ← Project overview & quickstart
├── .env.example                       ← Template for API keys
├── requirements.txt                   ← Python dependencies
│
├── data/
│   ├── raw/                           ← Untouched API responses per city
│   │   ├── pune/
│   │   ├── bengaluru/
│   │   ├── mumbai/
│   │   └── delhi/
│   ├── processed/                     ← Cleaned, spatial-indexed local data banks
│   │   ├── pune/
│   │   │   ├── stations_master.csv
│   │   │   └── poi_layers.parquet
│   │   ├── bengaluru/
│   │   ├── mumbai/
│   │   ├── delhi/
│   │   └── heatmaps/                  ← Pre-rendered H3 GeoJSONs (res 8)
│   │       ├── heatmap_pune.geojson
│   │       ├── heatmap_bengaluru.geojson
│   │       ├── heatmap_mumbai.geojson
│   │       └── heatmap_delhi.geojson
│   └── external/                      ← Ward boundaries, DISCOM substation shapefiles & municipal GIS layers
│
├── src/
│   ├── data/
│   │   ├── google_places_client.py    ← Google Places API batch collector (with opening_date field)
│   │   ├── ocm_client.py              ← OpenChargeMap API connector (kW filter >= 22kW)
│   │   ├── plugshare_client.py        ← PlugShare API supplementary EV collector
│   │   ├── discom_ingest.py           ← DISCOM shapefile loader (MSEDCL / BESCOM / BRPL)
│   │   ├── osm_ingest.py              ← OSM layer extractor (roads, POIs, power, parking)
│   │   ├── station_deduplicator.py    ← Spatial dedup of OCM + Google + PlugShare (<50m = same station)
│   │   └── spatial_index.py           ← KDTree builder for sub-15ms local queries
│   │
│   ├── models/
│   │   ├── target_engineering.py      ← PU labeler, Age-Corrected Throughput Index & AHP baseline
│   │   ├── spatial_cv.py              ← Spatial Block Cross-Validation splitter
│   │   ├── trainer.py                 ← City-specific & unified XGBoost training + MAPIE conformal wrapper
│   │   ├── predictor.py               ← Real-time inference engine + SHAP explainer + UQ interval
│   │   ├── recommender.py             ← Four-quadrant decision classifier
│   │   ├── sfds_engine.py             ← SFDS point estimate + bootstrap CI + UMAP scatter generator
│   │   └── ahp_baseline.py            ← AHP benchmark model with expert-assigned weights
│   │
│   ├── spatial/
│   │   ├── h3_hex_grid.py             ← H3 polyfill and centroid generator
│   │   └── constraints_filter.py      ← Hard setback & exclusion rule checker
│   │
│   └── api/
│       ├── main.py                    ← FastAPI application entrypoint
│       ├── routes/
│       │   ├── score.py               ← GET /api/score (Real-time location scorer)
│       │   ├── heatmap.py             ← GET /api/heatmap (Pre-computed GeoJSON)
│       │   ├── divergence.py          ← GET /api/divergence (SFDS & weights)
│       │   └── cities.py              ← GET /api/cities (City metadata & bboxes)
│       └── schemas.py                 ← Pydantic response models
│
├── frontend/                          ← Modern React Web Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── MapView.jsx            ← Leaflet map, H3 layer, click marker, comparison pin B
│   │   │   ├── ScoreCard.jsx          ← Dual petrol/EV gauge + UQ interval band + recommendation badge
│   │   │   ├── SiteComparison.jsx     ← Side-by-side dual ScoreCard + SHAP diff panel (Pin A vs Pin B)
│   │   │   ├── ShapWaterfall.jsx      ← Explainability waterfall chart
│   │   │   ├── ResearchView.jsx       ← SFDS + bootstrap CI bars, MCDA radar, UMAP scatter
│   │   │   └── CitySelector.jsx       ← Dropdown: Pune (active) | Bengaluru | Mumbai | Delhi
│   │   ├── hooks/
│   │   │   ├── useSiteScorer.js       ← Single-pin asynchronous API caller
│   │   │   └── useComparison.js       ← Two-pin state manager for Site Comparison Mode
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── scripts/
│   ├── 01_download_raw_data.py        ← Batch data collection pipeline
│   ├── 02_build_spatial_indices.py    ← Build local Parquet & KDTree caches
│   ├── 03_train_and_evaluate.py       ← Run Spatial CV, train models, save PKLs
│   └── 04_generate_heatmaps.py        ← Pre-compute H3 grid suitability GeoJSONs
│
└── models/                            ← Serialized model artifacts per city
    ├── pune/
    │   ├── pu_petrol.pkl
    │   ├── pu_ev.pkl
    │   ├── reg_petrol.pkl
    │   ├── reg_ev.pkl
    │   ├── shap_petrol.pkl
    │   ├── shap_ev.pkl
    │   └── metadata.json
    ├── bengaluru/
    ├── mumbai/
    ├── delhi/
    └── transfer_benchmarks.json       ← Cross-city zero-shot evaluation logs
```

---

## 12. HARD RULES & CONSTRAINTS FOR CODING AGENTS

1. **Zero Live External API Calls at Inference Time:**  
   The FastAPI endpoint `/api/score` must **never** call Google Places or Overpass API during a user request. All spatial calculations must query the local `scipy.spatial.cKDTree` cached from `poi_layers.parquet`. Inference must complete in $< 20\text{ milliseconds}$.
2. **Never Train on Raw Star Ratings (1.0–5.0):**  
   Always use the log-transformed Footfall Throughput Index $Y = \log_{10}(1 + \text{reviews}) \times (\text{rating} / 5.0)$.
3. **Always Run Spatial Block CV:**  
   Do not use `train_test_split(random_state=42)` or standard non-spatial K-Fold. Cross-validation must cluster coordinates spatially to prevent data leakage.
4. **Always Return SHAP Values:**  
   Every response from `/api/score` must include exact per-feature SHAP attributions for both Petrol and EV models.
5. **Enforce Separate Models:**  
   Never train a single monolithic model with `is_ev` as a feature. The core scientific contribution requires extracting independent weight vectors $W_{\text{petrol}}$ and $W_{\text{ev}}$ to compute SFDS.
6. **Zero Cloud Cost:**  
   All batch downloads must respect Google's free credit tier ($< \$2\text{ total}$), and OSM/OCM/PlugShare downloads must use respectful throttling (min 1s delay between requests).
7. **Always Supplement OSM Power Grid with DISCOM Data:**  
   Feature #11 (`power_grid_dist_m`) must be computed from DISCOM shapefiles where available. OSM `power=substation|transformer|line` may be merged as a fallback. Never rely on OSM power data alone — Indian OSM power coverage is 25–55% complete and will produce artificially inflated distance values.
8. **Always Use Age-Corrected Throughput Index:**  
   Apply the age correction $Y = Y_{\text{raw}} / \log_{10}(1 + M_{\text{open}})$ during target engineering. If `opening_date` is unavailable for a station, apply the city-median listing age as an imputed correction.
9. **Always Implement AHP Baseline for Paper Comparison:**  
   Train a simple AHP (Analytic Hierarchy Process) benchmark with expert-assigned weights (drawn from Indian municipal EV planning documents) as a comparison model. DSSE must outperform this baseline in AUC and SFDS interpretability to justify the Inverse MCDA learning claim.

---

## 13. RESEARCH PAPER STRATEGY (IEEE ACCESS / CEUS)

- **Working Title:**  
  *"Empirical Decoupling of Spatial Success Factors: An Explainable Machine Learning Framework for Dual-Infrastructure (Fossil vs. Electric) Siting Across Indian Metropolitan Archetypes"*
- **Target Venues:**  
  1. *Computers, Environment and Urban Systems (CEUS)* — Elsevier (IF: 6.8)
  2. *IEEE Access* (Special Section on Intelligent Transportation Systems)
  3. *ACM SIGSPATIAL 2027*

### Literature Gap & Differentiation (Mandatory — Reviewers Read This First)
DSSE must explicitly distinguish itself from the following existing categories of work:

| Prior Work Category | Representative Papers | What DSSE Does Differently |
|---|---|---|
| **AHP/MCDM-only EV siting** | Zhao et al. (2023, *Applied Energy*); Guo et al. (2022, *Sustainable Cities*) | DSSE *learns* MCDA weights from observed infrastructure outcomes (Inverse MCDA) instead of expert assignment; provides falsifiable weight validation via SFDS |
| **CNN / raster-based suitability** | Luo et al. (2023, *IJGIS*); Erbaş et al. (2022, *Energy*) | DSSE uses interpretable tree-ensemble spatial features + SHAP rather than black-box pixel convolution; produces per-feature causal attribution |
| **Single-infrastructure GNN/POI** | Wang et al. (2024, *KDD*); Liu et al. (2023, *AAAI*) | DSSE performs *dual-infrastructure comparative* analysis; SFDS metric specifically measures infrastructure decoupling which none of the above quantify |
| **PU Learning in other domains** | Mordelet & Vert (2014); Bekker & Davis (2020) | DSSE is first to apply PU Learning specifically to energy transition infrastructure siting across multiple heterogeneous Indian urban archetypes |

### Key Empirical Hypotheses to Test
  1. **H1 (Spatial Decoupling):** EV charging infrastructure exhibits a statistically significant divergence ($\text{SFDS} > 0.35$ [95% bootstrap CI non-overlapping with zero]) from fossil-fuel infrastructure, trading highway accessibility for human dwell-time amenities.
  2. **H2 (Transferability):** Spatial models trained on high-density EV cities (Bengaluru) can predict viable emerging charging nodes in transitional cities (Pune) with an AUC $\ge 0.78$.
  3. **H3 (Grid Sensitivity):** Proximity to medium-voltage electrical transformers is the dominant negative barrier for high-capacity EV DC fast-charging placement in dense Indian corridors (SHAP rank #1 for EV model, confirmed by DISCOM-sourced `power_grid_dist_m`).
  4. **H4 (Topology Heterogeneity):** Linear constrained cities (Mumbai) enforce significantly lower divergence ($\text{SFDS}_{\text{Mumbai}} < \text{SFDS}_{\text{Bengaluru}}$, non-overlapping 95% CIs) due to land scarcity forcing infrastructure co-location.

---

## 14. PATENT APPLICATION STRATEGY (INDIAN PATENT ACT SEC 3(K))

- **Patent Classification:** Computer-Implemented Method and Geospatial Planning System.
- **Overcoming Section 3(k) ("Computer programs per se / Algorithms"):**  
  The patent application does **not** claim the mathematical formula of Jensen-Shannon Divergence. Instead, it claims a **complete physical and digital engineering system** that transforms spatial physical geography and electrical constraints into concrete municipal infrastructure zoning allocations:
  - *"An automated geospatial energy-transition infrastructure allocation system comprising: an asynchronous spatial rasterizer, a dual-stage positive-unlabeled infrastructure demand estimator, a real-time local spatial tree indexer, and an automated multi-quadrant infrastructure state controller."*

---

## 15. STEP-BY-STEP IMPLEMENTATION ROADMAP

> ⚠️ **PUNE-FIRST ORDER:** Complete all phases for Pune before starting any other city. Do not parallelize city data collection until Pune Phase 3 is verified.

```
━━━ PUNE PHASE 1: Data Ingestion & Caching (Scripts 01 & 02) ━━━
   ├── [Pune] Fetch: Google Places (Petrol + EV, with opening_date) + OCM (kW>=22) + PlugShare
   ├── [Pune] Deduplicate EV stations via station_deduplicator.py (<50m proximity = same station)
   ├── [Pune] Download MSEDCL substation shapefile (data.maharashtra.gov.in)
   ├── [Pune] Download OSM extract: Pune.osm.pbf from Geofabrik India mirror
   ├── [Pune] Validate power grid: spot-check MSEDCL vs OSM on 5×5km sample; log coverage ratio
   └── [Pune] Build cKDTree spatial index → data/processed/pune/poi_layers.parquet (merged MSEDCL+OSM)

━━━ PUNE PHASE 2: Target Formulation & Model Training (Script 03) ━━━
   ├── Compute Age-Corrected Throughput Index (Y); impute city-median age where opening_date missing
   ├── Generate PU background samples (H3 Res-8 centroids across Pune bbox)
   ├── Apply Pune-Relative Z-Score standardization across all 16 features
   ├── Train AHP Baseline (weights from CESL/BEE Indian EV guidelines → ahp_baseline.py)
   ├── Execute Spatial Block CV (K=5 Pune administrative wards / KMeans spatial blocks)
   ├── Train Pune XGBoost PU Classifiers (pu_petrol.pkl, pu_ev.pkl) + MAPIE conformal wrapper
   ├── Train Pune XGBoost Footfall Regressors (reg_petrol.pkl, reg_ev.pkl) + MAPIE conformal wrapper
   ├── Fit TreeExplainer (SHAP) → shap_petrol.pkl, shap_ev.pkl
   ├── Compute SFDS_Pune: point estimate + 1000-bootstrap 95% CI → log to MLflow
   ├── Generate UMAP scatter (figures/umap_pune.png) — petrol vs EV feature space
   └── Compare DSSE AUC/SFDS vs AHP Baseline → log to models/pune/metadata.json

━━━ PUNE PHASE 3: H3 Heatmap Pre-computation (Script 04) ━━━
   ├── Generate H3 Res-8 hex polygons for Pune bbox (≈500 hexagons)
   ├── Batch score all hex centroids with Pune models (petrol + EV, PU + Reg)
   └── Export → data/processed/heatmaps/heatmap_pune.geojson

━━━ PUNE PHASE 4: FastAPI Backend ━━━
   ├── Implement /api/score → returns {petrol_score, ev_score, petrol_ci, ev_ci, shap_petrol, shap_ev, quadrant}
   ├── Implement /api/heatmap, /api/divergence (SFDS + CI), /api/cities
   └── Verify /api/score latency < 20ms on Pune data

━━━ PUNE PHASE 5: React Web Dashboard ━━━
   ├── MapView: Leaflet + Pune H3 heatmap + single-pin click + comparison Pin B toggle
   ├── ScoreCard: dual petrol/EV gauges + conformal interval band + 4-Quadrant badge
   ├── SiteComparison: side-by-side ScoreCard A vs B + SHAP attribution diff
   ├── ShapWaterfall: per-feature SHAP bar chart (positive = helps, negative = hurts)
   └── ResearchView: SFDS bar + 95% CI whiskers, MCDA radar chart, UMAP scatter

━━━ REPEAT FOR BENGALURU → MUMBAI → DELHI (after Pune is paper-ready) ━━━
```
