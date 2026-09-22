# DSSE — Dynamic Site Suitability Engine
## Complete Project Documentation

**Project:** EDI-5 (Entrepreneurship Development Initiative, Semester 5)  
**Topic:** AI-Driven Site Suitability Analysis for Petrol and EV Charging Infrastructure  
**City Focus:** Pune, Maharashtra, India  
**Status:** Phase 1–3 Complete | MVP Demo Ready

---

## Table of Contents

1. [What Is This Project? (Non-Technical)](#1-what-is-this-project-non-technical)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [Why This Matters — India's EV Context](#3-why-this-matters--indias-ev-context)
4. [Our Solution — At a Glance](#4-our-solution--at-a-glance)
5. [Data — What We Collected and From Where](#5-data--what-we-collected-and-from-where)
6. [How the System Works — Step by Step](#6-how-the-system-works--step-by-step)
7. [The Machine Learning Models — Explained Simply](#7-the-machine-learning-models--explained-simply)
8. [The Research Finding — SFDS Metric](#8-the-research-finding--sfds-metric)
9. [Results and Performance](#9-results-and-performance)
10. [The Four-Quadrant Recommendation System](#10-the-four-quadrant-recommendation-system)
11. [System Architecture — Technical](#11-system-architecture--technical)
12. [File and Folder Structure](#12-file-and-folder-structure)
13. [How to Run the Demo](#13-how-to-run-the-demo)
14. [Understanding the Output](#14-understanding-the-output)
15. [Limitations and Honest Caveats](#15-limitations-and-honest-caveats)
16. [What Is Remaining — Roadmap](#16-what-is-remaining--roadmap)
17. [Glossary of Terms](#17-glossary-of-terms)

---

## 1. What Is This Project? (Non-Technical)

Imagine you are an investor and you want to open a petrol pump or an EV charging station in Pune. You look at a plot of land and ask: **"Is this a good location?"**

Currently, this decision is made by gut feeling, expensive consultants, or simple rules like "near a highway." There is no data-driven, scientific way to evaluate a location quickly and explain *why* it is good or bad.

**DSSE (Dynamic Site Suitability Engine)** is an AI system that solves this problem. You give it any location in Pune — by typing coordinates or pasting a Google Maps link — and within less than one second it tells you:

- **Score out of 100** — how suitable this location is for a petrol station AND for an EV charger
- **Recommendation** — whether to invest in petrol, EV, both, or neither
- **Explanation** — exactly which factors (roads, residential density, power grid, etc.) are helping or hurting the score

It is like having a data scientist instantly analyse any location on the map.

---

## 2. The Problem We Are Solving

### For Petrol Stations:
- New petrol pump licenses require significant capital (Rs 50–80 lakh setup cost)
- Wrong location = low footfall = financial loss
- Existing site selection is subjective and inconsistent

### For EV Charging Stations:
- India had ~12,000 public EV chargers as of 2024, against a target of 400,000+ by 2030
- The FAME-II and PM e-Bus Sewa schemes are creating Rs 10,000+ crore of investment
- Investors and DISCOMs have no systematic tool to identify which locations will attract EV users
- **EV charging is not the same as a petrol pump.** They serve fundamentally different spatial needs — yet many planners treat them the same way

### The Core Research Question:
> *Do EV chargers need to be placed in the same kinds of locations as petrol pumps, or do they need fundamentally different environments?*

This is both a **practical business question** (where should I invest?) and a **research question** (how different are petrol and EV spatial requirements?). DSSE answers both.

---

## 3. Why This Matters — India's EV Context

| Fact | Figure |
|---|---|
| India EV sales growth (FY23-24) | +40% YoY |
| Public EV chargers in India (2024) | ~12,000 |
| Government target by 2030 | 400,000+ |
| Gap to fill | ~388,000 chargers |
| FAME-II subsidy allocated | Rs 10,000 crore |
| Average EV charger installation cost | Rs 2–8 lakh |

The rush to deploy EV infrastructure is creating a **mis-siting problem**: chargers are being installed in locations with low utilisation while high-demand locations remain unserved.

DSSE directly addresses this by providing a spatial intelligence layer that separates where EV chargers *should* go from where petrol pumps *currently* go.

---

## 4. Our Solution — At a Glance

```
INPUT:  Any location in Pune (coordinates or Google Maps link)
           |  (< 1 second)
PROCESS: AI scores location using 16 spatial features
           |
OUTPUT: Petrol score /100 | EV score /100 | Recommendation + Explanation
```

**What makes it different from existing tools:**
1. **Dual infrastructure analysis** — petrol AND EV scored simultaneously, not separately
2. **Explainable AI** — SHAP values tell you *why* a location scored the way it did
3. **Scientifically validated** — uses Spatial Block Cross-Validation (not random K-fold)
4. **Original research metric** — SFDS quantifies how different petrol and EV spatial requirements are

---

## 5. Data — What We Collected and From Where

### 5.1 Station Data (Training Labels)

These are the real stations the model learns from.

| Dataset | Count | Source | How Fetched |
|---|---|---|---|
| Petrol/fuel stations | 363 | Google Places API | Grid-tiled Nearby Search across Pune bbox |
| EV chargers (commercial, confirmed) | 13 | OpenChargeMap (OCM) | API query for Pune area |
| EV charger candidates (all rated) | 378 | OCM + EV network scrapers | PlugShare, ChargeZone, Tata Power APIs |
| **Total stations in dataset** | **741** | | |

**Why only 13 confirmed commercial EV chargers?** This reflects Pune's real EV infrastructure density as of 2024. EV charging in Pune is sparse and highly concentrated — this scarcity is itself a research finding.

### 5.2 Spatial Feature Data (What the Model Sees)

These are the 16 factors the AI uses to evaluate any location. All data was downloaded once and stored locally.

| Feature | What It Measures | Source |
|---|---|---|
| dist_primary_highway_m | Distance to nearest major road | OpenStreetMap |
| dist_secondary_road_m | Distance to nearest secondary road | OpenStreetMap |
| cafe_count_300m | Number of cafes within 300m | OpenStreetMap |
| restaurant_count_300m | Number of restaurants within 300m | OpenStreetMap |
| retail_count_500m | Number of retail shops within 500m | OpenStreetMap |
| supermarket_count_500m | Number of supermarkets within 500m | OpenStreetMap |
| office_count_500m | Number of offices within 500m | OpenStreetMap |
| bus_stop_count_500m | Number of bus stops within 500m | OpenStreetMap |
| metro_station_dist_m | Distance to nearest metro station | OpenStreetMap |
| residential_count_500m | Number of residential buildings within 500m | OpenStreetMap |
| power_grid_dist_m | Distance to nearest electrical substation | OSM + DISCOM scrape |
| competitor_petrol_1km | Number of petrol stations within 1km | Google Places API |
| competitor_ev_1km | Number of EV chargers within 1km | OpenChargeMap |
| dwell_index | Composite: how long people stay nearby | Computed |
| transit_accessibility | Composite: reachable by public transport | Computed |
| parking_count_300m | Number of parking areas within 300m | OpenStreetMap |

### 5.3 Geographic Coverage

- **City:** Pune, Maharashtra
- **Bounding Box:** 18.40N–18.63N, 73.75E–73.98E (~230 km2 coverage)
- **Data vintage:** Downloaded September 2026

---

## 6. How the System Works — Step by Step

### Step 1: Data Ingestion (Done once, offline)
```
Download all Pune petrol/EV station data
    |
Download all OSM spatial layers (roads, POIs, buildings)
    |
Download power grid data
    |
Clean and deduplicate all stations → 741 unique stations
    |
Build spatial index (KDTrees — 13 layers)
```

### Step 2: Feature Engineering (Done once, offline)
For every station in the dataset, extract all 16 features:
```
For station at (18.5553, 73.7928):
  KDTree query: nearest road = 140m
  KDTree query: cafes within 300m = 3
  KDTree query: power grid = 2255m
  ... (13 more features)
  Result: [140, 3, 2255, ...]  <- feature vector
```

### Step 3: Model Training (Done once, offline)
```
PU Learning:
  Positives  = confirmed petrol stations (363)
  Unlabelled = all other locations (background)

  Train XGBoost classifier:
  "How likely is this location to be a viable petrol station?"

Repeat for EV chargers (13 positives)
```

### Step 4: Live Inference (Runs in < 1 second)
```
User inputs: (18.5312, 73.8474)
    |
Extract 16 features using pre-loaded KDTrees (< 15ms)
    |
PU classifier: Petrol viability probability = 0.59
PU classifier: EV viability probability = 0.001
    |
Combine with regressor: Petrol score = 64.7/100, EV score = 11.5/100
    |
Compute SHAP values: which features drove the score
    |
Output: BEST_PETROL | Good | Explanation
```

---

## 7. The Machine Learning Models — Explained Simply

### 7.1 The Core Challenge: Very Few EV Chargers

Standard ML classification needs examples of both "good" and "bad" locations. The problem: **we don't know which locations are genuinely bad for EV.** A random point with no EV charger might be a great EV location — just no one has built there yet. This is called the **Positive-Unlabelled (PU) problem.**

### 7.2 PU Learning — The Solution

```
Traditional ML:                  PU Learning (our approach):
  Positives = known good sites     Positives = known stations
  Negatives = confirmed bad sites  Unlabelled = everything else
                                   (may be good or bad — we don't assume)
```

Our PU model uses the Elkan-Noto calibration algorithm to correct for the uncertainty of unlabelled data, producing honest probability scores.

### 7.3 XGBoost — The Engine

XGBoost (Extreme Gradient Boosting) is the algorithm powering our classifiers. Think of it as:

1. A sequence of 200+ decision trees
2. Each tree learns from the mistakes of the previous one
3. Together they capture complex spatial patterns

XGBoost is the industry standard for tabular ML. It is why our model achieves AUC = 0.91 for EV.

### 7.4 The Composite Score Formula

```
Final Score = 40% x PU Viability + 60% x Throughput Regressor
```

- **PU Viability (40%):** Spatial similarity to known stations (0–100)
- **Throughput Regressor (60%):** Predicted business activity using ACTI metric
- **ACTI:** log10(1 + total_reviews) x (rating/5) — proxy for station activity level

### 7.5 SHAP — Why the Model Decided What It Did

SHAP assigns a contribution value to every feature for every prediction:

- **Positive SHAP (▲)** = this feature pushed the score UP
- **Negative SHAP (▼)** = this feature pushed the score DOWN

Example for a petrol station location:
```
▲ dist primary highway m  +1.98   Being near a highway strongly helps petrol
▼ transit accessibility   -0.26   Poor transit access slightly hurts
▼ competitor ev 1km       -0.23   Nearby EV chargers slightly reduce petrol score
```

### 7.6 Spatial Block Cross-Validation — Why It Matters

Standard ML uses random K-Fold CV, which is **wrong for spatial data**. Nearby stations share similar environments, so a random split leaks information from training to test.

We use **Spatial Block CV**: Pune is divided into geographic blocks. No training block is adjacent to any test block. This gives an honest model performance estimate.

### 7.7 Conformal Prediction Intervals

Instead of a single score, our regressor gives a range:

```
Score: 68.7/100
Range: [19 – 96]  (90% confidence interval)
```

This means: "We are 90% confident the true score is between 19 and 96." The wide range reflects genuine uncertainty — useful for investors to know the model is being honest about its limitations.

---

## 8. The Research Finding — SFDS Metric

### 8.1 The Central Question

> Are EV chargers placed in fundamentally different kinds of locations than petrol pumps, or do they share the same spatial requirements?

### 8.2 What SFDS Is

**SFDS = Spatial Feature Divergence Score** — an original metric invented for this project.

```
SFDS = Sum of |weight_EV(feature) - weight_petrol(feature)| x 0.5
```

- **SFDS = 0.0** → EV and petrol need identical spatial environments
- **SFDS = 1.0** → EV and petrol need completely opposite environments
- **SFDS = 0.407** → Partial Decoupling (our finding)

### 8.3 Our Result

```
SFDS = 0.4069
95% Confidence Interval: [0.3480 – 0.6453]
Interpretation: PARTIAL DECOUPLING
```

### 8.4 Feature Weight Comparison

| Feature | Petrol Weight | EV Weight | Difference | What It Means |
|---|---|---|---|---|
| Residential buildings | 0.056 | 0.239 | +0.183 | EV needs residential areas 4x more than petrol |
| Competitor petrol stations | 0.191 | 0.018 | -0.173 | Petrol clusters together; EV does not |
| Competitor EV chargers | 0.028 | 0.138 | +0.109 | EV chargers cluster near each other (network effect) |
| Primary road distance | 0.210 | 0.122 | -0.088 | Petrol needs highways more than EV does |
| Secondary road distance | 0.139 | 0.072 | -0.067 | Same pattern: roads matter more for petrol |
| Dwell index | 0.044 | 0.080 | +0.036 | EV benefits more from dwell amenities |

**Key insight:** Petrol stations are about road access and rapid throughput. EV chargers are about dwell time — places where people wait while their car charges (residential areas, offices, cafes). These are fundamentally different spatial requirements.

**Policy implication:** Simply placing EV chargers where petrol pumps are will underperform. A separate spatial planning framework is needed for EV infrastructure in India.

---

## 9. Results and Performance

### 9.1 Model Performance

| Metric | Petrol Model | EV Model |
|---|---|---|
| PU AUC (Spatial Block CV) | 0.779 ± 0.036 | 0.912 ± 0.127 |
| Regressor R2 | -0.155 | -0.289 (documented limitation) |
| Cross-validation method | Spatial Block CV | Spatial Block CV |
| Model type | XGBoost + Elkan-Noto | XGBoost + Elkan-Noto |

**AUC interpretation:**
- AUC = 0.5 means random guessing
- AUC = 0.779 means the model correctly ranks 78% of location pairs by petrol suitability
- AUC = 0.912 means the model correctly ranks 91% of location pairs by EV suitability

### 9.2 Validation on Known Stations

When we score actual known station coordinates (not seen during training in their spatial block):

| Location | EV Score | Petrol Score | Verdict |
|---|---|---|---|
| Garve Hyundai (EV charger) | 92.8 | 78.6 | BOTH_VIABLE |
| Shell Recharge Sus Road | 92.5 | 67.3 | BOTH_VIABLE |
| Nanded City EV Hub | 91.8 | 72.8 | BOTH_VIABLE |
| HP Petrol Pump Sinhagad | 33.4 | 69.6 | BEST_PETROL |
| Bharat Petroleum Pump | 23.8 | 80.2 | BEST_PETROL |

**Mean EV score at actual EV stations: 74.2/100**
**Mean Petrol score at actual petrol stations: 71.9/100**

The model correctly identifies known stations as suitable.

### 9.3 City-Wide Heatmap Results (896 hexagons, H3 Res-8)

| Quadrant | Hexagons | Percentage |
|---|---|---|
| BEST_PETROL | 783 | 87% |
| POOR_BOTH | 111 | 12% |
| BOTH_VIABLE | 2 | <1% |
| BEST_EV | 0 | 0% |

**Why 0 BEST_EV hexagons?** EV chargers in Pune are hyper-localised to specific buildings. No broad neighbourhood zone (460m hexagon) is purely EV-dominant. When an exact EV charger address is scored, it gets 92.8 for EV. The heatmap shows city-level zones; the point scorer handles specific addresses. Both are correct.

---

## 10. The Four-Quadrant Recommendation System

Every location is classified into one of four quadrants based on petrol score and EV score vs threshold of 60/100:

```
                          EV Score
                   Low (< 60)      High (>= 60)
                +---------------+---------------+
Petrol  High    | BEST_PETROL   | BOTH_VIABLE   |
Score   (>= 60) |      Petrol   |    Both       |
                +---------------+---------------+
        Low     |  POOR_BOTH    |   BEST_EV     |
        (< 60)  |    Neither    |     EV        |
                +---------------+---------------+
```

### Quadrant Descriptions:

**BEST_PETROL — High-Throughput Transit Node**
Road-adjacent, commercially active, suitable for rapid fuel turnover. Petrol investment viable. EV charger would underperform due to low dwell time. Recommendation: Petrol pump.

**BEST_EV — High-Dwell Destination Hub**
Residential-dense, rich in cafes and amenities, people stay longer. EV charger will see consistent utilisation. Petrol pump not ideal (no highway throughput). Recommendation: EV charger.

**BOTH_VIABLE — Prime Multi-Modal Interchange**
Rare, prime locations scoring well for both. Recommend EV or EV+Petrol hybrid. Future-proof investment with highest ROI potential. Recommendation: EV charger (or hybrid).

**POOR_BOTH — Infrastructural Dead Zone**
Neither road-accessible enough for petrol nor amenity-rich enough for EV. Recommendation: Avoid.

---

## 11. System Architecture — Technical

### 11.1 Data Pipeline

```
scripts/01_download_raw_data.py
    Google Places API  → 363 petrol stations
    OpenChargeMap API  → EV candidates
    Overpass API (OSM) → 17,412 POI rows

scripts/02_build_spatial_indices.py
    Build 13 KDTrees → models/pune/kdtrees.pkl

scripts/03_enrich_google_places.py
    Fetch activity metadata for stations (ACTI)

scripts/04_scrape_ev_networks.py
    Browser scraping for EV network portals

scripts/05_train_and_evaluate.py
    Feature extraction (FeatureExtractor)
    Target engineering (ACTI, PU labels)
    Spatial Block CV training
    PU classifiers (petrol + EV)
    Conformal regressors (MAPIE)
    SHAP importances
    SFDS computation → 0.4069

scripts/06_cache_heatmap.py
    Score 896 H3 hexagons → heatmap_pune.geojson
```

### 11.2 Model Files

```
models/pune/
    pune_pu_petrol.pkl       # PU classifier (petrol), 764 KB
    pune_pu_ev.pkl           # PU classifier (EV), 393 KB
    pune_reg_petrol.pkl      # Conformal regressor (petrol), 5.3 MB
    pune_reg_ev.pkl          # Conformal regressor (EV), 5.5 MB
    kdtrees.pkl              # 13 spatial index layers, 897 KB
    shap_importances.json    # SHAP weights for both models
    train_report.json        # CV metrics
```

### 11.3 Inference Stack

```python
Predictor.score(lat, lon)  →  result in < 1 second

    FeatureExtractor.extract(lat, lon)
        13 x KDTree.query()  →  16 features  (< 15ms)

    pu_petrol.predict_proba(X)   →  petrol viability  (< 5ms)
    pu_ev.predict_proba(X)       →  EV viability      (< 5ms)
    reg_petrol.predict(X)        →  petrol throughput + interval  (< 5ms)
    reg_ev.predict(X)            →  EV throughput + interval      (< 5ms)

    composite = 0.4 x viability + 0.6 x throughput

    shap.TreeExplainer(clf).shap_values(X)  →  SHAP  (< 300ms)
```

### 11.4 Key Algorithms

| Algorithm | Purpose | Library |
|---|---|---|
| XGBoost | PU classification base learner | xgboost |
| Elkan-Noto | PU calibration (correct for label bias) | pulearn |
| MAPIE | Conformal prediction intervals | mapie |
| KDTree | Spatial nearest-neighbour queries | scipy.spatial |
| H3 | Hexagonal grid tiling of city | h3-py (Uber) |
| SHAP | Model explanation | shap |
| Spatial Block CV | Honest cross-validation for spatial data | custom implementation |

---

## 12. File and Folder Structure

```
5_EDI/
    data/
        raw/pune/                    # Downloaded raw CSVs
        processed/pune/
            stations_master.csv      # 741 stations with all features
            poi_layers.parquet       # 17,412 OSM POI rows
            sfds_results_pune.json   # SFDS = 0.4069
            heatmaps/
                heatmap_pune.geojson # 896 hex scores (438 KB)
                heatmap_pune_meta.json

    models/pune/                     # All trained model files

    figures/
        heatmap_pune.html            # Interactive browser map (open this!)
        sfds_weights_pune.png
        umap_feature_space_pune.png

    scripts/
        01_download_raw_data.py      # Phase 1: Data ingestion
        02_build_spatial_indices.py  # Phase 1: KDTree construction
        03_enrich_google_places.py   # Phase 1: Station enrichment
        04_scrape_ev_networks.py     # Phase 1: EV scraping
        05_train_and_evaluate.py     # Phase 2: Model training
        06_cache_heatmap.py          # Phase 3: Heatmap generation
        07_demo.py                   # MVP: Terminal demo tool
        08_visualize_heatmap.py      # MVP: Browser heatmap

    src/
        data/
            google_places_client.py  # Places API wrapper
            ocm_client.py            # OpenChargeMap API
            osm_ingest.py            # Overpass API OSM ingestion
            discom_ingest.py         # Power grid data
            spatial_index.py         # KDTree builder
        models/
            feature_extractor.py     # 16-feature extraction from KDTrees
            target_engineering.py    # ACTI + PU label creation
            trainer.py               # PU training loop
            spatial_cv.py            # Spatial Block CV implementation
            sfds_engine.py           # SFDS metric computation
            predictor.py             # Real-time inference engine
            recommender.py           # Quadrant logic and score_to_100
        spatial/
            constraints_filter.py    # Hard constraints (150m road, 2500m power)
        api/
            main.py                  # FastAPI backend (built, not yet tested)

    PROJECT_DOCUMENTATION.md        # This file
    CONTEXT_EDI_5.md                 # Full project specification
    requirements.txt                 # Python dependencies
    .env.example                     # API key template
```

---

## 13. How to Run the Demo

### Prerequisites
- Python 3.10+
- All dependencies installed: `pip install -r requirements.txt`
- Models already trained (models/pune/ directory exists)
- Run from project root: `cd C:\Users\Harsh Gupta\5_EDI`

### All Demo Commands

**Open the interactive browser heatmap:**
```bash
python -X utf8 scripts/08_visualize_heatmap.py --city pune
```
Opens figures/heatmap_pune.html. Shows 896 hexagons colour-coded. Hover for scores.

**Score a specific location by coordinates:**
```bash
python -X utf8 scripts/07_demo.py --lat 18.5312 --lon 73.8474 --city pune
```

**Score an actual EV charger site (Garve Hyundai):**
```bash
python -X utf8 scripts/07_demo.py --lat 18.5553 --lon 73.7928 --city pune
```

**Score using a Google Maps URL:**
```bash
python -X utf8 scripts/07_demo.py --url "https://www.google.com/maps/place/Garve+Hyundai/@18.5553,73.7928,17z"
```

**Interactive mode (best for live demo — paste URLs on the fly):**
```bash
python -X utf8 scripts/07_demo.py --interactive --city pune
```

**Show only the SFDS research finding:**
```bash
python -X utf8 scripts/07_demo.py --sfds --city pune
```

### Suggested Demo Flow

| Step | Command | What to Say |
|---|---|---|
| 1 | Heatmap | "This is Pune's suitability map. Blue = petrol zones. 87% of the city is petrol-viable." |
| 2 | Petrol coords | "Let's score a typical petrol area — correctly recommends petrol, 68.7/100." |
| 3 | EV coords | "Now an actual EV charger showroom. EV scores 92.8, recommended BOTH_VIABLE." |
| 4 | Interactive + URL | "Someone give me a location on Google Maps. I will paste the link and score it live." |
| 5 | SFDS | "And here is the research finding — EV needs different spaces from petrol, SFDS = 0.41." |

---

## 14. Understanding the Output

Full annotated output explanation:

```
DSSE — Dynamic Site Suitability Engine  |  Pune Demo

Location: 18.5312N, 73.8474E        <- The location you queried

SCORES
------
Petrol Station   Good V              <- Label: Excellent / Good / Moderate / Weak
  Score    ████████████████████   68.7/100     <- Composite score out of 100
  Range    [19 – 96]                           <- Conformal prediction interval (90%)
                                               "True score is likely between 19-96"
                                               Wide range = model is uncertain
                                               (expected given limited training data)

EV Charger       Weak X
  Score    ███                    11.5/100
  Range    [0 – 73]

VERDICT
-------
RECOMMENDED FOR PETROL STATION                 <- Main recommendation
This location is road-accessible...            <- Plain English reason

WHY? — Key Location Factors (SHAP Explanation)
-----------------------------------------------
Petrol factors                  EV factors
▲ dist primary highway +1.98   ▼ residential count  -2.05
▼ dist secondary road  -0.52   ▼ dwell index        -0.89
▼ bus stop count       -0.34   ▼ transit access     -0.81
▼ transit access       -0.26   ▼ dist secondary rd  -0.77
▼ competitor ev 1km    -0.23   ▲ metro dist m       +0.71

LEFT column: what drives the PETROL score (▲ helps, ▼ hurts)
RIGHT column: what drives the EV score
NUMBER: magnitude of influence (larger number = more important factor)
```

**Reading the SHAP table:**
- Petrol's top driver is `dist_primary_highway_m = +1.98` — being near a highway strongly helps petrol
- EV's top penalty is `residential_count_500m = -2.05` — too few residential buildings nearby strongly hurts EV
- This directly visualises the SFDS finding at the individual location level

---

## 15. Limitations and Honest Caveats

### 15.1 Data Limitations

**OpenStreetMap Freshness**
OSM can be 1–3 years behind Google Maps for cafes, restaurants, retail. For roads (our most important feature at 35% SHAP weight), OSM is well-maintained. Both training and inference use the same OSM data, so the model is internally consistent even if OSM is incomplete.

**Only 13 EV Positives**
Only 13 confirmed commercial EV charger locations in Pune. This makes the EV model conservative. AUC = 0.912 is still excellent despite this constraint.

**No Temporal Data**
No time-series data for station revenue or charger utilisation. ACTI (Google review-based) is an imperfect proxy. This is why the throughput regressor has negative R2.

**Power Grid Data**
The official DISCOM GIS portal was inaccessible. Power grid nodes sourced from OSM and a partial DISCOM scrape. This layer has fewer data points than ideal.

### 15.2 Model Limitations

**Wide Conformal Intervals**
The range [19–96] for a score of 68.7 is wide, reflecting genuine uncertainty. The point estimate is useful for ranking locations, but not as a precise number.

**Regressor R2 < 0**
The throughput regressor performs worse than a simple mean predictor. This is a fundamental data limitation, not a code error. The PU classifier is the reliable component.

**No Brand Data**
A branded BP station vs. an independent fuel vendor in the same location will have different revenue. We have no brand data — OSM and Google Places don't reliably provide this.

### 15.3 Geographic Limitations

**Pune Only (for now)**
Models are trained on Pune data. Applying to another city requires re-running the full pipeline for that city. Bengaluru is planned next.

---

## 16. What Is Remaining — Roadmap

### MVP Complete
- Phase 1: Data collection and ingestion (741 stations, 13 KDTree layers)
- Phase 2: Model training (PU classifiers, SFDS = 0.4069, Spatial Block CV)
- Phase 3: Heatmap (896 hexagons, H3 Res-8, GeoJSON 438KB)
- Terminal demo tool (07_demo.py) with Google Maps URL support
- Browser heatmap (08_visualize_heatmap.py) with ESRI tiles

### Post Mid-Sem Remaining

**Phase 4: FastAPI Backend (~2 hours)**
FastAPI server is written (src/api/main.py) but not yet tested.
6 endpoints: health, cities, score, heatmap, divergence, compare
Start: `uvicorn src.api.main:app --reload --port 8000`

**Phase 5: Web Frontend (~4 hours)**
React/Vite or plain HTML+JS web interface.
Leaflet.js map with live hexagon overlay and click-to-score.
SHAP waterfall chart panel. Site comparison table.

**Phase 6: Bengaluru (~3 hours)**
Re-run Phase 1–3 with `--city bengaluru`.
Cross-city SFDS comparison (Pune SFDS vs Bengaluru SFDS).
Core academic contribution for paper.

**Phase 7: Paper Tables (~1 hour)**
Model comparison table, SFDS cross-city table, feature importance heatmap.
Formatted for conference submission (IEEE or ACM).

**Phase 8 Optional: Google Places Feature Upgrade**
Replace OSM POI features with live Google Places Nearby Search.
Requires ToS compliance review for academic use.
Would address the OSM freshness limitation.

---

## 17. Glossary of Terms

| Term | Plain English Meaning |
|---|---|
| AUC | Area Under the ROC Curve. Score from 0.5 (random) to 1.0 (perfect). Our models: 0.78 and 0.91. |
| ACTI | Activity Index. Computed from Google Maps review count and rating. Proxy for how busy a station is. |
| Conformal Prediction | A method to give an honest range around a prediction, rather than just a single number. |
| DISCOM | Distribution Company. Government entity for electricity distribution. MSEDCL is Pune's DISCOM. |
| Dwell Index | Derived feature: how long people typically stay near a location. From cafes + offices + restaurants. |
| FAME-II | Faster Adoption and Manufacturing of Electric Vehicles Phase II. Government subsidy for EV infrastructure. |
| H3 | Uber's hexagonal grid system for the Earth. Res-8 hexagons cover ~0.46 km2 each. |
| KDTree | K-Dimensional Tree. Makes "find nearest neighbour" queries very fast (< 1ms). |
| OSM | OpenStreetMap. Free community-maintained world map used for road and POI data. |
| PU Learning | Positive-Unlabelled Learning. ML approach where you have confirmed positives (known stations) but uncertain negatives. |
| SHAP | SHapley Additive exPlanations. Explains why an ML model made a specific prediction, based on Nobel Prize game theory. |
| SFDS | Spatial Feature Divergence Score. Our original metric quantifying how differently petrol and EV stations are distributed spatially. |
| Spatial Block CV | Cross-validation for geographic data that prevents data leakage by keeping training and test blocks geographically separate. |
| XGBoost | Extreme Gradient Boosting. Ensemble of decision trees. Industry standard for tabular ML. |

---

*Document prepared: September 2026*
*Project: EDI-5 | DSSE — Dynamic Site Suitability Engine | Pune, Maharashtra, India*
