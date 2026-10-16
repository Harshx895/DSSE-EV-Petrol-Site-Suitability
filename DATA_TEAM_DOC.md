# DSSE — Data Collection Team Documentation
## Everything Your Team Needs for the Mid-Semester Review

**Project:** EDI-5 — Dynamic Site Suitability Engine (DSSE)  
**Your Role:** Data Collection, Cleaning, Feature Engineering, Spatial Indexing  
**Scripts You Own:** 01, 02, 03, 04  
**Output You Produced:** 832 stations | 17,412 POIs | 13 KDTree layers | 16 spatial features

---

## Table of Contents

1. [Your Contribution in One Line](#1-your-contribution-in-one-line)
2. [What Is "Data" in This Project?](#2-what-is-data-in-this-project)
3. [Data Source 1 — Google Places API (Petrol Stations)](#3-data-source-1--google-places-api-petrol-stations)
4. [Data Source 2 — OpenChargeMap (EV Chargers)](#4-data-source-2--openchargeMap-ev-chargers)
5. [Data Source 3 — OpenStreetMap (Spatial Features)](#5-data-source-3--openstreetmap-spatial-features)
6. [Data Source 4 — Power Grid (DISCOM + OSM)](#6-data-source-4--power-grid-discom--osm)
7. [Data Cleaning and Deduplication](#7-data-cleaning-and-deduplication)
8. [The 16 Spatial Features — What and Why](#8-the-16-spatial-features--what-and-why)
9. [KDTree — The Spatial Index](#9-kdtree--the-spatial-index)
10. [ACTI — Activity Index Computation](#10-acti--activity-index-computation)
11. [Scripts Reference](#11-scripts-reference)
12. [Final Data Statistics](#12-final-data-statistics)
13. [Q&A — Non-Technical Questions](#13-qa--non-technical-questions)
14. [Q&A — Technical Questions](#14-qa--technical-questions)

---

## 1. Your Contribution in One Line

> *"We collected, cleaned, and structured all real-world location data for Pune into a form the ML team could train on — 741 ground-truth stations, 17,412 points of interest, 13 spatial layers, and 16 engineered features."*

Without your data, the ML models have nothing to learn from. The quality of the ML output is directly determined by the quality of your data collection.

---

## 2. What Is "Data" in This Project?

The project needs two types of data:

### Type A: Training Labels (Ground Truth)
*"Where do stations actually exist today?"*

- 363 petrol/fuel stations in Pune (from Google Places)
- 13 confirmed commercial EV chargers (from OpenChargeMap)
- 469 EV candidates (rated EV locations)

These are the "answers" the ML model learns from. Think of it like a student studying past exam questions.

### Type B: Spatial Features (What Describes a Location)
*"What makes a location suitable?"*

For every station (and every point the model evaluates), we need to know:
- How far is it from the nearest road?
- How many cafes are within 300m?
- How far is the nearest power substation?
- etc. (16 features total)

These come from OpenStreetMap and Google Places.

---

## 3. Data Source 1 — Google Places API (Petrol Stations)

### What We Got
- **363 unique petrol/fuel stations** in Pune
- For each: name, latitude, longitude, rating, review count, place_id, address
- Also: 80 petrol stations with enriched metadata (opening hours, price level)

### How We Got It — Technical Detail

Google Places API does not return all places in a city in one call. It returns at most 60 results per search (3 pages × 20). To cover all of Pune (~230 km²), we used a **grid-tiled search strategy**:

```
Pune bounding box divided into a 6×6 grid = 36 tiles
For each tile:
    Nearby Search: type=gas_station, radius=4000m, center=tile_center
    Fetch page 1 → 20 results
    Fetch page 2 → 20 results
    Fetch page 3 → 20 results
Merge all results → deduplicate by place_id
Final: 363 unique stations
```

**Why grid tiling?** If you search "petrol stations near Pune" with a single large radius, Google truncates results at 60. Grid tiling ensures you cover the entire city without missing peripheral areas.

### API Details
- Endpoint: `https://maps.googleapis.com/maps/api/place/nearbysearch/json`
- Parameters: `location`, `radius`, `type=gas_station`, `key`
- Pagination: `pagetoken` (wait 2 seconds between pages, else empty response)
- Rate limit: 1 QPS (queries per second), 10,000 requests/day on free tier
- Cost: Free up to $200/month credit (~4,000 Nearby Search calls)

### Script: `src/data/google_places_client.py`

Key function:
```python
def search_nearby(lat, lon, radius=4000, place_type="gas_station"):
    # Returns up to 60 results via 3-page pagination
    # Handles rate limiting automatically
```

---

## 4. Data Source 2 — OpenChargeMap (EV Chargers)

### What We Got
- **469 EV candidate locations** in Pune (all rated locations)
- **13 confirmed commercial EV chargers** (our positive training set)
- For each: name, latitude, longitude, charger type (AC/DC), operator, power output (kW)

### How We Got It

**OpenChargeMap (OCM)** is the world's largest open database of EV charging locations, maintained by a community similar to OpenStreetMap.

```
OCM REST API:
  Endpoint: https://api.openchargemap.io/v3/poi/
  Parameters:
    latitude=18.5204
    longitude=73.8567
    distance=30  (km from center)
    distanceunit=km
    maxresults=500
    key=YOUR_OCM_KEY
```

### Why Only 13 Commercial EV Chargers?

OCM contains all submitted EV locations — many are home chargers, workplace chargers, or unverified submissions. We applied strict filters:

```
Filter criteria for "commercial" EV charger:
  - usage_type = "Public"
  - status_type = "Operational"
  - operator is NOT None
  - has_at_least_one_level2_or_dcfc_charger = True
Result: 13 locations
```

This is not a data quality failure — it accurately reflects Pune's sparse public EV infrastructure in 2024.

### EV Network Scraping
We also scraped EV network portals for additional Pune chargers:
- PlugShare (browser automation via Playwright)
- ChargeZone API (public endpoint)
- Tata Power EV portal
- Magenta ChargeGrid

These added to the 469 total EV (328 commercial, 141 candidate) pool but few were confirmed commercial-grade.

---

## 5. Data Source 3 — OpenStreetMap (Spatial Features)

### What Is OpenStreetMap?
OpenStreetMap (OSM) is a free, community-maintained world map — like Wikipedia, but for geographic data. It contains roads, buildings, businesses, bus stops, and more, contributed by volunteers worldwide.

### Why OSM Instead of Google Maps?
1. **Google Maps Terms of Service** prohibit using Places API data to train ML models
2. **Cost**: Google Places Nearby Search costs ~$32 per 1,000 calls; with 17 POI types × 896 hexagons × training points, this would cost ₹1.5 lakh+
3. **Reproducibility**: Academic papers must use data others can reproduce; OSM is openly versioned

### How We Got OSM Data — Overpass API

The Overpass API is a query interface for OSM. We queried it for 8 categories of points of interest (POIs) across Pune:

```
Query example for cafes within Pune bounding box:
[out:json][timeout:90];
(
  node["amenity"="cafe"](18.40, 73.75, 18.63, 73.98);
  way["amenity"="cafe"](18.40, 73.75, 18.63, 73.98);
);
out center;
```

### POI Layers Downloaded

| Layer | OSM Tag | Count | Radius Used in Feature |
|---|---|---|---|
| Cafes | amenity=cafe | 1,247 | 300m |
| Restaurants | amenity=restaurant | 3,892 | 300m |
| Retail shops | shop=* | 4,203 | 500m |
| Supermarkets | shop=supermarket | 412 | 500m |
| Offices | office=* or building=office | 2,156 | 500m |
| Bus stops | highway=bus_stop | 1,834 | 500m |
| Residential buildings | building=residential | 2,180 | 500m |
| Parking | amenity=parking | 387 | 300m |
| Metro stations | railway=station | 36 | (distance to nearest) |
| Primary roads | highway=primary/trunk | special | (distance to nearest) |
| Secondary roads | highway=secondary | special | (distance to nearest) |
| **Total POI rows** | | **~17,412** | |

### Script: `src/data/osm_ingest.py`

Key function:
```python
def fetch_layer(bbox, amenity_tag, timeout=90):
    # Queries Overpass API
    # Returns list of {lat, lon, name, tags}
    # Handles timeout and retry automatically
```

---

## 6. Data Source 4 — Power Grid (DISCOM + OSM)

### Why Power Grid Data?
EV chargers require significant electrical infrastructure. A DC fast charger (50kW+) needs a dedicated 3-phase connection from a nearby transformer substation. Distance to the power grid is a key cost factor:
- < 500m: Low capex, economically viable
- 500–2500m: Moderate capex, warrants cost-benefit analysis
- > 2500m: High capex, significant barrier to EV installation

### What We Got
- **45 power grid nodes** (transformer substations, 33/11 kV feeders) in Pune

### How We Got It
1. **Primary attempt**: MSEDCL (Maharashtra State Electricity Distribution Company Ltd) GIS portal at `mahagis.maharashtra.gov.in` — **blocked/inaccessible** at time of collection
2. **Fallback 1**: OSM query for `power=substation` and `power=transformer` within Pune bbox → 31 nodes
3. **Fallback 2**: DISCOM website scraping (MSEDCL consumer portal, feeder data) → 14 additional nodes
4. **Total**: 45 confirmed substation locations

### Limitation
45 nodes is likely an undercount — Pune has many small distribution substations not mapped in OSM. This is documented as a project limitation.

### Script: `src/data/discom_ingest.py` + `src/data/power_grid_client.py`

---

## 7. Data Cleaning and Deduplication

Raw data from multiple sources has duplicates, errors, and inconsistencies. Here is every cleaning step:

### 7.1 Station Deduplication

Problem: Google Places and OCM may list the same physical location under different names.

Solution — Spatial deduplication:
```python
def deduplicate_stations(df, radius_m=100):
    # If two stations are within 100m of each other AND
    # have the same infra_type (petrol/ev),
    # they are the same station → keep one
    # Uses KDTree for efficient pairwise distance computation
```

Result: Reduced from ~810 raw entries to 741 unique stations.

### 7.2 Coordinate Validation
```python
# Pune bounding box: lat [18.40, 18.63], lon [73.75, 73.98]
df = df[df["lat"].between(18.40, 18.63)]
df = df[df["lon"].between(73.75, 73.98)]
# Removes stations incorrectly geocoded outside Pune
```

### 7.3 Missing Value Handling
- Missing rating: filled with city median (4.1 for petrol, 4.3 for EV)
- Missing review count: filled with 0 (conservative — treated as inactive)
- Missing name: retained (name not used in ML features)

### 7.4 EV Label Assignment
```python
# Three-tier EV classification:
if usage_type == "Public" and status == "Operational" and has_dcfc:
    label = "commercial"   # 13 stations — used as ML positives
elif rating >= 3.0:
    label = "candidate"    # 378 stations — used for heatmap scoring
else:
    label = "unverified"   # excluded
```

### Output: `data/processed/pune/stations_master.csv`
741 rows × 24 columns, fully clean and labeled.

---

## 8. The 16 Spatial Features — What and Why

For every point (station or heatmap hexagon), we extract 16 features. Here is each one, why it matters, and how it is computed:

### Road Access Features (Most important for Petrol)

| Feature | Why It Matters | Computation |
|---|---|---|
| dist_primary_highway_m | Petrol pumps need main road access for throughput | KDTree nearest neighbour on primary/trunk road nodes |
| dist_secondary_road_m | Secondary roads = neighbourhood access | KDTree nearest neighbour on secondary road nodes |

**SHAP finding:** These two features alone account for 35% of petrol model's decision weight.

### Amenity Features (Most important for EV)

| Feature | Why It Matters | Computation |
|---|---|---|
| cafe_count_300m | EV charging takes 20–45 min; cafes = dwell time | KDTree radius query, count within 300m |
| restaurant_count_300m | Same dwell logic | KDTree radius query, count within 300m |
| retail_count_500m | Retail = destination visits (people stay) | KDTree radius query, count within 500m |
| office_count_500m | Office workers charge during 9–5 hours | KDTree radius query, count within 500m |
| parking_count_300m | Parked cars = captive charging opportunity | KDTree radius query, count within 300m |

### Transit and Accessibility

| Feature | Why It Matters | Computation |
|---|---|---|
| bus_stop_count_500m | Transit access for station staff and customers | KDTree count within 500m |
| metro_station_dist_m | Metro areas = dense, high-footfall | KDTree nearest neighbour |
| transit_accessibility | Composite score | 0.5 × bus_stop_count_norm + 0.5 × (1/metro_dist_norm) |

### Competitive Context

| Feature | Why It Matters | Computation |
|---|---|---|
| competitor_petrol_1km | Too many competitors = saturated market | KDTree count within 1km |
| competitor_ev_1km | EV chargers cluster (network effect) | KDTree count within 1km |

### Residential Context

| Feature | Why It Matters | Computation |
|---|---|---|
| residential_count_500m | EV preferred by residents (home charging supplement) | KDTree count within 500m |

**SHAP finding:** This is the #1 driver for EV model (weight = 0.239).

### Infrastructure

| Feature | Why It Matters | Computation |
|---|---|---|
| power_grid_dist_m | EV charger installation cost increases with distance | KDTree nearest neighbour on 45 substations |
| supermarket_count_500m | Supermarket = sustained footfall, 30–60 min dwell | KDTree count within 500m |

### Derived Index

| Feature | Formula | What It Captures |
|---|---|---|
| dwell_index | log1p(cafes × 2 + restaurants + offices) | Combined likelihood people stay long enough to charge |

---

## 9. KDTree — The Spatial Index

### The Problem Without KDTree
Without spatial indexing, to find "all cafes within 300m of this point," you'd compare the point against every one of 1,247 cafe coordinates. For 896 hexagons × 16 features × 832 stations = millions of comparisons. Too slow.

### What a KDTree Is

A KD-Tree (K-Dimensional Tree) is a data structure that organises points in space so you can find neighbours very fast.

**Analogy:** Imagine a phone book organised alphabetically. Finding "Sharma" takes seconds (binary search), not hours (reading every page). KDTree is like a phone book but for locations instead of names.

### How It Works

```
13 KDTrees built from:
  roads_primary  (nodes from OSM primary road geometries)
  roads_secondary
  cafes
  restaurants
  retail
  offices
  supermarkets
  bus_stops
  metro_stations
  residential
  parking
  substations (power grid)
  petrol_stations
  ev_chargers

For each KDTree, 2 operations available:
  tree.query(point)          → nearest neighbour + distance
  tree.query_ball_point(point, radius)  → all neighbours within radius
```

### Performance
- Build time: ~2 seconds for all 13 trees
- Query time per point: < 1ms per tree → 16 features extracted in < 15ms total
- This is why the full system scores a location in < 1 second

### File: `models/pune/kdtrees.pkl` (897 KB)

---

## 10. ACTI — Activity Index Computation

### Why We Need ACTI

The ML team's throughput regressor needs a target variable — something that measures "how busy/successful is this station?" We don't have revenue data or fuel dispensing volume. The closest available proxy is Google Maps review data.

### The Formula

```
ACTI = log10(1 + total_reviews) × (rating / 5.0)
```

Examples:
| Station | Reviews | Rating | ACTI |
|---|---|---|---|
| Busy HP pump (1000 reviews, 4.2★) | 1000 | 4.2 | 3.001 × 0.84 = 2.52 |
| Quiet pump (12 reviews, 3.8★) | 12 | 3.8 | 1.114 × 0.76 = 0.85 |
| No reviews (0, 0★) | 0 | 0 | 0.0 |

### Why log10?
Review counts follow a power-law distribution (a few pumps have thousands of reviews, most have <50). Log compression prevents the model from being dominated by a few outliers.

### Limitation
ACTI is an imperfect proxy. A new station or a station with a small number of customers who don't leave reviews will appear "low activity" even if it's commercially successful. This is documented as a project limitation, and is why the throughput regressor R² is negative.

### Script: `src/models/target_engineering.py`

---

## 11. Scripts Reference

### `scripts/01_download_raw_data.py`
**What it does:** Downloads all station data from Google Places and OpenChargeMap. Saves raw CSVs to `data/raw/pune/`.

**Key parameters:**
```bash
python -X utf8 scripts/01_download_raw_data.py --city pune
  --google-grid-size 6    # 6x6 grid = 36 tiles
  --ocm-radius 30         # 30km from Pune center
```

**Runtime:** ~15–20 minutes (API rate limits)
**Output:** `data/raw/pune/google_places_raw.csv`, `data/raw/pune/ocm_raw.csv`

---

### `scripts/02_build_spatial_indices.py`
**What it does:** Fetches all OSM POI layers via Overpass API, fetches power grid data, cleans everything, and builds 13 KDTrees.

```bash
python -X utf8 scripts/02_build_spatial_indices.py --city pune
```

**Runtime:** ~10 minutes (OSM download + KDTree construction)
**Output:** `models/pune/kdtrees.pkl`, `data/processed/pune/poi_layers.parquet`

---

### `scripts/03_enrich_google_places.py`
**What it does:** For each station in the master CSV, fetches detailed metadata from Google Places (review count, rating, opening hours). This data is used to compute ACTI.

```bash
python -X utf8 scripts/03_enrich_google_places.py --city pune
```

**Runtime:** ~5 minutes
**Output:** Enriched `stations_master.csv`

---

### `scripts/04_scrape_ev_networks.py`
**What it does:** Uses Playwright (browser automation) to scrape EV network portals (ChargeZone, PlugShare) for additional Pune EV charger data. Merges with OCM data.

```bash
python -X utf8 scripts/04_scrape_ev_networks.py --city pune
```

**Runtime:** ~10 minutes (browser-based)
**Output:** Additional EV candidates merged into `stations_master.csv`

---

## 12. Final Data Statistics

| Metric | Value |
|---|---|
| Total stations in dataset | 741 |
| Petrol stations | 363 |
| EV commercial chargers (ML positives) | 13 |
| EV candidates (total) | 378 |
| OSM POI rows collected | 17,412 |
| KDTree spatial layers | 13 |
| Spatial features per location | 16 |
| Power grid nodes | 45 |
| Pune area covered | ~230 km² |
| Geographic bounding box | 18.40N–18.63N, 73.75E–73.98E |
| Data collection date | September 2026 |

---

## 13. Q&A — Non-Technical Questions

**Q: What data did you collect?**
We collected data on 741 fuel and EV stations in Pune, along with 17,412 points of interest like cafes, restaurants, offices, and bus stops. We also mapped Pune's road network and power grid (45 substations). All of this describes the spatial environment around each location.

---

**Q: Why did you need so much data? Couldn't you just use intuition?**
Intuition fails at scale. A human can evaluate maybe 10 locations by visiting them. Our system evaluates 896 locations on a city map within 5 minutes, and gives consistent, explainable scores based on 16 quantitative factors. Data makes the evaluation objective and repeatable.

---

**Q: Why did you use Google Maps for petrol stations but OpenStreetMap for other things?**
Google Places has the most accurate and complete database of petrol stations in India — it's their core business. But using Google data to train ML models violates their Terms of Service, so we only use it for station locations (not as a spatial feature). For spatial context (roads, cafes, buildings), we use OpenStreetMap, which is openly licensed for ML training.

---

**Q: OpenStreetMap is maintained by volunteers — how do you trust its quality?**
For major roads, OSM is excellent — they are well-mapped everywhere. For small businesses (cafes, shops), OSM is about 60–70% as complete as Google Maps in Indian cities. However, since we use the same OSM data for both training and inference, the model is internally consistent: it learns patterns from OSM data and applies the same OSM data at scoring time. The relative spatial patterns are preserved.

---

**Q: Why only 13 EV chargers? Is that enough?**
13 reflects the real EV infrastructure situation in Pune in 2024 — the city genuinely has very few public commercial EV chargers. Our ML team used a technique called PU Learning specifically designed for situations with very few confirmed positives. Despite only 13 positives, the EV model achieves AUC = 0.91 (very high). The scarcity itself is a finding: EV infrastructure in Pune is highly concentrated.

---

**Q: How do you ensure there are no duplicate locations in your data?**
We applied spatial deduplication: any two stations of the same type within 100 metres of each other are treated as the same station (one is removed). We reduced ~810 raw entries to 741 unique stations this way.

---

**Q: What is ACTI and why did you invent it?**
ACTI (Activity Index) is our proxy for how busy or commercially successful a station is. Because we don't have access to actual fuel sales data or charger utilisation rates, we use Google Maps review count and star rating: `log(1 + reviews) × (rating/5)`. A busy, popular station gets a high ACTI score. This is one of our innovative contributions.

---

**Q: What is the power grid data used for?**
Installing an EV charger requires electrical infrastructure — specifically a connection to a transformer substation. If the nearest substation is 3km away, the installation costs can be prohibitive. We mapped 45 substations in Pune to give the model a "power grid distance" feature. Locations close to substations score better for EV.

---

**Q: Did you face any challenges in data collection?**
Yes, three key challenges:
1. Google Places API paginates results at 60 per query — we had to design a grid-tiling strategy to cover all of Pune
2. MSEDCL's GIS portal (Maharashtra's electricity distribution company) was inaccessible — we had to fall back to OSM and scraping for power grid data
3. EV charger data is fragmented across many operators (ChargeZone, Tata Power, Magenta) — we had to scrape multiple sources and deduplicate

---

## 14. Q&A — Technical Questions

**Q: What is the Overpass API and how did you use it?**
Overpass API is a read-only query interface for OpenStreetMap data. We sent structured QL (Query Language) queries specifying a geographic bounding box (Pune's coordinates) and OSM tags (e.g., `amenity=cafe`). The API returns GeoJSON with all matching features. We queried 8 POI types, totalling ~17,412 rows.

---

**Q: What is a KDTree and why did you use it?**
A KD-Tree (K-Dimensional Tree) is a binary tree data structure for organising points in k-dimensional space. Each node splits the dataset along one coordinate axis alternately (latitude, then longitude, then latitude...). This allows nearest-neighbour and radius queries in O(log n) time instead of O(n). For our use case: 13 KDTrees, each queried < 1ms, totalling < 15ms feature extraction per location.

---

**Q: How did you handle the Google Places API pagination?**
Google's Nearby Search returns at most 20 results per page and 3 pages per query (max 60 total). To collect all 363 petrol stations across Pune's 230 km², we divided the bounding box into a 6×6 grid of 36 tiles, queried each tile independently, merged all results, and deduplicated by `place_id` (Google's unique identifier). We added a 2-second sleep between page token requests, as required by Google's API contract.

---

**Q: What is spatial deduplication?**
Spatial deduplication removes records that represent the same physical location. We built a KDTree from all station coordinates, then for each station, found all others within 100m of the same type. Duplicate pairs were resolved by keeping the record with more metadata (reviews, name, etc.). This reduced ~810 raw entries to 741 unique stations.

---

**Q: How did you compute transit_accessibility and dwell_index?**
Both are composite indices derived from raw features:

`transit_accessibility = 0.5 × normalize(bus_stop_count_500m) + 0.5 × normalize(1 / metro_station_dist_m)`

`dwell_index = log1p(cafes_300m × 2 + restaurants_300m + offices_500m)`

The weights (0.5/0.5, cafe×2) were set based on domain knowledge: metro proximity matters as much as bus stop density, and cafes provide stronger dwell incentive than offices (people choose to stay in cafes).

---

**Q: What is ACTI and why log10?**
ACTI (Activity Index) = `log10(1 + total_reviews) × (rating / 5.0)`. The logarithm handles the power-law distribution of review counts — a few stations have thousands of reviews, most have under 50. Without log compression, the regressor would overfit to high-review outliers. Using log10 makes the distribution approximately normal, which is better for regression.

---

**Q: How did you handle missing values in the dataset?**
Three strategies depending on the feature:
1. **Missing rating/reviews**: Imputed with city median (4.1 for petrol, 0 reviews for new stations). Zero reviews = treated as inactive, which is conservative.
2. **Missing coordinates**: Record dropped entirely.
3. **Missing OSM POIs**: Zero count (interpreted as no POI of that type nearby, not a data error).

---

**Q: What is the coordinate reference system used?**
WGS84 (EPSG:4326) — decimal degrees latitude/longitude. This is the standard for GPS and Google Maps. All distance computations use the Haversine formula (accounts for Earth's curvature), converted to metres before being passed to the KDTree. For the KDTree itself, we project to approximate metres using a local flat-earth approximation (valid for the small Pune bounding box).

---

**Q: How many API calls did you make in total?**
- Google Places Nearby Search: 36 tiles × 3 pages = 108 calls
- Google Places Details: ~741 calls (one per station for enrichment)
- OpenChargeMap: 1 call (single large-radius query)
- Overpass API: 8 calls (one per POI type)
- DISCOM/power scraping: ~50 HTTP requests
- EV network scraping: ~20 browser sessions
- Total: ~930 API calls

---

**Q: What does `data/processed/pune/stations_master.csv` contain?**
741 rows × 24 columns:
- Identifiers: name, lat, lon, place_id, source
- Labels: infra_type (petrol/ev), ev_label (commercial/candidate/unverified)
- Raw features: all 16 spatial features pre-computed
- ACTI: activity index
- Metadata: rating, review_count, address

---

*Data Collection Team | DSSE — EDI-5 | September 2026*
