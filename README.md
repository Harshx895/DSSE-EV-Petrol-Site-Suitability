# Dynamic Site Suitability Engine (DSSE)

> **EDI 5 Capstone | Dual-Infrastructure Spatial ML for EV & Petrol Station Siting**  
> Pune-first implementation · IEEE Access / CEUS target · Provisional Patent pending

---

## 🗺️ What It Does

Given any map coordinate in Pune, DSSE instantly computes:
- **Petrol Station Viability Score** (0–100) + confidence interval
- **EV Fast-Charging Viability Score** (0–100) + confidence interval
- **4-Quadrant Decision:** `BEST_PETROL` | `BEST_EV` | `BOTH_VIABLE` | `POOR_BOTH`
- **SHAP Waterfall:** per-feature attribution explaining *why* the site scored this way
- **SFDS (Spatial Factor Divergence Score):** quantifies how differently EV vs petrol succeed spatially

---

## 🚀 Quickstart

### 1. Clone & Setup Python
```bash
git clone <repo-url>
cd 5_EDI
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Run Phase 1 — Data Ingestion (Pune)
```bash
python scripts/01_download_raw_data.py --city pune
python scripts/02_build_spatial_indices.py --city pune
```

### 3. Run Phase 2 — Train Models (Pune)
```bash
python scripts/03_train_and_evaluate.py --city pune
```

### 4. Run Phase 3 — Generate Heatmaps (Pune)
```bash
python scripts/04_generate_heatmaps.py --city pune
```

### 5. Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 6. Start the Frontend
```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

---

## 📁 Repository Structure

```
5_EDI/
├── data/               ← Raw & processed spatial data (gitignored)
├── models/             ← Serialized ML model artifacts (gitignored)
├── figures/            ← UMAP scatter plots, SHAP charts (generated)
├── scripts/            ← Numbered pipeline scripts (01–04)
├── src/
│   ├── data/           ← API clients & OSM ingestion
│   ├── models/         ← ML training, SHAP, SFDS, AHP baseline
│   ├── spatial/        ← H3 grid, constraint filters
│   └── api/            ← FastAPI backend (routes + schemas)
└── frontend/           ← React + Vite + Leaflet web dashboard
```

---

## 🔬 Implementation Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Pune data ingestion & spatial index | 🔄 In Progress |
| Phase 2 | Model training & SFDS | ⬜ Pending |
| Phase 3 | H3 heatmap generation | ⬜ Pending |
| Phase 4 | FastAPI backend | ⬜ Pending |
| Phase 5 | React frontend dashboard | ⬜ Pending |

---

## 📄 Full Specification

See [CONTEXT_EDI_5.md](./CONTEXT_EDI_5.md) — the master project specification. Feed this to any coding agent at the start of every session.
