"""
src/data/station_deduplicator.py
──────────────────────────────────
Deduplicates EV charging station records collected from three sources:
  1. Google Places API
  2. OpenChargeMap (OCM)
  3. PlugShare API (future)

Rule: two records within 50 metres of each other = same physical station.
Priority for canonical record: OCM > Google Places > PlugShare
(OCM has richer technical specs like kw_max, connector_types)

Input:  DataFrames from google_places_client, ocm_client
Output: A single deduplicated stations_master.csv with columns:
  canonical_id, name, lat, lon, rating, user_ratings_total,
  opening_date_est, kw_max, connector_types, is_fast_charge,
  infra_type, city, sources (pipe-separated list of contributing APIs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Two stations within this degree distance (~50m at Indian latitudes) = same station
DEDUP_THRESHOLD_DEG = 0.00045   # ~50m


def _haversine_approx_deg(lat1: float, lat2: float) -> float:
    """Approximate degrees-to-metres conversion (valid near equator ±20°)."""
    return (lat2 - lat1) * 111_320


def deduplicate(
    google_df: pd.DataFrame,
    ocm_df: pd.DataFrame,
    plugshare_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge and deduplicate stations from all sources.

    Strategy:
      1. Start with OCM records (highest technical quality).
      2. For each Google Places EV record, check if any OCM record is within 50m.
         - If yes: enrich OCM record with Google rating/review data, skip Google record.
         - If no:  add as new record (OCM may have missed it).
      3. Repeat for PlugShare.
      4. Petrol stations come only from Google (OCM/PlugShare don't cover petrol).
    """
    # ── Split Google records by type ─────────────────────────────────────────
    google_ev     = google_df[google_df["infra_type"] == "ev"].copy()
    google_petrol = google_df[google_df["infra_type"] == "petrol"].copy()

    # ── Prepare OCM base ─────────────────────────────────────────────────────
    ocm_df = ocm_df.copy()
    ocm_df["rating"]             = None
    ocm_df["user_ratings_total"] = None
    ocm_df["opening_date_est"]   = None
    ocm_df["sources"]            = "ocm"

    # ── Match Google EV → OCM ────────────────────────────────────────────────
    enriched_ocm = ocm_df.copy()
    google_ev_unmatched = []

    if not ocm_df.empty and not google_ev.empty:
        ocm_coords = ocm_df[["lat", "lon"]].values
        tree = cKDTree(ocm_coords)

        for _, g_row in google_ev.iterrows():
            # Pass a flat 1D array so dist/idx come back as scalars
            g_coord = np.array([g_row["lat"], g_row["lon"]])
            dist, ocm_idx = tree.query(g_coord, k=1)

            if dist <= DEDUP_THRESHOLD_DEG:
                # Enrich matching OCM record with Google ratings
                enriched_ocm.at[ocm_idx, "rating"]             = g_row.get("rating")
                enriched_ocm.at[ocm_idx, "user_ratings_total"] = g_row.get("user_ratings_total")
                enriched_ocm.at[ocm_idx, "opening_date_est"]   = g_row.get("opening_date_est")
                existing_sources = enriched_ocm.at[ocm_idx, "sources"] or "ocm"
                enriched_ocm.at[ocm_idx, "sources"] = existing_sources + "|google"
            else:
                # Google has an EV station OCM doesn't know about
                new_row = g_row.to_dict()
                new_row["ocm_id"]          = None
                new_row["kw_max"]          = None
                new_row["connector_types"] = None
                new_row["is_fast_charge"]  = None
                new_row["sources"]         = "google"
                google_ev_unmatched.append(new_row)
    else:
        # No OCM data — use all Google EV records
        for _, g_row in google_ev.iterrows():
            new_row = g_row.to_dict()
            new_row.update({"ocm_id": None, "kw_max": None,
                            "connector_types": None, "is_fast_charge": None,
                            "sources": "google"})
            google_ev_unmatched.append(new_row)

    # ── Petrol stations (Google only) ────────────────────────────────────────
    google_petrol = google_petrol.copy()
    google_petrol["ocm_id"]          = None
    google_petrol["kw_max"]          = None
    google_petrol["connector_types"] = None
    google_petrol["is_fast_charge"]  = False
    google_petrol["sources"]         = "google"

    # ── Assemble final dataset ───────────────────────────────────────────────
    parts = [enriched_ocm]
    if google_ev_unmatched:
        parts.append(pd.DataFrame(google_ev_unmatched))
    parts.append(google_petrol)
    if plugshare_df is not None and not plugshare_df.empty:
        plugshare_df = plugshare_df.copy()
        plugshare_df["sources"] = "plugshare"
        parts.append(plugshare_df)

    # Drop columns that are entirely NA before concat to avoid FutureWarning
    parts_clean = [p.dropna(axis=1, how="all") for p in parts if not p.empty]
    master = pd.concat(parts_clean, ignore_index=True)

    # Add a stable canonical ID
    master.insert(0, "canonical_id", [f"DSSE_{i:05d}" for i in range(len(master))])

    # ── Add ev_label for PU Learning ─────────────────────────────────────────
    # 'commercial' = OCM-verified, kw_max >= 22  → POSITIVE class
    # 'candidate'  = Google-only EV, kW unknown  → UNLABELED (may be positive)
    # 'petrol'     = petrol station              → UNLABELED background
    def _label(row):
        if row["infra_type"] == "petrol":
            return "petrol"
        if row.get("is_fast_charge") is True:
            return "commercial"   # OCM-verified ≥22kW
        return "candidate"        # Google EV, kW unknown

    master["ev_label"] = master.apply(_label, axis=1)

    # Summary
    n_commercial = (master["ev_label"] == "commercial").sum()
    n_candidate  = (master["ev_label"] == "candidate").sum()
    n_petrol     = (master["ev_label"] == "petrol").sum()
    print(f"[Deduplicator] Final: {len(master)} stations")
    print(f"  commercial EV (OCM-verified >=22kW): {n_commercial}")
    print(f"  candidate EV  (Google, kW unknown):  {n_candidate}")
    print(f"  petrol stations:                     {n_petrol}")
    return master


def save_master(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stations_master.csv"
    df.to_csv(path, index=False)
    print(f"[Deduplicator] Saved → {path}")
    return path


if __name__ == "__main__":
    # Quick test with empty DataFrames
    g = pd.DataFrame(columns=["place_id", "name", "lat", "lon", "rating",
                               "user_ratings_total", "opening_date_est",
                               "infra_type", "city"])
    o = pd.DataFrame(columns=["ocm_id", "name", "lat", "lon", "kw_max",
                               "connector_types", "is_fast_charge",
                               "infra_type", "city"])
    result = deduplicate(g, o)
    print(result)
