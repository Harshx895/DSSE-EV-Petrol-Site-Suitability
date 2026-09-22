"""
scripts/01b_enrich_opening_dates.py
--------------------------------------
Fetches review timestamps from Google Places API (New) for all stations
in stations_master.csv that have a place_id but no opening_date_est.

Uses the oldest review publishTime as a proxy for opening date.
This enables proper ACTI age-correction in Formulation 2.

Cost estimate: ~558 calls × $0.017 = ~$9.50 (well within $200 free credit)
API endpoint: GET https://places.googleapis.com/v1/places/{place_id}
Field mask:   reviews (only this field — cheapest billing tier)

Usage:
  python -X utf8 scripts/01b_enrich_opening_dates.py --city pune
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
API_KEY  = os.getenv("GOOGLE_PLACES_API_KEY")
BASE_URL = "https://places.googleapis.com/v1/places/{place_id}"


def fetch_oldest_review_date(place_id: str) -> str | None:
    """
    Fetch up to 5 reviews for a place and return the oldest publishTime.
    Returns ISO date string (e.g. '2019-03-15') or None if no reviews found.
    """
    url = BASE_URL.format(place_id=place_id)
    headers = {
        "X-Goog-Api-Key":    API_KEY,
        "X-Goog-FieldMask":  "reviews",
        "Accept":            "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        reviews = data.get("reviews", [])
        if not reviews:
            return None
        # publishTime is ISO 8601: "2019-03-15T10:22:33Z"
        times = [r.get("publishTime") or r.get("relativePublishTimeDescription") 
                 for r in reviews if r.get("publishTime")]
        if not times:
            return None
        # Return the oldest
        times_sorted = sorted(times)
        return times_sorted[0][:10]   # YYYY-MM-DD
    except Exception:
        return None


def main(city: str) -> None:
    if not API_KEY:
        print("ERROR: GOOGLE_PLACES_API_KEY not set in .env")
        sys.exit(1)

    master_path = Path("data/processed") / city / "stations_master.csv"
    df = pd.read_csv(master_path)

    # Only enrich stations that have a place_id but no opening_date_est
    needs_enrich = (
        df["place_id"].notna() &
        df["opening_date_est"].isna() &
        (df["source"].isin(["google_places", "google_ev", "google_petrol"]) |
         df["sources"].str.contains("google", na=False))
    )

    candidates = df[needs_enrich].copy()
    print(f"\n[Enrich] {len(candidates)} stations need opening date enrichment")
    print(f"         (out of {len(df)} total in {master_path})\n")

    if len(candidates) == 0:
        print("Nothing to enrich. All stations already have opening_date_est or no place_id.")
        return

    enriched = 0
    for idx, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Fetching reviews"):
        place_id = str(row["place_id"])
        if not place_id or place_id == "nan":
            continue

        date = fetch_oldest_review_date(place_id)
        if date:
            df.at[idx, "opening_date_est"] = date
            enriched += 1

        time.sleep(0.05)   # ~20 req/s — well within 60 QPM limit

    df.to_csv(master_path, index=False)

    n_total = df["opening_date_est"].notna().sum()
    print(f"\n[Enrich] Done. {enriched} new dates fetched.")
    print(f"         Total stations with opening_date_est: {n_total}/{len(df)}")
    print(f"         Saved → {master_path}")
    print(f"\n  Now retrain: python -X utf8 scripts/05_train_and_evaluate.py --city {city}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="pune",
                        choices=["pune", "bengaluru", "mumbai", "delhi"])
    args = parser.parse_args()
    main(args.city)
