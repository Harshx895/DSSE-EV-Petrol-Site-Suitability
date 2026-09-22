"""
scripts/08_visualize_heatmap.py
---------------------------------
Generates an interactive browser heatmap from the pre-cached H3 GeoJSON.

Features:
  - Four-quadrant colour coding (Petrol=Blue, EV=Green, Both=Gold, None=Gray)
  - Hover tooltip: scores + quadrant
  - Actual EV + petrol station markers
  - SFDS annotation panel
  - Auto-opens in default browser

Usage:
  python -X utf8 scripts/08_visualize_heatmap.py --city pune
"""

from __future__ import annotations
import argparse, json, sys, webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import folium
    from folium.plugins import FloatImage
except ImportError:
    print("Installing folium...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "folium", "-q"])
    import folium

QUADRANT_COLORS = {
    "BEST_PETROL":  "#3B82F6",   # blue
    "BEST_EV":      "#22C55E",   # green
    "BOTH_VIABLE":  "#F59E0B",   # amber/gold
    "POOR_BOTH":    "#6B7280",   # gray
}
QUADRANT_LABELS = {
    "BEST_PETROL": "⛽ Best for Petrol",
    "BEST_EV":     "⚡ Best for EV",
    "BOTH_VIABLE": "🏆 Both Viable",
    "POOR_BOTH":   "✖ Poor for Both",
}

PUNE_CENTER = [18.5204, 73.8567]


def build_map(city: str) -> folium.Map:
    heatmap_path = Path("data/processed/heatmaps") / f"heatmap_{city}.geojson"
    stations_path = Path("data/processed") / city / "stations_master.csv"
    sfds_path     = Path("data/processed") / city / f"sfds_results_{city}.json"

    if not heatmap_path.exists():
        raise FileNotFoundError(f"Run scripts/06_cache_heatmap.py --city {city} first.")

    with open(heatmap_path, encoding="utf-8") as f:
        heatmap = json.load(f)

    # Normalize H3 geometry: h3_to_geo_boundary returns coordinate list,
    # not a proper {"type":"Polygon","coordinates":[...]} dict
    for feat in heatmap["features"]:
        geom = feat["geometry"]
        if isinstance(geom, list):
            feat["geometry"] = {"type": "Polygon", "coordinates": [geom]}
        elif isinstance(geom, dict) and "coordinates" not in geom:
            # already has type but no coordinates key — shouldn't happen
            pass

    # ── Base map ──────────────────────────────────────────────────────────────
    m = folium.Map(
        location=PUNE_CENTER,
        zoom_start=12,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri | Sources: Esri, HERE, Garmin, FAO, NOAA, USGS",
        prefer_canvas=True,
    )

    # ── H3 hexagon layer (single GeoJson call — much faster) ─────────────────
    hex_group = folium.FeatureGroup(name="Suitability Heatmap", show=True)

    def hex_style(feature):
        q = feature["properties"].get("quadrant", "POOR_BOTH")
        return {
            "fillColor":   QUADRANT_COLORS.get(q, "#6B7280"),
            "color":       "#000000",
            "weight":      0.5,
            "fillOpacity": 0.65,
        }

    def hex_tooltip(feature):
        p     = feature["properties"]
        q     = p.get("quadrant", "?")
        label = QUADRANT_LABELS.get(q, q)
        ps    = p.get("petrol_score", 0)
        es    = p.get("ev_score", 0)
        warns = p.get("warnings", [])
        warn_html = ""
        if warns:
            warn_html = "<br><span style='color:#F59E0B'>⚠ " + "<br>⚠ ".join(warns) + "</span>"
        return folium.GeoJsonTooltip(
            fields=["quadrant", "petrol_score", "ev_score"],
            aliases=["Quadrant", "Petrol Score", "EV Score"],
            localize=True,
        )

    folium.GeoJson(
        heatmap,
        name="Suitability Hexagons",
        style_function=hex_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["quadrant", "petrol_score", "ev_score"],
            aliases=["Quadrant 🏷", "Petrol Score ⛽", "EV Score ⚡"],
            style="font-family:monospace;font-size:13px;",
            localize=True,
        ),
    ).add_to(hex_group)

    hex_group.add_to(m)


    # ── Station markers ───────────────────────────────────────────────────────
    if stations_path.exists():
        import pandas as pd
        df = pd.read_csv(stations_path).dropna(subset=["lat", "lon"])

        ev_group  = folium.FeatureGroup(name="⚡ EV Stations (known)", show=True)
        pet_group = folium.FeatureGroup(name="⛽ Petrol Stations",     show=False)

        for _, row in df[df["ev_label"] == "commercial"].iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6, color="#22C55E", fill=True, fill_color="#22C55E",
                fill_opacity=0.9, weight=2,
                tooltip=f"⚡ {row['name']}",
            ).add_to(ev_group)

        for _, row in df[df["infra_type"] == "petrol"].sample(min(80, len(df[df["infra_type"]=="petrol"])),
                                                               random_state=42).iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4, color="#3B82F6", fill=True, fill_color="#3B82F6",
                fill_opacity=0.7, weight=1,
                tooltip=f"⛽ {row['name']}",
            ).add_to(pet_group)

        ev_group.add_to(m)
        pet_group.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    sfds_str = ""
    if sfds_path.exists():
        with open(sfds_path) as f:
            sfds = json.load(f)
        sfds_str = (
            f"<hr style='border-color:#444;margin:6px 0'>"
            f"<b style='color:#A78BFA'>SFDS = {sfds['sfds']:.4f}</b><br>"
            f"<span style='font-size:11px;color:#9CA3AF'>"
            f"[{sfds['ci_low']:.3f}–{sfds['ci_high']:.3f}] — "
            f"Partial Decoupling</span>"
        )

    legend_html = f"""
    <div style='position:fixed;bottom:30px;left:20px;z-index:1000;
                background:#1F2937;border:1px solid #374151;border-radius:8px;
                padding:12px 16px;font-family:monospace;font-size:13px;color:#F9FAFB;
                box-shadow:0 4px 12px rgba(0,0,0,0.5)'>
      <b style='font-size:15px'>DSSE — Site Suitability</b><br>
      <span style='font-size:11px;color:#9CA3AF'>Pune, Maharashtra • H3 Res-8</span>
      <hr style='border-color:#374151;margin:6px 0'>
      <span style='color:#3B82F6'>█</span> ⛽ Best for Petrol<br>
      <span style='color:#22C55E'>█</span> ⚡ Best for EV<br>
      <span style='color:#F59E0B'>█</span> 🏆 Both Viable<br>
      <span style='color:#6B7280'>█</span> ✖  Poor for Both
      {sfds_str}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ─────────────────────────────────────────────────────────────────
    title_html = """
    <div style='position:fixed;top:12px;left:50%;transform:translateX(-50%);
                z-index:1000;background:#1F2937cc;border:1px solid #374151;
                border-radius:8px;padding:8px 20px;font-family:monospace;
                font-size:14px;color:#F9FAFB;text-align:center;'>
      <b>Dynamic Site Suitability Engine</b> &nbsp;|&nbsp; Pune Demo
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="pune")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    print(f"\n[Heatmap] Building interactive map for {args.city}...")
    m = build_map(args.city)

    out = Path("figures") / f"heatmap_{args.city}.html"
    out.parent.mkdir(exist_ok=True)
    m.save(str(out))

    print(f"[Heatmap] Saved → {out.resolve()}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
        print("[Heatmap] Opened in browser.")


if __name__ == "__main__":
    main()
