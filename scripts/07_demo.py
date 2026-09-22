"""
scripts/07_demo.py
-------------------
DSSE Terminal Demo — Mid-Semester Presentation

Usage:
  python -X utf8 scripts/07_demo.py --lat 18.5590 --lon 73.7868
  python -X utf8 scripts/07_demo.py --lat 18.5089 --lon 73.9260 --city pune
  python -X utf8 scripts/07_demo.py --interactive
"""

from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Google Maps URL parser ─────────────────────────────────────────────────────
def parse_gmaps_url(url: str) -> tuple[float, float]:
    """
    Extract (lat, lon) from any Google Maps URL format:
      • https://www.google.com/maps/place/.../@18.55,73.78,17z/...
      • https://www.google.com/maps?q=18.55,73.78
      • https://maps.app.goo.gl/XXXXX  (shortened — follows redirect)
    Raises ValueError if no coordinates found.
    """
    # Follow redirect for shortened links (goo.gl/maps, maps.app.goo.gl)
    if "goo.gl" in url or "maps.app" in url:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=8) as resp:
                url = resp.url
        except Exception:
            pass  # fall through to regex on original URL

    # Pattern 1: /@lat,lon,zoom  (most common)
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern 2: ?q=lat,lon  or  &q=lat,lon
    qs = parse_qs(urlparse(url).query)
    if "q" in qs:
        parts = qs["q"][0].split(",")
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass

    # Pattern 3: /maps/place/Name/lat,lon
    m = re.search(r"(-?\d{1,3}\.\d{4,}),(-?\d{1,3}\.\d{4,})", url)
    if m:
        return float(m.group(1)), float(m.group(2))

    raise ValueError(
        f"Could not extract coordinates from URL.\n"
        f"  Supported formats:\n"
        f"    maps.google.com/maps/place/.../@LAT,LON,...\n"
        f"    maps.google.com/maps?q=LAT,LON\n"
        f"    maps.app.goo.gl/XXXX  (shortened)\n"
        f"  Got: {url}"
    )

# ── Colour helpers (ANSI) ──────────────────────────────────────────────────────
R  = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B  = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"
W  = "\033[97m"; DIM = "\033[2m"; BOLD = "\033[1m"; RST = "\033[0m"

QUADRANT_COLOUR = {
    "BEST_PETROL": B,
    "BEST_EV":     G,
    "BOTH_VIABLE": Y,
    "POOR_BOTH":   DIM,
}
QUADRANT_ICON = {
    "BEST_PETROL": "⛽",
    "BEST_EV":     "⚡",
    "BOTH_VIABLE": "🏆",
    "POOR_BOTH":   "✖",
}


def bar(score: float, width: int = 30) -> str:
    filled = int(score / 100 * width)
    colour = G if score >= 60 else Y if score >= 40 else R
    return colour + "█" * filled + DIM + "░" * (width - filled) + RST


def print_header():
    print(f"\n{BOLD}{C}{'═'*62}{RST}")
    print(f"{BOLD}{C}  DSSE — Dynamic Site Suitability Engine  |  Pune Demo{RST}")
    print(f"{BOLD}{C}{'═'*62}{RST}\n")


def score_label(s: float) -> str:
    if s >= 75: return f"{G}Excellent ✓{RST}"
    if s >= 60: return f"{G}Good ✓{RST}"
    if s >= 40: return f"{Y}Moderate ~{RST}"
    return f"{R}Weak ✗{RST}"


def print_score(result: dict):
    q   = result["quadrant"]
    qc  = QUADRANT_COLOUR.get(q, W)
    qi  = QUADRANT_ICON.get(q, "?")
    lat, lon = result["lat"], result["lon"]

    print(f"  {DIM}📍 Location:{RST} {BOLD}{lat:.4f}°N, {lon:.4f}°E{RST}\n")

    # ── Scores ───────────────────────────────────────────────────────────────
    print(f"  {BOLD}{'─'*58}{RST}")
    print(f"  {BOLD}  SCORES{RST}")
    print(f"  {BOLD}{'─'*58}{RST}")

    for key, label, icon in [("petrol", "Petrol Station", "⛽"),
                              ("ev",     "EV Charger    ", "⚡")]:
        d  = result[key]
        sc = d["composite"]
        lo, hi = d["interval"]
        lbl = score_label(sc)
        print(f"\n  {BOLD}{icon} {label}{RST}   {lbl}")
        print(f"     Score    {bar(sc)} {BOLD}{sc:5.1f}/100{RST}")
        print(f"     Range    {DIM}[{lo:.0f} – {hi:.0f}] (conformal 90% interval){RST}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    ps = result["petrol"]["composite"]
    es = result["ev"]["composite"]

    VERDICT = {
        "BEST_PETROL":  f"  {B}{BOLD}✔  RECOMMENDED FOR PETROL STATION{RST}\n"
                        f"  {DIM}This location is road-accessible and commercially active.\n"
                        f"  NOT recommended for EV due to low residential density / dwell potential.{RST}",
        "BEST_EV":      f"  {G}{BOLD}✔  RECOMMENDED FOR EV CHARGER{RST}\n"
                        f"  {DIM}High residential density and dwell amenities favour EV.\n"
                        f"  NOT ideal for petrol (road access or throughput concern).{RST}",
        "BOTH_VIABLE":  f"  {Y}{BOLD}🏆  PRIME LOCATION — VIABLE FOR BOTH{RST}\n"
                        f"  {DIM}Recommend EV or EV+Petrol hybrid. Future-proof investment.{RST}",
        "POOR_BOTH":    f"  {R}{BOLD}✖  NOT RECOMMENDED FOR EITHER{RST}\n"
                        f"  {DIM}Insufficient road access, residential density, and amenities.{RST}",
    }
    print(f"\n  {'─'*58}")
    print(f"  {BOLD}  VERDICT{RST}")
    print(f"  {'─'*58}\n")
    print(VERDICT.get(q, "  Unknown quadrant"))

    # ── Warnings ─────────────────────────────────────────────────────────────
    for w in result.get("warnings", []):
        print(f"\n  {Y}⚠  {w}{RST}")

    # ── SHAP: Petrol + EV side by side ────────────────────────────────────────
    print(f"\n  {'─'*58}")
    print(f"  {BOLD}  WHY? — Key Location Factors (SHAP Explanation){RST}")
    print(f"  {'─'*58}")
    print(f"  {DIM}▲ = boosts score  |  ▼ = reduces score{RST}\n")

    shap_p = sorted(result["shap"]["petrol"],
                    key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
    shap_e = sorted(result["shap"]["ev"],
                    key=lambda x: abs(x["shap_value"]), reverse=True)[:5]

    print(f"  {BOLD}{'⛽ Petrol factors':<35}  {'⚡ EV factors'}{RST}")
    print(f"  {'─'*56}")
    for i in range(max(len(shap_p), len(shap_e))):
        # Petrol side
        if i < len(shap_p):
            sp = shap_p[i]
            vp = sp["shap_value"]
            cp = G if vp > 0 else R
            ap = "▲" if vp > 0 else "▼"
            fp = sp["feature"].replace("_", " ")[:26]
            left = f"{cp}{ap} {fp:<26} {vp:+.2f}{RST}"
        else:
            left = " " * 35

        # EV side
        if i < len(shap_e):
            se = shap_e[i]
            ve = se["shap_value"]
            ce = G if ve > 0 else R
            ae = "▲" if ve > 0 else "▼"
            fe = se["feature"].replace("_", " ")[:26]
            right = f"{ce}{ae} {fe:<26} {ve:+.2f}{RST}"
        else:
            right = ""

        print(f"  {left}  {right}")
    print()


def print_sfds(city: str):
    sfds_path = Path("data/processed") / city / f"sfds_results_{city}.json"
    if not sfds_path.exists():
        return
    with open(sfds_path) as f:
        s = json.load(f)

    print(f"{BOLD}{M}{'─'*62}{RST}")
    print(f"{BOLD}{M}  Research Finding: Spatial Feature Divergence Score (SFDS){RST}")
    print(f"{BOLD}{M}{'─'*62}{RST}")
    print(f"\n  {BOLD}SFDS = {s['sfds']:.4f}{RST}  [95% CI: {s['ci_low']:.4f}–{s['ci_high']:.4f}]")
    print(f"  {s['interpretation']}\n")
    print(f"  {DIM}Interpretation: EV stations occupy a measurably different{RST}")
    print(f"  {DIM}spatial environment from petrol stations in Pune.{RST}")

    # Top diverging features
    shap_path = Path("models") / city / "shap_importances.json"
    if shap_path.exists():
        with open(shap_path) as f:
            sh = json.load(f)
        import numpy as np
        names = sh["feature_names"]
        wp = np.array(sh["weights_petrol"])
        we = np.array(sh["weights_ev"])
        diffs = list(zip(names, wp, we))
        diffs.sort(key=lambda x: abs(x[2]-x[1]), reverse=True)

        print(f"\n  {BOLD}Feature weights (Petrol vs EV):{RST}")
        print(f"  {'Feature':<28}  {'Petrol':>8}  {'EV':>8}  {'Δ':>8}")
        print(f"  {'─'*56}")
        for name, p, e in diffs[:8]:
            d   = e - p
            col = G if d > 0.03 else R if d < -0.03 else DIM
            print(f"  {name:<28}  {p:>8.3f}  {e:>8.3f}  {col}{d:>+8.3f}{RST}")
    print()


def score_location(lat: float, lon: float, city: str):
    from src.models.predictor import Predictor
    from src.models.recommender import QUADRANT_LABELS

    print(f"\n  {DIM}Loading models…{RST}", end="", flush=True)
    t0 = time.time()
    pred = Predictor(city, Path("models"), Path(f"data/processed/{city}"))
    print(f"\r  {DIM}Models loaded in {time.time()-t0:.1f}s{RST}          ")

    t1 = time.time()
    result = pred.score(lat, lon)
    result["quadrant_label"] = QUADRANT_LABELS.get(result["quadrant"], "")
    ms = (time.time() - t1) * 1000
    print(f"  {DIM}Scored in {ms:.0f}ms{RST}\n")
    return result


def interactive_mode(city: str):
    print_header()
    print(f"  {DIM}Interactive mode — type 'quit' to exit{RST}\n")

    from src.models.predictor import Predictor
    from src.models.recommender import QUADRANT_LABELS
    pred = Predictor(city, Path("models"), Path(f"data/processed/{city}"))

    while True:
        try:
            raw = input(f"  {BOLD}Enter lat,lon or Google Maps URL (or 'quit'): {RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
        if raw.lower() in ("quit", "exit", "q"):
            break

        # Try URL first
        if raw.startswith("http") or "maps" in raw:
            try:
                lat, lon = parse_gmaps_url(raw)
                print(f"  {DIM}Parsed → {lat:.6f}, {lon:.6f}{RST}")
            except ValueError as e:
                print(f"  {R}{e}{RST}")
                continue
        else:
            try:
                lat, lon = map(float, raw.split(","))
            except ValueError:
                print(f"  {R}Invalid input. Use: lat,lon  OR  a Google Maps URL{RST}")
                continue

        result = pred.score(lat, lon)
        result["quadrant_label"] = QUADRANT_LABELS.get(result["quadrant"], "")
        print_score(result)


def main():
    parser = argparse.ArgumentParser(description="DSSE Terminal Demo")
    parser.add_argument("--lat",  type=float, default=None)
    parser.add_argument("--lon",  type=float, default=None)
    parser.add_argument("--url",  type=str,   default=None,
                        help="Google Maps URL or goo.gl link")
    parser.add_argument("--city", default="pune")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive mode (accepts lat,lon or Google Maps URL)")
    parser.add_argument("--sfds", action="store_true",
                        help="Print SFDS research finding only")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.city)
        return

    print_header()

    if args.sfds:
        print_sfds(args.city)
        return

    # Resolve coordinates
    if args.url:
        try:
            lat, lon = parse_gmaps_url(args.url)
            print(f"  {DIM}Parsed from URL → {lat:.6f}, {lon:.6f}{RST}\n")
        except ValueError as e:
            print(f"  {R}{e}{RST}")
            sys.exit(1)
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = 18.5590, 73.7868   # default: Baner, Pune

    result = score_location(lat, lon, args.city)
    print_score(result)
    print_sfds(args.city)
    print(f"{BOLD}{C}{'═'*62}{RST}\n")


if __name__ == "__main__":
    main()
