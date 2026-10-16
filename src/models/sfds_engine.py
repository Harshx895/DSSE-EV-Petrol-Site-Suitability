"""
src/models/sfds_engine.py
---------------------------
Spatial Factor Divergence Score (SFDS) — the core research contribution.

Quantifies how much the spatial success environments differ between
petrol stations and EV charging hubs using Jensen-Shannon Divergence
on the learned feature importance weight vectors.

Per CONTEXT §7:
  SFDS → 0: co-location archetype (same spatial drivers)
  SFDS → 1: decoupled archetype (EV migrated to dwell/commercial zones)

Outputs:
  - SFDS point estimate + 95% bootstrap CI (n=1000)
  - UMAP 2D feature space scatter (figures/umap_feature_space_{city}.png)
  - sfds_results_{city}.json for MLflow / paper reporting
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import jensenshannon
from sklearn.utils import resample

from src.models.feature_extractor import FEATURE_NAMES


def compute_sfds(
    weights_petrol: np.ndarray,
    weights_ev: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Compute SFDS point estimate + 95% bootstrap confidence interval.

    Parameters:
      weights_petrol / weights_ev : feature importance arrays (sum need not = 1)

    Returns dict with keys: sfds, ci_low, ci_high, report_str
    """
    rng = np.random.default_rng(random_state)

    # Normalize to probability simplex
    p = weights_petrol / (weights_petrol.sum() + 1e-12)
    q = weights_ev     / (weights_ev.sum()     + 1e-12)

    sfds_point = float(jensenshannon(p, q, base=2))

    # Bootstrap CI
    samples = []
    for _ in range(n_bootstrap):
        idx_p = rng.integers(0, len(p), len(p))
        idx_q = rng.integers(0, len(q), len(q))
        p_b = p[idx_p]; p_b /= p_b.sum()
        q_b = q[idx_q]; q_b /= q_b.sum()
        samples.append(jensenshannon(p_b, q_b, base=2))

    ci_low, ci_high = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

    # Interpretation
    if sfds_point < 0.20:
        interp = "Co-Location Archetype: EV and petrol share similar spatial drivers"
    elif sfds_point < 0.45:
        interp = "Partial Decoupling: EV shows measurable spatial differentiation"
    else:
        interp = "Decoupled Archetype: EV infrastructure has distinct spatial ecology"

    return {
        "sfds":       sfds_point,
        "ci_low":     ci_low,
        "ci_high":    ci_high,
        "report_str": f"SFDS = {sfds_point:.4f} [95% sensitivity range: {ci_low:.4f}–{ci_high:.4f}]",
        "interpretation": interp,
        "weights_petrol": p.tolist(),
        "weights_ev":     q.tolist(),
        "feature_names":  FEATURE_NAMES,
    }


def plot_sfds_comparison(result: dict, city: str, out_dir: Path) -> Path:
    """
    Generate horizontal bar chart comparing petrol vs EV feature weights.
    Suitable as a paper figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = result["feature_names"]
    p     = np.array(result["weights_petrol"])
    q     = np.array(result["weights_ev"])

    # Short display names
    labels = [
        "Primary Hwy Dist", "Secondary Rd Dist", "Cafes 300m",
        "Restaurants 300m", "Retail 500m", "Supermarket 500m",
        "Offices 500m", "Bus Stops 500m", "Metro Dist",
        "Residential 500m", "Power Grid Dist", "Competitor Petrol 1km",
        "Competitor EV 1km", "Dwell Index", "Transit Access", "Parking 300m"
    ]

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y_pos - 0.18, p, height=0.35, label="Petrol",
            color="#E8764B", alpha=0.85)
    ax.barh(y_pos + 0.18, q, height=0.35, label="EV Charging",
            color="#4B9FE8", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Normalized Feature Importance (SHAP)", fontsize=10)
    ax.set_title(
        f"Spatial Factor Divergence — {city.capitalize()}\n"
        f"{result['report_str']}", fontsize=11
    )
    ax.legend(fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    plt.tight_layout()

    out_path = out_dir / f"sfds_weights_{city}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SFDS] Weights plot → {out_path}")
    return out_path


def plot_umap_scatter(
    X_petrol: np.ndarray,
    X_ev: np.ndarray,
    city: str,
    out_dir: Path,
    random_state: int = 42,
) -> Path:
    """
    UMAP 2D scatter of petrol vs EV feature spaces — paper figure.
    """
    try:
        import umap
    except ImportError:
        print("  [UMAP] umap-learn not installed. Skipping scatter.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    X_all  = np.vstack([X_petrol, X_ev])
    labels = (["Petrol"] * len(X_petrol)) + (["EV Charging"] * len(X_ev))
    colors = ["#E8764B"] * len(X_petrol) + ["#4B9FE8"] * len(X_ev)

    reducer   = umap.UMAP(n_components=2, random_state=random_state,
                          n_neighbors=min(15, len(X_all)-1), min_dist=0.1)
    embedding = reducer.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1],
               c=colors, alpha=0.65, s=25, linewidths=0)
    p_patch = mpatches.Patch(color="#E8764B", label="Petrol Station")
    e_patch = mpatches.Patch(color="#4B9FE8", label="EV Charging Hub")
    ax.legend(handles=[p_patch, e_patch], fontsize=10)
    ax.set_title(f"UMAP Feature Space — {city.capitalize()}", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    out_path = out_dir / f"umap_feature_space_{city}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [UMAP] Scatter plot → {out_path}")
    return out_path


def save_sfds_results(result: dict, city: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sfds_results_{city}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [SFDS] Results → {path}")
