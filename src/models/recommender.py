"""
src/models/recommender.py
---------------------------
Four-quadrant decision logic (per CONTEXT §6).

Final Score = 0.4 × Viability_PU + 0.6 × Throughput_Regression

Quadrant thresholds (both normalized 0-100):
  BEST_PETROL  : petrol ≥ 60 AND ev < 60
  BEST_EV      : ev ≥ 60 AND petrol < 60
  BOTH_VIABLE  : petrol ≥ 60 AND ev ≥ 60  → recommend EV (future-proofing)
  POOR_BOTH    : petrol < 60 AND ev < 60
"""

from __future__ import annotations

THRESHOLD = 60.0


def score_to_100(raw: float) -> float:
    """
    Clip and scale a raw model output to [0, 100].
    PU calibrated probabilities are already [0,1]; regressor ACTI needs scaling.
    Handles both cases gracefully.
    """
    if raw <= 0:
        return 0.0
    if raw > 1.0:
        # ACTI regressor outputs — scale via sigmoid-like compression
        import numpy as np
        return float(min(100.0, 100.0 * (1 - np.exp(-raw))))
    return float(raw * 100.0)


def decide_quadrant(score_petrol: float, score_ev: float) -> str:
    """
    Returns one of: BEST_PETROL | BEST_EV | BOTH_VIABLE | POOR_BOTH.

    Parameters:
      score_petrol : composite petrol score [0–100]
      score_ev     : composite EV score [0–100]
    """
    p_ok = score_petrol >= THRESHOLD
    e_ok = score_ev     >= THRESHOLD

    if p_ok and e_ok:
        return "BOTH_VIABLE"
    if p_ok and not e_ok:
        return "BEST_PETROL"
    if e_ok and not p_ok:
        return "BEST_EV"
    return "POOR_BOTH"


QUADRANT_LABELS = {
    "BEST_PETROL": "High-Throughput Transit Node: Ideal for rapid fuel turnover.",
    "BEST_EV":     "High-Dwell Destination Hub: Strong EV siting; unsuitable for fuel tankers.",
    "BOTH_VIABLE": "Prime Multi-Modal Interchange: Viable for both — recommend EV/Hybrid.",
    "POOR_BOTH":   "Infrastructural Dead Zone: Unviable for either infrastructure type.",
}
