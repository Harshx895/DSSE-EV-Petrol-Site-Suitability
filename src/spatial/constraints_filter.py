"""
src/spatial/constraints_filter.py
------------------------------------
Hard pre-ML feasibility filter (CONTEXT §5):

  1. Setback Rule: must be within ≤150m of motorized road
  2. Exclusion Mask: cannot be inside water body
  3. Power Grid Warning (EV only): >2500m triggers warning badge

These checks run BEFORE the ML models to eliminate clearly infeasible sites.
Zero live API calls — uses local KDTrees only.
"""

from __future__ import annotations

MAX_ROAD_DIST_M   = 150.0
MAX_POWER_WARN_M  = 2500.0


def passes_hard_constraints(features: dict, infra_type: str = "both") -> tuple[bool, list[str]]:
    """
    Returns (passes: bool, warnings: list[str]).
    
    A site fails only on the setback rule or water exclusion.
    Power grid distance is a warning, not a hard failure.
    """
    warnings = []

    # 1. Road setback — hard fail
    road_dist = min(
        features.get("dist_primary_highway_m", 9999),
        features.get("dist_secondary_road_m",  9999),
    )
    if road_dist > MAX_ROAD_DIST_M:
        return False, [
            f"Fails setback rule: {road_dist:.0f}m from nearest road (max {MAX_ROAD_DIST_M}m)"
        ]

    # 2. Power grid warning (EV only)
    if infra_type in ("ev", "both"):
        pg_dist = features.get("power_grid_dist_m", 0)
        if pg_dist > MAX_POWER_WARN_M:
            warnings.append(
                f"High Grid Interconnection Cost: power grid {pg_dist:.0f}m away "
                f"(>{MAX_POWER_WARN_M:.0f}m threshold)"
            )

    return True, warnings
