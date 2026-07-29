"""Feature extraction for sim ↔ real comparison.

Both sim T_max fields (FieldResult) and real thermal frames produce a
FieldFeatures bag of summary statistics that are robust to spatial
resolution mismatch. The score module compares two FieldFeatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldFeatures:
    t_max_mean_K: float
    t_max_std_K: float
    t_max_p50_K: float
    t_max_p95_K: float
    t_max_max_K: float
    coverage_fraction: float        # T >= liquidus
    keyhole_fraction: float         # T >= 0.9 * boiling
    histogram_K: np.ndarray         # (n_bins,) normalised counts
    histogram_edges_K: np.ndarray   # (n_bins + 1,) bin edges
    n_cells: int


def extract_field_features(
    field_K: np.ndarray,
    liquidus_K: float,
    boiling_K: float,
    n_bins: int = 32,
    bin_range_K: tuple[float, float] | None = None,
) -> FieldFeatures:
    """Aggregate a 2D thermal field into comparable summary statistics."""
    if field_K.ndim != 2:
        raise ValueError(f"expected 2D field; got shape {field_K.shape}")
    flat = field_K[np.isfinite(field_K)]
    if flat.size == 0:
        raise ValueError("field has no finite cells")
    if bin_range_K is None:
        bin_range_K = (float(flat.min()), float(flat.max()))
    if bin_range_K[1] <= bin_range_K[0]:
        bin_range_K = (bin_range_K[0], bin_range_K[0] + 1.0)
    counts, edges = np.histogram(flat, bins=n_bins, range=bin_range_K)
    total = counts.sum()
    norm = counts.astype(float) / max(total, 1)
    p50, p95 = np.percentile(flat, [50, 95])
    return FieldFeatures(
        t_max_mean_K=float(flat.mean()),
        t_max_std_K=float(flat.std()),
        t_max_p50_K=float(p50),
        t_max_p95_K=float(p95),
        t_max_max_K=float(flat.max()),
        coverage_fraction=float((flat >= liquidus_K).mean()),
        keyhole_fraction=float((flat >= 0.9 * boiling_K).mean()),
        histogram_K=norm,
        histogram_edges_K=edges,
        n_cells=int(flat.size),
    )
