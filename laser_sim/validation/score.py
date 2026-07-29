"""Sim ↔ real distance metrics.

Used by the validation harness to score how close a simulation is to a
measurement. The composite SimRealScore aggregates several geometry-aware
metrics into one number; lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.validation.feature_extract import FieldFeatures


@dataclass(frozen=True)
class SimRealScore:
    rmse_K: float
    iou_melt: float
    iou_keyhole: float
    cosine_hist: float
    kl_hist: float
    composite: float


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("inf")
    return float(np.sqrt(((a[mask] - b[mask]) ** 2).mean()))


def iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) on histograms, with smoothing."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float((p * np.log(p / q)).sum())


def score_thermal_frame(
    sim_field_K: np.ndarray,
    real_field_K: np.ndarray,
    liquidus_K: float,
    boiling_K: float,
    sim_feat: FieldFeatures | None = None,
    real_feat: FieldFeatures | None = None,
) -> SimRealScore:
    """Composite distance: RMSE (normalized) + (1-IoU) + (1-cos) + KL.

    Both fields must be on the same grid (caller's responsibility — use
    register.resample_field_to_grid first if needed).
    """
    if sim_field_K.shape != real_field_K.shape:
        raise ValueError(
            f"sim/real grid shape mismatch: {sim_field_K.shape} vs {real_field_K.shape}; "
            "resample first via register.resample_field_to_grid"
        )
    pixel_rmse = rmse(sim_field_K, real_field_K)
    iou_melt = iou_mask(sim_field_K >= liquidus_K, real_field_K >= liquidus_K)
    iou_kh = iou_mask(sim_field_K >= 0.9 * boiling_K, real_field_K >= 0.9 * boiling_K)
    if sim_feat is None or real_feat is None:
        from laser_sim.validation.feature_extract import extract_field_features

        # use shared bin range so cosine/KL are meaningful
        lo = float(min(np.nanmin(sim_field_K), np.nanmin(real_field_K)))
        hi = float(max(np.nanmax(sim_field_K), np.nanmax(real_field_K)))
        sim_feat = extract_field_features(
            sim_field_K, liquidus_K=liquidus_K, boiling_K=boiling_K, bin_range_K=(lo, hi)
        )
        real_feat = extract_field_features(
            real_field_K, liquidus_K=liquidus_K, boiling_K=boiling_K, bin_range_K=(lo, hi)
        )
    cos = cosine_similarity(sim_feat.histogram_K, real_feat.histogram_K)
    kl = kl_divergence(sim_feat.histogram_K, real_feat.histogram_K)
    # normalise RMSE by liquidus (a scale-invariant T error)
    rmse_norm = pixel_rmse / max(liquidus_K, 1e-6)
    composite = (
        0.40 * rmse_norm
        + 0.20 * (1.0 - iou_melt)
        + 0.10 * (1.0 - iou_kh)
        + 0.15 * (1.0 - cos)
        + 0.15 * kl
    )
    return SimRealScore(
        rmse_K=pixel_rmse,
        iou_melt=iou_melt,
        iou_keyhole=iou_kh,
        cosine_hist=cos,
        kl_hist=kl,
        composite=float(composite),
    )
