"""Coordinate registration sim ↔ real.

Both sim and real frames live in the build-volume frame (ISO/ASTM 52921).
In practice the camera is offset and rotated relative to the laser scanner.
This module solves a planar rigid transform (translation + rotation) given
matched control points, using closed-form SVD — no scipy dependency.

Once a `RigidTransform` is known, `apply_to_grid` warps a sim grid into the
measurement frame so they can be compared cell-by-cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    rotation_deg: float
    translation_mm: tuple[float, float]
    rms_residual_mm: float

    def matrix(self) -> np.ndarray:
        rad = np.deg2rad(self.rotation_deg)
        c, s = np.cos(rad), np.sin(rad)
        M = np.array(
            [
                [c, -s, self.translation_mm[0]],
                [s, c, self.translation_mm[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        return M

    def apply(self, points_mm: np.ndarray) -> np.ndarray:
        """Apply transform to (N, 2) point array."""
        if points_mm.ndim != 2 or points_mm.shape[1] != 2:
            raise ValueError(f"expected (N,2) array, got {points_mm.shape}")
        rad = np.deg2rad(self.rotation_deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        return points_mm @ R.T + np.asarray(self.translation_mm)


def register_rigid_2d(
    src_mm: np.ndarray, dst_mm: np.ndarray
) -> RigidTransform:
    """Solve for the rigid 2D transform that maps `src` onto `dst`.

    Closed-form Procrustes analysis (Umeyama 1991, no scaling).

    src_mm: (N, 2), dst_mm: (N, 2). Returns RigidTransform mapping src -> dst.
    """
    if src_mm.shape != dst_mm.shape or src_mm.ndim != 2 or src_mm.shape[1] != 2:
        raise ValueError(
            f"src and dst must have matching (N, 2) shape; got {src_mm.shape}, {dst_mm.shape}"
        )
    if src_mm.shape[0] < 2:
        raise ValueError("need at least 2 control-point pairs")
    src_c = src_mm.mean(axis=0)
    dst_c = dst_mm.mean(axis=0)
    src0 = src_mm - src_c
    dst0 = dst_mm - dst_c
    H = src0.T @ dst0
    U, _, Vt = np.linalg.svd(H)
    # ensure proper rotation (det = +1)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c
    rot_deg = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))
    src_aligned = src_mm @ R.T + t
    rms = float(np.sqrt(((src_aligned - dst_mm) ** 2).sum(axis=1).mean()))
    return RigidTransform(
        rotation_deg=rot_deg,
        translation_mm=(float(t[0]), float(t[1])),
        rms_residual_mm=rms,
    )


def resample_field_to_grid(
    src_field: np.ndarray,
    src_grid_x_mm: np.ndarray,
    src_grid_y_mm: np.ndarray,
    target_grid_x_mm: np.ndarray,
    target_grid_y_mm: np.ndarray,
    transform: RigidTransform | None = None,
) -> np.ndarray:
    """Bilinear-resample src_field onto the target grid, after optional transform.

    `src_field` shape (Nx, Ny) on grid (src_grid_x_mm, src_grid_y_mm).
    """
    Tx, Ty = np.meshgrid(target_grid_x_mm, target_grid_y_mm, indexing="ij")
    pts = np.column_stack([Tx.ravel(), Ty.ravel()])
    if transform is not None:
        # Inverse transform: target -> src frame
        rad = np.deg2rad(transform.rotation_deg)
        c, s = np.cos(rad), np.sin(rad)
        R_inv = np.array([[c, s], [-s, c]])
        t = np.array(transform.translation_mm)
        pts = (pts - t) @ R_inv.T
    fx = (pts[:, 0] - src_grid_x_mm[0]) / max(src_grid_x_mm[-1] - src_grid_x_mm[0], 1e-12) * (
        len(src_grid_x_mm) - 1
    )
    fy = (pts[:, 1] - src_grid_y_mm[0]) / max(src_grid_y_mm[-1] - src_grid_y_mm[0], 1e-12) * (
        len(src_grid_y_mm) - 1
    )
    out = np.full(pts.shape[0], np.nan, dtype=float)
    valid = (fx >= 0) & (fx <= len(src_grid_x_mm) - 1) & (fy >= 0) & (fy <= len(src_grid_y_mm) - 1)
    fxv = fx[valid]
    fyv = fy[valid]
    ix = np.clip(fxv.astype(int), 0, len(src_grid_x_mm) - 2)
    iy = np.clip(fyv.astype(int), 0, len(src_grid_y_mm) - 2)
    a = fxv - ix
    b = fyv - iy
    out[valid] = (
        (1 - a) * (1 - b) * src_field[ix, iy]
        + a * (1 - b) * src_field[ix + 1, iy]
        + (1 - a) * b * src_field[ix, iy + 1]
        + a * b * src_field[ix + 1, iy + 1]
    )
    return out.reshape(Tx.shape)
