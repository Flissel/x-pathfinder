"""Measurement ingest: load thermal frames, pyrometer traces, micrographs.

Real LPBF lab data lives in many formats: HDF5 from coax cameras, CSV from
pyrometers, TIFF from cross-section micrographs. To keep the first slice
dependency-light, the canonical formats here are:

  - thermal frame:   .npz with keys 'frame_K' (H,W float64),
                     'pixel_um' (float), 'origin_xy_mm' (2,), 'sensor' (str)
  - pyrometer trace: .csv with header 'time_s,temperature_K[,sensor]'
  - measurement set: a directory containing one thermal.npz, one
                     pyrometer.csv, plus an optional metadata.json with
                     scenario_id and registration hint

HDF5 ingest is opt-in (requires h5py extra) and lives in load_thermal_h5,
which raises an informative error if the dep is missing.

Coordinate convention: ISO/ASTM 52921 build-volume frame (origin at substrate
centre, +Z up). 'origin_xy_mm' on a thermal frame is the (x, y) of the
pixel (0, 0) in machine coordinates.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThermalFrame:
    timestamp_s: float
    frame_K: np.ndarray
    pixel_um: float
    origin_xy_mm: tuple[float, float]
    sensor: str
    source_uri: str


@dataclass(frozen=True)
class PyrometerTrace:
    time_s: np.ndarray
    temperature_K: np.ndarray
    sensor: str = "spot_pyrometer"
    source_uri: str = ""

    def __post_init__(self) -> None:
        if self.time_s.shape != self.temperature_K.shape:
            raise ValueError(
                f"time_s and temperature_K must match: {self.time_s.shape} vs "
                f"{self.temperature_K.shape}"
            )


@dataclass(frozen=True)
class Measurement:
    """Single ingested artefact (thermal frame, pyrometer trace, etc.)."""

    kind: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementSet:
    """Collection of measurements, all bound to one experimental run."""

    experiment_id: str
    scenario_id: str | None
    items: tuple[Measurement, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def of_kind(self, kind: str) -> tuple[Measurement, ...]:
        return tuple(m for m in self.items if m.kind == kind)

    def first_thermal(self) -> ThermalFrame | None:
        thermals = self.of_kind("thermal")
        return thermals[0].payload if thermals else None

    def first_pyrometer(self) -> PyrometerTrace | None:
        pyros = self.of_kind("pyrometer")
        return pyros[0].payload if pyros else None


def load_thermal_npz(path: str | Path) -> ThermalFrame:
    """Load a .npz thermal frame (canonical format)."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    if "frame_K" not in data.files:
        raise ValueError(f"{path}: required key 'frame_K' missing; got {data.files}")
    frame = np.asarray(data["frame_K"], dtype=float)
    if frame.ndim != 2:
        raise ValueError(f"{path}: 'frame_K' must be 2D, got shape {frame.shape}")
    pixel_um = float(data["pixel_um"]) if "pixel_um" in data.files else 100.0
    if "origin_xy_mm" in data.files:
        origin = tuple(float(x) for x in np.asarray(data["origin_xy_mm"]).flatten()[:2])
    else:
        origin = (0.0, 0.0)
    sensor = str(data["sensor"]) if "sensor" in data.files else "unknown"
    timestamp = float(data["timestamp_s"]) if "timestamp_s" in data.files else 0.0
    return ThermalFrame(
        timestamp_s=timestamp,
        frame_K=frame,
        pixel_um=pixel_um,
        origin_xy_mm=origin,
        sensor=sensor,
        source_uri=str(path),
    )


def load_thermal_h5(path: str | Path) -> ThermalFrame:
    """Load thermal frame from HDF5 (requires h5py extra)."""
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError(
            "load_thermal_h5 requires h5py; install with `pip install -e \".[hf]\"`"
        ) from e
    path = Path(path)
    with h5py.File(path, "r") as f:
        frame = np.asarray(f["frame_K"][...], dtype=float)
        pixel_um = float(f["frame_K"].attrs.get("pixel_um", 100.0))
        origin = tuple(
            float(x) for x in np.asarray(f["frame_K"].attrs.get("origin_xy_mm", (0.0, 0.0)))[:2]
        )
        sensor = str(f["frame_K"].attrs.get("sensor", "h5"))
        ts = float(f["frame_K"].attrs.get("timestamp_s", 0.0))
    return ThermalFrame(
        timestamp_s=ts,
        frame_K=frame,
        pixel_um=pixel_um,
        origin_xy_mm=origin,
        sensor=sensor,
        source_uri=str(path),
    )


def load_pyrometer_csv(path: str | Path) -> PyrometerTrace:
    """Read a CSV with header 'time_s,temperature_K[,sensor]'."""
    path = Path(path)
    times: list[float] = []
    temps: list[float] = []
    sensors: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "time_s" not in reader.fieldnames or "temperature_K" not in reader.fieldnames:
            raise ValueError(
                f"{path}: expected CSV header with 'time_s' and 'temperature_K'; got {reader.fieldnames}"
            )
        for row in reader:
            times.append(float(row["time_s"]))
            temps.append(float(row["temperature_K"]))
            if "sensor" in row and row["sensor"]:
                sensors.append(row["sensor"])
    sensor = sensors[0] if sensors else "spot_pyrometer"
    return PyrometerTrace(
        time_s=np.array(times, dtype=float),
        temperature_K=np.array(temps, dtype=float),
        sensor=sensor,
        source_uri=str(path),
    )


def load_measurement_set(directory: str | Path) -> MeasurementSet:
    """Load every recognised measurement from a directory.

    Conventions:
      <dir>/metadata.json      optional, sets scenario_id, experiment_id
      <dir>/thermal*.npz       loaded as ThermalFrame
      <dir>/pyrometer*.csv     loaded as PyrometerTrace
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    meta_path = directory / "metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    items: list[Measurement] = []
    for p in sorted(directory.glob("thermal*.npz")):
        items.append(Measurement(kind="thermal", payload=load_thermal_npz(p), metadata={"path": str(p)}))
    for p in sorted(directory.glob("pyrometer*.csv")):
        items.append(Measurement(kind="pyrometer", payload=load_pyrometer_csv(p), metadata={"path": str(p)}))
    return MeasurementSet(
        experiment_id=str(metadata.get("experiment_id", directory.name)),
        scenario_id=metadata.get("scenario_id"),
        items=tuple(items),
        metadata=metadata,
    )
