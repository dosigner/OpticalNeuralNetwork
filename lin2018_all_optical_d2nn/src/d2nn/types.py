"""Common dataclasses for configuration and detector definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectorRegionConfig:
    """Detector region definition in physical coordinates.

    Units:
        center_xy: meters
        size_xy: meters
    """

    name: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]


@dataclass
class RunConfig:
    """Top-level runtime config container.

    This class is intentionally permissive to support YAML-driven schemas.
    """

    experiment: dict[str, Any] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    detector_layout: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)
    viz: dict[str, Any] = field(default_factory=dict)
    error_model: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
