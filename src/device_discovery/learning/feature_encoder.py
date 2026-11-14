from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType

BOOLEAN_PROBES = (ProbeType.ROUTING, ProbeType.LLDP, ProbeType.SNMP, ProbeType.VLAN)

VENDOR_VALUES = ("cisco", "juniper", "arista", "hpe")

INTERFACE_VALUES = ("low", "medium", "high")

FEATURE_ORDER = tuple(ProbeType)


def _parse_csv_value(value: str):
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def observation_from_row(row: Mapping[str, str]) -> Observation:
    """Build an observation using only visible CSV features."""
    device_id = row["device_id"]

    evidence = {}

    for probe in FEATURE_ORDER:
        raw = row[probe.value]
        value = _parse_csv_value(raw)

        if value is not None:
            evidence[probe] = value

    return Observation(device_id=device_id, evidence=evidence)


def load_observations(csv_path: str | Path) -> list[Observation]:
    """Read visible observations without using class labels."""
    observations = []

    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            observations.append(observation_from_row(row))

    return observations


@dataclass
class FeatureEncoder:
    """Encode visible evidence into a fixed-size numeric vector."""

    firmware_values: tuple[str, ...] = field(default_factory=tuple)
    fitted: bool = False

    def fit(self, observations: Iterable[Observation]) -> FeatureEncoder:
        """Learn firmware vocabulary from training observations only."""
        firmware = set()

        for observation in observations:
            value = observation.evidence.get(ProbeType.FIRMWARE)

            if isinstance(value, str):
                firmware.add(value.lower())

        self.firmware_values = tuple(sorted(firmware))
        self.fitted = True

        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()

        names = []

        for probe in FEATURE_ORDER:
            prefix = probe.value

            if probe == ProbeType.VENDOR:
                categories = (*VENDOR_VALUES, "__unknown__", "__missing__")

            elif probe in BOOLEAN_PROBES:
                categories = ("false", "true", "__missing__")

            elif probe == ProbeType.INTERFACES:
                categories = (*INTERFACE_VALUES, "__unknown__", "__missing__")

            elif probe == ProbeType.FIRMWARE:
                categories = (*self.firmware_values, "__unknown__", "__missing__")

            else:
                raise ValueError(f"Unsupported probe: {probe}")

            names.extend(f"{prefix}={category}" for category in categories)

        return tuple(names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, observation: Observation) -> list[float]:
        """Encode one observation without inspecting hidden data."""
        self._require_fitted()

        vector = []

        for probe in FEATURE_ORDER:
            value = observation.evidence.get(probe)

            if probe == ProbeType.VENDOR:
                categories = (*VENDOR_VALUES, "__unknown__", "__missing__")
                selected = self._categorical_value(value, VENDOR_VALUES)

            elif probe in BOOLEAN_PROBES:
                categories = ("false", "true", "__missing__")

                if value is None:
                    selected = "__missing__"
                elif type(value) is bool:
                    selected = "true" if value else "false"
                else:
                    raise TypeError(f"{probe.value} expects bool or None")

            elif probe == ProbeType.INTERFACES:
                categories = (*INTERFACE_VALUES, "__unknown__", "__missing__")
                selected = self._categorical_value(value, INTERFACE_VALUES)

            elif probe == ProbeType.FIRMWARE:
                categories = (*self.firmware_values, "__unknown__", "__missing__")
                selected = self._categorical_value(value, self.firmware_values)

            else:
                raise ValueError(f"Unsupported probe: {probe}")

            vector.extend(1.0 if category == selected else 0.0 for category in categories)

        return vector

    def transform_many(self, observations: Iterable[Observation]) -> list[list[float]]:
        return [self.transform(observation) for observation in observations]

    @staticmethod
    def _categorical_value(value, allowed_values: tuple[str, ...]) -> str:
        if value is None:
            return "__missing__"

        if not isinstance(value, str):
            raise TypeError("Categorical feature expects str or None")

        normalized = value.lower()

        if normalized in allowed_values:
            return normalized

        return "__unknown__"

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("FeatureEncoder must be fitted first")
