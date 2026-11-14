from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES


@dataclass(frozen=True)
class SymbolicConfig:
    """Non-negative weights for soft symbolic rules.

    Default values preserve the Phase 3.4 configuration.
    A zero weight disables the corresponding rule group.
    """

    vendor_weight: float = 1.0
    routing_weight: float = 1.0
    vlan_weight: float = 0.25

    def __post_init__(self) -> None:
        for name in ("vendor_weight", "routing_weight", "vlan_weight"):
            value = getattr(self, name)

            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


class SoftSymbolicConstraints:
    """Compute additive symbolic scores for device classes.

    Rules are soft preferences, not hard exclusions.
    Missing or unrecognized evidence contributes zero.
    """

    def __init__(self, config: SymbolicConfig | None = None) -> None:
        self.config = config if config is not None else SymbolicConfig()

        if not isinstance(self.config, SymbolicConfig):
            raise TypeError("config must be a SymbolicConfig instance")

    def score(self, observation: Observation) -> dict[str, float]:
        evidence = observation.evidence

        vendor_hint = evidence.get(ProbeType.VENDOR)
        routing = evidence.get(ProbeType.ROUTING)
        vlan = evidence.get(ProbeType.VLAN)

        if isinstance(vendor_hint, str):
            vendor_hint = vendor_hint.strip().lower()
        else:
            vendor_hint = None

        scores: dict[str, float] = {}

        for label in DEVICE_CLASSES:
            vendor, role = label.rsplit("_", 1)
            value = 0.0

            # Rule 1: observed vendor matches class vendor.
            if vendor_hint == vendor:
                value += self.config.vendor_weight

            # Rule 2: routing evidence supports a role.
            if routing is True and role == "router":
                value += self.config.routing_weight

            elif routing is False and role == "switch":
                value += self.config.routing_weight

            # Rule 3: VLAN evidence provides a weak role hint.
            if vlan is True and role == "switch":
                value += self.config.vlan_weight

            elif vlan is False and role == "router":
                value += self.config.vlan_weight

            scores[label] = value

        return scores

    def score_many(self, observations: Iterable[Observation]) -> list[dict[str, float]]:
        return [self.score(observation) for observation in observations]
