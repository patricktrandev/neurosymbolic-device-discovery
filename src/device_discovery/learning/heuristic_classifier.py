from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Mapping

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType

VENDORS = ("cisco", "juniper", "arista", "hpe")
ROLES = ("router", "switch")

DEVICE_CLASSES = tuple(f"{vendor}_{role}" for vendor in VENDORS for role in ROLES)

# Hand-designed weights, not learned from test labels.
VENDOR_WEIGHT = 3.0
FIRMWARE_WEIGHT = 1.5
ROUTING_WEIGHT = 2.0
VLAN_WEIGHT = 1.0
INTERFACE_WEIGHT = 0.5

# Example firmware associations.
# These are illustrative assumptions for the baseline.
FIRMWARE_VENDOR = {"network_os_a": "cisco", "network_os_b": "juniper", "network_os_c": "arista", "network_os_d": "hpe"}

INTERFACE_ROLE = {"low": "router", "high": "switch"}


@dataclass(frozen=True)
class HeuristicPrediction:
    predicted_label: str
    scores: dict[str, float]
    confidence: float
    is_ambiguous: bool


class HeuristicClassifier:
    """Deterministic, hand-designed scoring baseline.

    Uses only visible Observation.evidence.

    Confidence is a normalized score, NOT a calibrated
    probability or a statistical coverage guarantee.
    """

    def __init__(
        self,
        vendor_weight: float = VENDOR_WEIGHT,
        firmware_weight: float = FIRMWARE_WEIGHT,
        routing_weight: float = ROUTING_WEIGHT,
        vlan_weight: float = VLAN_WEIGHT,
        interface_weight: float = INTERFACE_WEIGHT,
    ) -> None:
        weights = (vendor_weight, firmware_weight, routing_weight, vlan_weight, interface_weight)

        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Heuristic weights must be finite and nonnegative")

        self.vendor_weight = vendor_weight
        self.firmware_weight = firmware_weight
        self.routing_weight = routing_weight
        self.vlan_weight = vlan_weight
        self.interface_weight = interface_weight

    def score(self, observation: Observation) -> dict[str, float]:
        """Score all eight classes using visible evidence."""
        scores = {label: 0.0 for label in DEVICE_CLASSES}

        evidence = observation.evidence

        vendor_hint = self._normalize(evidence.get(ProbeType.VENDOR))
        firmware = self._normalize(evidence.get(ProbeType.FIRMWARE))
        routing = evidence.get(ProbeType.ROUTING)
        vlan = evidence.get(ProbeType.VLAN)
        interface = self._normalize(evidence.get(ProbeType.INTERFACES))

        for label in DEVICE_CLASSES:
            vendor, role = label.rsplit("_", 1)

            if vendor_hint == vendor:
                scores[label] += self.vendor_weight

            firmware_vendor = FIRMWARE_VENDOR.get(firmware) if firmware is not None else None

            if firmware_vendor == vendor:
                scores[label] += self.firmware_weight

            if routing is True and role == "router":
                scores[label] += self.routing_weight

            elif routing is False and role == "switch":
                scores[label] += self.routing_weight

            if vlan is True and role == "switch":
                scores[label] += self.vlan_weight

            elif vlan is False and role == "router":
                scores[label] += self.vlan_weight

            if interface is not None:
                expected_role = INTERFACE_ROLE.get(interface)

                if expected_role == role:
                    scores[label] += self.interface_weight

        return scores

    def predict(self, observation: Observation) -> HeuristicPrediction:
        scores = self.score(observation)

        # DEVICE_CLASSES defines deterministic tie-breaking.
        best_label = max(DEVICE_CLASSES, key=lambda label: scores[label])

        best_score = scores[best_label]

        tied_labels = [label for label, score in scores.items() if abs(score - best_score) < 1e-12]

        probabilities = self._softmax(scores)

        return HeuristicPrediction(predicted_label=best_label, scores=scores, confidence=probabilities[best_label], is_ambiguous=len(tied_labels) > 1)

    def predict_many(self, observations: list[Observation]) -> list[HeuristicPrediction]:
        return [self.predict(observation) for observation in observations]

    @staticmethod
    def _normalize(value: object) -> str | None:
        if isinstance(value, str):
            return value.strip().lower()
        return None

    @staticmethod
    def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
        maximum = max(scores.values())

        exponentials = {label: exp(score - maximum) for label, score in scores.items()}

        total = sum(exponentials.values())

        return {label: value / total for label, value in exponentials.items()}
