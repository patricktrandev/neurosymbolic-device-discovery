"""Bayes-optimal reference classifier for the synthetic simulator.

The oracle knows the exact generative process of ``DeviceSimulator``
(feature distributions, missing rate and noise rate). It never sees
labels or hidden probe outcomes, only visible ``Observation.evidence``.

Because every probe is sampled independently given the device class,
and both probe selection and probe failure are class-independent,
the exact posterior factorizes over the visible probes:

    P(class | evidence) ∝ P(class) * Π_p P(observed value_p | class)

An absent probe contributes a class-independent factor and can be
ignored. The oracle is therefore the accuracy ceiling for any model
on data produced by the same scenario. It is an upper-bound reference,
not a competing method.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType, Vendor
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.simulation.profiles import LLDP_ENABLED_PROBABILITY, ROLE_DISTRIBUTIONS, SNMP_AVAILABLE_PROBABILITY, VENDOR_FIRMWARE_DISTRIBUTIONS, VENDOR_HINT_UNKNOWN_PROBABILITY
from device_discovery.simulation.scenarios import get_scenario

ProbeValue = bool | str

# Value domains used by the simulator when it injects noise.
FIRMWARE_DOMAIN: tuple[str, ...] = tuple(sorted({firmware for distribution in VENDOR_FIRMWARE_DISTRIBUTIONS.values() for firmware in distribution}))

VALUE_DOMAINS: dict[ProbeType, tuple[ProbeValue, ...]] = {
    ProbeType.VENDOR: (*(vendor.value for vendor in Vendor), "unknown"),
    ProbeType.ROUTING: (True, False),
    ProbeType.LLDP: (True, False),
    ProbeType.SNMP: (True, False),
    ProbeType.INTERFACES: ("low", "medium", "high"),
    ProbeType.VLAN: (True, False),
    ProbeType.FIRMWARE: FIRMWARE_DOMAIN,
}


def _bernoulli(probability: float) -> dict[ProbeValue, float]:
    return {True: probability, False: 1.0 - probability}


def true_feature_distribution(label: str, probe: ProbeType) -> dict[ProbeValue, float]:
    """Distribution of a device's true feature value, before probing."""
    vendor, role = label.rsplit("_", 1)
    role_distribution = ROLE_DISTRIBUTIONS[role]

    if probe == ProbeType.VENDOR:
        return {vendor: 1.0 - VENDOR_HINT_UNKNOWN_PROBABILITY, "unknown": VENDOR_HINT_UNKNOWN_PROBABILITY}

    if probe == ProbeType.ROUTING:
        return _bernoulli(role_distribution.routing_probability)

    if probe == ProbeType.LLDP:
        return _bernoulli(LLDP_ENABLED_PROBABILITY)

    if probe == ProbeType.SNMP:
        return _bernoulli(SNMP_AVAILABLE_PROBABILITY)

    if probe == ProbeType.INTERFACES:
        return dict(role_distribution.interface_bands)

    if probe == ProbeType.VLAN:
        return _bernoulli(role_distribution.vlan_probability)

    if probe == ProbeType.FIRMWARE:
        return dict(VENDOR_FIRMWARE_DISTRIBUTIONS[Vendor(vendor)])

    raise ValueError(f"Unsupported probe: {probe}")


def observed_value_distribution(label: str, probe: ProbeType, noise_rate: float) -> dict[ProbeValue, float]:
    """P(observed value | class, probe succeeded) under the simulator's noise model.

    With probability ``noise_rate`` the true value is replaced by a
    uniformly chosen different value from the probe's domain.
    """
    domain = VALUE_DOMAINS[probe]
    true_distribution = true_feature_distribution(label, probe)

    observed = {value: 0.0 for value in domain}

    for true_value, probability in true_distribution.items():
        observed[true_value] += probability * (1.0 - noise_rate)

        alternatives = [value for value in domain if value != true_value]

        for alternative in alternatives:
            observed[alternative] += probability * noise_rate / len(alternatives)

    return observed


@dataclass(frozen=True)
class OraclePrediction:
    predicted_label: str
    probabilities: dict[str, float]


class BayesOracleClassifier:
    """Exact posterior under the simulator's generative model."""

    def __init__(self, scenario: str = "moderate", noise_rate: float | None = None) -> None:
        self.scenario = scenario
        self.noise_rate = get_scenario(scenario).noise_rate if noise_rate is None else float(noise_rate)

        if not 0.0 <= self.noise_rate <= 1.0:
            raise ValueError("noise_rate must be between 0 and 1")

        self._likelihoods = {label: {probe: observed_value_distribution(label, probe, self.noise_rate) for probe in ProbeType} for label in DEVICE_CLASSES}

    def log_likelihood(self, label: str, observation: Observation) -> float:
        total = 0.0

        for probe, value in observation.evidence.items():
            if value is None:
                continue

            normalized = value.strip().lower() if isinstance(value, str) else value
            probability = self._likelihoods[label][ProbeType(probe)].get(normalized, 0.0)

            if probability <= 0.0:
                return -math.inf

            total += math.log(probability)

        return total

    def predict(self, observation: Observation) -> OraclePrediction:
        # Uniform class prior: the dataset builder balances classes.
        log_scores = [self.log_likelihood(label, observation) for label in DEVICE_CLASSES]
        maximum = max(log_scores)

        if maximum == -math.inf:
            probabilities = [1.0 / len(DEVICE_CLASSES)] * len(DEVICE_CLASSES)
        else:
            weights = [math.exp(score - maximum) for score in log_scores]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]

        best_index = max(range(len(DEVICE_CLASSES)), key=lambda index: probabilities[index])

        return OraclePrediction(predicted_label=DEVICE_CLASSES[best_index], probabilities=dict(zip(DEVICE_CLASSES, probabilities)))

    def predict_many(self, observations: list[Observation]) -> list[OraclePrediction]:
        return [self.predict(observation) for observation in observations]


def bayes_accuracy_ceiling(visible_probe_count: int, scenario: str = "moderate") -> float:
    """Exact expected accuracy of the Bayes-optimal classifier.

    Assumes the materializer's process: ``visible_probe_count`` probes
    chosen uniformly at random, each failing with the scenario's
    missing rate and corrupted with its noise rate.
    """
    probes = list(ProbeType)

    if not 0 <= visible_probe_count <= len(probes):
        raise ValueError("Invalid visible_probe_count")

    config = get_scenario(scenario)
    oracle = BayesOracleClassifier(scenario=scenario)
    class_count = len(DEVICE_CLASSES)

    subsets = list(itertools.combinations(probes, visible_probe_count))
    total = 0.0

    for subset in subsets:
        # None means the probe failed (missing).
        spaces = [(*VALUE_DOMAINS[probe], None) for probe in subset]

        for outcome in itertools.product(*spaces):
            best_joint = 0.0

            for label in DEVICE_CLASSES:
                joint = 1.0

                for probe, value in zip(subset, outcome):
                    if value is None:
                        joint *= config.missing_rate
                    else:
                        joint *= (1.0 - config.missing_rate) * oracle._likelihoods[label][probe][value]

                best_joint = max(best_joint, joint)

            total += best_joint / class_count

    return total / len(subsets)


def bayes_accuracy_ceiling_range(min_count: int, max_count: int, scenario: str = "moderate") -> float:
    """Ceiling when the visible probe count is uniform on [min_count, max_count]."""
    if min_count > max_count:
        raise ValueError("min_count must not exceed max_count")

    counts = range(min_count, max_count + 1)

    return sum(bayes_accuracy_ceiling(count, scenario) for count in counts) / len(counts)
