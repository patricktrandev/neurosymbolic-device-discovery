import json
import random

import pytest

from core.domain import Observation
from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor
from device_discovery.data.builder import build_device_splits
from device_discovery.data.materializer import materialize_observations
from device_discovery.learning.bayes_oracle import VALUE_DOMAINS, BayesOracleClassifier, bayes_accuracy_ceiling, observed_value_distribution
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.simulation.simulator import DeviceSimulator


@pytest.mark.parametrize("label", DEVICE_CLASSES)
@pytest.mark.parametrize("probe", list(ProbeType))
def test_observed_distributions_sum_to_one(label, probe):
    distribution = observed_value_distribution(label, probe, noise_rate=0.1)

    assert set(distribution) == set(VALUE_DOMAINS[probe])
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_empty_observation_is_uniform():
    prediction = BayesOracleClassifier().predict(Observation(device_id="d", evidence={}))

    for probability in prediction.probabilities.values():
        assert probability == pytest.approx(1 / len(DEVICE_CLASSES))


def test_strong_evidence_identifies_class():
    observation = Observation(device_id="d", evidence={ProbeType.VENDOR: "juniper", ProbeType.FIRMWARE: "network_os_b", ProbeType.ROUTING: True, ProbeType.INTERFACES: "low"})

    prediction = BayesOracleClassifier().predict(observation)

    assert prediction.predicted_label == "juniper_router"
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_ceiling_increases_with_visible_probes():
    ceilings = [bayes_accuracy_ceiling(k) for k in range(0, 4)]

    assert ceilings[0] == pytest.approx(1 / len(DEVICE_CLASSES))
    assert all(a < b for a, b in zip(ceilings, ceilings[1:]))


def test_ceiling_matches_monte_carlo_simulation():
    """Oracle accuracy on simulated devices matches the exact ceiling."""
    simulator = DeviceSimulator(seed=7, scenario="moderate")
    oracle = BayesOracleClassifier(scenario="moderate")
    rng = random.Random(7)

    correct = 0
    total = 0

    for vendor in Vendor:
        for role in DeviceRole:
            for index in range(1500):
                device = simulator.create_device(f"{vendor.value}-{role.value}-{index}", vendor, role)
                evidence = {}

                for probe in rng.sample(list(ProbeType), k=2):
                    result = simulator.probe(device, probe)

                    if result.success:
                        evidence[probe] = result.value

                prediction = oracle.predict(Observation(device_id=device.device_id, evidence=evidence))
                correct += prediction.predicted_label == device.label
                total += 1

    # 12,000 samples: standard error is about 0.004.
    assert correct / total == pytest.approx(bayes_accuracy_ceiling(2, "moderate"), abs=0.02)


def test_materializer_random_probe_count(tmp_path):
    splits = build_device_splits(samples_per_class=20, seed=1)

    materialize_observations(splits, output_dir=tmp_path, seed=1, initial_observation_count=1, max_initial_observation_count=7)

    with (tmp_path / "initial_attempts.json").open(encoding="utf-8") as file:
        attempts = json.load(file)

    with (tmp_path / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    counts = {len(probes) for probes in attempts.values()}

    assert min(counts) >= 1 and max(counts) <= 7
    assert len(counts) > 3
    assert manifest["max_initial_observation_count"] == 7


def test_materializer_rejects_invalid_range(tmp_path):
    splits = build_device_splits(samples_per_class=20, seed=1)

    with pytest.raises(ValueError):
        materialize_observations(splits, output_dir=tmp_path, initial_observation_count=4, max_initial_observation_count=3)
