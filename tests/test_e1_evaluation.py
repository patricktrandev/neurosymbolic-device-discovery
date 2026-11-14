import pytest

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.evaluation.e1_evaluator import accuracy, evidence_change_rate, flip_rate, macro_f1
from device_discovery.evaluation.perturbations import PROBE_VALUES, PerturbationConfig, PerturbationKind, perturb_observations


@pytest.fixture
def observations():
    return [
        Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco", ProbeType.ROUTING: True, ProbeType.VLAN: False}),
        Observation(device_id="device-2", evidence={ProbeType.VENDOR: "juniper", ProbeType.INTERFACES: "high"}),
    ]


def test_zero_missing_rate_preserves_evidence(observations):
    result = perturb_observations(observations, PerturbationConfig(PerturbationKind.MISSING, rate=0.0))

    assert [item.evidence for item in result] == [item.evidence for item in observations]


def test_full_missing_rate_removes_all_evidence(observations):
    result = perturb_observations(observations, PerturbationConfig(PerturbationKind.MISSING, rate=1.0))

    assert all(not item.evidence for item in result)
    assert evidence_change_rate(observations, result) == 1.0


def test_full_noise_changes_all_observed_values(observations):
    result = perturb_observations(observations, PerturbationConfig(PerturbationKind.NOISE, rate=1.0))

    assert evidence_change_rate(observations, result) == 1.0

    for before, after in zip(observations, result):
        assert set(before.evidence) == set(after.evidence)

        for probe, value in after.evidence.items():
            assert value in PROBE_VALUES[probe]
            assert value != before.evidence[probe]


def test_perturbation_does_not_mutate_original(observations):
    original = [dict(item.evidence) for item in observations]

    perturb_observations(observations, PerturbationConfig(PerturbationKind.NOISE, rate=1.0))

    assert [item.evidence for item in observations] == original


def test_perturbation_is_reproducible(observations):
    config = PerturbationConfig(PerturbationKind.NOISE, rate=0.5, seed=42)

    first = perturb_observations(observations, config)
    second = perturb_observations(observations, config)

    assert [item.evidence for item in first] == [item.evidence for item in second]


def test_invalid_perturbation_rate():
    with pytest.raises(ValueError):
        PerturbationConfig(PerturbationKind.MISSING, rate=1.5)


def test_accuracy():
    assert accuracy(["cisco_router", "hpe_switch"], ["cisco_router", "cisco_router"]) == pytest.approx(0.5)


def test_macro_f1_perfect():
    actual = ["cisco_router", "hpe_switch"]

    # Macro-F1 includes all 8 configured classes, including
    # classes absent from this tiny synthetic example.
    assert macro_f1(actual, actual) == pytest.approx(0.25)


def test_flip_rate():
    assert flip_rate(["cisco_router", "hpe_switch"], ["cisco_router", "cisco_switch"]) == pytest.approx(0.5)


def test_empty_evidence_change_rate():
    before = [Observation(device_id="empty", evidence={})]
    after = [Observation(device_id="empty", evidence={})]

    assert evidence_change_rate(before, after) == 0.0
