
import csv
import json

import pytest

from device_discovery.data.builder import build_device_splits
from device_discovery.data.materializer import materialize_observations
from core.utils.constants.device_constants import ProbeType
from device_discovery.environment import load_environment
from core.domain import Observation


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp(
        "environment_leakage"
    )

    splits = build_device_splits(
        samples_per_class=20,
        seed=42,
    )

    materialize_observations(
        splits=splits,
        output_dir=output_dir,
        seed=42,
        scenario="hard",
        cost_mode="heterogeneous",
        initial_observation_count=2,
    )

    return output_dir


def test_test_csv_does_not_expose_labels(dataset_dir):
    with (dataset_dir / "test.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames is not None
        assert "label" not in reader.fieldnames


def test_observation_does_not_expose_ground_truth(dataset_dir):
    with (dataset_dir / "test.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        device_id = next(reader := csv.DictReader(file))[
            "device_id"
        ]

    env = load_environment(dataset_dir, split="test")
    observation = env.observe(device_id)

    assert isinstance(observation, Observation)
    assert not hasattr(observation, "label")
    assert not hasattr(observation, "vendor")
    assert not hasattr(observation, "role")
    assert not hasattr(observation, "true_features")


def test_unqueried_probe_remains_hidden(dataset_dir):
    with (dataset_dir / "test.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        device_id = next(csv.DictReader(file))["device_id"]

    with (
        dataset_dir / "initial_attempts.json"
    ).open(encoding="utf-8") as file:
        attempts = json.load(file)

    env = load_environment(dataset_dir, split="test")
    observation = env.observe(device_id)

    unqueried = [
        probe for probe in ProbeType
        if probe.value not in attempts[device_id]
    ]

    assert all(
        probe not in observation.evidence
        for probe in unqueried
    )


def test_observation_mutation_cannot_leak_into_environment(
    dataset_dir,
):
    with (dataset_dir / "test.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        device_id = next(csv.DictReader(file))["device_id"]

    env = load_environment(dataset_dir, split="test")

    observation = env.observe(device_id)
    observation.evidence[ProbeType.VENDOR] = "fake_vendor"

    fresh = env.observe(device_id)

    assert fresh.evidence.get(ProbeType.VENDOR) != "fake_vendor"


def test_query_does_not_return_device_label(dataset_dir):
    with (dataset_dir / "test.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        device_id = next(csv.DictReader(file))["device_id"]

    with (
        dataset_dir / "initial_attempts.json"
    ).open(encoding="utf-8") as file:
        attempts = json.load(file)

    env = load_environment(dataset_dir, split="test")

    probe = next(
        p for p in ProbeType
        if p.value not in attempts[device_id]
    )

    result = env.query(device_id, probe)

    assert not hasattr(result, "label")
    assert not hasattr(result, "vendor")
    assert not hasattr(result, "role")
    assert not hasattr(result, "true_features")


def test_failed_query_does_not_expose_value(dataset_dir):
    with (dataset_dir / "test.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        device_ids = [
            row["device_id"]
            for row in csv.DictReader(file)
        ]

    with (
        dataset_dir / "initial_attempts.json"
    ).open(encoding="utf-8") as file:
        attempts = json.load(file)

    with (
        dataset_dir / "probe_outcomes.json"
    ).open(encoding="utf-8") as file:
        outcomes = json.load(file)

    candidate = None

    for device_id in device_ids:
        for probe in ProbeType:
            if probe.value in attempts[device_id]:
                continue

            if not outcomes[device_id][probe.value]["success"]:
                candidate = (device_id, probe)
                break

        if candidate is not None:
            break

    assert candidate is not None, (
        "Expected at least one failed probe"
    )

    device_id, probe = candidate

    env = load_environment(dataset_dir, split="test")
    result = env.query(device_id, probe)

    assert result.success is False
    assert result.value is None
    assert probe not in env.observe(device_id).evidence
