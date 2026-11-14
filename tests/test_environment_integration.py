import csv
import json

import pytest

from core.utils.constants.device_constants import ProbeType
from device_discovery.data.builder import build_device_splits
from device_discovery.data.materializer import materialize_observations
from device_discovery.environment import load_environment


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("environment_integration")

    splits = build_device_splits(samples_per_class=20, seed=42)

    materialize_observations(splits=splits, output_dir=output_dir, seed=42, scenario="moderate", cost_mode="heterogeneous", initial_observation_count=2)

    return output_dir


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("split,expected_count", [("train", 112), ("val", 24), ("test", 24)])
def test_environment_loads_each_split(dataset_dir, split, expected_count):
    rows = read_csv(dataset_dir / f"{split}.csv")
    env = load_environment(dataset_dir, split=split)

    assert len(rows) == expected_count

    for row in rows:
        observation = env.observe(row["device_id"])
        assert observation.device_id == row["device_id"]


def test_initial_observation_matches_csv(dataset_dir):
    rows = read_csv(dataset_dir / "test.csv")
    env = load_environment(dataset_dir, split="test")

    for row in rows:
        observation = env.observe(row["device_id"])

        for probe in ProbeType:
            raw_value = row[probe.value]

            if raw_value == "":
                assert probe not in observation.evidence
            else:
                assert probe in observation.evidence

                expected = raw_value
                if raw_value == "True":
                    expected = True
                elif raw_value == "False":
                    expected = False

                assert observation.evidence[probe] == expected


def test_query_matches_hidden_outcome(dataset_dir):
    rows = read_csv(dataset_dir / "test.csv")
    device_id = rows[0]["device_id"]

    attempts = read_json(dataset_dir / "initial_attempts.json")
    outcomes = read_json(dataset_dir / "probe_outcomes.json")

    env = load_environment(dataset_dir, split="test")

    probe = next(p for p in ProbeType if p.value not in attempts[device_id])

    expected = outcomes[device_id][probe.value]

    result = env.query(device_id, probe)

    assert result.success == expected["success"]
    assert result.value == expected["value"]
    assert result.cost == expected["cost"]


def test_query_updates_observation(dataset_dir):
    rows = read_csv(dataset_dir / "test.csv")
    device_id = rows[0]["device_id"]

    attempts = read_json(dataset_dir / "initial_attempts.json")

    env = load_environment(dataset_dir, split="test")

    before = env.observe(device_id)

    probe = next(p for p in ProbeType if p.value not in attempts[device_id])

    result = env.query(device_id, probe)
    after = env.observe(device_id)

    if result.success:
        assert probe not in before.evidence
        assert after.evidence[probe] == result.value
    else:
        assert probe not in after.evidence


def test_initial_probe_cannot_be_repeated(dataset_dir):
    rows = read_csv(dataset_dir / "test.csv")
    device_id = rows[0]["device_id"]

    attempts = read_json(dataset_dir / "initial_attempts.json")

    env = load_environment(dataset_dir, split="test")

    initial_probe = ProbeType(attempts[device_id][0])

    with pytest.raises(ValueError):
        env.query(device_id, initial_probe)


def test_query_cost_accumulates(dataset_dir):
    rows = read_csv(dataset_dir / "test.csv")
    device_id = rows[0]["device_id"]

    attempts = read_json(dataset_dir / "initial_attempts.json")

    env = load_environment(dataset_dir, split="test")

    available = [p for p in ProbeType if p.value not in attempts[device_id]]

    first = env.query(device_id, available[0])
    second = env.query(device_id, available[1])

    assert env.get_total_cost(device_id) == pytest.approx(first.cost + second.cost)


def test_split_isolation(dataset_dir):
    train_rows = read_csv(dataset_dir / "train.csv")
    test_rows = read_csv(dataset_dir / "test.csv")

    train_device = train_rows[0]["device_id"]
    test_device = test_rows[0]["device_id"]

    test_env = load_environment(dataset_dir, split="test")

    assert test_env.observe(test_device)

    with pytest.raises(KeyError):
        test_env.observe(train_device)


def test_invalid_split_rejected(dataset_dir):
    with pytest.raises(ValueError):
        load_environment(dataset_dir, split="unknown")
