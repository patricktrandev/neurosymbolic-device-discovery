import csv

import pytest

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.feature_encoder import FeatureEncoder, load_observations, observation_from_row


@pytest.fixture
def encoder():
    train = [
        Observation(device_id="train-1", evidence={ProbeType.VENDOR: "cisco", ProbeType.FIRMWARE: "ios"}),
        Observation(device_id="train-2", evidence={ProbeType.VENDOR: "juniper", ProbeType.FIRMWARE: "junos"}),
    ]

    return FeatureEncoder().fit(train)


def as_features(encoder, observation):
    return dict(zip(encoder.feature_names, encoder.transform(observation)))


def test_encoder_requires_fit():
    encoder = FeatureEncoder()

    with pytest.raises(RuntimeError):
        encoder.transform(Observation(device_id="device-1"))


def test_encoder_returns_fixed_length(encoder):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco"})

    vector = encoder.transform(observation)

    assert len(vector) == encoder.n_features
    assert all(value in (0.0, 1.0) for value in vector)


def test_boolean_false_is_not_missing(encoder):
    observation = Observation(device_id="device-1", evidence={ProbeType.ROUTING: False})

    features = as_features(encoder, observation)

    assert features["routing_enabled=false"] == 1.0
    assert features["routing_enabled=true"] == 0.0
    assert features["routing_enabled=__missing__"] == 0.0


def test_missing_boolean_has_own_category(encoder):
    observation = Observation(device_id="device-1", evidence={})

    features = as_features(encoder, observation)

    assert features["routing_enabled=__missing__"] == 1.0
    assert features["routing_enabled=false"] == 0.0


def test_unknown_vendor_is_not_missing(encoder):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "unrecognized_vendor"})

    features = as_features(encoder, observation)

    assert features["vendor_hint=__unknown__"] == 1.0
    assert features["vendor_hint=__missing__"] == 0.0


def test_firmware_vocabulary_comes_from_train(encoder):
    assert encoder.firmware_values == ("ios", "junos")

    observation = Observation(device_id="test-1", evidence={ProbeType.FIRMWARE: "new_firmware"})

    features = as_features(encoder, observation)

    assert features["firmware_family=__unknown__"] == 1.0
    assert "new_firmware" not in encoder.firmware_values


def test_transform_does_not_modify_observation(encoder):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco"})

    before = observation.evidence.copy()

    encoder.transform(observation)

    assert observation.evidence == before


def test_transform_many(encoder):
    observations = [Observation(device_id="device-1"), Observation(device_id="device-2")]

    vectors = encoder.transform_many(observations)

    assert len(vectors) == 2
    assert all(len(vector) == encoder.n_features for vector in vectors)


def test_observation_from_row_ignores_label():
    row = {"device_id": "device-1", "label": "cisco_router", **{probe.value: "" for probe in ProbeType}}

    row["vendor_hint"] = "cisco"
    row["routing_enabled"] = "False"

    observation = observation_from_row(row)

    assert observation.device_id == "device-1"
    assert observation.evidence[ProbeType.VENDOR] == "cisco"
    assert observation.evidence[ProbeType.ROUTING] is False
    assert not hasattr(observation, "label")


def test_load_observations_from_csv(tmp_path):
    path = tmp_path / "train.csv"

    fieldnames = ["device_id", "label", *[probe.value for probe in ProbeType]]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerow({**{name: "" for name in fieldnames}, "device_id": "device-1", "label": "cisco_router", "vendor_hint": "cisco"})

    observations = load_observations(path)

    assert len(observations) == 1
    assert observations[0].device_id == "device-1"
    assert observations[0].evidence == {ProbeType.VENDOR: "cisco"}


def test_invalid_boolean_value_rejected(encoder):
    observation = Observation(device_id="device-1", evidence={ProbeType.ROUTING: "maybe"})

    with pytest.raises(TypeError):
        encoder.transform(observation)
