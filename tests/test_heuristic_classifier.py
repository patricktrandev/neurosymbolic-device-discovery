import pytest

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES, HeuristicClassifier


@pytest.fixture
def classifier():
    return HeuristicClassifier()


def test_has_eight_classes():
    assert len(DEVICE_CLASSES) == 8
    assert len(set(DEVICE_CLASSES)) == 8


def test_vendor_hint_favors_correct_vendor(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco"})

    scores = classifier.score(observation)

    assert scores["cisco_router"] > scores["juniper_router"]
    assert scores["cisco_switch"] > scores["arista_switch"]


def test_routing_true_favors_router(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.ROUTING: True})

    scores = classifier.score(observation)

    assert scores["cisco_router"] > scores["cisco_switch"]


def test_routing_false_favors_switch(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.ROUTING: False})

    scores = classifier.score(observation)

    assert scores["cisco_switch"] > scores["cisco_router"]


def test_missing_routing_does_not_imply_false(classifier):
    observation = Observation(device_id="device-1", evidence={})

    scores = classifier.score(observation)

    assert scores["cisco_router"] == scores["cisco_switch"]


def test_vlan_true_favors_switch(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VLAN: True})

    scores = classifier.score(observation)

    assert scores["hpe_switch"] > scores["hpe_router"]


def test_firmware_favors_associated_vendor(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.FIRMWARE: "network_os_a"})

    scores = classifier.score(observation)

    assert scores["cisco_router"] > scores["juniper_router"]


def test_prediction_has_all_scores(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "arista"})

    prediction = classifier.predict(observation)

    assert prediction.predicted_label in DEVICE_CLASSES
    assert set(prediction.scores) == set(DEVICE_CLASSES)
    assert 0.0 <= prediction.confidence <= 1.0


def test_empty_observation_is_ambiguous(classifier):
    observation = Observation(device_id="device-1", evidence={})

    prediction = classifier.predict(observation)

    assert prediction.is_ambiguous is True
    assert prediction.predicted_label == DEVICE_CLASSES[0]
    assert prediction.confidence == pytest.approx(1 / 8)


def test_clear_prediction_not_ambiguous(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco", ProbeType.ROUTING: True})

    prediction = classifier.predict(observation)

    assert prediction.predicted_label == "cisco_router"
    assert prediction.is_ambiguous is False


def test_prediction_is_deterministic(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "juniper", ProbeType.ROUTING: False})

    first = classifier.predict(observation)
    second = classifier.predict(observation)

    assert first == second


def test_prediction_does_not_modify_observation(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco"})

    before = observation.evidence.copy()

    classifier.predict(observation)

    assert observation.evidence == before


def test_predict_many(classifier):
    observations = [Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco"}), Observation(device_id="device-2", evidence={ProbeType.VENDOR: "hpe"})]

    predictions = classifier.predict_many(observations)

    assert len(predictions) == 2
    assert all(prediction.predicted_label in DEVICE_CLASSES for prediction in predictions)


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        HeuristicClassifier(vendor_weight=-1)


def test_nonfinite_weight_rejected():
    with pytest.raises(ValueError):
        HeuristicClassifier(vendor_weight=float("nan"))


def test_conflicting_vendor_and_firmware(classifier):
    observation = Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco", ProbeType.FIRMWARE: "network_os_b"})

    scores = classifier.score(observation)

    assert scores["cisco_router"] > scores["juniper_router"]
    assert scores["juniper_router"] > scores["arista_router"]
