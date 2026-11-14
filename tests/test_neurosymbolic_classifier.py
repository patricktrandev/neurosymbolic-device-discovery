import math

import pytest
import torch

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.feature_encoder import FeatureEncoder
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.mlp_classifier import MLPClassifier
from device_discovery.learning.neurosymbolic_classifier import NeurosymbolicClassifier, load_neurosymbolic_classifier
from device_discovery.learning.symbolic_constraints import SoftSymbolicConstraints, SymbolicConfig


@pytest.fixture
def observation():
    return Observation(device_id="device-1", evidence={ProbeType.VENDOR: "cisco", ProbeType.ROUTING: True})


@pytest.fixture
def encoder():
    return FeatureEncoder().fit([Observation(device_id="train-1", evidence={ProbeType.FIRMWARE: "network_os_a"})])


@pytest.fixture
def model(encoder):
    torch.manual_seed(42)
    return MLPClassifier(input_dim=encoder.n_features, dropout=0.0)


def test_symbolic_scores_cover_all_classes(observation):
    constraints = SoftSymbolicConstraints()

    scores = constraints.score(observation)

    assert set(scores) == set(DEVICE_CLASSES)


def test_symbolic_rules_favor_matching_class(observation):
    scores = SoftSymbolicConstraints().score(observation)

    assert scores["cisco_router"] > scores["cisco_switch"]
    assert scores["cisco_router"] > scores["juniper_router"]


def test_missing_evidence_is_neutral():
    observation = Observation(device_id="empty", evidence={})

    scores = SoftSymbolicConstraints().score(observation)

    assert all(value == 0.0 for value in scores.values())


def test_conflicting_evidence_does_not_remove_classes():
    observation = Observation(device_id="conflict", evidence={ProbeType.VENDOR: "cisco", ProbeType.ROUTING: False, ProbeType.VLAN: False})

    scores = SoftSymbolicConstraints().score(observation)

    assert len(scores) == 8
    assert all(math.isfinite(v) for v in scores.values())


def test_invalid_symbolic_weight():
    with pytest.raises(ValueError):
        SoftSymbolicConstraints(SymbolicConfig(vendor_weight=-1))


def test_fusion_returns_valid_probabilities(model, encoder, observation):
    classifier = NeurosymbolicClassifier(model=model, encoder=encoder)

    prediction = classifier.predict(observation)

    assert prediction.predicted_label in DEVICE_CLASSES
    assert set(prediction.probabilities) == set(DEVICE_CLASSES)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_zero_strength_matches_neural_model(model, encoder, observation):
    classifier = NeurosymbolicClassifier(model=model, encoder=encoder, symbolic_strength=0.0)

    prediction = classifier.predict(observation)

    for label in DEVICE_CLASSES:
        assert prediction.probabilities[label] == pytest.approx(prediction.neural_probabilities[label], abs=1e-6)


def test_stronger_symbolic_weight_changes_distribution(model, encoder, observation):
    baseline = NeurosymbolicClassifier(model=model, encoder=encoder, symbolic_strength=0.0)

    fused = NeurosymbolicClassifier(model=model, encoder=encoder, symbolic_strength=3.0)

    p0 = baseline.predict(observation)
    p1 = fused.predict(observation)

    assert p1.probabilities["cisco_router"] > (p0.probabilities["cisco_router"])


def test_predict_many(model, encoder, observation):
    classifier = NeurosymbolicClassifier(model=model, encoder=encoder)

    predictions = classifier.predict_many([observation, Observation(device_id="device-2", evidence={})])

    assert len(predictions) == 2


def test_empty_batch(model, encoder):
    classifier = NeurosymbolicClassifier(model=model, encoder=encoder)

    assert classifier.predict_many([]) == []


def test_invalid_strength(model, encoder):
    with pytest.raises(ValueError):
        NeurosymbolicClassifier(model=model, encoder=encoder, symbolic_strength=-1)


def test_checkpoint_roundtrip(model, encoder, observation, tmp_path):
    checkpoint_path = tmp_path / "model.pt"

    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": encoder.n_features, "hidden_dim": 64, "dropout": 0.0, "classes": list(DEVICE_CLASSES), "firmware_values": list(encoder.firmware_values)},
        checkpoint_path,
    )

    classifier = load_neurosymbolic_classifier(checkpoint_path)

    prediction = classifier.predict(observation)

    assert prediction.predicted_label in DEVICE_CLASSES
