import pytest
import torch

from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.mlp_classifier import MLPClassifier


def test_output_shape():
    model = MLPClassifier(input_dim=30)
    features = torch.zeros(4, 30)

    logits = model(features)

    assert logits.shape == (4, 8)


def test_single_observation():
    model = MLPClassifier(input_dim=30)
    features = torch.zeros(30)

    prediction = model.predict(features)

    assert len(prediction) == 1
    assert prediction[0] in DEVICE_CLASSES


def test_probabilities_sum_to_one():
    model = MLPClassifier(input_dim=30)
    features = torch.randn(5, 30)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (5, 8)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)


def test_predict_returns_valid_labels():
    model = MLPClassifier(input_dim=30)
    features = torch.randn(3, 30)

    predictions = model.predict(features)

    assert len(predictions) == 3
    assert all(label in DEVICE_CLASSES for label in predictions)


def test_invalid_input_dimension():
    with pytest.raises(ValueError):
        MLPClassifier(input_dim=0)


def test_invalid_dropout():
    with pytest.raises(ValueError):
        MLPClassifier(input_dim=30, dropout=1.0)


def test_backward_pass():
    model = MLPClassifier(input_dim=30)

    features = torch.randn(4, 30)
    targets = torch.tensor([0, 1, 2, 3])

    logits = model(features)

    loss = torch.nn.functional.cross_entropy(logits, targets)

    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
