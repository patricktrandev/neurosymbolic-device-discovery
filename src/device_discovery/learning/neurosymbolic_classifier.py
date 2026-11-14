from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from core.domain import Observation
from device_discovery.learning.feature_encoder import FeatureEncoder
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.mlp_classifier import MLPClassifier
from device_discovery.learning.symbolic_constraints import SoftSymbolicConstraints, SymbolicConfig


@dataclass(frozen=True)
class NeurosymbolicPrediction:
    predicted_label: str
    probabilities: dict[str, float]
    neural_probabilities: dict[str, float]
    symbolic_scores: dict[str, float]


def resolve_constraints(constraints: SoftSymbolicConstraints | SymbolicConfig | None) -> SoftSymbolicConstraints:
    """Normalize a configuration or constraint engine."""
    if constraints is None:
        return SoftSymbolicConstraints()

    if isinstance(constraints, SoftSymbolicConstraints):
        return constraints

    if isinstance(constraints, SymbolicConfig):
        return SoftSymbolicConstraints(constraints)

    raise TypeError("constraints must be SoftSymbolicConstraints, " "SymbolicConfig, or None")


class NeurosymbolicClassifier:
    """MLP logits combined with additive symbolic scores."""

    def __init__(self, model: MLPClassifier, encoder: FeatureEncoder, constraints: SoftSymbolicConstraints | SymbolicConfig | None = None, symbolic_strength: float = 1.0, device: str = "cpu") -> None:
        if not isfinite(symbolic_strength):
            raise ValueError("symbolic_strength must be finite")

        if symbolic_strength < 0.0:
            raise ValueError("symbolic_strength must be non-negative")

        if not encoder.fitted:
            raise ValueError("FeatureEncoder must be fitted")

        if model.input_dim != encoder.n_features:
            raise ValueError("Model input dimension does not match encoder")

        if model.num_classes != len(DEVICE_CLASSES):
            raise ValueError("Model output dimension does not match classes")

        self.model = model
        self.encoder = encoder
        self.constraints = resolve_constraints(constraints)
        self.symbolic_strength = float(symbolic_strength)
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_many(self, observations: list[Observation]) -> list[NeurosymbolicPrediction]:
        if not observations:
            return []

        self.model.eval()

        encoded = self.encoder.transform_many(observations)

        features = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)

        neural_logits = self.model(features)

        if neural_logits.ndim == 1:
            neural_logits = neural_logits.unsqueeze(0)

        expected_shape = (len(observations), len(DEVICE_CLASSES))

        if tuple(neural_logits.shape) != expected_shape:
            raise ValueError("Unexpected neural logits shape: " f"{tuple(neural_logits.shape)}")

        symbolic_scores = self.constraints.score_many(observations)

        symbolic_values = [[scores[label] for label in DEVICE_CLASSES] for scores in symbolic_scores]

        symbolic_tensor = torch.tensor(symbolic_values, dtype=neural_logits.dtype, device=self.device)

        fused_logits = neural_logits + self.symbolic_strength * symbolic_tensor

        neural_probabilities = torch.softmax(neural_logits, dim=-1)

        fused_probabilities = torch.softmax(fused_logits, dim=-1)

        predicted_indices = fused_probabilities.argmax(dim=-1)

        neural_rows = neural_probabilities.detach().cpu().tolist()

        fused_rows = fused_probabilities.detach().cpu().tolist()

        indices = predicted_indices.detach().cpu().tolist()

        results: list[NeurosymbolicPrediction] = []

        for index, label_index in enumerate(indices):
            results.append(
                NeurosymbolicPrediction(
                    predicted_label=DEVICE_CLASSES[label_index],
                    probabilities={label: float(value) for label, value in zip(DEVICE_CLASSES, fused_rows[index])},
                    neural_probabilities={label: float(value) for label, value in zip(DEVICE_CLASSES, neural_rows[index])},
                    symbolic_scores=dict(symbolic_scores[index]),
                )
            )

        return results

    def predict(self, observation: Observation) -> NeurosymbolicPrediction:
        return self.predict_many([observation])[0]


def _load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")

    return checkpoint


def load_neurosymbolic_classifier(
    checkpoint_path: str | Path, symbolic_strength: float = 1.0, device: str = "cpu", constraints: SoftSymbolicConstraints | SymbolicConfig | None = None
) -> NeurosymbolicClassifier:
    """Load an MLP and optionally inject symbolic constraints.

    Backward compatible:
        load_neurosymbolic_classifier(path)
        load_neurosymbolic_classifier(path, 0.5, "cuda")

    New:
        load_neurosymbolic_classifier(
            path,
            constraints=SymbolicConfig(...),
        )
    """
    checkpoint = _load_checkpoint(checkpoint_path)

    checkpoint_classes = tuple(checkpoint["classes"])

    if checkpoint_classes != tuple(DEVICE_CLASSES):
        raise ValueError("Checkpoint device class order does not match " "the current DEVICE_CLASSES")

    firmware_values = tuple(checkpoint["firmware_values"])

    encoder = FeatureEncoder(firmware_values=firmware_values, fitted=True)

    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    dropout = float(checkpoint["dropout"])

    model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return NeurosymbolicClassifier(model=model, encoder=encoder, constraints=constraints, symbolic_strength=symbolic_strength, device=device)
