from __future__ import annotations

import torch
from torch import nn

from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES


class MLPClassifier(nn.Module):
    """Small neural baseline for device identification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.num_classes = len(DEVICE_CLASSES)

        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, self.num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw logits, not probabilities."""
        return self.network(features)

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Return uncalibrated class probabilities."""
        self.eval()
        logits = self(features)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> list[str]:
        self.eval()
        logits = self(features)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        indices = logits.argmax(dim=-1).tolist()

        return [DEVICE_CLASSES[index] for index in indices]
