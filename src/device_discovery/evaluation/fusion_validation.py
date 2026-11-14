from __future__ import annotations

import argparse
import csv
from pathlib import Path

from device_discovery.learning.feature_encoder import load_observations
from device_discovery.learning.neurosymbolic_classifier import load_neurosymbolic_classifier


def load_labels(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row["label"] for row in reader]


def accuracy(predictions: list[str], labels: list[str]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("Prediction count mismatch")

    if not labels:
        raise ValueError("Empty validation dataset")

    correct = sum(predicted == actual for predicted, actual in zip(predictions, labels))

    return correct / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/generated/moderate")
    parser.add_argument("--checkpoint", default="artifacts/mlp/model.pt")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    val_path = data_dir / "val.csv"

    observations = load_observations(val_path)
    labels = load_labels(val_path)

    strengths = [0.0, 0.25, 0.5, 1.0, 2.0]

    results = []

    for strength in strengths:
        classifier = load_neurosymbolic_classifier(checkpoint_path=args.checkpoint, symbolic_strength=strength, device=args.device)

        predictions = classifier.predict_many(observations)

        predicted_labels = [prediction.predicted_label for prediction in predictions]

        score = accuracy(predicted_labels, labels)
        results.append((strength, score))

        print(f"lambda={strength:.2f} | " f"val_accuracy={score:.4f}")

    # Prefer smaller lambda when accuracies are tied.
    best_strength, best_accuracy = max(results, key=lambda result: (result[1], -result[0]))

    print()
    print(f"Best lambda: {best_strength}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
