from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable

from device_discovery.evaluation.perturbations import PerturbationConfig, PerturbationKind, perturb_observations
from device_discovery.learning.bayes_oracle import BayesOracleClassifier
from device_discovery.learning.feature_encoder import load_observations
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES, HeuristicClassifier
from device_discovery.learning.neurosymbolic_classifier import load_neurosymbolic_classifier


def read_labels(path: Path) -> list[str]:
    """Read labels from a labeled evaluation CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Evaluation CSV has no label column: {path}")

        labels = [row["label"] for row in reader]

    if not labels:
        raise ValueError("Evaluation dataset is empty")

    unknown = set(labels) - set(DEVICE_CLASSES)

    if unknown:
        raise ValueError(f"Unknown evaluation labels: {sorted(unknown)}")

    return labels


def accuracy(actual: list[str], predicted: list[str]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Invalid accuracy inputs")

    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def macro_f1(actual: list[str], predicted: list[str]) -> float:
    """Compute macro-F1 across all configured device classes."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Invalid macro-F1 inputs")

    scores = []

    for label in DEVICE_CLASSES:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))

        denominator = 2 * tp + fp + fn
        score = 2 * tp / denominator if denominator > 0 else 0.0

        scores.append(score)

    return sum(scores) / len(scores)


def flip_rate(baseline: list[str], perturbed: list[str]) -> float:
    if len(baseline) != len(perturbed) or not baseline:
        raise ValueError("Invalid flip-rate inputs")

    return sum(before != after for before, after in zip(baseline, perturbed)) / len(baseline)


def evidence_change_rate(original: list, perturbed: list) -> float:
    """Fraction of originally observed probe values changed."""
    if len(original) != len(perturbed):
        raise ValueError("Observation count mismatch")

    total = 0
    changed = 0

    for before, after in zip(original, perturbed):
        if before.device_id != after.device_id:
            raise ValueError("Device ordering mismatch")

        for probe, value in before.evidence.items():
            total += 1

            if after.evidence.get(probe) != value:
                changed += 1

    return changed / total if total else 0.0


def make_predictors(checkpoint: Path, device: str, scenario: str = "moderate") -> dict[str, Callable[[list], list[str]]]:
    """Create model predictors with consistent interfaces.

    ``bayes_oracle`` knows the simulator's generative model and is an
    upper-bound reference for the clean condition. Under E1
    perturbations its noise model no longer matches exactly.
    """
    heuristic = HeuristicClassifier()
    oracle = BayesOracleClassifier(scenario=scenario)

    neural = load_neurosymbolic_classifier(checkpoint_path=checkpoint, symbolic_strength=0.0, device=device)

    fusion = load_neurosymbolic_classifier(checkpoint_path=checkpoint, symbolic_strength=0.5, device=device)

    def predict_heuristic(observations: list) -> list[str]:
        return [heuristic.predict(observation).predicted_label for observation in observations]

    def predict_neural(observations: list) -> list[str]:
        return [prediction.predicted_label for prediction in neural.predict_many(observations)]

    def predict_fusion(observations: list) -> list[str]:
        return [prediction.predicted_label for prediction in fusion.predict_many(observations)]

    def predict_oracle(observations: list) -> list[str]:
        return [prediction.predicted_label for prediction in oracle.predict_many(observations)]

    return {"bayes_oracle": predict_oracle, "heuristic": predict_heuristic, "mlp": predict_neural, "mlp_symbolic_0.5": predict_fusion}


def evaluate_experiment(data_path: Path, checkpoint: Path, output_dir: Path, device: str = "cpu", seeds: tuple[int, ...] = (42, 43, 44, 45, 46)) -> tuple[list[dict], list[dict]]:
    """Evaluate clean, missing and noisy evidence conditions."""
    observations = load_observations(data_path)
    labels = read_labels(data_path)

    if len(observations) != len(labels):
        raise ValueError("Observation and label counts differ")

    if not seeds:
        raise ValueError("At least one seed is required")

    manifest_path = data_path.parent / "manifest.json"
    scenario = "moderate"

    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as file:
            scenario = json.load(file).get("scenario", scenario)

    predictors = make_predictors(checkpoint, device, scenario)

    baseline_predictions = {name: predict(observations) for name, predict in predictors.items()}

    rows: list[dict] = []

    # Clean baseline: one row per model, not repeated by seed.
    for name, predictions in baseline_predictions.items():
        rows.append(
            {
                "model": name,
                "condition": "baseline",
                "rate": 0.0,
                "seed": -1,
                "accuracy": accuracy(labels, predictions),
                "macro_f1": macro_f1(labels, predictions),
                "flip_rate": 0.0,
                "evidence_change_rate": 0.0,
            }
        )

    conditions = {PerturbationKind.MISSING: (0.10, 0.25, 0.50, 0.75), PerturbationKind.NOISE: (0.10, 0.25, 0.50)}

    for kind, rates in conditions.items():
        for rate in rates:
            for seed in seeds:
                changed_observations = perturb_observations(observations, PerturbationConfig(kind=kind, rate=rate, seed=seed))

                change_rate = evidence_change_rate(observations, changed_observations)

                for name, predict in predictors.items():
                    predictions = predict(changed_observations)

                    rows.append(
                        {
                            "model": name,
                            "condition": kind.value,
                            "rate": rate,
                            "seed": seed,
                            "accuracy": accuracy(labels, predictions),
                            "macro_f1": macro_f1(labels, predictions),
                            "flip_rate": flip_rate(baseline_predictions[name], predictions),
                            "evidence_change_rate": change_rate,
                        }
                    )

    grouped = defaultdict(list)

    for row in rows:
        key = (row["model"], row["condition"], row["rate"])
        grouped[key].append(row)

    summary = []

    for (model, condition, rate), group in sorted(grouped.items()):
        result = {"model": model, "condition": condition, "rate": rate, "runs": len(group)}

        for metric in ("accuracy", "macro_f1", "flip_rate", "evidence_change_rate"):
            values = [row[metric] for row in group]

            result[f"{metric}_mean"] = statistics.mean(values)

            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

        summary.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "e1_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "e1_summary.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"evaluation_dataset": str(data_path), "checkpoint": str(checkpoint), "device": device, "seeds": list(seeds), "summary": summary}, file, indent=2)

    print()
    print("E1 Robustness Evaluation")
    print("=" * 88)

    print(f"{'Model':<19} {'Condition':<10} " f"{'Rate':>5} {'Accuracy':>10} " f"{'Macro-F1':>10} {'Flip':>9}")

    print("-" * 88)

    for row in summary:
        print(f"{row['model']:<19} " f"{row['condition']:<10} " f"{row['rate']:>5.2f} " f"{row['accuracy_mean']:>10.4f} " f"{row['macro_f1_mean']:>10.4f} " f"{row['flip_rate_mean']:>9.4f}")

    print()
    print(f"Detailed results: {csv_path}")
    print(f"Summary: {json_path}")

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 robustness evaluation")

    parser.add_argument("--data-path", default="data/generated/moderate/val.csv")
    parser.add_argument("--checkpoint", default="artifacts/mlp/model.pt")
    parser.add_argument("--output-dir", default="artifacts/e1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))

    args = parser.parse_args()

    evaluate_experiment(data_path=Path(args.data_path), checkpoint=Path(args.checkpoint), output_dir=Path(args.output_dir), device=args.device)


if __name__ == "__main__":
    main()
