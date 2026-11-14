"""E2 — Low-data learning (hypothesis H2).

Question: does symbolic knowledge help when only a handful of labeled
devices are available, and does the benefit shrink as data grows?

Protocol, per (train size n per class, seed):
    1. Draw a class-stratified subset of n training devices per class.
    2. Draw an equally small validation subset (n per class, at least 1)
       for early stopping, so no model sees more labels than the
       low-data budget allows.
    3. Train an MLP from scratch.
    4. Evaluate on the full held-out test split. Test labels are read
       from ground_truth.json and used only for scoring.

Models compared:
    bayes_oracle      exact posterior of the simulator (upper bound)
    heuristic         hand-written scoring rules, no training
    symbolic_only     soft symbolic rules alone, no training
    mlp               neural network only
    mlp_symbolic_<λ>  MLP logits + λ * symbolic scores

The feature-encoder vocabulary (firmware names) is fitted on all
training observations WITHOUT labels. It only defines one-hot slots,
so it does not leak label information.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from core.domain import Observation
from device_discovery.evaluation.e1_evaluator import accuracy, macro_f1
from device_discovery.learning.bayes_oracle import BayesOracleClassifier, bayes_accuracy_ceiling, bayes_accuracy_ceiling_range
from device_discovery.learning.feature_encoder import FeatureEncoder, load_observations
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES, HeuristicClassifier
from device_discovery.learning.neurosymbolic_classifier import NeurosymbolicClassifier
from device_discovery.learning.symbolic_constraints import SoftSymbolicConstraints
from device_discovery.learning.trainer import LABEL_TO_INDEX, TrainingConfig, fit_mlp, load_labels, set_seed

DEFAULT_TRAIN_SIZES = (2, 4, 8, 16, 32, 64, 140)
DEFAULT_STRENGTHS = (0.5, 1.0, 2.0)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def load_test_split(data_dir: Path) -> tuple[list[Observation], list[str]]:
    """Load test observations and their evaluator-only labels."""
    observations = load_observations(data_dir / "test.csv")

    with (data_dir / "ground_truth.json").open(encoding="utf-8") as file:
        ground_truth = json.load(file)

    labels = []

    for observation in observations:
        record = ground_truth[observation.device_id]

        if record["split"] != "test":
            raise ValueError(f"{observation.device_id} is not a test device")

        labels.append(record["label"])

    return observations, labels


def stratified_subset(observations: list[Observation], labels: list[str], per_class: int, rng: random.Random) -> tuple[list[Observation], list[str]]:
    by_class: dict[str, list[int]] = defaultdict(list)

    for index, label in enumerate(labels):
        by_class[label].append(index)

    chosen: list[int] = []

    for label in DEVICE_CLASSES:
        indices = by_class[label]

        if len(indices) < per_class:
            raise ValueError(f"Only {len(indices)} examples for {label}, need {per_class}")

        chosen.extend(rng.sample(indices, per_class))

    rng.shuffle(chosen)

    return [observations[index] for index in chosen], [labels[index] for index in chosen]


def mean_nll(labels: list[str], probabilities: list[dict[str, float]]) -> float:
    """Mean negative log-likelihood of the true class (lower is better)."""
    return statistics.mean(-math.log(max(row[label], 1e-12)) for label, row in zip(labels, probabilities))


def softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {label: math.exp(value - maximum) for label, value in scores.items()}
    total = sum(weights.values())

    return {label: value / total for label, value in weights.items()}


def score_rows(name: str, labels: list[str], predicted: list[str], probabilities: list[dict[str, float]] | None) -> dict:
    return {"model": name, "accuracy": accuracy(labels, predicted), "macro_f1": macro_f1(labels, predicted), "nll": mean_nll(labels, probabilities) if probabilities is not None else float("nan")}


def evaluate_static_models(test_observations: list[Observation], test_labels: list[str], scenario: str) -> list[dict]:
    """Models that need no training: evaluated once."""
    rows = []

    oracle = BayesOracleClassifier(scenario=scenario)
    oracle_predictions = oracle.predict_many(test_observations)
    rows.append(score_rows("bayes_oracle", test_labels, [p.predicted_label for p in oracle_predictions], [p.probabilities for p in oracle_predictions]))

    heuristic = HeuristicClassifier()
    heuristic_predictions = heuristic.predict_many(test_observations)
    rows.append(score_rows("heuristic", test_labels, [p.predicted_label for p in heuristic_predictions], [softmax_scores(p.scores) for p in heuristic_predictions]))

    constraints = SoftSymbolicConstraints()
    symbolic_probabilities = [softmax_scores(scores) for scores in constraints.score_many(test_observations)]
    symbolic_predicted = [max(DEVICE_CLASSES, key=lambda label: row[label]) for row in symbolic_probabilities]
    rows.append(score_rows("symbolic_only", test_labels, symbolic_predicted, symbolic_probabilities))

    return rows


def run_e2(
    data_dir: Path,
    output_dir: Path,
    train_sizes: tuple[int, ...] = DEFAULT_TRAIN_SIZES,
    strengths: tuple[float, ...] = DEFAULT_STRENGTHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str = "cpu",
    epochs: int = 200,
    patience: int = 20,
) -> tuple[list[dict], list[dict]]:
    with (data_dir / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    scenario = manifest["scenario"]
    min_probes = manifest["initial_observation_count"]
    max_probes = manifest.get("max_initial_observation_count") or min_probes
    ceiling = bayes_accuracy_ceiling(min_probes, scenario) if min_probes == max_probes else bayes_accuracy_ceiling_range(min_probes, max_probes, scenario)

    train_observations = load_observations(data_dir / "train.csv")
    train_labels = load_labels(data_dir / "train.csv")
    val_observations = load_observations(data_dir / "val.csv")
    val_labels = load_labels(data_dir / "val.csv")
    test_observations, test_labels = load_test_split(data_dir)

    encoder = FeatureEncoder().fit(train_observations)
    torch_device = torch.device(device)

    rows: list[dict] = []

    for row in evaluate_static_models(test_observations, test_labels, scenario):
        rows.extend({**row, "train_per_class": n, "seed": -1} for n in train_sizes)

    val_per_class_available = min(val_labels.count(label) for label in DEVICE_CLASSES)

    for n in train_sizes:
        for seed in seeds:
            rng = random.Random(seed * 10_007 + n)

            sub_train_obs, sub_train_labels = stratified_subset(train_observations, train_labels, n, rng)
            sub_val_obs, sub_val_labels = stratified_subset(val_observations, val_labels, max(1, min(n, val_per_class_available)), rng)

            def tensors(observations: list[Observation], labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
                features = torch.tensor(encoder.transform_many(observations), dtype=torch.float32)
                targets = torch.tensor([LABEL_TO_INDEX[label] for label in labels], dtype=torch.long)
                return features, targets

            x_train, y_train = tensors(sub_train_obs, sub_train_labels)
            x_val, y_val = tensors(sub_val_obs, sub_val_labels)

            config = TrainingConfig(seed=seed, epochs=epochs, patience=patience, batch_size=min(32, len(y_train)))
            set_seed(seed)

            model, best_epoch, _, _ = fit_mlp(x_train, y_train, x_val, y_val, input_dim=encoder.n_features, config=config, device=torch_device, verbose=False)

            for strength in (0.0, *strengths):
                classifier = NeurosymbolicClassifier(model=model, encoder=encoder, symbolic_strength=strength, device=device)
                predictions = classifier.predict_many(test_observations)
                name = "mlp" if strength == 0.0 else f"mlp_symbolic_{strength:g}"

                row = score_rows(name, test_labels, [p.predicted_label for p in predictions], [p.probabilities for p in predictions])
                rows.append({**row, "train_per_class": n, "seed": seed, "best_epoch": best_epoch})

            print(f"n={n:>4}/class seed={seed} best_epoch={best_epoch:>3} mlp_acc={rows[-1 - len(strengths)]['accuracy']:.3f}")

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for row in rows:
        grouped[(row["model"], row["train_per_class"])].append(row)

    summary = []

    for (model, n), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        result = {"model": model, "train_per_class": n, "runs": len(group)}

        for metric in ("accuracy", "macro_f1", "nll"):
            values = [row[metric] for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

        summary.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "train_per_class", "seed", "accuracy", "macro_f1", "nll", "best_epoch"]

    with (output_dir / "e2_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "e2_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "data_dir": str(data_dir),
                "scenario": scenario,
                "visible_probes": [min_probes, max_probes],
                "theoretical_accuracy_ceiling": ceiling,
                "test_samples": len(test_labels),
                "seeds": list(seeds),
                "summary": summary,
            },
            file,
            indent=2,
        )

    print()
    print(f"E2 Low-data Evaluation  (test n={len(test_labels)}, theoretical ceiling={ceiling:.4f})")
    print("=" * 78)
    print(f"{'n/class':>7}  {'Model':<18} {'Accuracy':>15} {'Macro-F1':>9} {'NLL':>15}")
    print("-" * 78)

    for row in summary:
        print(
            f"{row['train_per_class']:>7}  {row['model']:<18} "
            f"{row['accuracy_mean']:>7.4f} ±{row['accuracy_std']:.4f} "
            f"{row['macro_f1_mean']:>9.4f} "
            f"{row['nll_mean']:>7.4f} ±{row['nll_std']:.4f}"
        )

    print()
    print(f"Results: {output_dir / 'e2_results.csv'}")
    print(f"Summary: {output_dir / 'e2_summary.json'}")

    return rows, summary


def _parse_tuple(text: str, cast) -> tuple:
    return tuple(cast(part) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 low-data evaluation (H2)")
    parser.add_argument("--data-dir", default="data/generated/moderate")
    parser.add_argument("--output-dir", default="artifacts/e2")
    parser.add_argument("--train-sizes", default=",".join(map(str, DEFAULT_TRAIN_SIZES)), help="Comma-separated training examples per class")
    parser.add_argument("--strengths", default=",".join(map(str, DEFAULT_STRENGTHS)), help="Comma-separated symbolic strengths")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    run_e2(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        train_sizes=_parse_tuple(args.train_sizes, int),
        strengths=_parse_tuple(args.strengths, float),
        seeds=_parse_tuple(args.seeds, int),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
