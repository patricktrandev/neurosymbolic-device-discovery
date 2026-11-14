
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from device_discovery.evaluation.h3_incorrect_rules import (
    CorruptedConstraints,
)
from device_discovery.learning.feature_encoder import (
    observation_from_row,
)
from device_discovery.learning.heuristic_classifier import (
    DEVICE_CLASSES,
)
from device_discovery.learning.neurosymbolic_classifier import (
    load_neurosymbolic_classifier,
)
from device_discovery.learning.symbolic_constraints import (
    SoftSymbolicConstraints,
)


FRACTIONS = (5, 10, 25, 50, 100)
SEEDS = tuple(range(42, 52))


def load_test_data(test_path, truth_path):
    with Path(test_path).open(
        newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    with Path(truth_path).open(
        encoding="utf-8"
    ) as file:
        truth = json.load(file)

    if not isinstance(truth, dict):
        raise ValueError("Ground truth must be a dictionary")

    if not rows:
        raise ValueError("Test CSV is empty")

    if len(rows) != 240:
        raise ValueError(
            f"Expected 240 test rows, got {len(rows)}"
        )

    ids = [row["device_id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate device_id in test CSV")

    labels = []

    for device_id in ids:
        if device_id not in truth:
            raise ValueError(
                f"Missing ground truth: {device_id}"
            )

        entry = truth[device_id]

        if entry.get("split") != "test":
            raise ValueError(
                f"Wrong split for {device_id}: "
                f"{entry.get('split')}"
            )

        label = entry.get("label")

        if label not in DEVICE_CLASSES:
            raise ValueError(
                f"Invalid label for {device_id}: {label}"
            )

        labels.append(label)

    # Verify that all ground-truth test IDs are present.
    truth_test_ids = {
        device_id
        for device_id, entry in truth.items()
        if entry.get("split") == "test"
    }

    if set(ids) != truth_test_ids:
        raise ValueError(
            "Test CSV IDs do not exactly match "
            "ground-truth test IDs"
        )

    observations = [
        observation_from_row(row)
        for row in rows
    ]

    # No ground-truth information enters observations.
    return observations, np.asarray(labels)


def evaluate(checkpoint, observations, labels, device, lam, constraints):
    classifier = load_neurosymbolic_classifier(
        checkpoint_path=checkpoint,
        symbolic_strength=lam,
        device=device,
        constraints=constraints,
    )

    results = classifier.predict_many(observations)

    predictions = np.asarray([
        item.predicted_label
        for item in results
    ])

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(DEVICE_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "predictions": predictions,
    }


def paired_changes(reference, candidate, labels):
    a = reference["predictions"]
    b = candidate["predictions"]

    a_correct = a == labels
    b_correct = b == labels

    return {
        "prediction_flips": int(np.sum(a != b)),
        "correct_to_wrong": int(
            np.sum(a_correct & ~b_correct)
        ),
        "wrong_to_correct": int(
            np.sum(~a_correct & b_correct)
        ),
    }


def summarize(records):
    df = pd.DataFrame(records)
    summary = []

    for (fraction, condition), group in df.groupby(
        ["fraction", "condition"], sort=True
    ):
        deltas = group["delta_vs_correct"].to_numpy(
            dtype=float
        )

        mean_delta = float(deltas.mean())
        n = len(deltas)

        std = (
            float(deltas.std(ddof=1))
            if n > 1 else 0.0
        )

        # Approximate 95% CI across training seeds.
        margin = (
            1.96 * std / np.sqrt(n)
            if n > 1 else None
        )

        summary.append({
            "fraction": float(fraction),
            "condition": condition,
            "n_runs": n,
            "accuracy_mean": float(
                group["accuracy"].mean()
            ),
            "macro_f1_mean": float(
                group["macro_f1"].mean()
            ),
            "delta_vs_correct_mean": mean_delta,
            "delta_ci95_low": (
                mean_delta - margin
                if margin is not None else None
            ),
            "delta_ci95_high": (
                mean_delta + margin
                if margin is not None else None
            ),
            "prediction_flips_mean": float(
                group["prediction_flips"].mean()
            ),
            "correct_to_wrong_mean": float(
                group["correct_to_wrong"].mean()
            ),
            "wrong_to_correct_mean": float(
                group["wrong_to_correct"].mean()
            ),
        })

    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        default="data/generated/moderate/test.csv",
    )
    parser.add_argument(
        "--truth",
        default="data/generated/moderate/ground_truth.json",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="artifacts/h2/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/final_test",
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if not np.isfinite(args.lam) or args.lam < 0:
        parser.error("--lam must be finite and nonnegative")

    observations, labels = load_test_data(
        args.test,
        args.truth,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "neural": (0.0, None),
        "correct": (
            args.lam,
            SoftSymbolicConstraints(),
        ),
        "corrupt_routing": (
            args.lam,
            CorruptedConstraints("routing"),
        ),
        "corrupt_all": (
            args.lam,
            CorruptedConstraints("all"),
        ),
    }

    records = []

    print("\nFINAL HELD-OUT TEST EVALUATION")
    print("=" * 70)
    print(f"Test observations: {len(labels)}")
    print(f"Lambda: {args.lam}")
    print(f"Device: {args.device}")

    for fraction in FRACTIONS:
        for seed in SEEDS:
            checkpoint = (
                Path(args.checkpoints_dir)
                / f"fraction_{fraction}"
                / f"seed_{seed}.pt"
            )

            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

            evaluated = {}

            for name, (lam, constraints) in conditions.items():
                evaluated[name] = evaluate(
                    checkpoint,
                    observations,
                    labels,
                    args.device,
                    lam,
                    constraints,
                )

            correct = evaluated["correct"]

            for name, result in evaluated.items():
                changes = paired_changes(
                    correct, result, labels
                )

                records.append({
                    "fraction": fraction / 100,
                    "seed": seed,
                    "condition": name,
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "delta_vs_correct": (
                        result["accuracy"]
                        - correct["accuracy"]
                    ),
                    **changes,
                })

            print(
                f"{fraction:3d}% seed={seed} | "
                f"Neural={evaluated['neural']['accuracy']:.4f} | "
                f"Correct={correct['accuracy']:.4f} | "
                f"Routing={evaluated['corrupt_routing']['accuracy']:.4f} | "
                f"All={evaluated['corrupt_all']['accuracy']:.4f}"
            )

    pd.DataFrame(records).to_csv(
        output_dir / "test_results.csv",
        index=False,
    )

    summary = summarize(records)

    summary.to_csv(
        output_dir / "test_summary.csv",
        index=False,
    )

    manifest = {
        "experiment": "final_held_out_test",
        "test_path": args.test,
        "ground_truth_path": args.truth,
        "test_size": len(labels),
        "lambda": args.lam,
        "fractions": list(FRACTIONS),
        "seeds": list(SEEDS),
        "conditions": list(conditions),
        "note": (
            "Ground truth is used only for evaluation. "
            "Checkpoints and symbolic rules are frozen. "
            "Confidence intervals describe training-seed "
            "variation on one fixed synthetic test set."
        ),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL TEST SUMMARY")
    print("=" * 70)
    print(
        summary[
            [
                "fraction",
                "condition",
                "accuracy_mean",
                "macro_f1_mean",
                "delta_vs_correct_mean",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
