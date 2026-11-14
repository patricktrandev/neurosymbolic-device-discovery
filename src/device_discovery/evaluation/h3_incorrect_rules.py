
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.feature_encoder import observation_from_row
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.neurosymbolic_classifier import (
    load_neurosymbolic_classifier,
)
from device_discovery.learning.symbolic_constraints import (
    SoftSymbolicConstraints,
    SymbolicConfig,
)


# A fixed cyclic permutation. No vendor maps to itself.
WRONG_VENDOR = {
    "cisco": "juniper",
    "juniper": "arista",
    "arista": "hpe",
    "hpe": "cisco",
}


class CorruptedConstraints(SoftSymbolicConstraints):
    """Controlled corruption of the original soft rules."""

    def __init__(
        self,
        mode: str,
        config: SymbolicConfig | None = None,
    ):
        super().__init__(config)

        if mode not in ("routing", "all"):
            raise ValueError(f"Unknown corruption mode: {mode}")

        self.mode = mode

    def score(self, observation):
        evidence = observation.evidence

        vendor_hint = evidence.get(ProbeType.VENDOR)
        routing = evidence.get(ProbeType.ROUTING)
        vlan = evidence.get(ProbeType.VLAN)

        if isinstance(vendor_hint, str):
            vendor_hint = vendor_hint.strip().lower()
        else:
            vendor_hint = None

        scores = {}

        for label in DEVICE_CLASSES:
            vendor, role = label.rsplit("_", 1)
            value = 0.0

            # Vendor: correct in routing-only corruption,
            # deliberately incorrect in all-rule corruption.
            if self.mode == "all":
                target_vendor = WRONG_VENDOR.get(vendor_hint)
            else:
                target_vendor = vendor_hint

            if target_vendor == vendor:
                value += self.config.vendor_weight

            # Routing: reversed in both corruption modes.
            if routing is True and role == "switch":
                value += self.config.routing_weight

            elif routing is False and role == "router":
                value += self.config.routing_weight

            # VLAN: reversed only in all-rule corruption.
            if self.mode == "all":
                if vlan is True and role == "router":
                    value += self.config.vlan_weight

                elif vlan is False and role == "switch":
                    value += self.config.vlan_weight
            else:
                if vlan is True and role == "switch":
                    value += self.config.vlan_weight

                elif vlan is False and role == "router":
                    value += self.config.vlan_weight

            scores[label] = value

        return scores


def load_validation(path):
    with Path(path).open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Validation dataset is empty")

    observations = [
        observation_from_row(row)
        for row in rows
    ]

    labels = [row["label"] for row in rows]

    if any(label not in DEVICE_CLASSES for label in labels):
        raise ValueError("Unknown validation label")

    return observations, labels


def evaluate_condition(
    checkpoint,
    observations,
    labels,
    device,
    lam,
    constraints,
):
    classifier = load_neurosymbolic_classifier(
        checkpoint_path=checkpoint,
        symbolic_strength=lam,
        device=device,
        constraints=constraints,
    )

    predictions = classifier.predict_many(observations)

    predicted_labels = [
        item.predicted_label
        for item in predictions
    ]

    return {
        "accuracy": float(
            accuracy_score(labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predicted_labels,
                labels=list(DEVICE_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "predictions": np.asarray(predicted_labels),
    }


def compare_predictions(reference, candidate, labels):
    reference_pred = reference["predictions"]
    candidate_pred = candidate["predictions"]
    truth = np.asarray(labels)

    reference_correct = reference_pred == truth
    candidate_correct = candidate_pred == truth

    return {
        "prediction_flips": int(
            np.sum(reference_pred != candidate_pred)
        ),
        "correct_to_wrong": int(
            np.sum(reference_correct & ~candidate_correct)
        ),
        "wrong_to_correct": int(
            np.sum(~reference_correct & candidate_correct)
        ),
    }


def run(args):
    observations, labels = load_validation(args.val)

    checkpoints_dir = Path(args.checkpoints_dir)

    if args.fraction is not None:
        fractions = [args.fraction]
    else:
        fractions = [5, 10, 25, 50, 100]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print("\nPHASE 3.9 — H3 INCORRECT SYMBOLIC KNOWLEDGE")
    print("=" * 65)
    print(f"Validation observations: {len(labels)}")
    print(f"Symbolic strength: {args.lam}")
    print(f"Device: {args.device}")

    for fraction in fractions:
        for seed in range(args.seed_start, args.seed_end + 1):
            checkpoint = (
                checkpoints_dir
                / f"fraction_{fraction}"
                / f"seed_{seed}.pt"
            )

            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Missing H2 checkpoint: {checkpoint}"
                )

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

            evaluations = {}

            for name, (strength, constraints) in conditions.items():
                evaluations[name] = evaluate_condition(
                    checkpoint=checkpoint,
                    observations=observations,
                    labels=labels,
                    device=args.device,
                    lam=strength,
                    constraints=constraints,
                )

            reference = evaluations["correct"]

            for condition, metrics in evaluations.items():
                flips = compare_predictions(
                    reference,
                    metrics,
                    labels,
                )

                records.append({
                    "fraction": fraction / 100.0,
                    "seed": seed,
                    "condition": condition,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "delta_vs_correct": (
                        metrics["accuracy"]
                        - reference["accuracy"]
                    ),
                    **flips,
                })

            print(
                f"{fraction:3d}% | seed={seed} | "
                f"MLP={evaluations['neural']['accuracy']:.4f} | "
                f"Correct={reference['accuracy']:.4f} | "
                f"Routing={evaluations['corrupt_routing']['accuracy']:.4f} | "
                f"All={evaluations['corrupt_all']['accuracy']:.4f}"
            )

            pd.DataFrame(records).to_csv(
                output_dir / "h3_results.csv",
                index=False,
            )

    df = pd.DataFrame(records)

    summary_rows = []

    for (fraction, condition), group in df.groupby(
        ["fraction", "condition"],
        sort=True,
    ):
        values = group["delta_vs_correct"].to_numpy(
            dtype=float
        )

        mean_delta = float(values.mean())

        std = (
            float(values.std(ddof=1))
            if len(values) > 1
            else 0.0
        )

        margin = (
            1.96 * std / np.sqrt(len(values))
            if len(values) > 1
            else None
        )

        summary_rows.append({
            "fraction": float(fraction),
            "condition": condition,
            "n_runs": int(len(group)),
            "accuracy_mean": float(
                group["accuracy"].mean()
            ),
            "macro_f1_mean": float(
                group["macro_f1"].mean()
            ),
            "delta_vs_correct_mean": mean_delta,
            "delta_ci95_low": (
                mean_delta - margin
                if margin is not None
                else None
            ),
            "delta_ci95_high": (
                mean_delta + margin
                if margin is not None
                else None
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

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output_dir / "h3_summary.csv",
        index=False,
    )

    manifest = {
        "experiment": "H3_incorrect_symbolic_knowledge",
        "lambda": args.lam,
        "validation_path": str(args.val),
        "validation_size": len(labels),
        "fractions": fractions,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "corruption": {
            "routing": "reverse router/switch routing rule",
            "all": "cyclic vendor mapping plus reversed routing and VLAN rules",
        },
        "reference": "correct symbolic fusion",
        "note": (
            "CI describes variation across training seeds "
            "on a fixed validation dataset."
        ),
    }

    (output_dir / "h3_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                "fraction",
                "condition",
                "accuracy_mean",
                "delta_vs_correct_mean",
                "prediction_flips_mean",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints-dir",
        default="artifacts/h2/checkpoints",
    )

    parser.add_argument(
        "--val",
        default="data/generated/moderate/val.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/h3",
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

    parser.add_argument(
        "--fraction",
        type=int,
        choices=[5, 10, 25, 50, 100],
        default=None,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--seed-end",
        type=int,
        default=51,
    )

    args = parser.parse_args()

    if not np.isfinite(args.lam) or args.lam < 0:
        parser.error("--lam must be finite and nonnegative")

    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start")

    run(args)


if __name__ == "__main__":
    main()
