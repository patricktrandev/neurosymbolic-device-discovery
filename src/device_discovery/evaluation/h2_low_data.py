
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from device_discovery.learning.feature_encoder import (
    FeatureEncoder,
    observation_from_row,
)
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.neurosymbolic_classifier import (
    load_neurosymbolic_classifier,
)
from device_discovery.learning.trainer import (
    TrainingConfig,
    fit_mlp,
    set_seed,
)


FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
SEEDS = tuple(range(42, 52))


def load_rows(path):
    with Path(path).open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def get_labels(rows):
    return [row["label"] for row in rows]


def split_training_rows(rows, seed=2026):
    labels = get_labels(rows)

    train_idx, stop_idx = train_test_split(
        np.arange(len(rows)),
        test_size=0.10,
        random_state=seed,
        stratify=labels,
    )

    train_rows = [rows[i] for i in train_idx]
    stop_rows = [rows[i] for i in stop_idx]

    return train_rows, stop_rows


def stratified_subset(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("Invalid training fraction")

    if fraction == 1:
        return list(rows)

    rng = random.Random(seed)
    grouped = {}

    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    selected = []

    for label in DEVICE_CLASSES:
        group = grouped[label]

        count = max(
            1,
            round(len(group) * fraction),
        )

        selected.extend(
            rng.sample(group, count)
        )

    rng.shuffle(selected)
    return selected


def encode_rows(rows, encoder, device="cpu"):
    observations = [
        observation_from_row(row)
        for row in rows
    ]

    x = torch.tensor(
        encoder.transform_many(observations),
        dtype=torch.float32,
    )

    label_to_index = {
        label: i
        for i, label in enumerate(DEVICE_CLASSES)
    }

    y = torch.tensor(
        [
            label_to_index[row["label"]]
            for row in rows
        ],
        dtype=torch.long,
    )

    return x, y


def save_checkpoint(model, encoder, config, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": state,
            "input_dim": encoder.n_features,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "classes": list(DEVICE_CLASSES),
            "firmware_values": list(
                encoder.firmware_values
            ),
        },
        path,
    )


def evaluate_fusion(
    checkpoint,
    rows,
    symbolic_strength,
    device,
):
    classifier = load_neurosymbolic_classifier(
        checkpoint_path=str(checkpoint),
        symbolic_strength=symbolic_strength,
        device=device,
    )

    observations = [
        observation_from_row(row)
        for row in rows
    ]

    predictions = classifier.predict_many(
        observations
    )

    y_true = get_labels(rows)
    y_pred = [
        prediction.predicted_label
        for prediction in predictions
    ]

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(DEVICE_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "predictions": y_pred,
    }


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_train_rows = load_rows(
        Path(args.data_dir) / "train.csv"
    )

    val_rows = load_rows(
        Path(args.data_dir) / "val.csv"
    )

    train_pool, stop_rows = split_training_rows(
        all_train_rows,
        seed=args.split_seed,
    )

    print("H2 LOW-DATA LEARNING")
    print("=" * 65)
    print(f"Training pool: {len(train_pool)}")
    print(f"Early stopping: {len(stop_rows)}")
    print(f"Evaluation: {len(val_rows)}")
    print(f"Device: {args.device}")
    print(f"Symbolic strength: {args.lam}")

    device = torch.device(args.device)
    results = []

    for fraction in FRACTIONS:
        for seed in SEEDS:
            print(
                f"\nFraction={fraction:.0%} "
                f"| Seed={seed}"
            )

            set_seed(seed)

            subset = stratified_subset(
                train_pool,
                fraction=fraction,
                seed=seed,
            )

            observations = [
                observation_from_row(row)
                for row in subset
            ]

            encoder = FeatureEncoder().fit(
                observations
            )

            x_train, y_train = encode_rows(
                subset, encoder
            )

            x_stop, y_stop = encode_rows(
                stop_rows, encoder
            )

            config = TrainingConfig(
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                hidden_dim=64,
                dropout=0.1,
            )

            model, best_epoch, _, _ = fit_mlp(
                x_train=x_train,
                y_train=y_train,
                x_val=x_stop,
                y_val=y_stop,
                input_dim=encoder.n_features,
                config=config,
                device=device,
                verbose=False,
            )

            checkpoint = (
                output_dir
                / "checkpoints"
                / f"fraction_{int(fraction * 100)}"
                / f"seed_{seed}.pt"
            )

            save_checkpoint(
                model,
                encoder,
                config,
                checkpoint,
            )

            mlp = evaluate_fusion(
                checkpoint,
                val_rows,
                symbolic_strength=0.0,
                device=args.device,
            )

            fusion = evaluate_fusion(
                checkpoint,
                val_rows,
                symbolic_strength=args.lam,
                device=args.device,
            )

            mlp_correct = np.array(
                mlp["predictions"]
            ) == np.array(get_labels(val_rows))

            fusion_correct = np.array(
                fusion["predictions"]
            ) == np.array(get_labels(val_rows))

            result = {
                "fraction": fraction,
                "seed": seed,
                "n_train": len(subset),
                "best_epoch": best_epoch,
                "mlp_accuracy": mlp["accuracy"],
                "fusion_accuracy": fusion["accuracy"],
                "mlp_macro_f1": mlp["macro_f1"],
                "fusion_macro_f1": fusion["macro_f1"],
                "accuracy_delta": (
                    fusion["accuracy"]
                    - mlp["accuracy"]
                ),
                "macro_f1_delta": (
                    fusion["macro_f1"]
                    - mlp["macro_f1"]
                ),
                "mlp_correct_fusion_wrong": int(
                    np.sum(
                        mlp_correct & ~fusion_correct
                    )
                ),
                "mlp_wrong_fusion_correct": int(
                    np.sum(
                        ~mlp_correct & fusion_correct
                    )
                ),
            }

            results.append(result)

            print(
                f"n={len(subset)} "
                f"| epoch={best_epoch} "
                f"| MLP={mlp['accuracy']:.4f} "
                f"| Fusion={fusion['accuracy']:.4f} "
                f"| Δ={result['accuracy_delta']:+.4f}"
            )

            # Save after every run to preserve progress.
            pd.DataFrame(results).to_csv(
                output_dir / "h2_results.csv",
                index=False,
            )

    df = pd.DataFrame(results)

    summary = []

    for fraction, group in df.groupby(
        "fraction", sort=True
    ):
        entry = {
            "fraction": float(fraction),
            "n_runs": int(len(group)),
            "n_train": int(group["n_train"].iloc[0]),
        }

        for metric in (
            "mlp_accuracy",
            "fusion_accuracy",
            "accuracy_delta",
            "mlp_macro_f1",
            "fusion_macro_f1",
            "macro_f1_delta",
        ):
            values = group[metric].to_numpy(
                dtype=float
            )

            mean = float(values.mean())

            std = float(
                values.std(ddof=1)
            ) if len(values) > 1 else 0.0

            # Normal-approximation CI over paired seeds.
            # Not an independent-dataset confidence interval.
            margin = (
                1.96 * std / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )

            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_ci95_low"] = (
                mean - margin
            )
            entry[f"{metric}_ci95_high"] = (
                mean + margin
            )

        summary.append(entry)

    summary_path = output_dir / "h2_summary.json"

    summary_path.write_text(
        json.dumps(
            {
                "experiment": "H2_low_data",
                "lambda": args.lam,
                "split_seed": args.split_seed,
                "training_seeds": list(SEEDS),
                "fractions": list(FRACTIONS),
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")
    print("=" * 65)

    for entry in summary:
        print(
            f"{entry['fraction']:.0%} | "
            f"MLP={entry['mlp_accuracy_mean']:.4f} | "
            f"Fusion={entry['fusion_accuracy_mean']:.4f} | "
            f"Δ={entry['accuracy_delta_mean']:+.4f} "
            f"[{entry['accuracy_delta_ci95_low']:+.4f}, "
            f"{entry['accuracy_delta_ci95_high']:+.4f}]"
        )

    print(f"\nResults: {output_dir / 'h2_results.csv'}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data/generated/moderate",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/h2",
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
        "--split-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    if args.lam < 0:
        parser.error("--lam must be nonnegative")

    run_experiment(args)


if __name__ == "__main__":
    main()
