
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle,
)

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.neurosymbolic_classifier import (
    load_neurosymbolic_classifier,
)


PROBE_NAMES = tuple(p.value for p in ProbeType)
LEVELS = (1, 2, 3, 5, 7)
FRACTIONS = (5, 10, 25, 50, 100)
SEEDS = tuple(range(42, 52))


def stable_seed(device_id, seed):
    payload = f"{seed}:{device_id}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    )


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_rows(path):
    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def make_probe_order(device_id, initial, seed):
    if len(initial) != 2 or len(set(initial)) != 2:
        raise ValueError(
            f"Expected two unique initial probes: {device_id}"
        )

    if not set(initial).issubset(PROBE_NAMES):
        raise ValueError(
            f"Unknown initial probe: {device_id}"
        )

    remaining = [
        p for p in PROBE_NAMES if p not in initial
    ]

    import random
    rng = random.Random(stable_seed(device_id, seed))
    rng.shuffle(remaining)

    return list(initial) + remaining


def make_observations(rows, outcomes, attempts, level, seed):
    observations = []
    labels = []
    observed_counts = []
    costs = []

    for row in rows:
        device_id = row["device_id"]

        if device_id not in outcomes or device_id not in attempts:
            raise ValueError(
                f"Missing stored probe data: {device_id}"
            )

        if row.get("label") not in DEVICE_CLASSES:
            raise ValueError(
                f"Missing/invalid validation label: {device_id}"
            )

        order = make_probe_order(
            device_id, attempts[device_id], seed
        )

        # At k=1, use the first original attempted probe.
        # At k=2, reproduce both original attempts.
        selected = order[:level]
        evidence = {}
        cost = 0.0

        for probe_name in selected:
            result = outcomes[device_id][probe_name]
            cost += float(result["cost"])

            if result["success"]:
                value = result["value"]

                if value is not None:
                    evidence[ProbeType(probe_name)] = value

        # Validate original baseline at k=2.
        if level == 2:
            for probe_name in PROBE_NAMES:
                actual = evidence.get(ProbeType(probe_name))
                raw = row.get(probe_name, "")

                if raw is None or raw == "":
                    expected = None
                elif raw == "True":
                    expected = True
                elif raw == "False":
                    expected = False
                else:
                    expected = raw

                if actual != expected:
                    raise ValueError(
                        f"Baseline mismatch: "
                        f"{device_id}/{probe_name}: "
                        f"{actual!r} != {expected!r}"
                    )

        observations.append(
            Observation(
                device_id=device_id,
                evidence=evidence,
            )
        )
        labels.append(row["label"])
        observed_counts.append(len(evidence))
        costs.append(cost)

    return (
        observations,
        labels,
        float(np.mean(observed_counts)),
        float(np.mean(costs)),
    )


def metrics(predictions, labels):
    from sklearn.metrics import accuracy_score, f1_score

    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(
            labels,
            predictions,
            labels=list(DEVICE_CLASSES),
            average="macro",
            zero_division=0,
        )),
    )


def save_chart(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fraction in FRACTIONS:
        subset = summary[
            summary["fraction"] == fraction
        ].sort_values("attempted_probes")

        ax.plot(
            subset["attempted_probes"],
            subset["neural_accuracy_mean"] * 100,
            marker="o",
            label=f"MLP trained with {fraction}%",
        )

    ax.set_title("E3-A: Fixed MLP with increasing evidence")
    ax.set_xlabel("Attempted probes per device")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(LEVELS)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = output_dir / "e3_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_report(summary, chart_path, output_dir):
    path = output_dir / "e3_report.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=(595, 842),
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(
            "E3-A: Evidence Availability Report",
            styles["Title"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Fixed-checkpoint evaluation on synthetic "
            "validation observations.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Protocol: reuse pretrained H2 checkpoints, "
            "preserve original two attempted probes, "
            "then reveal additional stored probe outcomes. "
            "No model retraining.",
            styles["Normal"],
        ),
        Spacer(1, 16),
        Image(str(chart_path), width=7 * inch, height=4.28 * inch),
        Spacer(1, 16),
    ]

    header = [
        "Train %",
        "Probes",
        "Observed",
        "MLP acc.",
        "Fusion acc.",
    ]
    data = [header]

    for _, row in summary.iterrows():
        data.append([
            str(int(row["fraction"])),
            str(int(row["attempted_probes"])),
            f'{row["observed_probes_mean"]:.2f}',
            f'{row["neural_accuracy_mean"]:.3f}',
            f'{row["fusion_accuracy_mean"]:.3f}',
        ])

    table = Table(
        data,
        colWidths=[65, 65, 80, 95, 95],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Limitations: Additional evidence comes from stored "
        "simulator outcomes, not new independent probe "
        "measurements. More probes may reveal noisy or "
        "incorrect values. Training and evaluation evidence "
        "distributions differ at higher probe counts. "
        "All seeds share one validation split.",
        styles["Normal"],
    ))

    doc.build(story)
    return path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data/generated/moderate",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="artifacts/h2/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/e3",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--probe-seed", type=int, default=2026)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_dir / "val.csv")
    outcomes = load_json(data_dir / "probe_outcomes.json")
    attempts = load_json(data_dir / "initial_attempts.json")

    manifest = load_json(data_dir / "manifest.json")
    if manifest["initial_observation_count"] != 2:
        raise ValueError(
            "This E3-A script requires a two-probe baseline"
        )

    datasets = {
        level: make_observations(
            rows, outcomes, attempts,
            level, args.probe_seed,
        )
        for level in LEVELS
    }

    records = []

    for fraction in FRACTIONS:
        for seed in SEEDS:
            checkpoint = (
                Path(args.checkpoints_dir)
                / f"fraction_{fraction}"
                / f"seed_{seed}.pt"
            )

            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

            neural = load_neurosymbolic_classifier(
                checkpoint,
                symbolic_strength=0.0,
                device=args.device,
            )
            fusion = load_neurosymbolic_classifier(
                checkpoint,
                symbolic_strength=args.lam,
                device=args.device,
            )

            for level in LEVELS:
                observations, labels, observed, cost = (
                    datasets[level]
                )

                neural_predictions = [
                    p.predicted_label
                    for p in neural.predict_many(observations)
                ]
                fusion_predictions = [
                    p.predicted_label
                    for p in fusion.predict_many(observations)
                ]

                n_acc, n_f1 = metrics(
                    neural_predictions, labels
                )
                f_acc, f_f1 = metrics(
                    fusion_predictions, labels
                )

                records.append({
                    "fraction": fraction,
                    "seed": seed,
                    "attempted_probes": level,
                    "observed_probes_mean": observed,
                    "attempted_cost_mean": cost,
                    "neural_accuracy": n_acc,
                    "neural_macro_f1": n_f1,
                    "fusion_accuracy": f_acc,
                    "fusion_macro_f1": f_f1,
                })

            print(
                f"Completed fraction={fraction}% seed={seed}"
            )

    results = pd.DataFrame(records)
    results.to_csv(
        output_dir / "e3_results.csv",
        index=False,
    )

    summary = results.groupby(
        ["fraction", "attempted_probes"],
        as_index=False,
    ).agg(
        observed_probes_mean=("observed_probes_mean", "mean"),
        attempted_cost_mean=("attempted_cost_mean", "mean"),
        neural_accuracy_mean=("neural_accuracy", "mean"),
        neural_accuracy_std=("neural_accuracy", "std"),
        neural_macro_f1_mean=("neural_macro_f1", "mean"),
        fusion_accuracy_mean=("fusion_accuracy", "mean"),
        fusion_accuracy_std=("fusion_accuracy", "std"),
        fusion_macro_f1_mean=("fusion_macro_f1", "mean"),
    )

    summary.to_csv(
        output_dir / "e3_summary.csv",
        index=False,
    )

    chart_path = save_chart(summary, output_dir)
    report_path = save_report(
        summary, chart_path, output_dir
    )

    (output_dir / "manifest.json").write_text(
        json.dumps({
            "experiment": "E3-A",
            "evaluation_split": "val",
            "probe_levels": list(LEVELS),
            "fractions": list(FRACTIONS),
            "seeds": list(SEEDS),
            "probe_seed": args.probe_seed,
            "symbolic_strength": args.lam,
            "checkpoint_source": args.checkpoints_dir,
            "note": (
                "Fixed sparse-trained MLPs. "
                "Nested attempted probes with stored outcomes. "
                "No retraining or test-set model selection."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    print("\nE3-A SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
