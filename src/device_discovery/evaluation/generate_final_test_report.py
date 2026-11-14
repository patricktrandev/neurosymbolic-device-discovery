
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


CONDITIONS = (
    "neural",
    "correct",
    "corrupt_routing",
    "corrupt_all",
)

COLORS = {
    "neural": "#64748b",
    "correct": "#2563eb",
    "corrupt_routing": "#f59e0b",
    "corrupt_all": "#dc2626",
}

LABELS = {
    "neural": "Neural only",
    "correct": "Correct fusion",
    "corrupt_routing": "Corrupt routing",
    "corrupt_all": "Corrupt all",
}


def page_title(ax, title, subtitle=None):
    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    if subtitle:
        ax.text(
            0, 1.01, subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color="#64748b",
        )


def add_text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    fig.text(
        0.10, 0.91, title,
        fontsize=21,
        fontweight="bold",
    )

    y = 0.84
    import textwrap

    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=90):
            fig.text(0.10, y, line, fontsize=11)
            y -= 0.026
        y -= 0.035

    pdf.savefig(fig)
    plt.close(fig)


def plot_accuracy(summary, output_dir, pdf):
    pivot = summary.pivot(
        index="fraction",
        columns="condition",
        values="accuracy_mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for condition in CONDITIONS:
        ax.plot(
            pivot.index * 100,
            pivot[condition] * 100,
            marker="o",
            linewidth=2,
            label=LABELS[condition],
            color=COLORS[condition],
        )

    page_title(
        ax,
        "Held-out test accuracy",
        "Mean accuracy across 10 training seeds",
    )

    ax.set_xlabel("Training fraction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(pivot.index * 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "accuracy_by_fraction.png",
        dpi=180,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_fusion_gain(summary, output_dir, pdf):
    pivot = summary.pivot(
        index="fraction",
        columns="condition",
        values="accuracy_mean",
    ).sort_index()

    gains = (
        pivot["correct"] - pivot["neural"]
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        [str(int(x * 100)) + "%" for x in gains.index],
        gains.values,
        color=COLORS["correct"],
    )

    for bar, value in zip(bars, gains.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"+{value:.2f} pp",
            ha="center",
            fontsize=10,
        )

    page_title(
        ax,
        "Benefit of correct symbolic fusion",
        "Accuracy gain over neural-only baseline",
    )

    ax.set_ylabel("Accuracy improvement (pp)")
    ax.set_ylim(0, max(gains) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        output_dir / "fusion_gain.png",
        dpi=180,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_corruption(summary, output_dir, pdf):
    pivot = summary.pivot(
        index="fraction",
        columns="condition",
        values="accuracy_mean",
    ).sort_index()

    x = np.arange(len(pivot))
    width = 0.34

    routing_loss = (
        pivot["correct"] - pivot["corrupt_routing"]
    ) * 100

    all_loss = (
        pivot["correct"] - pivot["corrupt_all"]
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        x - width / 2,
        routing_loss,
        width,
        label="Corrupt routing",
        color=COLORS["corrupt_routing"],
    )

    ax.bar(
        x + width / 2,
        all_loss,
        width,
        label="Corrupt all",
        color=COLORS["corrupt_all"],
    )

    page_title(
        ax,
        "Impact of incorrect symbolic knowledge",
        "Accuracy loss relative to correct fusion",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{v * 100:g}%" for v in pivot.index]
    )
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Accuracy loss (pp)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        output_dir / "corruption_loss.png",
        dpi=180,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_transitions(summary, output_dir, pdf):
    subset = summary[
        summary["condition"] == "corrupt_all"
    ].sort_values("fraction")

    x = np.arange(len(subset))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        x - width / 2,
        subset["correct_to_wrong_mean"],
        width,
        label="Correct to wrong",
        color=COLORS["corrupt_all"],
    )

    ax.bar(
        x + width / 2,
        subset["wrong_to_correct_mean"],
        width,
        label="Wrong to correct",
        color="#16a34a",
    )

    page_title(
        ax,
        "Prediction transitions: corrupt all",
        "Mean number of changes across 10 seeds",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{v * 100:g}%"
        for v in subset["fraction"]
    ])

    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Mean prediction count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        output_dir / "prediction_transitions.png",
        dpi=180,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        default="artifacts/final_test/test_summary.csv",
    )
    parser.add_argument(
        "--results",
        default="artifacts/final_test/test_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/final_test",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    results = pd.read_csv(args.results)

    required = {
        "fraction",
        "condition",
        "accuracy_mean",
        "macro_f1_mean",
        "correct_to_wrong_mean",
        "wrong_to_correct_mean",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Missing summary columns: {sorted(missing)}"
        )

    if len(results) != 200:
        raise ValueError(
            f"Expected 200 runs, got {len(results)}"
        )

    pivot = summary.pivot(
        index="fraction",
        columns="condition",
        values="accuracy_mean",
    ).sort_index()

    gain_5 = (
        pivot.loc[0.05, "correct"]
        - pivot.loc[0.05, "neural"]
    ) * 100

    loss_5 = (
        pivot.loc[0.05, "correct"]
        - pivot.loc[0.05, "corrupt_all"]
    ) * 100

    pdf_path = output_dir / "final_test_report.pdf"

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Final Held-out Test Report",
            [
                "Neurosymbolic Device Discovery",
                "Evaluation setup: 240 held-out synthetic "
                "observations, 50 trained MLP checkpoints, "
                "10 seeds per training fraction, "
                "and symbolic strength lambda = 0.5.",
                "Conditions: neural-only, correct symbolic "
                "fusion, corrupted routing rules, and "
                "corrupted vendor/routing/VLAN rules.",
                f"At 5% training data, correct fusion "
                f"improved accuracy over neural-only "
                f"by {gain_5:.2f} percentage points.",
                f"At the same fraction, corrupting all "
                f"symbolic rules reduced accuracy "
                f"by {loss_5:.2f} percentage points "
                f"relative to correct fusion.",
                "The test set was not used for training "
                "or hyperparameter selection in this "
                "evaluation.",
            ],
        )

        plot_accuracy(summary, output_dir, pdf)
        plot_fusion_gain(summary, output_dir, pdf)
        plot_corruption(summary, output_dir, pdf)
        plot_transitions(summary, output_dir, pdf)

        add_text_page(
            pdf,
            "Discussion and limitations",
            [
                "The held-out synthetic test reproduces "
                "the main trends observed during "
                "validation: symbolic fusion is most "
                "helpful with scarce labeled data, "
                "and incorrect symbolic knowledge "
                "can degrade performance.",
                "The dataset is generated by a simulator "
                "rather than collected from real "
                "network devices.",
                "All 10 training seeds are evaluated "
                "on the same 240 test observations; "
                "they are not independent test sets.",
                "The fixed lambda value and symbolic "
                "rules were explored during development. "
                "The held-out results should therefore "
                "be interpreted as a single synthetic "
                "test evaluation, not evidence of "
                "real-world deployment performance.",
                "Future work should evaluate independent "
                "simulator settings, rule reliability, "
                "and real network device telemetry.",
            ],
        )

    print(f"PDF report: {pdf_path}")
    print(f"Figures: {output_dir}")


if __name__ == "__main__":
    main()
