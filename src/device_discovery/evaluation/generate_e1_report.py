from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

MODEL_NAMES = {"bayes_oracle": "Bayes oracle (ceiling)", "heuristic": "Heuristic", "mlp": "MLP", "mlp_symbolic_0.5": "MLP + Symbolic"}

MODEL_COLORS = {"bayes_oracle": "#16a34a", "heuristic": "#64748b", "mlp": "#2563eb", "mlp_symbolic_0.5": "#ea580c"}

MODEL_ORDER = list(MODEL_NAMES)


def load_results(path: Path) -> list[dict]:
    """Load E1 summary metrics from CSV."""
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required = {"model", "condition", "rate", "accuracy", "macro_f1", "flip_rate"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required CSV columns: {required}")

        for row in reader:
            rows.append(
                {
                    "model": row["model"],
                    "condition": row["condition"],
                    "rate": float(row["rate"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "flip_rate": float(row["flip_rate"]),
                }
            )

    if not rows:
        raise ValueError("E1 results CSV is empty")

    return rows


def aggregate(rows: list[dict]) -> dict:
    """Average metrics across seeds for each condition."""
    grouped = defaultdict(list)

    for row in rows:
        key = (row["model"], row["condition"], row["rate"])
        grouped[key].append(row)

    summary = {}

    for key, group in grouped.items():
        summary[key] = {metric: sum(row[metric] for row in group) / len(group) for metric in ("accuracy", "macro_f1", "flip_rate")}

    return summary


def configure_chart(ax, title: str) -> None:
    """Apply consistent visual styling."""
    ax.set_title(title, fontsize=17, fontweight="bold", pad=20)

    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelsize=10)


def plot_baseline(summary: dict):
    """Figure 1: Baseline accuracy and macro-F1."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = list(range(len(MODEL_ORDER)))
    width = 0.32

    accuracies = [summary[(model, "baseline", 0.0)]["accuracy"] for model in MODEL_ORDER]

    f1_scores = [summary[(model, "baseline", 0.0)]["macro_f1"] for model in MODEL_ORDER]

    bars1 = ax.bar([value - width / 2 for value in x], accuracies, width, label="Accuracy", color="#2563eb")

    bars2 = ax.bar([value + width / 2 for value in x], f1_scores, width, label="Macro-F1", color="#ea580c")

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=10, xytext=(0, 5), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])

    ax.set_ylim(0, max(accuracies + f1_scores) * 1.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    configure_chart(ax, "Baseline Classification Performance")

    ax.legend(frameon=False)
    fig.tight_layout()

    return fig


def plot_condition(summary: dict, condition: str, metric: str, title: str):
    """Plot a metric across perturbation rates."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    rates = sorted({rate for model, kind, rate in summary if kind == condition})

    if not rates:
        raise ValueError(f"No data for condition: {condition}")

    for model in MODEL_ORDER:
        values = []

        for rate in rates:
            key = (model, condition, rate)

            if key not in summary:
                raise ValueError(f"Missing result: {key}")

            values.append(summary[key][metric])

        ax.plot([rate * 100 for rate in rates], [value * 100 for value in values], marker="o", linewidth=2.5, markersize=7, label=MODEL_NAMES[model], color=MODEL_COLORS[model])

    ax.set_xlabel("Additional perturbation rate (%)", fontsize=11)

    ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else "Prediction flip rate (%)", fontsize=11)

    ax.set_xticks([rate * 100 for rate in rates])

    if metric == "flip_rate":
        ax.set_ylim(bottom=0)

    configure_chart(ax, title)
    ax.legend(frameon=False)
    fig.tight_layout()

    return fig


def plot_tradeoff(summary: dict):
    """Figure 6: Accuracy vs prediction stability."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    condition = "missing"
    rate = 0.5

    for model in MODEL_ORDER:
        key = (model, condition, rate)

        if key not in summary:
            raise ValueError(f"Missing result: {key}")

        values = summary[key]

        x = values["flip_rate"] * 100
        y = values["accuracy"] * 100

        ax.scatter(x, y, s=170, color=MODEL_COLORS[model], label=MODEL_NAMES[model], zorder=3)

        ax.annotate(MODEL_NAMES[model], (x, y), xytext=(7, 8), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Prediction flip rate (%) — lower is more stable", fontsize=11)

    ax.set_ylabel("Accuracy (%) — higher is better", fontsize=11)

    ax.margins(x=0.25, y=0.35)

    configure_chart(ax, "Accuracy vs Stability — 50% Additional Missingness")

    fig.tight_layout()
    return fig


def add_chart_page(pdf: canvas.Canvas, figure, number: int, title: str, description: str) -> None:
    """Render one figure and short explanation per PDF page."""
    page_width, page_height = landscape(A4)

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, page_height - 82, page_width, 82, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(42, page_height - 48, title)

    buffer = io.BytesIO()

    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")

    buffer.seek(0)

    image = ImageReader(buffer)

    pdf.drawImage(image, 40, 128, width=page_width - 80, height=page_height - 235, preserveAspectRatio=True, anchor="c")

    pdf.setStrokeColor(colors.HexColor("#e2e8f0"))
    pdf.line(42, 113, page_width - 42, 113)

    pdf.setFillColor(colors.HexColor("#334155"))
    pdf.setFont("Helvetica", 10)

    # Descriptions are deliberately short.
    # Split on newline to support up to two lines.
    for index, line in enumerate(description.split("\n")):
        pdf.drawString(42, 91 - index * 15, line)

    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor("#64748b"))

    pdf.drawString(42, 25, "Neurosymbolic Device Discovery | Phase 3 - E1")

    pdf.drawRightString(page_width - 42, 25, f"Figure {number}")

    pdf.showPage()
    plt.close(figure)
    buffer.close()


def generate_report(input_csv: Path, output_pdf: Path) -> None:
    """Generate a six-page PDF with E1 figures."""
    rows = load_results(input_csv)
    summary = aggregate(rows)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_pdf), pagesize=landscape(A4))

    pdf.setTitle("Neurosymbolic Device Discovery - E1 Visual Report")

    figures = [
        (plot_baseline(summary), "01 | Baseline Performance", "MLP slightly outperforms the heuristic baseline in accuracy and macro-F1.\n" "Soft symbolic fusion does not improve baseline accuracy."),
        (
            plot_condition(summary, "missing", "accuracy", "Accuracy Under Missing Evidence"),
            "02 | Missing Evidence - Accuracy",
            "Accuracy decreases as more observed evidence is removed.\n" "All three methods approach similar accuracy at high missingness.",
        ),
        (
            plot_condition(summary, "missing", "flip_rate", "Prediction Stability Under Missing Evidence"),
            "03 | Missing Evidence - Stability",
            "The heuristic baseline changes predictions less frequently than MLP.\n" "Lower flip rate alone does not guarantee higher accuracy.",
        ),
        (
            plot_condition(summary, "noise", "accuracy", "Accuracy Under Noisy Evidence"),
            "04 | Noisy Evidence - Accuracy",
            "Random corruption of observed probe values reduces accuracy.\n" "Soft symbolic fusion shows no consistent improvement over MLP.",
        ),
        (
            plot_condition(summary, "noise", "flip_rate", "Prediction Stability Under Noisy Evidence"),
            "05 | Noisy Evidence - Stability",
            "MLP predictions are more sensitive to noisy evidence than heuristic predictions.\n" "Fusion produces a similar sensitivity profile to the neural baseline.",
        ),
        (
            plot_tradeoff(summary),
            "06 | Accuracy vs Stability",
            "At 50% additional missingness, accuracy is similar across models.\n" "The heuristic baseline exhibits substantially fewer prediction flips.",
        ),
    ]

    for number, (figure, title, description) in enumerate(figures, start=1):
        add_chart_page(pdf, figure, number, title, description)

    pdf.save()

    print(f"Generated PDF: {output_pdf}")
    print(f"Figures: {len(figures)}")
    print(f"Pages: {len(figures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 E1 visual PDF report")

    parser.add_argument("--input", default="artifacts/e1/e1_results.csv")

    parser.add_argument("--output", default="artifacts/reports/e1_visual_report.pdf")

    args = parser.parse_args()

    generate_report(input_csv=Path(args.input), output_pdf=Path(args.output))


if __name__ == "__main__":
    main()
