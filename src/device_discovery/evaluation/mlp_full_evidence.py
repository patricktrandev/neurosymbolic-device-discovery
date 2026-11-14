from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType
from device_discovery.learning.neurosymbolic_classifier import load_neurosymbolic_classifier

PROBES = tuple(ProbeType)


def load_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_value(probe, value):
    if value in ("", "None", "null"):
        return None

    if probe in (ProbeType.ROUTING, ProbeType.LLDP, ProbeType.SNMP, ProbeType.VLAN):
        if value not in ("True", "False"):
            raise ValueError(f"Invalid boolean: {probe.value}={value}")
        return value == "True"

    return value


def to_observation(row):
    evidence = {}

    for probe in PROBES:
        value = parse_value(probe, row[probe.value])

        if value is not None:
            evidence[probe] = value

    return Observation(device_id=row["device_id"], evidence=evidence)


def evaluate(model, rows):
    observations = [to_observation(row) for row in rows]

    predictions = model.predict_many(observations)

    y_true = [row["label"] for row in rows]
    y_pred = [prediction.predicted_label for prediction in predictions]

    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="artifacts/mlp/model.pt")

    parser.add_argument("--sparse-val", default="data/generated/moderate/val.csv")

    parser.add_argument("--full-val", default="data/generated/moderate_full/val.csv")

    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    sparse_rows = load_rows(Path(args.sparse_val))

    full_rows = load_rows(Path(args.full_val))

    sparse_ids = [row["device_id"] for row in sparse_rows]

    full_ids = [row["device_id"] for row in full_rows]

    if sparse_ids != full_ids:
        raise ValueError("Sparse/full validation rows do not align")

    model = load_neurosymbolic_classifier(checkpoint_path=args.checkpoint, symbolic_strength=0.0, device=args.device)

    sparse_metrics = evaluate(model, sparse_rows)

    full_metrics = evaluate(model, full_rows)

    print("\nSPARSE-TRAINED MLP")
    print("=" * 50)

    print("Sparse validation accuracy:", f"{sparse_metrics['accuracy']:.4f}")

    print("Full validation accuracy:", f"{full_metrics['accuracy']:.4f}")

    print("Full validation Macro-F1:", f"{full_metrics['macro_f1']:.4f}")

    print("Accuracy difference:", f"{full_metrics['accuracy'] - sparse_metrics['accuracy']:+.4f}")


if __name__ == "__main__":
    main()
