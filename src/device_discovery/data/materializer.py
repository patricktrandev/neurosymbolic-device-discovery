import csv
import json
import random
from pathlib import Path

from core.utils.constants.device_constants import ProbeType
from device_discovery.simulation.simulator import DeviceSimulator


def to_json_value(value):
    """Convert enum keys or values to JSON-compatible values."""
    return value.value if hasattr(value, "value") else value


def materialize_observations(splits: dict, output_dir: str | Path = "data/generated", seed: int = 42, missing_rate: float = 0.20, noise_rate: float = 0.05, initial_observation_count: int = 2) -> None:
    if not 0 <= missing_rate <= 1:
        raise ValueError("Invalid missing_rate")
    if not 0 <= noise_rate <= 1:
        raise ValueError("Invalid noise_rate")
    if not 0 <= initial_observation_count <= len(ProbeType):
        raise ValueError("Invalid initial_observation_count")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator = DeviceSimulator(seed=seed)
    rng = random.Random(seed + 1)

    ground_truth = {}
    probe_outcomes = {}
    split_counts = {}

    feature_names = [probe.value for probe in ProbeType]
    csv_fields = ["device_id", *feature_names, "label"]

    for split_name, devices in splits.items():
        rows = []

        for device in devices:
            # Internal ground truth: never passed to the model.
            ground_truth[device.device_id] = {
                "vendor": device.vendor.value,
                "role": device.role.value,
                "label": device.label,
                "true_features": {probe.value: to_json_value(value) for probe, value in device.true_features.items()},
            }

            # Precompute all possible query results.
            outcomes = {}

            for probe in ProbeType:
                result = simulator.probe(device=device, probe_type=probe, missing_rate=missing_rate, noise_rate=noise_rate)

                outcomes[probe.value] = {"value": to_json_value(result.value), "success": result.success, "cost": result.cost}

            probe_outcomes[device.device_id] = outcomes

            # Only reveal a small subset initially.
            initially_visible = set(rng.sample(list(ProbeType), k=initial_observation_count))

            row = {"device_id": device.device_id, "label": device.label}

            for probe in ProbeType:
                result = outcomes[probe.value]

                row[probe.value] = result["value"] if probe in initially_visible and result["success"] else None

            rows.append(row)

        csv_path = output_path / f"{split_name}.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(rows)

        split_counts[split_name] = len(rows)

    artifacts = {
        "ground_truth.json": ground_truth,
        "probe_outcomes.json": probe_outcomes,
        "manifest.json": {
            "seed": seed,
            "missing_rate": missing_rate,
            "noise_rate": noise_rate,
            "initial_observation_count": initial_observation_count,
            "split_counts": split_counts,
            "feature_names": feature_names,
            "classes": sorted({device.label for devices in splits.values() for device in devices}),
        },
    }

    for filename, content in artifacts.items():
        with (output_path / filename).open("w", encoding="utf-8") as file:
            json.dump(content, file, indent=2, sort_keys=True)
