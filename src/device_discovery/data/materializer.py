import csv
import json
import random
from pathlib import Path

from core.domain import DeviceProfile
from core.utils.constants.device_constants import ProbeType
from device_discovery.simulation.simulator import DeviceSimulator

FEATURE_NAMES = [probe.value for probe in ProbeType]


def materialize_observations(
    splits: dict[str, list[DeviceProfile]], output_dir: str | Path = "data/generated", seed: int = 42, scenario: str = "moderate", cost_mode: str = "heterogeneous", initial_observation_count: int = 2
) -> None:
    """
    Materialize synthetic device observations.

    CSV files contain only initially visible evidence.
    Hidden probe results are stored separately for oracle use.
    """

    if not 0 <= initial_observation_count <= len(ProbeType):
        raise ValueError("Invalid initial_observation_count")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulator = DeviceSimulator(seed=seed, scenario=scenario, cost_mode=cost_mode)

    rng = random.Random(seed)

    ground_truth = {}
    probe_outcomes = {}
    initial_attempts = {}
    split_counts = {}

    for split_name, devices in splits.items():
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split_name}")

        split_counts[split_name] = len(devices)

        rows = []

        for device in devices:
            device_id = device.device_id

            if device_id in ground_truth:
                raise ValueError(f"Duplicate device_id: {device_id}")

            ground_truth[device_id] = {"split": split_name, "label": device.label, "vendor": device.vendor.value, "role": device.role.value}

            outcomes = {}

            for probe_type in ProbeType:
                result = simulator.probe(device=device, probe_type=probe_type)

                outcomes[probe_type.value] = {"value": result.value, "success": result.success, "cost": result.cost}

            probe_outcomes[device_id] = outcomes

            # Randomly choose initially available probes.
            selected_probes = rng.sample(list(ProbeType), k=initial_observation_count)
            initial_attempts[device_id] = [
    probe.value for probe in selected_probes
]
            row = {"device_id": device_id}

            # Training and validation labels are visible.
            # Test labels remain evaluator-only.
            if split_name != "test":
                row["label"] = device.label

            for probe_type in ProbeType:
                row[probe_type.value] = ""

            for probe_type in selected_probes:
                result = outcomes[probe_type.value]

                if result["success"]:
                    row[probe_type.value] = result["value"]

            rows.append(row)

        fieldnames = ["device_id"]

        if split_name != "test":
            fieldnames.append("label")

        fieldnames.extend(FEATURE_NAMES)

        csv_path = output_path / f"{split_name}.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(rows)

    with (output_path / "ground_truth.json").open("w", encoding="utf-8") as file:
        json.dump(ground_truth, file, indent=2)

    with (output_path / "probe_outcomes.json").open("w", encoding="utf-8") as file:
        json.dump(probe_outcomes, file, indent=2)
    with (output_path / "initial_attempts.json").open("w",encoding="utf-8") as file:
        json.dump(initial_attempts, file, indent=2)
    manifest = {"seed": seed, "scenario": scenario, "cost_mode": cost_mode, "initial_observation_count": initial_observation_count, "split_counts": split_counts, "feature_names": FEATURE_NAMES}

    with (output_path / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
