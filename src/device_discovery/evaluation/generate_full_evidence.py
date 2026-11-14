from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.utils.constants.device_constants import ProbeType
from device_discovery.data.builder import build_device_splits

PROBES = tuple(ProbeType)
PROBE_COLUMNS = [probe.value for probe in PROBES]

LABEL_CANDIDATES = ("label", "device_class", "class_label", "target", "true_label")


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing sparse dataset: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        rows = list(reader)
        columns = list(reader.fieldnames)

    return columns, rows


def find_label_column(columns):
    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate

    raise ValueError("Cannot find label column. Available: " + ", ".join(columns))


def device_label(device):
    return f"{device.vendor.value}_{device.role.value}"


def serialize_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"

    if value is None:
        raise ValueError("Full evidence must not contain None")

    return str(value)


def make_full_rows(sparse_path: Path, devices):
    columns, sparse_rows = read_csv(sparse_path)

    if "device_id" not in columns:
        raise ValueError(f"device_id column is required: {sparse_path}")

    label_column = find_label_column(columns)

    missing_columns = [name for name in PROBE_COLUMNS if name not in columns]

    if missing_columns:
        raise ValueError("Missing probe columns: " + ", ".join(missing_columns))

    device_lookup = {device.device_id: device for device in devices}

    if len(device_lookup) != len(devices):
        raise ValueError("Duplicate device IDs in generated split")

    csv_ids = [row["device_id"] for row in sparse_rows]

    if len(csv_ids) != len(set(csv_ids)):
        raise ValueError(f"Duplicate device IDs in {sparse_path}")

    if set(csv_ids) != set(device_lookup):
        missing = set(csv_ids) - set(device_lookup)
        extra = set(device_lookup) - set(csv_ids)

        raise ValueError(
            "Sparse CSV does not match regenerated split. " f"Missing generated IDs: {sorted(missing)[:5]}; " f"extra generated IDs: {sorted(extra)[:5]}. " "Check seed and samples_per_class."
        )

    full_rows = []

    for sparse_row in sparse_rows:
        device_id = sparse_row["device_id"]
        device = device_lookup[device_id]

        expected_label = device_label(device)
        actual_label = sparse_row[label_column]

        if actual_label != expected_label:
            raise ValueError(f"Label mismatch for {device_id}: " f"CSV={actual_label!r}, " f"generated={expected_label!r}")

        full_row = dict(sparse_row)

        for probe in PROBES:
            if probe not in device.true_features:
                raise ValueError(f"Missing true feature {probe.value} " f"for {device_id}")

            full_row[probe.value] = serialize_value(device.true_features[probe])

        full_rows.append(full_row)

    return columns, full_rows


def write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=("Generate paired full-evidence datasets " "using DeviceProfile.true_features"))

    parser.add_argument("--sparse-dir", default="data/generated/moderate")

    parser.add_argument("--output-dir", default="data/generated/moderate_full")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--samples-per-class", type=int, default=200)

    parser.add_argument("--force", action="store_true", help="Allow overwriting existing output CSVs")

    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir)
    output_dir = Path(args.output_dir)

    if sparse_dir.resolve() == output_dir.resolve():
        parser.error("Output directory must differ from sparse directory")

    splits = build_device_splits(samples_per_class=args.samples_per_class, seed=args.seed)

    # Prepare and validate BOTH splits before writing.
    prepared = {}

    for split in ("train", "val"):
        sparse_path = sparse_dir / f"{split}.csv"

        columns, rows = make_full_rows(sparse_path, splits[split])

        prepared[split] = (columns, rows)

    train_ids = {row["device_id"] for row in prepared["train"][1]}

    val_ids = {row["device_id"] for row in prepared["val"][1]}

    if train_ids & val_ids:
        raise ValueError("Train/validation device leakage detected")

    output_paths = [output_dir / "train.csv", output_dir / "val.csv"]

    if not args.force:
        existing = [path for path in output_paths if path.exists()]

        if existing:
            raise FileExistsError("Output already exists: " + ", ".join(str(p) for p in existing) + ". Use --force to overwrite.")

    for split in ("train", "val"):
        columns, rows = prepared[split]

        output_path = output_dir / f"{split}.csv"

        write_csv(output_path, columns, rows)

        print(f"{split}: {len(rows)} observations " f"-> {output_path}")

    manifest = {
        "generator": "build_device_splits",
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "source": "DeviceProfile.true_features",
        "missingness": 0.0,
        "probe_noise": 0.0,
        "probe_columns": PROBE_COLUMNS,
        "train_count": len(prepared["train"][1]),
        "val_count": len(prepared["val"][1]),
        "test_split_used": False,
        "warning": ("Matching device IDs and labels does not " "prove identical latent features if the " "original dataset used another simulator seed."),
    }

    manifest_path = output_dir / "manifest.json"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print("Full-evidence generation completed.")


if __name__ == "__main__":
    main()
