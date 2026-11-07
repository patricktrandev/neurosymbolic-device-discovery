import csv
import json

from device_discovery.data.materializer import materialize_observations
from src.data.builder import build_device_splits


def test_materialized_dataset_has_expected_structure(tmp_path):
    splits = build_device_splits(samples_per_class=20, seed=42)

    materialize_observations(splits=splits, output_dir=tmp_path, seed=42, missing_rate=0.0, noise_rate=0.0, initial_observation_count=2)

    with (tmp_path / "train.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 8 * 14

    features = ["vendor_hint", "routing_enabled", "lldp_enabled", "snmp_available", "interface_count_band", "vlan_capable", "firmware_family"]

    for row in rows:
        observed = [feature for feature in features if row[feature] != ""]
        assert len(observed) == 2
        assert row["label"]

    with (tmp_path / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["seed"] == 42
    assert manifest["split_counts"]["train"] == 112
