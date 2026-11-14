
import csv
import json
from pathlib import Path

from core.utils.constants.device_constants import ProbeType
from device_discovery.environment.discovery_environment import (
    DiscoveryEnvironment,
)


def parse_value(value: str):
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def load_environment(
    data_dir: str | Path = "data/generated/moderate",
    split: str = "test",
) -> DiscoveryEnvironment:

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    path = Path(data_dir)

    initial_observations = {}

    with (path / f"{split}.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        for row in csv.DictReader(file):
            device_id = row["device_id"]

            initial_observations[device_id] = {
                probe: parse_value(row[probe.value])
                for probe in ProbeType
                if row[probe.value] != ""
            }

    with (path / "probe_outcomes.json").open(
        encoding="utf-8",
    ) as file:
        all_outcomes = json.load(file)

    with (path / "initial_attempts.json").open(
        encoding="utf-8",
    ) as file:
        all_attempts = json.load(file)

    outcomes = {
        device_id: all_outcomes[device_id]
        for device_id in initial_observations
    }

    attempts = {
        device_id: [
            ProbeType(value)
            for value in all_attempts[device_id]
        ]
        for device_id in initial_observations
    }

    return DiscoveryEnvironment(
        initial_observations=initial_observations,
        probe_outcomes=outcomes,
        initial_attempts=attempts,
    )
