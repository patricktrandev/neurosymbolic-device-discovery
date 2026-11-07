import random

from core.utils.constants.device_constants import DeviceRole, Vendor
from device_discovery.simulation.simulator import DeviceSimulator


def build_device_splits(samples_per_class: int = 200, seed: int = 42):
    simulator = DeviceSimulator(seed=seed)
    devices = []

    for vendor in Vendor:
        for role in DeviceRole:
            for index in range(samples_per_class):
                device = simulator.create_device(device_id=(f"{vendor.value}-{role.value}-{index}"), vendor=vendor, role=role)
                devices.append(device)

    rng = random.Random(seed)
    rng.shuffle(devices)

    # Stratified by class.
    splits = {"train": [], "val": [], "test": []}

    for vendor in Vendor:
        for role in DeviceRole:
            group = [d for d in devices if d.vendor == vendor and d.role == role]
            n = len(group)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            splits["train"].extend(group[:train_end])
            splits["val"].extend(group[train_end:val_end])
            splits["test"].extend(group[val_end:])

    return splits
