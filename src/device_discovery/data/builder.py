import random

from core.domain import DeviceProfile
from core.utils.constants.device_constants import DeviceRole, Vendor
from device_discovery.simulation.simulator import DeviceSimulator


def build_device_splits(samples_per_class: int = 200, seed: int = 42) -> dict[str, list[DeviceProfile]]:
    """
    Generate balanced synthetic device datasets.

    Splits:
        train: 70%
        val:   15%
        test:  15%

    Each split contains all 8 device classes.
    """

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    train_count = int(samples_per_class * 0.70)
    val_count = int(samples_per_class * 0.15)
    test_count = samples_per_class - train_count - val_count

    if min(train_count, val_count, test_count) == 0:
        raise ValueError("samples_per_class is too small " "to populate all splits")

    simulator = DeviceSimulator(seed=seed)
    rng = random.Random(seed)

    splits: dict[str, list[DeviceProfile]] = {"train": [], "val": [], "test": []}

    for vendor in Vendor:
        for role in DeviceRole:
            devices = []

            for index in range(samples_per_class):
                device = simulator.create_device(device_id=(f"{vendor.value}-" f"{role.value}-{index:05d}"), vendor=vendor, role=role)
                devices.append(device)

            rng.shuffle(devices)

            train_end = train_count
            val_end = train_end + val_count

            splits["train"].extend(devices[:train_end])
            splits["val"].extend(devices[train_end:val_end])
            splits["test"].extend(devices[val_end:])

    # Shuffle class ordering inside each split.
    for devices in splits.values():
        rng.shuffle(devices)

    return splits
