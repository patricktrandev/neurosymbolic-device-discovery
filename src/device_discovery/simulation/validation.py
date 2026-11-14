from collections import Counter
from dataclasses import dataclass

from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor
from device_discovery.simulation.simulator import DeviceSimulator


@dataclass(frozen=True)
class ValidationReport:
    device_count: int
    probe_count: int
    missing_rate: float
    noise_rate: float
    ambiguous_vendor_rate: float
    ambiguous_firmware_rate: float
    total_cost: float
    class_counts: dict[str, int]


def validate_simulator(samples_per_class: int = 500, seed: int = 42, scenario: str = "moderate", cost_mode: str = "heterogeneous") -> ValidationReport:

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    simulator = DeviceSimulator(seed=seed, scenario=scenario, cost_mode=cost_mode)

    class_counts = Counter()

    missing_count = 0
    noise_count = 0
    success_count = 0

    ambiguous_vendor_count = 0
    ambiguous_firmware_count = 0

    total_cost = 0.0
    probe_count = 0

    for vendor in Vendor:
        for role in DeviceRole:
            for index in range(samples_per_class):

                device = simulator.create_device(device_id=(f"{vendor.value}-" f"{role.value}-{index}"), vendor=vendor, role=role)

                class_counts[device.label] += 1

                if device.true_features[ProbeType.VENDOR] == "unknown":
                    ambiguous_vendor_count += 1

                if device.true_features[ProbeType.FIRMWARE] == "generic_network_os":
                    ambiguous_firmware_count += 1

                for probe_type in ProbeType:
                    result = simulator.probe(device, probe_type)

                    probe_count += 1
                    total_cost += result.cost

                    if not result.success:
                        missing_count += 1
                        continue

                    success_count += 1

                    if result.value != device.true_features[probe_type]:
                        noise_count += 1

    device_count = sum(class_counts.values())

    return ValidationReport(
        device_count=device_count,
        probe_count=probe_count,
        missing_rate=missing_count / probe_count,
        noise_rate=(noise_count / success_count if success_count else 0.0),
        ambiguous_vendor_rate=(ambiguous_vendor_count / device_count),
        ambiguous_firmware_rate=(ambiguous_firmware_count / device_count),
        total_cost=total_cost,
        class_counts=dict(class_counts),
    )
