import random

from core.domain import DeviceProfile, ProbeResult
from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor
from device_discovery.simulation.profiles import ROLE_DISTRIBUTIONS, VENDOR_FIRMWARE


def sample_category(rng: random.Random, probabilities: dict[str, float]) -> str:
    return rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]


class DeviceSimulator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def create_device(self, device_id: str, vendor: Vendor, role: DeviceRole) -> DeviceProfile:
        distribution = ROLE_DISTRIBUTIONS[role.value]

        features = {
            ProbeType.VENDOR: vendor.value,
            ProbeType.ROUTING: (self.rng.random() < distribution.routing_probability),
            ProbeType.LLDP: self.rng.random() < 0.85,
            ProbeType.SNMP: self.rng.random() < 0.90,
            ProbeType.INTERFACES: sample_category(self.rng, distribution.interface_bands),
            ProbeType.VLAN: (self.rng.random() < distribution.vlan_probability),
            ProbeType.FIRMWARE: VENDOR_FIRMWARE[vendor.value],
        }

        return DeviceProfile(device_id=device_id, vendor=vendor, role=role, true_features=features)

    def probe(self, device: DeviceProfile, probe_type: ProbeType, missing_rate: float = 0.20, noise_rate: float = 0.05) -> ProbeResult:
        if not 0 <= missing_rate <= 1:
            raise ValueError("Invalid missing_rate")
        if not 0 <= noise_rate <= 1:
            raise ValueError("Invalid noise_rate")

        if self.rng.random() < missing_rate:
            return ProbeResult(probe=probe_type, value=None, success=False, cost=1.0)

        value = device.true_features[probe_type]

        if self.rng.random() < noise_rate:
            if isinstance(value, bool):
                value = not value
            elif probe_type == ProbeType.VENDOR:
                alternatives = [v.value for v in Vendor if v.value != value]
                value = self.rng.choice(alternatives)
            elif probe_type == ProbeType.FIRMWARE:
                alternatives = [v for v in VENDOR_FIRMWARE.values() if v != value]
                value = self.rng.choice(alternatives)
            elif probe_type == ProbeType.INTERFACES:
                alternatives = [v for v in ("low", "medium", "high") if v != value]
                value = self.rng.choice(alternatives)

        return ProbeResult(probe=probe_type, value=value, success=True, cost=1.0)
