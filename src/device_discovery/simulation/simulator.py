import random

from core.domain import DeviceProfile, ProbeResult
from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor
from device_discovery.simulation.probe_costs import HETEROGENEOUS_COSTS, UNIFORM_COSTS
from device_discovery.simulation.profiles import LLDP_ENABLED_PROBABILITY, ROLE_DISTRIBUTIONS, SNMP_AVAILABLE_PROBABILITY, VENDOR_FIRMWARE_DISTRIBUTIONS, VENDOR_HINT_UNKNOWN_PROBABILITY
from device_discovery.simulation.scenarios import ScenarioConfig, get_scenario


def sample_category(rng: random.Random, probabilities: dict[str, float]) -> str:
    return rng.choices(population=list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]


class DeviceSimulator:
    def __init__(self, seed: int = 42, scenario: str = "moderate", cost_mode: str = "heterogeneous"):
        self.rng = random.Random(seed)
        self.scenario: ScenarioConfig = get_scenario(scenario)

        if cost_mode == "uniform":
            self.probe_costs = UNIFORM_COSTS.copy()
        elif cost_mode == "heterogeneous":
            self.probe_costs = HETEROGENEOUS_COSTS.copy()
        else:
            raise ValueError(f"Unknown cost mode: {cost_mode}")

    def create_device(self, device_id: str, vendor: Vendor, role: DeviceRole) -> DeviceProfile:
        distribution = ROLE_DISTRIBUTIONS[role]

        features = {
            ProbeType.VENDOR: self._sample_vendor_hint(vendor),
            ProbeType.ROUTING: (self.rng.random() < distribution.routing_probability),
            ProbeType.LLDP: self.rng.random() < LLDP_ENABLED_PROBABILITY,
            ProbeType.SNMP: self.rng.random() < SNMP_AVAILABLE_PROBABILITY,
            ProbeType.INTERFACES: sample_category(self.rng, distribution.interface_bands),
            ProbeType.VLAN: (self.rng.random() < distribution.vlan_probability),
            ProbeType.FIRMWARE: sample_category(self.rng, VENDOR_FIRMWARE_DISTRIBUTIONS[vendor]),
        }

        return DeviceProfile(device_id=device_id, vendor=vendor, role=role, true_features=features)

    def _sample_vendor_hint(self, vendor: Vendor) -> str:
        if self.rng.random() < VENDOR_HINT_UNKNOWN_PROBABILITY:
            return "unknown"

        return vendor.value

    def probe(self, device: DeviceProfile, probe_type: ProbeType) -> ProbeResult:
        cost = self.probe_costs[probe_type]

        if self.rng.random() < self.scenario.missing_rate:
            return ProbeResult(probe=probe_type, value=None, success=False, cost=cost)

        value = device.true_features[probe_type]

        if self.rng.random() < self.scenario.noise_rate:
            value = self._apply_noise(probe_type, value)

        return ProbeResult(probe=probe_type, value=value, success=True, cost=cost)

    def _apply_noise(self, probe_type: ProbeType, value: bool | int | str | None) -> bool | int | str | None:
        if isinstance(value, bool):
            return not value

        if probe_type == ProbeType.VENDOR:
            options = [vendor.value for vendor in Vendor] + ["unknown"]

        elif probe_type == ProbeType.FIRMWARE:
            options = sorted({firmware for distribution in VENDOR_FIRMWARE_DISTRIBUTIONS.values() for firmware in distribution})

        elif probe_type == ProbeType.INTERFACES:
            options = ["low", "medium", "high"]

        else:
            return value

        alternatives = [option for option in options if option != value]

        if not alternatives:
            return value

        return self.rng.choice(alternatives)
