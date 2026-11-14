from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from core.domain import Observation
from core.utils.constants.device_constants import ProbeType


class PerturbationKind(str, Enum):
    MISSING = "missing"
    NOISE = "noise"


@dataclass(frozen=True)
class PerturbationConfig:
    kind: PerturbationKind
    rate: float
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be between 0 and 1")


# Values verified against the generated training dataset.
# Noise replaces an existing value with a DIFFERENT valid value.
PROBE_VALUES: dict[ProbeType, tuple[str | bool, ...]] = {
    ProbeType.VENDOR: ("cisco", "juniper", "arista", "hpe", "unknown"),
    ProbeType.ROUTING: (True, False),
    ProbeType.LLDP: (True, False),
    ProbeType.SNMP: (True, False),
    ProbeType.INTERFACES: ("low", "medium", "high"),
    ProbeType.VLAN: (True, False),
    ProbeType.FIRMWARE: ("network_os_a", "network_os_b", "network_os_c", "network_os_d", "generic_network_os"),
}


def perturb_observations(observations: list[Observation], config: PerturbationConfig) -> list[Observation]:
    """Perturb visible evidence without changing the originals.

    Missing:
        Independently remove each observed probe with probability
        config.rate.

    Noise:
        Independently replace each observed probe with probability
        config.rate, choosing a different value from its domain.

    No new probe is introduced. Device IDs remain unchanged.
    """
    rng = random.Random(config.seed)
    results: list[Observation] = []

    for observation in observations:
        evidence = dict(observation.evidence)

        for probe, original_value in observation.evidence.items():
            if rng.random() >= config.rate:
                continue

            if config.kind == PerturbationKind.MISSING:
                evidence.pop(probe, None)

            elif config.kind == PerturbationKind.NOISE:
                valid_values = PROBE_VALUES.get(probe)

                if valid_values is None:
                    raise ValueError(f"No noise domain defined for {probe}")

                alternatives = [value for value in valid_values if value != original_value]

                if not alternatives:
                    raise ValueError(f"No alternative value for {probe}: " f"{original_value!r}")

                evidence[probe] = rng.choice(alternatives)

            else:
                raise ValueError(f"Unsupported perturbation: {config.kind}")

        results.append(Observation(device_id=observation.device_id, evidence=evidence))

    return results
