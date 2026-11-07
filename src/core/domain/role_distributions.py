from dataclasses import dataclass


@dataclass(frozen=True)
class RoleDistribution:
    routing_probability: float
    vlan_probability: float
    interface_bands: dict[str, float]
