from core.domain.role_distributions import RoleDistribution

# Illustrative simulation assumptions
ROLE_DISTRIBUTIONS = {
    "router": RoleDistribution(routing_probability=0.90, vlan_probability=0.55, interface_bands={"low": 0.55, "medium": 0.35, "high": 0.10}),
    "switch": RoleDistribution(routing_probability=0.30, vlan_probability=0.95, interface_bands={"low": 0.10, "medium": 0.35, "high": 0.55}),
}

VENDOR_FIRMWARE = {"cisco": "ios_family", "juniper": "junos_family", "arista": "eos_family", "hpe": "hpe_os_family"}
