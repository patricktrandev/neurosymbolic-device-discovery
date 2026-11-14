from core.domain.role_distributions import RoleDistribution
from core.utils.constants.device_constants import Vendor

# Illustrative simulation assumptions
ROLE_DISTRIBUTIONS = {
    "router": RoleDistribution(routing_probability=0.90, vlan_probability=0.55, interface_bands={"low": 0.55, "medium": 0.35, "high": 0.10}),
    "switch": RoleDistribution(routing_probability=0.30, vlan_probability=0.95, interface_bands={"low": 0.10, "medium": 0.35, "high": 0.55}),
}


# Different vendors can share firmware families.
# These are synthetic categories, not real OS names.
VENDOR_FIRMWARE_DISTRIBUTIONS = {
    Vendor.CISCO: {"network_os_a": 0.70, "generic_network_os": 0.30},
    Vendor.JUNIPER: {"network_os_b": 0.70, "generic_network_os": 0.30},
    Vendor.ARISTA: {"network_os_c": 0.70, "generic_network_os": 0.30},
    Vendor.HPE: {"network_os_d": 0.70, "generic_network_os": 0.30},
}
