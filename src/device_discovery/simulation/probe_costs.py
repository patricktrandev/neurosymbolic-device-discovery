from core.utils.constants.device_constants import ProbeType

# Baseline: all probes have equal cost
UNIFORM_COSTS: dict[ProbeType, float] = {probe: 1.0 for probe in ProbeType}


# Different probes have different costs
HETEROGENEOUS_COSTS: dict[ProbeType, float] = {
    ProbeType.VENDOR: 1.0,
    ProbeType.ROUTING: 1.0,
    ProbeType.LLDP: 2.0,
    ProbeType.SNMP: 3.0,
    ProbeType.INTERFACES: 2.0,
    ProbeType.VLAN: 2.0,
    ProbeType.FIRMWARE: 4.0,
}
