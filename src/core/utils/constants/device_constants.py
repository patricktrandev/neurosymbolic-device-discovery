from enum import Enum
from typing import TypeAlias


class Vendor(str, Enum):
    CISCO = "cisco"
    JUNIPER = "juniper"
    ARISTA = "arista"
    HPE = "hpe"


class DeviceRole(str, Enum):
    ROUTER = "router"
    SWITCH = "switch"


class ProbeType(str, Enum):
    VENDOR = "vendor_hint"
    ROUTING = "routing_enabled"
    LLDP = "lldp_enabled"
    SNMP = "snmp_available"
    INTERFACES = "interface_count_band"
    VLAN = "vlan_capable"
    FIRMWARE = "firmware_family"
