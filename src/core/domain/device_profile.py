from dataclasses import dataclass
from typing import TypeAlias

from ..utils.constants.device_constants import DeviceRole, ProbeType, Vendor

ProbeValue: TypeAlias = bool | int | str | None


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    vendor: Vendor
    role: DeviceRole
    true_features: dict[ProbeType, ProbeValue]

    @property
    def label(self) -> str:
        return f"{self.vendor.value}_{self.role.value}"
