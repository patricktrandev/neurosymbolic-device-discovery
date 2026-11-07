from dataclasses import dataclass
from typing import TypeAlias

from ..utils.constants.device_constants import ProbeType

ObservationValue: TypeAlias = str | bool | None


@dataclass(frozen=True)
class ProbeResult:
    probe: ProbeType
    value: ObservationValue
    success: bool
    cost: float
