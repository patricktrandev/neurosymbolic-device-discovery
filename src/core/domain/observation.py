from dataclasses import dataclass
from typing import TypeAlias

from ..utils.constants.device_constants import ProbeType

ObservationValue: TypeAlias = str | bool | None


@dataclass(frozen=True)
class Observation:
    device_id: str
    evidence: dict[ProbeType, ObservationValue]
