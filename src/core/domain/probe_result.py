from dataclasses import dataclass
from typing import TypeAlias

from ..utils.constants.device_constants import ProbeType

ProbeValue: TypeAlias = bool | int | str | None


@dataclass(frozen=True)
class ProbeResult:
    probe: ProbeType
    value: ProbeValue
    success: bool
    cost: float

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("Probe cost cannot be negative")

        if not self.success and self.value is not None:
            raise ValueError("Failed probe must not expose a value")
