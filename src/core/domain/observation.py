from dataclasses import dataclass, field
from typing import TypeAlias

from core.utils.constants.device_constants import ProbeType

ProbeValue: TypeAlias = bool | int | str | None


@dataclass(frozen=True)
class Observation:
    device_id: str
    evidence: dict[ProbeType, ProbeValue] = field(default_factory=dict)

    def add_evidence(self, probe: ProbeType, value: ProbeValue) -> None:
        if value is not None:
            self.evidence[probe] = value

    def has_evidence(self, probe: ProbeType) -> bool:
        return probe in self.evidence and self.evidence[probe] is not None
