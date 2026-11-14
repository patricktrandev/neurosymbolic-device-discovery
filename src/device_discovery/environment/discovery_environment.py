from copy import deepcopy

from core.domain import Observation, ProbeResult
from core.utils.constants.device_constants import ProbeType


class DiscoveryEnvironment:
    """
    Controls which evidence is visible to the learner.

    Ground truth is never stored in this class.
    """

    def __init__(self, initial_observations: dict[str, dict[ProbeType, bool | int | str | None]], probe_outcomes: dict[str, dict[str, dict]], initial_attempts: dict[str, list[ProbeType]]):
        self._visible = deepcopy(initial_observations)
        self._outcomes = deepcopy(probe_outcomes)

        self._queried = {device_id: set(initial_attempts[device_id]) for device_id in initial_observations}

        self._total_cost = {device_id: 0.0 for device_id in initial_observations}

    def observe(self, device_id: str) -> Observation:
        if device_id not in self._visible:
            raise KeyError(f"Unknown device: {device_id}")

        return Observation(device_id=device_id, evidence=deepcopy(self._visible[device_id]))

    def query(self, device_id: str, probe_type: ProbeType) -> ProbeResult:
        if device_id not in self._visible:
            raise KeyError(f"Unknown device: {device_id}")

        if probe_type in self._queried[device_id]:
            raise ValueError(f"Probe already attempted: {probe_type.value}")

        outcome = self._outcomes[device_id][probe_type.value]

        result = ProbeResult(probe=probe_type, value=outcome["value"], success=outcome["success"], cost=float(outcome["cost"]))

        self._queried[device_id].add(probe_type)
        self._total_cost[device_id] += result.cost

        if result.success:
            self._visible[device_id][probe_type] = result.value

        return result

    def get_total_cost(self, device_id: str) -> float:
        if device_id not in self._total_cost:
            raise KeyError(f"Unknown device: {device_id}")

        return self._total_cost[device_id]
