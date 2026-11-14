from dataclasses import dataclass

from src.core.utils.constants.simulation_constants import ScenarioName


@dataclass(frozen=True)
class ScenarioConfig:
    missing_rate: float
    noise_rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.missing_rate <= 1.0:
            raise ValueError("missing_rate must be between 0 and 1")

        if not 0.0 <= self.noise_rate <= 1.0:
            raise ValueError("noise_rate must be between 0 and 1")


SCENARIOS: dict[ScenarioName, ScenarioConfig] = {
    ScenarioName.EASY: ScenarioConfig(missing_rate=0.10, noise_rate=0.02),
    ScenarioName.MODERATE: ScenarioConfig(missing_rate=0.20, noise_rate=0.05),
    ScenarioName.HARD: ScenarioConfig(missing_rate=0.35, noise_rate=0.12),
}


def get_scenario(name: ScenarioName | str) -> ScenarioConfig:
    try:
        scenario_name = ScenarioName(name)
    except ValueError as exc:
        raise ValueError(f"Unknown scenario: {name}") from exc

    return SCENARIOS[scenario_name]
