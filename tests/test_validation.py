import pytest

from device_discovery.simulation.validation import validate_simulator


@pytest.fixture(scope="module")
def report():
    return validate_simulator(samples_per_class=500, seed=42, scenario="moderate", cost_mode="heterogeneous")


def test_device_count(report):
    assert report.device_count == 4000


def test_probe_count(report):
    assert report.probe_count == 4000 * 7


def test_class_balance(report):
    assert len(report.class_counts) == 8

    assert all(count == 500 for count in report.class_counts.values())


def test_missing_rate(report):
    assert report.missing_rate == pytest.approx(0.20, abs=0.02)


def test_noise_rate(report):
    assert report.noise_rate == pytest.approx(0.05, abs=0.02)


def test_vendor_ambiguity(report):
    assert report.ambiguous_vendor_rate == pytest.approx(0.25, abs=0.04)


def test_firmware_ambiguity(report):
    assert report.ambiguous_firmware_rate == pytest.approx(0.30, abs=0.04)


def test_total_cost(report):
    assert report.total_cost == pytest.approx(4000 * 15.0)


def test_deterministic_replay():
    first = validate_simulator(samples_per_class=20, seed=42)

    second = validate_simulator(samples_per_class=20, seed=42)

    assert first == second


def test_invalid_sample_count():
    with pytest.raises(ValueError):
        validate_simulator(samples_per_class=0)


@pytest.mark.parametrize("scenario,expected_missing,expected_noise", [("easy", 0.10, 0.02), ("moderate", 0.20, 0.05), ("hard", 0.35, 0.12)])
def test_scenario_rates(scenario, expected_missing, expected_noise):
    result = validate_simulator(samples_per_class=500, seed=42, scenario=scenario)

    assert result.missing_rate == pytest.approx(expected_missing, abs=0.02)

    assert result.noise_rate == pytest.approx(expected_noise, abs=0.02)
