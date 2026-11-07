from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor
from device_discovery.simulation.simulator import DeviceSimulator


def create_test_device(simulator):
    return simulator.create_device(device_id="test-device-1", vendor=Vendor.CISCO, role=DeviceRole.ROUTER)


def test_same_seed_produces_same_devices():
    first = DeviceSimulator(seed=42)
    second = DeviceSimulator(seed=42)

    device_a = create_test_device(first)
    device_b = create_test_device(second)

    assert device_a == device_b


def test_missing_rate_one_returns_none():
    simulator = DeviceSimulator(seed=42)
    device = create_test_device(simulator)

    result = simulator.probe(device=device, probe_type=ProbeType.VENDOR, missing_rate=1.0, noise_rate=0.0)

    assert result.value is None
    assert result.success is False


def test_noise_rate_zero_preserves_value():
    simulator = DeviceSimulator(seed=42)
    device = create_test_device(simulator)

    for probe_type in ProbeType:
        result = simulator.probe(device=device, probe_type=probe_type, missing_rate=0.0, noise_rate=0.0)

        assert result.success is True
        assert result.value == device.true_features[probe_type]
