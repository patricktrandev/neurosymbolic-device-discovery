from dataclasses import FrozenInstanceError

import pytest

from core.domain import DeviceProfile, Observation, ProbeResult
from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor


def test_device_profile_label():
    device = DeviceProfile(device_id="device-001", vendor=Vendor.CISCO, role=DeviceRole.ROUTER, true_features={ProbeType.VENDOR: "cisco", ProbeType.ROUTING: True})

    assert device.label == "cisco_router"


def test_device_profile_is_frozen():
    device = DeviceProfile(device_id="device-001", vendor=Vendor.CISCO, role=DeviceRole.ROUTER, true_features={})

    with pytest.raises(FrozenInstanceError):
        setattr(device, "vendor", Vendor.JUNIPER)

    assert device.vendor == Vendor.CISCO


def test_observation_starts_empty():
    observation = Observation(device_id="device-001")

    assert observation.evidence == {}


def test_observation_add_evidence():
    observation = Observation(device_id="device-001")

    observation.add_evidence(ProbeType.ROUTING, True)

    assert observation.has_evidence(ProbeType.ROUTING)

    assert observation.evidence[ProbeType.ROUTING] is True


def test_none_is_not_added_as_evidence():
    observation = Observation(device_id="device-001")

    observation.add_evidence(ProbeType.FIRMWARE, None)

    assert not observation.has_evidence(ProbeType.FIRMWARE)


def test_observation_has_no_ground_truth():
    observation = Observation(device_id="device-001")

    assert not hasattr(observation, "vendor")
    assert not hasattr(observation, "role")
    assert not hasattr(observation, "label")
    assert not hasattr(observation, "true_features")


def test_successful_probe():
    result = ProbeResult(probe=ProbeType.VENDOR, value="cisco", success=True, cost=1.0)

    assert result.success is True
    assert result.value == "cisco"
    assert result.cost == 1.0


def test_failed_probe():
    result = ProbeResult(probe=ProbeType.SNMP, value=None, success=False, cost=3.0)

    assert result.success is False
    assert result.value is None
    assert result.cost == 3.0


def test_negative_probe_cost_rejected():
    with pytest.raises(ValueError):
        ProbeResult(probe=ProbeType.VENDOR, value="cisco", success=True, cost=-1.0)


def test_failed_probe_cannot_expose_value():
    with pytest.raises(ValueError):
        ProbeResult(probe=ProbeType.VENDOR, value="cisco", success=False, cost=1.0)


def test_observations_have_independent_evidence():
    first = Observation(device_id="device-001")
    second = Observation(device_id="device-002")

    first.add_evidence(ProbeType.VENDOR, "cisco")

    assert second.evidence == {}
