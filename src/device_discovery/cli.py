import argparse

from core.utils.constants.device_constants import DeviceRole, ProbeType, Vendor
from device_discovery.data.builder import build_device_splits
from device_discovery.data.materializer import materialize_observations
from device_discovery.simulation.simulator import DeviceSimulator
from device_discovery.simulation.validation import validate_simulator


def run_generation(args: argparse.Namespace) -> None:
    splits = build_device_splits(samples_per_class=args.samples_per_class, seed=args.seed)

    materialize_observations(
        splits=splits,
        output_dir=args.output,
        seed=args.seed,
        scenario=args.scenario,
        cost_mode=args.cost_mode,
        initial_observation_count=args.initial_observations,
        max_initial_observation_count=args.max_initial_observations,
    )

    print("Dataset generation completed.")
    print(f"Output: {args.output}")

    for name, devices in splits.items():
        print(f"{name}: {len(devices)} devices")


def run_simulation(args: argparse.Namespace) -> None:
    simulator = DeviceSimulator(seed=args.seed, scenario=args.scenario, cost_mode=args.cost_mode)

    device = simulator.create_device(device_id="device-001", vendor=Vendor(args.vendor), role=DeviceRole(args.role))

    print(f"Scenario: {args.scenario}")
    print(f"Cost mode: {args.cost_mode}")
    print(f"Device: {device.device_id}")
    print(f"Ground truth: {device.label}")
    print("-" * 65)

    total_cost = 0.0

    for probe_type in ProbeType:
        result = simulator.probe(device, probe_type)
        total_cost += result.cost

        print(f"{probe_type.value:22} " f"success={str(result.success):5} " f"value={str(result.value):20} " f"cost={result.cost:.1f}")

    print("-" * 65)
    print(f"Total cost: {total_cost:.1f}")


def run_validation(args: argparse.Namespace) -> None:
    report = validate_simulator(samples_per_class=args.samples_per_class, seed=args.seed, scenario=args.scenario, cost_mode=args.cost_mode)

    print("=" * 55)
    print("SIMULATOR VALIDATION REPORT")
    print("=" * 55)

    print(f"Scenario: {args.scenario}")
    print(f"Cost mode: {args.cost_mode}")
    print(f"Seed: {args.seed}")
    print(f"Samples per class: {args.samples_per_class}")

    print("-" * 55)

    print(f"Total devices: {report.device_count}")
    print(f"Total probes: {report.probe_count}")
    print(f"Missing rate: {report.missing_rate:.2%}")
    print(f"Noise rate: {report.noise_rate:.2%}")

    print(f"Ambiguous vendor rate: " f"{report.ambiguous_vendor_rate:.2%}")
    print(f"Ambiguous firmware rate: " f"{report.ambiguous_firmware_rate:.2%}")

    print(f"Total probe cost: {report.total_cost:.1f}")

    print("-" * 55)
    print("CLASS DISTRIBUTION")

    for label, count in sorted(report.class_counts.items()):
        print(f"{label:20} {count}")

    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic network device simulation and validation")

    parser.add_argument("--mode", choices=["simulate", "validate", "generate"], default="simulate", help="Run a single-device simulation or bulk validation")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--scenario", choices=["easy", "moderate", "hard"], default="moderate")

    parser.add_argument("--cost-mode", choices=["uniform", "heterogeneous"], default="heterogeneous")

    parser.add_argument("--vendor", choices=[vendor.value for vendor in Vendor], default="cisco")

    parser.add_argument("--role", choices=[role.value for role in DeviceRole], default="router")

    parser.add_argument("--samples-per-class", type=int, default=500, help="Number of devices per class in validation mode")
    parser.add_argument("--output", default="data/generated/moderate")

    parser.add_argument("--initial-observations", type=int, default=2, help="Initial visible probes per device (minimum when --max-initial-observations is set)")
    parser.add_argument("--max-initial-observations", type=int, default=None, help="If set, draw the initial probe count uniformly from [--initial-observations, this value]")
    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulation(args)

    elif args.mode == "validate":
        run_validation(args)
    elif args.mode == "generate":
        run_generation(args)


if __name__ == "__main__":
    main()
