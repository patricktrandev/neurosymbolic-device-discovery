import argparse

from data.builder import build_device_splits
from device_discovery.data.materializer import materialize_observations


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic device discovery data")

    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/generated")

    args = parser.parse_args()

    splits = build_device_splits(samples_per_class=args.samples_per_class, seed=args.seed)

    materialize_observations(splits=splits, output_dir=args.output, seed=args.seed)

    print("Dataset generation completed.")

    for name, devices in splits.items():
        print(f"{name}: {len(devices)} devices")


if __name__ == "__main__":
    main()
