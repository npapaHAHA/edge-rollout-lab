import argparse
from pathlib import Path

from edge_rollout.experiment import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark incident models and edge rollouts.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 67, 89, 113])
    parser.add_argument("--devices", type=int, default=180)
    parser.add_argument("--days", type=int, default=36)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.devices < 1 or args.days < 6:
        parser.error("use at least 1 device and 6 days")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")

    report = run_benchmark(args.seeds, args.devices, args.days, args.output)
    print(f"Synthetic benchmark written to {args.output.resolve()}")
    for key, stats in report["summary"]["rollouts"].items():
        print(f"{key:>9}  {stats['mean_utility'] / 1000:>10,.1f} weighted accuracy-days")


if __name__ == "__main__":
    main()
