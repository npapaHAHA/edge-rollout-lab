"""Run paired benchmarks with validation-driven plans and held-out scoring."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from statistics import mean

from edge_rollout.learning import evaluate
from edge_rollout.scheduler import Device, Fleet, Release, Rollout, plan, validate_and_score


MODEL_IDS = ("global", "local", "rule_bank")
STRATEGIES = ("fifo", "gain", "bandwidth")


def make_fleet(seed: int, device_count: int, days: int, validation: dict) -> Fleet:
    """Use accuracy estimates from validation as release quality."""
    rng = Random(seed)
    site_mb = {"harbor": 48, "ridge": 35, "terminal": 42, "plant": 55, "remote": 29}
    devices = tuple(
        Device(f"edge-{i + 1:03d}", rng.choice(tuple(site_mb)), rng.randint(1, 6),
               1 if rng.random() < 0.72 else 2)
        for i in range(device_count)
    )
    releases = tuple(
        Release(model, day, size, round(validation[model]["accuracy"] * 1000))
        for model, day, size in zip(MODEL_IDS, (0, days // 6, days // 3), (9, 15, 21), strict=True)
    )
    return Fleet(days, 115, site_mb, devices, releases)


def scoring_fleet(fleet: Fleet, metrics: dict) -> Fleet:
    return replace(fleet, releases=tuple(
        replace(release, quality=round(metrics[release.id]["accuracy"] * 1000))
        for release in fleet.releases
    ))


def write_schedule(path: Path, fleet: Fleet, actions: tuple[Rollout, ...]) -> None:
    devices = {device.id: device for device in fleet.devices}
    releases = {release.id: release for release in fleet.releases}
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('day', 'device_id', 'site', 'release_id', 'size_mb'))
        for action in actions:
            writer.writerow((action.day, action.device_id, devices[action.device_id].site,
                             action.release_id, releases[action.release_id].size_mb))


def run_benchmark(seeds: list[int], device_count: int, days: int, output: Path) -> dict:
    from edge_rollout.plot import save_benchmark_figure

    runs = []
    sample = None
    for seed in seeds:
        metrics = evaluate(seed)
        fleet = make_fleet(seed, device_count, days, metrics['validation'])
        replay_fleet = scoring_fleet(fleet, metrics['test'])
        results = {}
        for strategy in STRATEGIES:
            actions = plan(fleet, strategy)
            forecast = validate_and_score(fleet, actions)
            realized = validate_and_score(replay_fleet, actions)
            results[strategy] = {**asdict(realized), 'planned_utility': forecast.utility}
            if seed == seeds[0] and strategy == 'bandwidth':
                sample = fleet, actions
        runs.append({
            'seed': seed,
            'models': metrics,
            'releases': [asdict(release) for release in fleet.releases],
            'rollouts': results,
        })

    model_summary = {
        model: {
            metric: mean(run['models']['test'][model][metric] for run in runs)
            for metric in ('accuracy', 'episode_exact_match')
        }
        for model in MODEL_IDS
    }
    rollout_summary = {
        strategy: {
            'mean_utility': mean(run['rollouts'][strategy]['utility'] for run in runs),
            'mean_megabytes': mean(run['rollouts'][strategy]['megabytes'] for run in runs),
            'mean_installed_devices': mean(run['rollouts'][strategy]['installed_devices'] for run in runs),
        }
        for strategy in STRATEGIES
    }
    report = {
        'device_count': device_count,
        'days': days,
        'seeds': seeds,
        'utility_unit': 'weighted milli-accuracy-days',
        'sample_schedule_seed': seeds[0],
        'runs': runs,
        'summary': {'models': model_summary, 'rollouts': rollout_summary},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    assert sample is not None
    write_schedule(output / 'schedule.csv', *sample)
    save_benchmark_figure(report, output / 'benchmark.png')
    return report
