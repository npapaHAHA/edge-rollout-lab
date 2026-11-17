"""Schedule model updates under per-day transfer limits and device quotas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heapify, heappop, heappush


@dataclass(frozen=True)
class Device:
    id: str
    site: str
    priority: int
    update_budget: int


@dataclass(frozen=True)
class Release:
    id: str
    available_day: int
    size_mb: int
    quality: int


@dataclass(frozen=True)
class Fleet:
    days: int
    global_mb: int
    site_mb: dict[str, int]
    devices: tuple[Device, ...]
    releases: tuple[Release, ...]


@dataclass(frozen=True)
class Rollout:
    day: int
    device_id: str
    release_id: str


@dataclass(frozen=True)
class Score:
    utility: int
    megabytes: int
    installed_devices: int
    daily_utility: tuple[int, ...]


def _validate_fleet(fleet: Fleet) -> None:
    if fleet.days < 0 or fleet.global_mb < 0 or any(limit < 0 for limit in fleet.site_mb.values()):
        raise ValueError("planning days and bandwidth limits must be non-negative")
    if len({device.id for device in fleet.devices}) != len(fleet.devices):
        raise ValueError("device IDs must be unique")
    if len({release.id for release in fleet.releases}) != len(fleet.releases):
        raise ValueError("release IDs must be unique")
    for device in fleet.devices:
        if device.site not in fleet.site_mb:
            raise ValueError(f"unknown site {device.site!r} for device {device.id!r}")
        if device.priority < 0 or device.update_budget < 0:
            raise ValueError(f"priority and update budget must be non-negative for {device.id!r}")
    for release in fleet.releases:
        if release.available_day < 0 or release.size_mb < 0 or release.quality < 0:
            raise ValueError(f"release day, size and quality must be non-negative for {release.id!r}")


def validate_and_score(fleet: Fleet, actions: Iterable[Rollout]) -> Score:
    """Replay a schedule by day and raise ValueError on any infeasible update."""
    _validate_fleet(fleet)
    devices = {device.id: device for device in fleet.devices}
    releases = {release.id: release for release in fleet.releases}
    by_day: list[list[Rollout]] = [[] for _ in range(fleet.days)]
    for action in actions:
        if not isinstance(action.day, int) or not 0 <= action.day < fleet.days:
            raise ValueError(f"day outside the planning horizon: {action.day!r}")
        by_day[action.day].append(action)

    installed: dict[str, int] = {}
    versions: set[tuple[str, str]] = set()
    budgets = {device.id: device.update_budget for device in fleet.devices}
    daily_utility = []
    active_utility = megabytes = 0
    for day, scheduled in enumerate(by_day):
        global_used = 0
        site_used: dict[str, int] = {}
        updated: set[str] = set()
        for action in scheduled:
            device = devices.get(action.device_id)
            release = releases.get(action.release_id)
            if device is None or release is None:
                raise ValueError(f"unknown device or release in {action!r}")
            if day < release.available_day:
                raise ValueError(f"release {release.id!r} is unavailable on day {day}")
            if device.id in updated:
                raise ValueError(f"device {device.id!r} can be updated only once on day {day}")
            if (device.id, release.id) in versions:
                raise ValueError(f"duplicate release {release.id!r} for device {device.id!r}")
            if budgets[device.id] == 0:
                raise ValueError(f"update budget exhausted for device {device.id!r}")
            global_used += release.size_mb
            site_used[device.site] = site_used.get(device.site, 0) + release.size_mb
            if global_used > fleet.global_mb:
                raise ValueError(f"global bandwidth exceeded on day {day}")
            if site_used[device.site] > fleet.site_mb[device.site]:
                raise ValueError(f"site {device.site!r} bandwidth exceeded on day {day}")

            previous_quality = installed.get(device.id, 0)
            active_utility += device.priority * (release.quality - previous_quality)
            installed[device.id] = release.quality
            budgets[device.id] -= 1
            updated.add(device.id)
            versions.add((device.id, release.id))
        megabytes += global_used
        daily_utility.append(active_utility)

    return Score(sum(daily_utility), megabytes, len(installed), tuple(daily_utility))


def _fifo(fleet: Fleet) -> tuple[Rollout, ...]:
    releases = sorted(fleet.releases, key=lambda release: release.available_day)
    budgets = {device.id: device.update_budget for device in fleet.devices}
    quality = {device.id: 0 for device in fleet.devices}
    actions = []
    for day in range(fleet.days):
        global_left = fleet.global_mb
        site_left = fleet.site_mb.copy()
        for device in fleet.devices:
            if budgets[device.id] == 0:
                continue
            for release in releases:
                if release.available_day > day:
                    break
                available_mb = min(global_left, site_left[device.site])
                if release.quality <= quality[device.id] or release.size_mb > available_mb:
                    continue
                actions.append(Rollout(day, device.id, release.id))
                quality[device.id] = release.quality
                budgets[device.id] -= 1
                global_left -= release.size_mb
                site_left[device.site] -= release.size_mb
                break
    return tuple(actions)


def _greedy(fleet: Fleet, size_exponent: float) -> tuple[Rollout, ...]:
    global_left = [fleet.global_mb] * fleet.days
    site_left = {site: [limit] * fleet.days for site, limit in fleet.site_mb.items()}
    schedules: list[list[tuple[int, int]]] = [[] for _ in fleet.devices]
    size_factor = [max(1, release.size_mb) ** size_exponent for release in fleet.releases]

    def earliest_offer(device_index: int, release_index: int) -> tuple[int, int, int] | None:
        device = fleet.devices[device_index]
        visits = schedules[device_index]
        if len(visits) >= device.update_budget or device.priority == 0:
            return None

        release = fleet.releases[release_index]
        start = release.available_day
        end = fleet.days
        quality = 0
        position = 0
        # A new release fits between its lower and higher quality neighbors.
        # Its gain ends when the next higher quality release takes over.
        for day, planned_index in visits:
            planned_quality = fleet.releases[planned_index].quality
            if planned_quality == release.quality:
                return None
            if planned_quality > release.quality:
                end = day
                break
            quality = planned_quality
            start = max(start, day + 1)
            position += 1
        if release.quality <= quality:
            return None

        for day in range(start, end):
            if release.size_mb <= min(global_left[day], site_left[device.site][day]):
                gain = (end - day) * device.priority * (release.quality - quality)
                return gain, day, position
        return None

    heap = []
    for device_index, _ in enumerate(fleet.devices):
        for release_index, _ in enumerate(fleet.releases):
            offer = earliest_offer(device_index, release_index)
            if offer is not None:
                gain, day, _ = offer
                heap.append((-gain / size_factor[release_index], day, device_index, release_index, gain))
    heapify(heap)

    while heap:
        _, _, device_index, release_index, previous_gain = heappop(heap)
        offer = earliest_offer(device_index, release_index)
        if offer is None:
            continue
        gain, day, position = offer
        # Inserting updates can only reduce capacity or shorten this benefit window.
        if gain < previous_gain:
            heappush(heap, (-gain / size_factor[release_index], day, device_index, release_index, gain))
            continue
        device = fleet.devices[device_index]
        release = fleet.releases[release_index]
        schedules[device_index].insert(position, (day, release_index))
        global_left[day] -= release.size_mb
        site_left[device.site][day] -= release.size_mb

    actions = [
        Rollout(day, device.id, fleet.releases[release_index].id)
        for device, visits in zip(fleet.devices, schedules)
        for day, release_index in visits
    ]
    return tuple(sorted(actions, key=lambda action: action.day))


def plan(fleet: Fleet, strategy: str) -> tuple[Rollout, ...]:
    """Plan increasing-quality rollouts with FIFO, gain, or bandwidth ranking."""
    _validate_fleet(fleet)
    if strategy == "fifo":
        return _fifo(fleet)
    if strategy == "gain":
        return _greedy(fleet, 0.0)
    if strategy == "bandwidth":
        candidates = (_greedy(fleet, exponent) for exponent in (0.0, 0.5, 1.0))
        return max(candidates, key=lambda actions: validate_and_score(fleet, actions).utility)
    raise ValueError(f"unknown rollout strategy {strategy!r}")
