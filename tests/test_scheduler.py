import unittest
from itertools import product

from edge_rollout.scheduler import Device, Fleet, Release, Rollout, plan, validate_and_score


class SchedulerTests(unittest.TestCase):
    def test_score_counts_install_day_and_replaces_quality(self):
        fleet = Fleet(
            4, 6, {"east": 5, "west": 5},
            (Device("a", "east", 3, 2), Device("b", "west", 2, 1)),
            (Release("first", 0, 2, 6), Release("older", 2, 3, 2)),
        )
        actions = (
            Rollout(2, "a", "older"), Rollout(1, "b", "first"), Rollout(0, "a", "first"),
        )

        score = validate_and_score(fleet, actions)

        self.assertEqual(tuple(score.daily_utility), (18, 30, 18, 18))
        self.assertEqual(score.utility, 84)
        self.assertEqual(score.megabytes, 7)
        self.assertEqual(score.installed_devices, 2)

        empty = validate_and_score(fleet, ())
        self.assertEqual(tuple(empty.daily_utility), (0, 0, 0, 0))
        self.assertEqual((empty.utility, empty.megabytes, empty.installed_devices), (0, 0, 0))

    def test_evaluator_rejects_invalid_rollouts(self):
        fleet = Fleet(
            3, 7, {"east": 6, "west": 7},
            (Device("a", "east", 3, 2), Device("b", "east", 2, 1), Device("c", "west", 1, 2)),
            (Release("base", 0, 4, 5), Release("mid", 1, 3, 8), Release("late", 2, 4, 10)),
        )
        cases = {
            "day outside horizon": (Rollout(3, "a", "base"),),
            "unknown device": (Rollout(0, "missing", "base"),),
            "unknown release": (Rollout(0, "a", "missing"),),
            "before release": (Rollout(1, "c", "late"),),
            "same device and day": (Rollout(1, "c", "base"), Rollout(1, "c", "mid")),
            "repeated device version": (Rollout(0, "a", "base"), Rollout(1, "a", "base")),
            "device quota": (Rollout(0, "b", "base"), Rollout(1, "b", "mid")),
            "site budget": (Rollout(1, "a", "base"), Rollout(1, "b", "mid")),
            "global budget": (Rollout(0, "a", "base"), Rollout(0, "c", "base")),
        }
        for label, actions in cases.items():
            with self.subTest(rule=label), self.assertRaises(ValueError):
                validate_and_score(fleet, actions)

    def test_strategies_respect_limits_and_can_reach_tiny_optimum(self):
        fleet = Fleet(
            3, 4, {"north": 4},
            (Device("low", "north", 1, 1), Device("high", "north", 4, 2)),
            (Release("early", 0, 4, 2), Release("better", 1, 4, 5)),
        )
        optimum = self.brute_utility(fleet)
        self.assertEqual(optimum, 53)
        quality_by_release = {release.id: release.quality for release in fleet.releases}

        for strategy in ("fifo", "gain", "bandwidth"):
            with self.subTest(strategy=strategy):
                actions = plan(fleet, strategy)
                self.assertIsInstance(actions, tuple)
                self.assertEqual(plan(fleet, strategy), actions)
                score = validate_and_score(fleet, actions)
                self.assertGreater(score.utility, 0)
                self.assertLessEqual(score.utility, optimum)
                if strategy == "gain":
                    self.assertEqual(score.utility, optimum)

                installed = {}
                for action in sorted(actions, key=lambda rollout: rollout.day):
                    quality = quality_by_release[action.release_id]
                    self.assertGreater(quality, installed.get(action.device_id, 0))
                    installed[action.device_id] = quality

    @staticmethod
    def brute_utility(fleet):
        # Each release consumes the full daily bandwidth in this fixture.
        choices = (None, *product(fleet.devices, fleet.releases))
        best = 0
        for days in product(choices, repeat=fleet.days):
            history = {device.id: set() for device in fleet.devices}
            quality = {device.id: 0 for device in fleet.devices}
            utility = 0
            for day, choice in enumerate(days):
                if choice is not None:
                    device, release = choice
                    seen = history[device.id]
                    if day < release.available_day or release.id in seen or len(seen) == device.update_budget:
                        break
                    seen.add(release.id)
                    quality[device.id] = release.quality
                utility += sum(device.priority * quality[device.id] for device in fleet.devices)
            else:
                best = max(best, utility)
        return best

    def test_strategies_skip_release_that_cannot_fit_site(self):
        fleet = Fleet(
            2, 2, {"small": 1}, (Device("d", "small", 3, 2),), (Release("large", 0, 2, 5),),
        )
        for strategy in ("fifo", "gain", "bandwidth"):
            with self.subTest(strategy=strategy):
                self.assertEqual(plan(fleet, strategy), ())


if __name__ == "__main__":
    unittest.main()
