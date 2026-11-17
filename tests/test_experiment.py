from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from edge_rollout.experiment import MODEL_IDS, STRATEGIES, run_benchmark


def metrics(accuracies):
    return {
        model: {'accuracy': accuracy, 'episode_exact_match': accuracy / 2}
        for model, accuracy in zip(MODEL_IDS, accuracies, strict=True)
    }


class BenchmarkTests(TestCase):
    def test_planning_depends_on_validation_quality(self):
        validation = metrics([0.55, 0.70, 0.95])
        with TemporaryDirectory() as folder:
            original, shifted = Path(folder) / 'original', Path(folder) / 'shifted'
            with patch('edge_rollout.experiment.evaluate') as evaluate:
                evaluate.return_value = {'validation': validation, 'test': metrics([0.60, 0.72, 0.92])}
                left = run_benchmark([17], 12, 18, original)
                evaluate.return_value = {'validation': validation, 'test': metrics([0.30, 0.36, 0.46])}
                right = run_benchmark([17], 12, 18, shifted)
            self.assertEqual((original / 'schedule.csv').read_bytes(), (shifted / 'schedule.csv').read_bytes())
            self.assertTrue((original / 'benchmark.png').is_file())
            for strategy in STRATEGIES:
                first = left['runs'][0]['rollouts'][strategy]
                second = right['runs'][0]['rollouts'][strategy]
                self.assertEqual(first['planned_utility'], second['planned_utility'])
                self.assertGreater(first['utility'], second['utility'])
                self.assertEqual(first['utility'], sum(first['daily_utility']))
