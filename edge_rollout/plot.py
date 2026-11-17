"""Render a static companion for the machine-readable benchmark."""

from itertools import accumulate
from pathlib import Path
from statistics import mean

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter


COLORS = ('#3A536B', '#A85312', '#006C59')
INK = '#172A35'
MUTED = '#4B6270'


def save_benchmark_figure(report: dict, path: Path) -> None:
    runs = report['runs']
    figure = Figure(figsize=(11.6, 4.75), dpi=160, facecolor='#F4F6F4')
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2, gridspec_kw={'width_ratios': [0.9, 1.25]})
    figure.subplots_adjust(left=.078, right=.96, bottom=.185, top=.745, wspace=.37)
    figure.text(.079, .915, 'Edge Rollout Lab', color=INK, fontsize=23, weight='bold')
    count = len(runs)
    figure.text(.081, .842, f'Synthetic telemetry and fleet benchmark  /  {count} independent seeds  /  '
                f'{report["device_count"]} devices per fleet', color=MUTED, fontsize=10)

    for axis in (left, right):
        axis.set_facecolor('#FFFFFF')
        axis.grid(axis='y', color='#E1E7E4', linewidth=.8)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.spines['bottom'].set_color('#ABB8B7')
        axis.tick_params(axis='both', length=0, colors=MUTED, labelsize=9)

    models = ('global', 'local', 'rule_bank')
    for run in runs:
        values = [run['models']['test'][name]['episode_exact_match'] for name in models]
        left.plot(range(len(models)), values, color='#97A8AF', linewidth=.75, alpha=.7, zorder=2)
    for i, (name, color) in enumerate(zip(models, COLORS, strict=True)):
        values = [run['models']['test'][name]['episode_exact_match'] for run in runs]
        left.scatter([i] * count, values, color=color, s=18, alpha=.66, zorder=3)
        left.scatter(i, mean(values), marker='D', s=56, color=color, edgecolor='white', zorder=4)
    left.set(title='Model quality on unseen sites', ylabel='Episode exact match', xlim=(-.38, 2.38), ylim=(0, 1))
    left.set_xticks(range(3), ['Global', 'Local', 'Policy bank'])
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    labels = ('First-in, first-out', 'Gain first', 'Bandwidth-aware')
    patterns = (':', '--', '-')
    for strategy, label, color, pattern in zip(('fifo', 'gain', 'bandwidth'), labels, COLORS, patterns, strict=True):
        curves = [tuple(accumulate(run['rollouts'][strategy]['daily_utility'])) for run in runs]
        average = [0.0, *(mean(day) / 1000 for day in zip(*curves, strict=True))]
        right.plot(range(len(average)), average, label=label, color=color, linestyle=pattern, linewidth=2)
    right.set(title='Cumulative fleet utility', xlabel='Days elapsed', ylabel='Weighted accuracy-days',
              xlim=(0, report['days']), ylim=(0, None))
    right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    right.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK)
    for axis in (left, right):
        axis.title.set_fontsize(12)
        axis.title.set_fontweight('bold')
        axis.title.set_color(INK)
        axis.xaxis.label.set_color(MUTED)
        axis.yaxis.label.set_color(MUTED)
    figure.text(.08, .049, 'Left: each line is one paired seed. Diamonds show means. '
                'Right: means across the same seeds.', color=MUTED, fontsize=9)
    figure.savefig(path, facecolor=figure.get_facecolor(), metadata={'Title': 'Edge Rollout Lab benchmark'})
