"""Render the repository overview from approved seed summaries, without field data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_paths import safe_output_directory
INK = '#182d44'
MUTED = '#536579'
TEAL = '#087f86'
BLUE = '#3868a6'
EDGE = '#d9e1ea'


def box(ax, x: float, y: float, width: float, height: float,
        color: str = 'white', edge: str = EDGE, padding: float = .008) -> None:
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                              boxstyle=f'round,pad={padding},rounding_size=0.012',
                              facecolor=color, edgecolor=edge, linewidth=1.1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'outputs/overview')
    args = parser.parse_args()
    output = safe_output_directory(ROOT, args.out_dir)
    with (ROOT/'tables/primary_metric_per_seed.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    names = ['TA-ANP-full', 'TA-ANP-noK']
    values = {}
    for name in names:
        subset = sorted((r for r in rows if r['family'] == name), key=lambda r: int(r['seed']))
        assert [int(r['seed']) for r in subset] == list(range(10)), 'Expected ten archived seed IDs'
        values[name] = np.array([float(r['future_plume_ssim']) for r in subset])
        assert np.isfinite(values[name]).all()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'ta-anp-research-overview',
                         'svg.fonttype': 'path', 'font.size': 12})
    fig = plt.figure(figsize=(14, 10), facecolor='#f6f8fb')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis('off')
    canvas.text(.045, .952, 'GROUNDWATER PLUME FORECASTING', color=TEAL,
                fontsize=12, fontweight='bold', va='center')
    canvas.text(.045, .906, 'Learning from observed plume history', color=INK,
                fontsize=27, fontweight='bold', va='center')
    canvas.text(.045, .862, 'Temporal Attentive Neural Processes | Loc K. Nguyen and Binh P. Nguyen',
                color=MUTED, fontsize=13, va='center')

    box(canvas, .05, .604, .9, .21)
    canvas.text(.07, .777, 'A   Forecasting protocol', color=INK, fontsize=14, fontweight='bold')
    start, step, width = .073, .034, .027
    for index in range(25):
        color = TEAL if index < 15 else BLUE
        box(canvas, start+step*index, .699, width, .025, color, color, padding=.001)
    canvas.text(.07, .741, 'Observed context: C[0] to C[14]', color=TEAL, fontsize=12, fontweight='bold')
    canvas.text(.582, .741, 'Forecast targets: C[15] to C[24]', color=BLUE, fontsize=12, fontweight='bold')
    canvas.text(.07, .667, '15 snapshots | days 1–75', color=MUTED, fontsize=12)
    canvas.text(.582, .667, '10 snapshots | days 90–365', color=MUTED, fontsize=12)
    canvas.text(.07, .628, 'Future fields supervise training realizations; they are excluded from evaluated inference inputs.',
                color=MUTED, fontsize=10.5)

    box(canvas, .05, .345, .9, .218)
    canvas.text(.07, .524, 'B   Model at inference', color=INK, fontsize=14, fontweight='bold')
    blocks = [(.076, 'History encoding', 'Frames, times and\nhistory-derived flow'),
              (.382, 'Attention + latent prior', 'Query-time attention\nDeterministic prior mean'),
              (.688, 'Residual decoding', 'Forecast residual added\nto the last observed frame')]
    for x, title, detail in blocks:
        box(canvas, x, .406, .234, .079, '#eef4f8')
        canvas.text(x+.117, .464, title, ha='center', color=INK, fontsize=12, fontweight='bold')
        canvas.text(x+.117, .422, detail, ha='center', color=MUTED, fontsize=10.5, linespacing=1.5)
    for x in [.316, .622]:
        canvas.add_patch(FancyArrowPatch((x, .445), (x+.056, .445),
                                        arrowstyle='-|>', mutation_scale=16, linewidth=1.3, color=MUTED))
    canvas.text(.07, .369, 'noK: history + query time.  full: the same inputs + explicit conductivity K.',
                color=MUTED, fontsize=11)

    box(canvas, .05, .053, .52, .249)
    canvas.text(.07, .264, 'C   Within-family comparison', color=INK, fontsize=14, fontweight='bold')
    ax = fig.add_axes([.207, .133, .326, .102], facecolor='white')
    for seed in range(10):
        ax.plot([values[names[0]][seed], values[names[1]][seed]], [0, 1], color='#b9c5d1', alpha=.65, lw=.9)
    for y, name, color in [(0, names[0], BLUE), (1, names[1], TEAL)]:
        ax.scatter(values[name], np.full(10, y), s=29, color=color, edgecolors='white', linewidths=.6, zorder=3)
        ax.errorbar(values[name].mean(), y+.17, xerr=values[name].std(ddof=1), fmt='D',
                    color=INK, markersize=4, capsize=3, elinewidth=1.5, zorder=4)
        canvas.text(.071, .153+y*.0567, name, color=color, fontsize=11, fontweight='bold', va='center')
    ax.set_xlim(.78, .85)
    ax.set_ylim(-.35, 1.45)
    ax.set_yticks([])
    ax.set_xticks([.78, .80, .82, .84])
    ax.tick_params(axis='x', labelsize=10, colors=MUTED)
    ax.grid(axis='x', color='#edf0f4')
    ax.set_xlabel('Future plume SSIM (higher is better)', fontsize=10, color=MUTED, labelpad=5)
    for side in ['top', 'left', 'right']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(EDGE)
    full, nok = values[names[0]], values[names[1]]
    canvas.text(.07, .237, f'noK {nok.mean():.3f} ± {nok.std(ddof=1):.3f}    full {full.mean():.3f} ± {full.std(ddof=1):.3f}',
                color=MUTED, fontsize=11)
    canvas.text(.07, .073, '10 seeds | dots: runs; diamonds/error bars: mean ± sample SD', color=MUTED, fontsize=9.5)

    box(canvas, .612, .053, .338, .249, '#fffaf3', '#e7ddcb')
    canvas.text(.634, .264, 'D   Scope of the evidence', color=INK, fontsize=14, fontweight='bold')
    canvas.text(.634, .222, 'Dense simulated observations\nShared conductivity geology\nIndependent geology untested\nLatent intervals uncalibrated',
                color=MUTED, fontsize=12, linespacing=1.9, va='top')
    canvas.text(.05, .022, 'Archived study results. Seed variation does not measure variability across independent data splits.',
                color=MUTED, fontsize=9.5)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        path = output/f'research_overview.{suffix}'
        options = {'dpi': 180} if suffix == 'png' else {'metadata': {'Date': None}}
        fig.savefig(path, facecolor=fig.get_facecolor(), **options)
        if suffix == 'svg':
            lines = path.read_text(encoding='utf-8').splitlines()
            with path.open('w', encoding='utf-8', newline='\n') as handle:
                handle.write('\n'.join(line.rstrip() for line in lines)+'\n')
        print(f'[saved] {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
