"""Plot the normalized covariance residual with faculty-bootstrap intervals."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FIELDS = {'all': '전체', 'mathematics': '수학', 'physics': '물리',
          'chemistry': '화학', 'biology': '생물'}
COMPONENTS = (
    ('excess', '전체', '#153c65', 'o'),
    ('within', '주제 내부', '#146696', 's'),
    ('between', '주제 사이', '#a05223', 'D'),
)


def draw_panel(ax, rows, labels, title):
    y = np.arange(len(labels))
    for offset, (key, label, color, marker) in zip((-.21, 0, .21), COMPONENTS, strict=True):
        value, low, high = np.asarray([r[key]['estimate'] for r in rows]).T
        ax.errorbar(value, y + offset, xerr=[value - low, high - value], fmt=marker, color=color,
                    markersize=4, linewidth=1.3, capsize=2.5, label=label)
    ax.axvline(0, color='#8798ac', linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc='left', fontsize=12)
    ax.set_xlabel('배경 흐름 대비 정규화된 공분산 차이')
    ax.xaxis.grid(color='#e7edf4')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#bac8d7')
    ax.tick_params(axis='both', length=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    if args.font:
        font_manager.fontManager.addfont(args.font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'text.color': '#152b44',
                         'axes.labelcolor': '#152b44', 'axes.titlecolor': '#152b44'})
    data = json.loads(args.summary.read_text())
    fig, (field, dim) = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    rows = [data['1024'][f]['estimates'] for f in FIELDS]
    names = [f"{label} · {r['excess']['n']:,}행" for label, r in zip(FIELDS.values(), rows, strict=True)]
    draw_panel(field, rows, names, '분야별 · 1024차원')
    draw_panel(dim, [row['all']['estimates'] for row in data.values()],
               [f'{int(d):,}' for d in data], '차원별 · 전체')
    common_limits = min(field.get_xlim()[0], dim.get_xlim()[0]), max(field.get_xlim()[1], dim.get_xlim()[1])
    field.set_xlim(common_limits)
    dim.set_xlim(common_limits)
    dim.legend(loc='upper right', bbox_to_anchor=(1, 1.14), ncol=3, frameon=False)
    fig.suptitle('첫 임용 전후 주제 분포의 공분산 변화', x=.02, ha='left', fontsize=16)
    fig.savefig(args.output, dpi=180, facecolor='white')


if __name__ == '__main__':
    main()
