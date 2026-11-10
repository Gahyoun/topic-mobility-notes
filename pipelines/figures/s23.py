"""Plot paired teacher-level contrasts on a common B axis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=root/'b-aggregation-comparison.png')
parser.add_argument('--input', type=Path, default=root/'results.json')
args = parser.parse_args()
for style in ('Regular', 'Bold'):
    fontManager.addfont(f'/Library/Fonts/NanumHuman{style}.ttf')
plt.rcParams.update({'font.family': ['NanumHuman TTF', 'DejaVu Sans'],
                     'text.color': '#182d48', 'xtick.color': '#4b6077'})
data = json.loads(args.input.read_text())
fields = [('all', '전체'), ('mathematics', '수학'), ('physics', '물리'),
          ('chemistry', '화학'), ('biology', '생물')]
methods = [('windows', '2년 창의 평균', '#147b9b', 'o', .12),
           ('pooled', '같은 논문을 한 번에 묶음', '#244981', 'D', -.12)]
all_ends = [v for sample in data.values() for r in sample.values()
            for key, *_ in methods for v in r['delta'][key][1:]]
limit = np.ceil(1.1 * max(abs(v) for v in all_ends) * 100) / 100

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.55), facecolor='#ffffff',
                         gridspec_kw={'left': .17, 'right': .97, 'top': .73,
                                      'bottom': .17, 'wspace': .48})
fig.text(.05, .91, '같은 교원·논문·좌표에서 집계 방식만 바꾼 비교', fontsize=19, weight='bold')
handles = [Line2D([], [], color=color, marker=marker, markersize=5, linewidth=1.5, label=label)
           for _, label, color, marker, _ in methods]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.043, .887),
           frameon=False, ncol=2, handlelength=1.8, columnspacing=1.9, fontsize=10)
for (sample, title), ax in zip([('pilot','기존 908명 파일럿'), ('database','KRI·CV 등 DB 임용연도 표본')], axes):
    ax.set_title(title, loc='left', pad=10, size=12, weight='bold')
    ax.set(xlim=(-limit, limit), ylim=(-.45, 4.45), yticks=[])
    ax.xaxis.set_major_locator(MultipleLocator(.01 if limit <= .035 else .02))
    ax.tick_params(axis='x', length=3, labelsize=9.5)
    for spine in ('left','top','right'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cad9e9')
    ax.axvline(0, linestyle=(0,(3,3)), color='#8298b2', linewidth=1)
    for y in (.5,1.5,2.5,3.5):
        ax.axhline(y, color='#edf2f8', linewidth=.8)
    for j,(field, label) in enumerate(fields):
        stats = data[sample][field]
        y = 4-j
        ax.text(-.07, y, f"{label} {stats['n']:,}행", va='center', ha='right',
                transform=ax.get_yaxis_transform(), size=10)
        for key, _, color, marker, offset in methods:
            point, lo, hi = stats['delta'][key]
            ax.errorbar(point, y+offset, xerr=[[point-lo], [hi-point]], color=color, fmt=marker,
                        markersize=5.2, elinewidth=1.5, capsize=3)
fig.text(.05,.055,'임용 3년차 이후의 B에서 직후 2년을 포함한 B를 뺀 값. 가로선은 교원 단위 95% 재표집 구간.',
         size=10, color='#4b6077')
fig.savefig(args.output, dpi=180)
