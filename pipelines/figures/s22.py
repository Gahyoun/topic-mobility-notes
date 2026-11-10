"""Render the anonymized paired-window summary used in the methodology page."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

root = Path(__file__).resolve().parent
fontManager.addfont('/Library/Fonts/NanumHumanRegular.ttf')
fontManager.addfont('/Library/Fonts/NanumHumanBold.ttf')
plt.rcParams.update({'font.family': ['NanumHuman TTF', 'DejaVu Sans'], 'axes.unicode_minus': True,
                     'text.color': '#182d48', 'xtick.color': '#53677e'})
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--career', action='store_true')
parser.add_argument('--input', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
career = args.career
if career:
    summary = json.loads((args.input or root/'career.json').read_text())['event']['database']
    file = 'career-reference-comparison.png'
else:
    summary = json.loads((args.input or root/'results.json').read_text())['results']
    file = 'reference-window-comparison.png'
fields = [
    ('all', '전체', '#1F4E79'),
    ('mathematics', '수학', '#FE5D21'),
    ('physics', '물리', '#009D5E'),
    ('chemistry', '화학', '#3E00F9'),
    ('biology', '생물', '#9437FF'),
]
metrics = [
    ('W', '논문 내부 변동  W', 1, (-.008, .010), [-.005, 0, .005]),
    ('B', '논문 사이 변동  B', 1, (-.011, .019), [-.01, 0, .01]),
    ('B_share', '논문 사이 비중  B/T  (%p)', 100, (-2.0, 3.0), [-2, 0, 2]),
]
if career:
    metrics = [
        ('W', '논문 내부 변동  W', 1, (-.006, .009), [-.005, 0, .005]),
        ('B', '논문 사이 변동  B', 1, (-.028, .005), [-.02, -.01, 0]),
        ('B_share', '논문 사이 비중  B/T  (%p)', 100, (-4.3, .5), [-4, -2, 0]),
    ]
fig = plt.figure(figsize=(11, 4.35), facecolor='white')
grid = fig.add_gridspec(1, 4, width_ratios=[1.04, 1, 1, 1], left=.045, right=.97,
                       bottom=.17, top=.72, wspace=.2)
labels = fig.add_subplot(grid[0, 0])
labels.set(xlim=(0, 1), ylim=(-.48, 4.48))
labels.axis('off')
fig.text(.045, .91, '임용 직후 2년을 빼면 얼마나 달라지는가', size=20, weight='bold')
fig.text(.045, .84, ('전체 후기 논문을 묶어 비교 · KRI·CV 등 DB 임용연도' if career
                     else '같은 교원에서 3년차 이후 평균 − 1~2년차를 포함한 평균'), size=12, color='#52667c')
labels.text(.08, 1.04, ('분야 · 짝지은 교원 / 연도 근거' if career else '분야 · 짝지은 교원 / 파일럿'), transform=labels.transAxes,
            size=10, color='#52667c', weight='bold')
for i, (field, name, color) in enumerate(fields):
    row = summary[field]
    y = 4 - i
    labels.scatter([.019], [y], color=color, s=27, clip_on=False)
    denominator = row['anchored'] if career else row['original']
    early = row['early_included'] if career else row['early']
    detail = f'직후 2년 논문 포함 {early:,}명' if career else f'직후 2년 산출 {early:,}명'
    labels.text(.08, y + .105, f"{name}  {row['n']:,}/{denominator:,}", va='center', weight='bold', size=12)
    labels.text(.08, y - .22, detail, va='center', color='#52667c', size=9.5)
for i, (metric, title, scale, limits, ticks) in enumerate(metrics, 1):
    ax = fig.add_subplot(grid[0, i], sharey=labels)
    ax.set(xlim=limits, xticks=ticks)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', length=3, width=.7, labelsize=9.5)
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#d8e3f1')
    ax.axvline(0, color='#8599b2', linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.set_title(title, fontsize=11.5, weight='bold', pad=11, loc='left')
    for y in (3.5, 2.5, 1.5, .5):
        ax.axhline(y, color='#edf2f8', linewidth=.75, zorder=0)
    for j, (field, _, color) in enumerate(fields):
        point, lo, hi = [v * scale for v in summary[field]['delta'][metric]]
        ax.errorbar(point, 4-j, xerr=[[point-lo], [hi-point]], color=color,
                    fmt='o', markersize=5.8, elinewidth=1.5, capsize=3, zorder=3)
fig.text(.045, .045, ('짝차이 = 3년차 이후 − 1~2년차 포함. 선은 교원 단위 95% 재표집 구간입니다.' if career
          else '점은 평균 짝차이, 선은 교원 단위 95% 재표집 구간입니다. 점선은 차이 0입니다.'),
         fontsize=10, color='#52667c')
fig.savefig(args.output or root/'docs'/'figures'/file, dpi=180, facecolor=fig.get_facecolor())
