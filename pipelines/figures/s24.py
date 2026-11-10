"""Draw the within-window and between-window contrasts for both time grids."""
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
parser.add_argument('--output', type=Path, default=root/'b-window-width-comparison.png')
parser.add_argument('--input', type=Path, default=root/'results.json')
args = parser.parse_args()
for font in ('Regular', 'Bold'):
    fontManager.addfont(f'/Library/Fonts/NanumHuman{font}.ttf')
plt.rcParams.update({'font.family': ['NanumHuman TTF', 'DejaVu Sans'],
                     'text.color': '#182d48', 'xtick.color': '#4b6077',
                     'ytick.color': '#4b6077'})
data = json.loads(args.input.read_text())
samples = [('database', 'DB 임용연도 표본'), ('pilot', '기존 파일럿')]
styles = [('fixed_start', '1년차부터 같은 폭으로 구획', '#225287', 'o', '-', -.07),
          ('separate_early', '직후 2년을 별도 구간으로 유지', '#08768c', 'D', '--', .07)]
metrics = [('within', '같은 창 안의 논문 사이 변동'), ('between', '창 사이의 주제 중심 차이')]
widths = np.array([2,3,5])
ends = [v for sample,_ in samples for rule,*_ in styles for w in widths for k,_ in metrics
        for v in data[sample]['all'][rule][str(w)]['delta'][k][1:]]
ymin = np.floor(min(ends)*1000-2)/1000
ymax = np.ceil(max(ends)*1000+2)/1000
fig, axes = plt.subplots(2,2, figsize=(11.7,6.7), facecolor='white', sharey=True,
                         gridspec_kw={'left':.11,'right':.96,'top':.77,
                                      'bottom':.13,'wspace':.24,'hspace':.46})
fig.text(.065,.942,'2년·3년·5년 창에서 주제 병행과 중심 차이', fontsize=19, weight='bold')
fig.text(.065,.893,'임용 3년차 이후에서 직후 2년을 포함한 값을 뺀 차이', fontsize=11, color='#4b6077')
handles = [Line2D([],[],color=color,marker=marker,linestyle=line,markersize=5,
                  linewidth=1.5,label=label) for _,label,color,marker,line,_ in styles]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.058,.878),frameon=False,
           ncol=2,handlelength=2.1,columnspacing=1.8,fontsize=10.5)
for i,(sample,label) in enumerate(samples):
    row = data[sample]['all']
    for j,(metric,title) in enumerate(metrics):
        ax=axes[i,j]
        ax.set_title(title,loc='left',size=11.5,weight='bold',pad=9)
        ax.set(xlim=(1.68,5.32),ylim=(ymin,ymax),xticks=widths,xticklabels=['2년','3년','5년'])
        ax.yaxis.set_major_locator(MultipleLocator(.01))
        ax.tick_params(axis='both',labelsize=9,length=3)
        for spine in ('top','right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#cfdbeb')
        ax.spines['bottom'].set_color('#cfdbeb')
        ax.yaxis.grid(color='#edf2f8',linewidth=.75)
        ax.axhline(0,color='#8c9fb4',linestyle=(0,(3,3)),linewidth=1)
        if j==0:
            ax.set_ylabel(f"{label} {row['n']:,}행",size=10.5,labelpad=12)
        for rule,_,color,marker,line,offset in styles:
            vals=np.array([row[rule][str(w)]['delta'][metric] for w in widths])
            point,lo,hi=vals.T
            ax.errorbar(widths+offset,point,yerr=[point-lo,hi-point],color=color,marker=marker,
                        linestyle=line,markersize=5.1,linewidth=1.25,capsize=2.8)
fig.text(.065,.046,'왼쪽은 논문 수로 가중한 창 내부 변동, 오른쪽은 창 중심 사이 변동입니다. 오차막대는 교원 단위 95% 재표집 구간입니다.',
         size=9.8,color='#4b6077')
fig.savefig(args.output,dpi=175,facecolor='white')
