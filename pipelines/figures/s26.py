"""Draw angular turns and radial-state changes for fixed paper cohorts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=root/'topic-turns-comparison.png')
p.add_argument('--input', type=Path, default=root/'results.json')
args = p.parse_args()
for face in ('Regular','Bold'):
    fontManager.addfont(f'/Library/Fonts/NanumHuman{face}.ttf')
plt.rcParams.update({'font.family':['NanumHuman TTF','DejaVu Sans'],
                     'text.color':'#182d48','xtick.color':'#4b6077'})
data=json.loads(args.input.read_text())
fields=(('all','전체'),('mathematics','수학'),('physics','물리'),
        ('chemistry','화학'),('biology','생물'))
widths=(('2','#235283','o',.22),('3','#08778e','s',0.),('5','#7b548d','D',-.22))
metrics=(('turn90','연속 변위가 90°를 넘어 꺾인 비율'),
         ('switch','초기 중심에 대한 diff·conv 전환율'))
fig,axes=plt.subplots(2,2,figsize=(11.7,7.25),facecolor='white',
                      gridspec_kw={'left':.16,'right':.965,'top':.75,'bottom':.12,
                                   'hspace':.55,'wspace':.25})
fig.text(.055,.943,'진행 방향의 꺾임과 초기 주제와의 거리 변화',size=18.5,weight='bold')
fig.text(.055,.899,'임용 3년차 이후에 인접한 세 창에서 연속 변위 두 개를 비교',size=11,color='#4b6077')
handles=[Line2D([],[],color=c,marker=m,linewidth=1.4,markersize=5,label=f'{w}년 창') for w,c,m,_ in widths]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.047,.886),frameon=False,ncol=3,
           fontsize=10.5,columnspacing=1.65,handlelength=1.7)
for i,(sample,label) in enumerate((('database','DB 임용연도 표본 · 세 폭 공통 교원'),
                                 ('pilot','기존 파일럿 · 2년 창에서 비교 가능한 교원'))):
    for j,(metric,title) in enumerate(metrics):
        ax=axes[i,j]
        ax.set_title(title,loc='left',size=11.7,weight='bold',pad=10)
        ax.set(xlim=(0,1),ylim=(-.48,4.48),yticks=[])
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
        ax.tick_params(axis='x',length=3,labelsize=9)
        ax.xaxis.grid(color='#f0f3f8',linewidth=.7)
        ax.set_axisbelow(True)
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#cad9e9')
        for y in (.5,1.5,2.5,3.5):
            ax.axhline(y,color='#edf2f8',linewidth=.75)
        for rank,(field,name) in enumerate(fields):
            row=data[sample][field]
            n=row['common'] if sample=='database' else row['widths']['2']['available']
            y=4-rank
            if j==0:
                ax.text(-.048,y,f'{name} {n:,}행',ha='right',va='center',
                        transform=ax.get_yaxis_transform(),size=10)
            if n<10:
                ax.text(.5,y,'표본 10행 미만',ha='center',va='center',size=9,color='#61758b')
                continue
            for w,c,m,offset in widths if sample=='database' else (widths[0],):
                est=row['widths'][w]['common_estimate' if sample=='database' else 'available_estimate']
                val,lo,hi=est[metric]
                ax.errorbar(val,y+(offset if sample=='database' else 0),xerr=[[val-lo],[hi-val]],
                            fmt=m,color=c,markersize=4.9,elinewidth=1.45,capsize=2.3)
        if j==0:
            fig.text(.055,ax.get_position().y1+.056,label,fontsize=11.7,weight='bold',color='#235283')
fig.text(.055,.045,'가로선은 교원 단위 95% 재표집 구간입니다. 파일럿의 3·5년 창은 후기 세 창을 확보할 관측 기간이 부족합니다.',
         size=9.6,color='#4b6077')
fig.savefig(args.output,dpi=175,facecolor='white')
