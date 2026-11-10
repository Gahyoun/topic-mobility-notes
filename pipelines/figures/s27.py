"""Make the aggregate field figure from the portfolio pipeline JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('summary',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--font',type=Path,nargs='+',
               default=[Path('/Library/Fonts/NanumHumanRegular.ttf'),Path('/Library/Fonts/NanumHumanBold.ttf')])
a=p.parse_args()
for f in a.font:
    fontManager.addfont(f)
from matplotlib.font_manager import FontProperties
plt.rcParams.update({'font.family':[FontProperties(fname=a.font[0]).get_name(),'DejaVu Sans'],
                     'text.color':'#182d48','xtick.color':'#4b6077'})
d=json.loads(a.summary.read_text())
fields=('all','mathematics','physics','chemistry','biology')
names=('전체','수학','물리','화학','생물')
fig,ax=plt.subplots(2,2,figsize=(12.2,7.35),facecolor='white',
                   gridspec_kw={'left':.115,'right':.965,'top':.785,'bottom':.105,'hspace':.62,'wspace':.39})
fig.text(.054,.944,'전체 논문 분포로 본 연구 궤적',fontsize=21,weight='bold')
fig.text(.054,.9,'명부 전체를 유지하고 같은 분야·시기의 논문 배경에 맞춰 비교',fontsize=11,color='#4b6077')


def style(g,title):
    g.set_title(title,loc='left',fontsize=11.5,weight='bold',pad=7)
    g.set(yticks=np.arange(5),yticklabels=names,ylim=(4.52,-.52))
    g.tick_params(axis='y',length=0,labelsize=10)
    g.tick_params(axis='x',length=3,labelsize=9)
    for k in ('top','left','right'):
        g.spines[k].set_visible(False)
    g.spines['bottom'].set_color('#cddbea')
    g.set_axisbelow(True)
    g.xaxis.grid(color='#edf2f8',linewidth=.7)


def stack(g,blocks,colors,labels):
    left=np.zeros(5)
    for v,c,label in zip(blocks,colors,labels,strict=True):
        g.barh(np.arange(5),v,left=left,color=c,height=.55,label=label)
        left+=v
    g.legend(loc='upper left',bbox_to_anchor=(-.01,1.025),frameon=False,
             fontsize=8.1,ncol=len(labels),handlelength=.9,columnspacing=1.05,
             borderaxespad=0)
    g.xaxis.set_major_formatter(PercentFormatter(1))


g=ax[0,0]
style(g,'01  명부의 관측 범위')
g.title.set_position((0,1.14))
stack(g,np.array([[(d[f]['roster']-d[f]['linked'])/d[f]['roster'] for f in fields],
                   [(d[f]['linked']-d[f]['with_geometry_2_total'])/d[f]['roster'] for f in fields],
                   [(d[f]['with_geometry_2_total']-d[f]['paired_faculty'])/d[f]['roster'] for f in fields],
                   [d[f]['paired_faculty']/d[f]['roster'] for f in fields]]),
      ('#e0e7ef','#9db3ca','#5e87b4','#204b7e'),
      ('연결 없음','기하 부족','기하 있음','전후 비교'))
g.set(xlim=(0,1))
g.set_xlabel('분야별 명부 행의 비중',fontsize=9.5,labelpad=5)

g=ax[0,1]
style(g,'02  논문 분포 변화의 구성')
g.title.set_position((0,1.14))
blocks=np.array([[d[f]['transport_share'][k] for f in fields] for k in ('location','scale','shape')])
stack(g,blocks,('#245384','#438ca4','#9472a7'),('중심 이동','폭 변화','모양 변화'))
g.set(xlim=(0,1))
for i,v in enumerate(blocks[0]):
    g.text(v/2,i,f'{100*v:.0f}%',color='white',ha='center',va='center',fontsize=9)
g.set_xlabel('완결 창 전환에서 측정한 분포 이동량의 비중',fontsize=9.5,labelpad=5)

g=ax[1,0]
style(g,'03  수렴·확산 판정')
g.title.set_position((0,1.14))
blocks=np.array([[d[f][k]/d[f]['geometry_pairs_complete'] for f in fields] for k in ('con','div')])
stack(g,np.vstack((blocks,1-blocks.sum(0))),('#5e8bb1','#1b6f83','#e4eaf2'),
      ('수렴','확산','기준 이내'))
g.set(xlim=(0,1))
g.set_xlabel('완결 5년 창 쌍 중 해당 신호의 비중',fontsize=9.5,labelpad=5)

g=ax[1,1]
style(g,'04  관측 마감 전 연결 논문 0편')
g.title.set_position((0,1.14))
for j,(k,c,label) in enumerate((('zero_last_3','#325c91','최근 3년'),('zero_last_5','#819ab7','최근 5년'))):
    v=[d[f][k] for f in fields]
    y=np.arange(5)+(j-.5)*.32
    g.barh(y,v,height=.24,color=c,label=label)
    for yy,vv in zip(y,v,strict=True):
        g.text(vv+6,yy,str(vv),ha='left',va='center',fontsize=8.5)
g.legend(loc='upper left',bbox_to_anchor=(-.01,1.035),frameon=False,fontsize=8.4,ncol=2,borderaxespad=0)
g.set(xlim=(0,max(d['all']['zero_last_3']*1.2,1)))
g.set_xlabel('이전에 연결 논문이 있었던 교원 행 수',fontsize=9.5,labelpad=5)
fig.text(.054,.033,'관측 마감 2025년 말. 기하 비교는 두 창 각각에 좌표가 있는 논문이 두 편 이상인 경우입니다.',
         fontsize=9.5,color='#4b6077')
fig.savefig(a.output,dpi=170,facecolor='white')
