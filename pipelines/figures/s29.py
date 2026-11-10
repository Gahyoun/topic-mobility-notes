"""Render four paired first-appointment signals with conservative uncertainty."""
import argparse
import json
from math import ceil, floor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FuncFormatter

FIELDS = [('all','전체','#1F4E79'),('mathematics','수학','#FE5D21'),('physics','물리','#009D5E'),
          ('chemistry','화학','#3E00F9'),('biology','생물','#9437FF')]
SIGNALS = [('con','수렴'),('div','확산'),('multi','병행'),('branch','분기')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--font',type=Path,nargs='+',
                   default=[Path('/Library/Fonts/NanumHumanRegular.ttf'),Path('/Library/Fonts/NanumHumanBold.ttf')])
    a=p.parse_args()
    for f in a.font:
        fontManager.addfont(f)
    plt.rcParams.update({'font.family':[FontProperties(fname=a.font[0]).get_name(),'DejaVu Sans'],
                         'text.color':'#142944','axes.edgecolor':'#bdd0e5','xtick.color':'#526780'})
    data=json.loads(a.summary.read_text())['dimensions']['1024']
    fig,axes=plt.subplots(2,2,figsize=(12.2,6.8),facecolor='white')
    fig.subplots_adjust(left=.12,right=.96,top=.80,bottom=.13,hspace=.60,wspace=.38)
    fig.text(.054,.93,'첫 임용 전후의 주제 궤적',size=24,weight='bold')
    n_all=data['all']['matched']['n']
    fig.text(.054,.873,f'같은 교원 {n_all:,}행 · 1024차원 · 임용 다음 두 해는 분리',size=12,color='#526780')
    included = [stat['metrics'][m]['delta'] for f,_,_ in FIELDS for m,_ in SIGNALS
                if (stat:=data[f]['matched'])['n']>=20]
    left=min(-5,5*floor(20*min(x[1] for x in included)))
    right=max(5,5*ceil(20*max(x[2] for x in included)))
    for ax,(key,label) in zip(axes.flat,SIGNALS,strict=True):
        ax.set_title(label,loc='left',fontsize=15,weight='bold',pad=8,color='#142944')
        ax.set_xlim(left-1,right+1)
        ax.set_ylim(-.55,4.55)
        ax.set_xticks(range(left,right+1,5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:+.0f}' if v else '0'))
        ax.axvline(0,color='#7992ad',lw=1.2,zorder=1)
        ax.grid(axis='x',color='#e4ecf4',linewidth=.7)
        ax.tick_params(axis='y',length=0,pad=7,labelcolor='#304764')
        ax.tick_params(axis='x',length=0,labelsize=9)
        ax.set_yticks(range(5),[f"{name} {data[f]['matched']['n']:,}" for f,name,_ in FIELDS][::-1])
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
        for i,(f,name,color) in enumerate(FIELDS):
            n=data[f]['matched']['n']
            v,low,high=[100*x for x in data[f]['matched']['metrics'][key]['delta']]
            y=4-i
            if n<20:
                ax.text(.02,y,f'사례 부족 · {v:+.1f}%p',transform=ax.get_yaxis_transform(),
                        size=9,color='#526780',va='center')
                continue
            ax.hlines(y,low,high,color=color,lw=2,alpha=.72,zorder=2)
            ax.scatter(v,y,s=43,color=color,edgecolor='white',linewidth=.8,zorder=3)
    fig.text(.54,.073,'임용 후 비율 − 임용 전 비율 (%p)',size=11,ha='center',color='#304764')
    fig.text(.054,.025,'선은 두 방향의 전환 건수로 계산한 보수적 95% 구간이다. 수학은 표에 건수와 구간을 싣는다.',size=9,color='#526780')
    fig.savefig(a.output,dpi=170,facecolor='white')


if __name__=='__main__':
    main()
