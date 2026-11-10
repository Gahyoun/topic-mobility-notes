"""Draw faculty-matched speed and signed radial motion by discipline."""
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

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'topic-center-motion.png')
parser.add_argument('--input', type=Path, default=root/'results.json')
args=parser.parse_args()
for face in ('Regular','Bold'):
    fontManager.addfont(f'/Library/Fonts/NanumHuman{face}.ttf')
plt.rcParams.update({'font.family':['NanumHuman TTF','DejaVu Sans'],
                     'text.color':'#182d48','xtick.color':'#4b6077'})
data=json.loads(args.input.read_text())
samples=(('database','DB 임용연도 표본'),('pilot','기존 파일럿'))
fields=(('all','전체'),('mathematics','수학'),('physics','물리'),
        ('chemistry','화학'),('biology','생물'))
widths=(('2','#235283','o',.23),('3','#08778e','s',0),('5','#7b548d','D',-.23))
metrics=(('speed','중심 이동 속도'),('radial','직후 2년 중심 기준의 방향'))
ends={metric:[v for s,_ in samples for f,_ in fields for w,*_ in widths
              for v in data[s][f]['widths'][w]['estimate'][metric][1:]]
      for metric,_ in metrics}
span=max(abs(v) for v in ends['radial'])
side=np.ceil(span*100)/100
limits={'radial':(-side,side),'speed':(0,np.ceil(max(ends['speed'])*100)/100)}
fig,axes=plt.subplots(2,2,figsize=(11.7,7.25),facecolor='white',
                      gridspec_kw={'left':.17,'right':.965,'top':.755,'bottom':.12,
                                   'hspace':.53,'wspace':.22})
fig.text(.055,.945,'후기 연구의 주제 중심은 얼마나 빠르게 움직이는가',size=18.5,weight='bold')
fig.text(.055,.902,'임용 직후 2년을 기준점으로 두고, 임용 3년차부터 인접한 창을 연결',size=11,color='#4b6077')
handles=[Line2D([],[],color=c,marker=m,linewidth=1.4,markersize=5,label=f'{w}년 창') for w,c,m,_ in widths]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.048,.885),frameon=False,ncol=3,
           fontsize=10.5,columnspacing=1.65,handlelength=1.7)
for i,(sample,label) in enumerate(samples):
    for j,(metric,title) in enumerate(metrics):
        ax=axes[i,j]
        ax.set_title(title,loc='left',size=11.8,weight='bold',pad=10)
        ax.set(xlim=limits[metric],ylim=(-.47,4.47),yticks=[])
        major=.05 if metric=='speed' else limits[metric][1]/2
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.tick_params(axis='x',length=3,labelsize=9)
        if metric=='radial':
            ax.axvline(0,color='#879ab0',linestyle=(0,(3,3)),linewidth=1)
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#cad9e9')
        for y in (.5,1.5,2.5,3.5):
            ax.axhline(y,color='#edf2f8',linewidth=.75)
        for rank,(field,name) in enumerate(fields):
            r=data[sample][field]
            y=4-rank
            if j==0:
                ax.text(-.045,y,f"{name} {r['common']:,}행",ha='right',va='center',
                        transform=ax.get_yaxis_transform(),size=10)
            for width,color,mark,offset in widths:
                value,lo,hi=r['widths'][width]['estimate'][metric]
                ax.errorbar(value,y+offset,xerr=[[value-lo],[hi-value]],fmt=mark,color=color,
                            markersize=4.8,elinewidth=1.45,capsize=2.35)
        if j==0:
            fig.text(.055, ax.get_position().y1+.053, label, fontsize=11.7,weight='bold',
                     color='#235283')
        ax.set_xlabel('임베딩 거리 / 년' if j==0 else '← 초기 중심에 가까워짐     멀어짐 →',
                      fontsize=9.5,color='#4b6077',labelpad=5)
fig.text(.055,.04,'세 창 폭 모두에서 비교할 수 있는 교원으로 맞췄습니다. 가로선은 교원 단위 95% 재표집 구간입니다.',
         size=9.6,color='#4b6077')
fig.savefig(args.output,dpi=175,facecolor='white')
