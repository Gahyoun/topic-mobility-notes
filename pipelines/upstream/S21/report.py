"""Render a field comparison without faculty identifiers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

root=Path(__file__).resolve().parent
data=json.loads((root/'summary.json').read_text())
labels={'mathematics':'수학','physics':'물리','chemistry':'화학','biology':'생물','all':'전체'}
modes=(('official','공식 임용연도'),('proxy','기존 근거와 추정 연도 포함'))
def pretty(x): return f'{x:+.3f}'.replace('-','−')
def interval(a): return ' ~ '.join(map(pretty,a)) if a else '—'
def get(mode): return data[mode]['spring']['common']
def row(field,cell):
    return f"| {labels[field]} | {cell['n']:,} | {pretty(cell['delta_inclusive'])} | {pretty(cell['delta_late'])} | {pretty(cell['difference'])} | {interval(cell['ci'][2] if cell['ci'] else None)} |"
def table(mode):
    return '\n'.join(row(f,r) for f,r in get(mode).items())
a,b=get('official')['all'],get('proxy')['all']
other=data['baseline']['spring']['common']['all']
all_a,all_b=(data[m]['spring']['all_late_ready']['all'] for m,_ in modes)
first_a,first_b=(data[m]['spring']['common_first_late']['all'] for m,_ in modes)
report=f"""# 임용 이행기 포함 여부와 전후 주제 퍼짐

2026-11-09 · 2026년 봄 명부 · 출판 마감 2025-12-31

공유 참고문헌 TF–IDF·SVD 128차원 좌표를 고정했다. 임용 전은 −10~−1년의 비중첩 2년 창, 임용 후는 +3~+4년, +5~+6년, +7~+8년의 완결된 창을 사용했다. 이행기 포함 사양은 임용 다음 두 달력연도(+1~+2년) 창을 같은 후기 창에 추가한다. 임용연도 자체는 두 사양에서 모두 제외한다. 논문 출판연도로 임용연도를 복원했을 때 복원에 쓴 논문이 결과에 직접 들어오는 것을 피하기 위해서다.

논문 좌표가 두 편 이상 있는 창에서 평균 쌍별 제곱거리 $\\widehat{{Q}}$를 계산했다. 임용 전, 초기 임용 후, 후기 임용 후에 각각 적격 창이 있는 교원만 주 비교에 포함했다. 개인별 창을 동일 가중으로 평균한 뒤 전후 차이를 구하고, 이를 교원에 동일 가중으로 평균했다. 짝차이는 3년 이후 기준의 전후 차이에서 이행기 포함 기준의 전후 차이를 뺀 값이다. 95% 구간은 교원 단위 10,000회 짝재표집으로 구했다. 전체는 분야별 표본 수를 고정했다.

## 공식 임용연도

| 분야 | 공통 교원 행 | 1~2년 포함 후−전 | 3년 이후 후−전 | 짝차이 | 짝차이 95% 구간 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table('official')}

전체 46행의 평균은 {pretty(a['delta_inclusive'])}에서 {pretty(a['delta_late'])}로 달라졌다. 짝차이 {pretty(a['difference'])}의 95% 구간은 {interval(a['ci'][2])}이었다. 수학은 3행의 기술값만 제시했다.

## 기존 근거와 추정 연도 포함

| 분야 | 공통 교원 행 | 1~2년 포함 후−전 | 3년 이후 후−전 | 짝차이 | 짝차이 95% 구간 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table('proxy')}

전체 1,481행의 평균은 {pretty(b['delta_inclusive'])}에서 {pretty(b['delta_late'])}로 높아졌다. 짝차이는 {pretty(b['difference'])}, 95% 구간은 {interval(b['ci'][2])}이다. 공식 확인과 기존 고신뢰 복원 연도만 사용한 별도 공통 표본 {other['n']:,}행의 짝차이도 {pretty(other['difference'])}였다. 두 범위에 포함된 교원 구성이 다르므로, 표 사이의 차이를 추정 연도 사용 자체의 효과로 해석하기는 어렵다.

## 비교 범위 점검

| 설정 | 공식 연도 공통 행 | 공식 연도 짝차이 | 추정 포함 공통 행 | 추정 포함 짝차이 |
| --- | ---: | ---: | ---: | ---: |
| 후기 퍼짐이 산출되는 행 전부 | {all_a['n']:,} | {pretty(all_a['difference'])} | {all_b['n']:,} | {pretty(all_b['difference'])} |
| 첫 후기 창 +3~+4년만 사용 | {first_a['n']:,} | {pretty(first_a['difference'])} | {first_b['n']:,} | {pretty(first_b['difference'])} |

후기 퍼짐이 산출되는 행 전부를 비교할 때 +1~+2년 퍼짐이 미산출인 경우에는 두 사양에서 같은 후기 창을 사용한다. 이 행은 표본에 남고 짝차이는 0이다. 임용 후 논문 미관측 행은 기존 추적 자료에 별도 상태로 보존한다.

![분야별 전후 퍼짐과 이행기 제외 시 짝차이](appointment-window-comparison.png)
"""
(root/'appointment-window-comparison.md').write_text(report)

font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),
                     'axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white',
                     'text.color':'#172B46','axes.labelcolor':'#293F5B',
                     'xtick.color':'#50667D','ytick.color':'#324C69'})
fig,(ax,changes)=plt.subplots(1,2,figsize=(13.6,8.2),gridspec_kw={'width_ratios':[1.0,1.03]})
fig.subplots_adjust(left=.215,right=.968,top=.758,bottom=.177,wspace=.21)
fig.text(.055,.932,'임용 직후 2년을 빼면 주제 퍼짐은 얼마나 달라지나',fontsize=24,weight='bold')
fig.text(.055,.882,'2026년 봄 명부 · 각 사양에서 같은 교원과 같은 참고문헌 좌표로 비교',fontsize=12,color='#50667D')
fig.legend(handles=[
    Line2D([],[],marker='o',ls='none',color='#7CA9D0',label='임용 후 1~2년 포함'),
    Line2D([],[],marker='o',ls='none',color='#123F73',label='임용 후 3년부터'),
],loc='upper left',bbox_to_anchor=(.047,.856),frameon=False,ncol=2,fontsize=11,handletextpad=.4,columnspacing=1.3)
left_color,right_color='#7CA9D0','#123F73'
y_values=[]
y_labels=[]
for (mode,group),y0 in zip(modes,(10,4)):
    first=get(mode)
    for index,f in enumerate(labels):
        r=first[f]; y=y0-index
        y_values.append(y); y_labels.append(f"{labels[f]}  {r['n']:,}행")
        q0,q1,d=(r[k] for k in ('delta_inclusive','delta_late','difference'))
        ax.plot([q0,q1],[y+.08,y-.08],color='#A5B6C9',lw=2.0,zorder=2)
        ax.scatter(q0,y+.08,s=43,color=left_color,edgecolor='white',linewidth=.6,zorder=3)
        ax.scatter(q1,y-.08,s=46,color=right_color,edgecolor='white',linewidth=.6,zorder=4)
        ci=r['ci']
        if ci:
            lo,hi=ci[2]
            changes.errorbar(d,y,xerr=[[d-lo],[hi-d]],color=right_color,fmt='o',capsize=3,
                             lw=1.75,markersize=5.2,zorder=3)
        else:
            changes.scatter(d,y,s=38,edgecolor=right_color,facecolor='white',zorder=3)
        if f=='all':
            for axis in (ax,changes):
                axis.axhspan(y-.36,y+.36,color='#EAF2FA',zorder=0)
    ax.text(-.49,y0+.61,group,transform=ax.get_yaxis_transform(),weight='bold',fontsize=11,color='#254E78')
for axis in (ax,changes):
    axis.axvline(0,color='#879BB0',ls='--',lw=1,zorder=1)
    axis.set_ylim(-.55,10.8)
    axis.spines[['top','left','right']].set_visible(False)
    axis.spines['bottom'].set_color('#C2D0DF')
    axis.tick_params(length=0,labelsize=10)
    axis.set_axisbelow(True)
    axis.grid(axis='x',color='#EDF1F6')
ax.set(xlim=(-.12,.68),xticks=[0,.2,.4,.6],xlabel='임용 전후 퍼짐 차이 (후 - 전)')
ax.set_yticks(y_values,y_labels,fontsize=10.2)
changes.set(xlim=(-.154,.085),xlabel='3년부터 본 값 - 첫 2년을 포함한 값')
changes.xaxis.set_major_locator(MultipleLocator(.05))
changes.set_yticks([])
for axis,title in [(ax,'두 사양의 전후 변화'),(changes,'이행기를 제외했을 때의 짝차이')]:
    axis.set_title(title,loc='left',fontsize=13,weight='bold',pad=11)
fig.text(.055,.075,'오른쪽 선은 교원 단위 짝재표집 95% 구간. 빈 점은 표본이 5행 미만이라 구간을 표시하지 않았다.',fontsize=10.2,color='#50667D')
fig.savefig(root/'appointment-window-comparison.png',dpi=170)
