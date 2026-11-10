"""Render anonymized field summaries and a figure for the paired validation."""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
dates=json.loads((root/'dates.json').read_text())
labels={'mathematics':'수학','physics':'물리','chemistry':'화학','biology':'생물','all':'전체'}
fields=list(labels)
def signed(x):
    return f'{x:+.3f}'.replace('-', '−')
def interval(x):
    return f'{signed(x[0])} ~ {signed(x[1])}' if x else '—'
def delta_row(label,r):
    a,b,d=r['delta']
    return f"| {label} | {r['n']} | {signed(a)} | {signed(b)} | {signed(d)} | {interval(r['ci'][2] if r['ci'] else None)} |"
def date_row(field):
    x=s['lag'][field]
    return f"| {labels[field]} | {x['linked']}/{x['n']} | {x['lt']} / {x['eq']} / {x['gt']} | +{int(x['median'])}년 | {x['mae']:.2f}년 | {x['abs_le1']} |"
lag='\n'.join(map(date_row,fields))
main='\n'.join(delta_row(labels[f],s['main'][f]) for f in fields)
tri=s['triple']
settings='\n'.join([
 delta_row('전후 4년 범위',s['near']['all']),
 delta_row('각 기준연도 자체를 제외',s['skip_zero']['all']),
 delta_row('기준별 적격 창을 각각 사용',s['independent_bins']['all']),
 delta_row('세 기준 공통 표본, 공식 기관의 첫 논문',tri['official_institution']),
 delta_row('세 기준 공통 표본, 명부 기관의 첫 논문',tri['roster_institution']),
])
report=f"""# 첫 소속 논문은 임용연도를 얼마나 잘 대체하는가

2026-11-09 · 한국 기초자연과학 교원 명부 · 논문 출판 마감 2025-12-31

## 첫 전임 기관의 소속 논문과 공식 임용연도

대학 공식 이력으로 첫 전임 직위와 시작 연도를 확인한 100행을 대상으로 했다. 해당 교원 자신의 논문 소속을 **공식 첫 전임 기관**과 대조하고, 저자 판별을 통과한 학술지 논문 가운데 가장 이른 출판연도를 썼다. 공식 임용 전 논문도 포함해 소속이 임용보다 먼저 관측되는 경우를 확인했다. 첫 임용기관의 소속 논문을 찾지 못한 14행과 공유 OpenAlex 저자 ID로 개인 확인이 필요한 5행을 제외해 81행이 남았다.

| 분야 | 대조 가능 / 공식 확인 | 논문이 더 이름 / 같은 해 / 더 늦음 | 연도 차이 중앙값 | 평균 절대 차이 | 1년 이내 |
| --- | ---: | ---: | ---: | ---: | ---: |
{lag}

차이는 **첫 소속 논문의 출판연도 − 공식 임용연도**다. 중앙값은 모든 분야에서 +1년이었다. 정확히 일치한 경우는 14/81행(17.3%), 1년 이내는 34/81행(42.0%)이다. 차이의 전체 범위는 −11년에서 +6년이었다.

## 같은 교원의 전후 퍼짐을 두 번 계산

공유 참고문헌의 TF–IDF·SVD 128차원 좌표를 고정했다. 같은 교원의 논문 풀을 두 기준연도로 각각 정렬했다. 기준연도 −10년부터 +9년까지 겹치지 않는 2년 창으로 나누고, 각 기준에서 좌표가 있는 논문이 두 편 이상인 동일 상대시점의 창만 사용했다. 전후 모두 한 개 이상의 창이 남은 45행이 주 분석에 들어갔다. −2~−1과 0~+1은 각각 임용 전과 후의 창이며, 공식 기준과 논문 기준에 들어가는 실제 출판연도는 기준연도 차이만큼 달라진다.

각 창에서 논문 쌍의 평균 제곱거리 Q̂를 구했다. 개인별로 전후 각 창의 Q̂를 평균한 뒤, **후−전**을 교원에 동일한 가중치로 평균했다. 짝차이는 논문 기준의 후−전에서 공식 기준의 후−전을 뺀 값이다. 95% 구간은 교원 단위로 10,000회 짝재표집해 구했다. 전체 표본에서는 분야별 행 수를 고정했다.

| 분야 | 공통 행 | 공식 기준 후−전 | 논문 기준 후−전 | 짝차이 | 짝차이 95% 구간 |
| --- | ---: | ---: | ---: | ---: | ---: |
{main}

전체 평균 방향은 두 기준에서 모두 양수였다. 짝차이는 +0.021이었고 95% 구간은 −0.084에서 +0.128이었다. 다만 기준연도가 실제로 달라진 34행 가운데 11행에서 개별 전후 변화의 부호가 바뀌었다. 수학은 2행이라 기술값만 제시했다. 분야별 짝차이 구간은 모두 0을 포함했다.

## 기존 대체 규칙과의 비교

기존 대체 규칙은 공식 이력이 부족할 때 **명부에서 교수 직급으로 처음 관측된 기관**을 사용한다. 직급 기록이 없으면 최초 명부 기관을 쓰고, 학위 시점이나 최초 명부 관측과 충돌하는 논문은 보류한다. 공식 연도가 있는 100행에 이 규칙을 그대로 대입했을 때 49행에서 논문 기준연도가 나왔으며, 공식 연도보다 중앙값 2년 늦었다.

아래 마지막 두 행은 공식 연도, 공식 첫 전임 기관의 논문 연도, 명부 기준 기관의 논문 연도에서 **세 기준에 모두 공통인 교원 35행과 상대시점 창**을 사용한다. 이 중 14행은 명부 기준 기관과 공식 첫 전임 기관이 달랐다.

| 설정 | 공통 행 | 공식 기준 후−전 | 논문 기준 후−전 | 짝차이 | 짝차이 95% 구간 |
| --- | ---: | ---: | ---: | ---: | ---: |
{settings}

명부 기준 기관까지 바꾸면 같은 35행에서 전후 평균 차이는 +0.189에서 +0.031로 줄었다. 두 기관이 달랐던 14행에서는 +0.379에서 +0.061로 줄었다. 공식 첫 전임 기관을 알고 있을 때의 검증과, 그 기관을 명부 관측으로 대신할 때의 검증은 서로 다른 결과를 보인다.

이 수치는 공식 이력을 교차 확인할 수 있었던 표본의 기술 비교다. 소속 논문이 학위·박사후연구나 겸직 시기를 반영하거나 실제 부임보다 늦게 출판될 수 있다. 명부 기준 기관이 이직 후 기관이면 대체 연도는 첫 임용 대신 뒤의 경력 전환을 가리킬 수 있다. 이번 비교는 임용의 인과 효과를 추정하지 않았다.

![공식 임용연도와 첫 논문 연도의 차이, 그리고 기준연도를 바꿨을 때의 전후 퍼짐 짝차이](faculty-anchor-validation.png)
"""
(root/'faculty-anchor-validation.md').write_text(report)

font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),
                     'axes.unicode_minus':False,'figure.facecolor':'white',
                     'axes.facecolor':'white','text.color':'#182B45','axes.labelcolor':'#273D58',
                     'xtick.color':'#53657A','ytick.color':'#53657A'})
fig,(left,right)=plt.subplots(1,2,figsize=(12.6,5.4),gridspec_kw={'width_ratios':[1,1.16]})
fig.subplots_adjust(left=.072,right=.98,top=.78,bottom=.19,wspace=.57)
fig.text(.072,.93,'첫 소속 논문을 임용 기준으로 쓰면',fontsize=22,weight='bold')
fig.text(.072,.86,'공식 임용연도가 확인된 기초자연과학 교원 명부',fontsize=12,color='#52657C')
signed_lags=Counter(d['paper_year']-d['official_year'] for d in dates if d['paper_year'] is not None and not d['shared_author_id'])
lag_x=list(range(min(signed_lags),max(signed_lags)+1))
left.bar(lag_x,[signed_lags[x] for x in lag_x],color=['#86B4DF' if x<0 else '#23456C' if x==0 else '#3D7AB4' for x in lag_x],width=.79)
left.axvline(0,color='#233955',lw=.75,alpha=.45,zorder=1)
left.set(title='첫 전임 기관의 논문 출판연도 차이 (81행)',xlabel='첫 소속 논문 연도 - 공식 임용연도 (년)',ylabel='명부 행 수')
left.set_xticks([-11,-8,-5,-2,0,2,4,6])
left.yaxis.set_major_locator(MultipleLocator(5))
left.set_ylim(0,max(signed_lags.values())+4)
comparisons=[('공식 기관 · 45행',s['main']['all']),
             ('공식 기관 · 35행',tri['official_institution']),
             ('명부 기관 · 35행',tri['roster_institution'])]
for y,(label,r) in enumerate(comparisons[::-1]):
    d=r['delta'][2]
    lo,hi=r['ci'][2]
    color='#172E54' if label.startswith('명부') else '#4383BC'
    right.errorbar(d,y,xerr=[[d-lo],[hi-d]],color=color,fmt='o',capsize=4,lw=2.15,markersize=7)
right.axvline(0,color='#677B91',lw=1,linestyle='--',zorder=1)
right.set_yticks(range(3),[x[0] for x in comparisons[::-1]])
right.set(ylim=(-.55,2.55),xlim=(-.34,.18),title='퍼짐의 전후 변화가 달라진 크기',
          xlabel='전후 퍼짐 변화의 짝차이 (논문 기준 - 공식 기준)')
right.xaxis.set_major_locator(MultipleLocator(.1))
for axis in [left,right]:
    axis.spines[['top','right','left']].set_visible(False)
    axis.spines['bottom'].set_color('#C5D2DE')
    axis.tick_params(length=0,labelsize=10)
    axis.title.set_fontsize(13)
    axis.title.set_weight('bold')
    axis.title.set_position((0,1.03))
    axis.title.set_ha('left')
    axis.set_axisbelow(True)
left.grid(axis='y',color='#E8EDF3')
right.grid(axis='x',color='#E8EDF3')
fig.text(.072,.065,'오른쪽 선은 교원 단위 짝재표집 95% 구간이다. 35행 비교에는 세 기준의 공통 교원과 상대시점 창을 사용했다.',fontsize=10,color='#52657C')
fig.savefig(root/'faculty-anchor-validation.png',dpi=170)
