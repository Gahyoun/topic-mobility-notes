"""Compare first-appointment metrics in their available and common faculty samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

FIELDS = {'all':'전체','mathematics':'수학','physics':'물리','chemistry':'화학','biology':'생물'}
METRICS = {'con':'수렴','div':'확산','parallel':'연간 병행','branch':'분기'}
BOOTSTRAPS = 10_000
SEED = 421110


def exact_rate(k, n, alpha=.05):
    lower = 0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    upper = 1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return [lower,upper]


def quantiles(draws):
    return np.quantile(draws,[.025,.975]).tolist()


def parallel_intervals(common, additional, rng):
    a,b=len(common),len(additional)
    full,selected,change=[],[],[]
    for _ in range(BOOTSTRAPS//50):
        c=common[rng.integers(a,size=(50,a))].mean(1)
        more=additional[rng.integers(b,size=(50,b))].mean(1) if b else np.zeros(50)
        f=(a*c+b*more)/(a+b)
        selected.extend(c)
        full.extend(f)
        change.extend(c-f)
    return quantiles(full),quantiles(selected),quantiles(change)


def compare_metric(series, common_mask, metric, rng):
    full=series.dropna().to_numpy()
    common=series[common_mask].to_numpy()
    additional=series[~common_mask].dropna().to_numpy()
    n,nc,na=len(full),len(common),len(additional)
    assert n==nc+na and np.isfinite(common).all() and nc>0
    full_value,common_value=float(full.mean()),float(common.mean())
    result={'full':{'n':n,'value':full_value},'common':{'n':nc,'value':common_value},
            'additional':{'n':na,'value':float(additional.mean()) if na else None},
            'change':{'value':common_value-full_value}}
    if metric=='parallel':
        f_ci,c_ci,delta=parallel_intervals(common,additional,rng)
    else:
        k,kc,ka=(int(v.sum()) for v in (full,common,additional))
        assert k==kc+ka and np.isin(full,(0,1)).all()
        for key,count in zip(('full','common','additional'),(k,kc,ka),strict=True):
            result[key]['events']=count
        f_ci,c_ci=exact_rate(k,n),exact_rate(kc,nc)
        if na:
            cl,cu=exact_rate(kc,nc,.025)
            al,au=exact_rate(ka,na,.025)
            delta=[na/n*(cl-au),na/n*(cu-al)]
        else:
            delta=[0.,0.]
    for key,ci in zip(('full','common','change'),(f_ci,c_ci,delta),strict=True):
        result[key]['ci95']=ci
    return result


def summarize(data):
    assert data.uid.is_unique
    assert data.con.notna().equals(data['div'].notna()) and data.con.notna().equals(data.excess.notna())
    assert (data[['con','div']].dropna().sum(axis=1)<=1).all()
    data=data.assign(parallel=data.post_multi-data.pre_multi)
    rng=np.random.default_rng(SEED)
    result={}
    for field in FIELDS:
        group=data if field=='all' else data[data.field==field]
        common=group[list(METRICS)].notna().all(1)
        by_metric={}
        for metric in METRICS:
            by_metric[metric]=compare_metric(group[metric],common,metric,rng)
            if metric=='parallel':
                for name,subset in (('full',group.dropna(subset=['parallel'])),('common',group[common])):
                    by_metric[metric][name]['pre']=float(subset.pre_multi.mean())
                    by_metric[metric][name]['post']=float(subset.post_multi.mean())
        omitted=group[group.excess.notna()&~common]
        by_metric['availability']={
            'transport_outside_common':len(omitted),
            'transport_outside_common_missing_parallel':int(omitted.parallel.isna().sum()),
            'transport_outside_common_missing_branch':int(omitted.branch.isna().sum()),
            'outside_no_pre_multi_year':int(omitted.pre_multi_years.eq(0).sum()),
            'outside_no_post_multi_year':int(omitted.post_multi_years.eq(0).sum()),
            'outside_no_multi_year_either_side':int((omitted.pre_multi_years.eq(0)&omitted.post_multi_years.eq(0)).sum()),
        }
        result[field]=by_metric
    return result


def render(summary, output, fonts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    for font in fonts:
        fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=fonts[0]).get_name(),'DejaVu Sans'],
                         'text.color':'#152b45','axes.unicode_minus':True})
    fig,axes=plt.subplots(2,2,figsize=(12.8,7.8),facecolor='white')
    fig.subplots_adjust(left=.16,right=.96,top=.753,bottom=.13,hspace=.62,wspace=.52)
    fig.text(.045,.939,'공통 표본에서 연구 궤적은 다르게 보이는가',size=24,weight='bold')
    n_common=summary['all']['con']['common']['n']
    fig.text(.045,.884,f'첫 임용 전후 · 인용 DAG 1024차원 · 지표별 전체 표본과 공통 {n_common:,}행 비교',
             size=12,color='#48617f')
    colors={'full':'#678aaf','common':'#21599a'}
    handles=[Line2D([],[],color=colors[k],marker='o',markersize=6,
                    markerfacecolor='white' if k=='full' else colors[k],lw=1.5,label=l)
             for k,l in [('full','지표별 전체'),('common','세 분석 공통')]]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.038,.852),frameon=False,ncol=2,
               fontsize=11,handletextpad=.5,columnspacing=1.6)
    fig.text(.37,.833,'분야 옆 숫자는 전체 / 공통 교원 행 수',size=10,color='#48617f')

    for axis,(metric,title) in zip(axes.flat,METRICS.items(),strict=True):
        axis.set_title(title,loc='left',size=15,weight='bold',pad=8)
        ticks=[]
        for i,(f,label) in enumerate(FIELDS.items()):
            stat=summary[f][metric]
            y=4-i
            ticks.append(f"{label}   {stat['full']['n']:,} / {stat['common']['n']:,}")
            for key,dy in [('full',.15),('common',-.15)]:
                v=stat[key]['value']*100
                lo,hi=[100*x for x in stat[key]['ci95']]
                axis.hlines(y+dy,lo,hi,color=colors[key],lw=1.55,alpha=.9,zorder=2)
                axis.plot(v,y+dy,'o',ms=5.6,mew=1.15,mec=colors[key],
                          mfc='white' if key=='full' else colors[key],zorder=3,clip_on=False)
        axis.set_yticks(range(5),ticks[::-1])
        axis.set_ylim(-.55,4.55)
        intervals=[v[metric][key]['ci95'] for v in summary.values() for key in ('full','common')]
        lo=100*min(c[0] for c in intervals)
        hi=100*max(c[1] for c in intervals)
        margin=.05*(hi-lo)
        axis.set_xlim(min(0,lo-margin) if metric=='parallel' else 0,hi+margin)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=4))
        axis.grid(axis='x',color='#e3edf7',lw=.8,zorder=0)
        if metric=='parallel':
            axis.axvline(0,color='#94a8bf',lw=1.1,zorder=1)
        axis.set_xlabel('임용 후 − 전, 교원별 연간 병행률 평균 (%p)' if metric=='parallel' else '관측 전환 중 해당 신호 비율 (%)',
                        fontsize=9.6,color='#314d6d',labelpad=8)
        axis.tick_params(axis='both',length=0,pad=7,labelsize=9.5,labelcolor='#304964')
        for edge in ('top','right','left'):
            axis.spines[edge].set_visible(False)
        axis.spines['bottom'].set_color('#b9cce1')
    fig.text(.045,.042,'선은 각 추정치의 95% 구간이다. 두 표본의 겹침을 반영한 차이의 구간은 방법론 표에 제시한다.',
             size=10,color='#48617f')
    fig.savefig(output,dpi=170,facecolor='white')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('faculty_csv',type=Path)
    p.add_argument('summary',type=Path)
    p.add_argument('--figure',type=Path)
    p.add_argument('--font',type=Path,nargs='+',default=[
        Path('/Library/Fonts/NanumHumanRegular.ttf'),Path('/Library/Fonts/NanumHumanBold.ttf')])
    a=p.parse_args()
    summary=summarize(pd.read_csv(a.faculty_csv))
    a.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if a.figure:
        render(summary,a.figure,a.font)


if __name__=='__main__':
    main()
