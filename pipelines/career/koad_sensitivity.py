"""Compare the two saved GCC analyses on a common set of faculty rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from appointment_dag import exact_difference
from motion import bootstrap
from native_dag import DIMENSIONS

FIELDS = {'all': '전체', 'mathematics': '수학', 'physics': '물리', 'chemistry': '화학', 'biology': '생물'}
METRICS = {'con': '수렴', 'div': '확산', 'parallel': '연간 병행', 'branch': '분기'}
METADATA = ['field', 'dated', 'event', 'basis', 'followup']
COLUMNS = [*METADATA, 'uid', 'con', 'div', 'pre_multi', 'post_multi', 'branch']
SEED = 6111126


def read_result(directory, dimension):
    frame = pd.read_csv(directory / f'faculty-d{dimension:04}.csv', usecols=COLUMNS)
    frame = frame.set_index('uid', verify_integrity=True).sort_index()
    if frame.index.hasnans or not frame.field.isin(set(FIELDS) - {'all'}).all():
        raise ValueError('Missing faculty identifier or unrecognized field')
    if not frame.con.isna().equals(frame['div'].isna()):
        raise ValueError('Convergence and divergence must share availability')
    for column in ('con', 'div', 'branch'):
        if not frame[column].dropna().isin((0, 1)).all():
            raise ValueError(f'{column} must be a binary indicator')
    for column in ('pre_multi', 'post_multi'):
        if not frame[column].dropna().between(0, 1).all():
            raise ValueError(f'{column} must be a share')
    return frame.assign(parallel=frame.post_multi - frame.pre_multi)


def common_results(before, after):
    versions = {d: (read_result(before, d), read_result(after, d)) for d in DIMENSIONS}
    roster = versions[DIMENSIONS[0]][0][METADATA]
    eligible = []
    for pair in versions.values():
        for frame in pair:
            if not frame[METADATA].equals(roster):
                raise ValueError('The faculty roster, fields and appointment anchors must agree')
        eligible.append(tuple(frame.dropna(subset=list(METRICS)).index for frame in pair))
    common = roster.index.intersection(eligible[0][0])
    for old, new in eligible:
        common = common.intersection(old).intersection(new)
    availability = {d: {'before': len(a), 'after': len(b), 'paired': len(a.intersection(b))}
                    for d, (a,b) in zip(DIMENSIONS, eligible, strict=True)}
    return versions, roster.loc[common, 'field'], availability


def summarize(before, after):
    versions, common_fields, availability = common_results(before, after)
    common = common_fields.index
    rng = np.random.default_rng(SEED)
    output = {}
    for field in FIELDS:
        ids = common if field == 'all' else common[common_fields == field]
        n = len(ids)
        data = {'n': n, 'dimensions': {}}
        output[field] = data
        if not n:
            raise ValueError(f'No common faculty rows for {field}')
        # Share the paired faculty resamples across dimensions.
        parallel = np.column_stack([
            (new.loc[ids, 'parallel'] - old.loc[ids, 'parallel']).to_numpy()
            for old, new in versions.values()
        ])
        intervals = bootstrap(parallel, rng)
        for j, (dimension, (old, new)) in enumerate(versions.items()):
            results = {}
            data['dimensions'][dimension] = results
            for metric in METRICS:
                a, b = (frame.loc[ids, metric].to_numpy() for frame in (old, new))
                delta = b - a
                result = {'before': float(a.mean()), 'after': float(b.mean()),
                          'changed': int(np.count_nonzero(np.abs(delta) > 1e-12))}
                if metric == 'parallel':
                    result['delta_ci95'] = intervals[j].tolist()
                else:
                    gain = int((delta == 1).sum())
                    loss = int((delta == -1).sum())
                    result.update(gain=gain, loss=loss, delta_ci95=exact_difference(gain, loss, n))
                results[metric] = result
    return {'fields': output, 'availability': availability}


def render(summary, output, font):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.ticker import MaxNLocator

    fontManager.addfont(font)
    plt.rcParams.update({'font.family': [FontProperties(fname=font).get_name(), 'DejaVu Sans'],
                         'text.color': '#172d47'})
    rows = summary['fields']
    fig, axes = plt.subplots(len(FIELDS), len(METRICS), figsize=(13.3, 9.6), sharex='col')
    fig.subplots_adjust(left=.18, right=.97, top=.845, bottom=.13, wspace=.17, hspace=.20)
    fig.text(.045, .961, '동명이인 정정 후 연구 궤적은 얼마나 달라졌나', fontsize=23, weight='bold')
    fig.text(.045, .920, f"두 버전과 여섯 차원의 공통 교원 {rows['all']['n']:,}행 · 점은 정정 후 − 정정 전, 선은 95% 구간",
             fontsize=12, color='#4a617d')
    fig.text(.045, .890, '남색은 1024차원 · 세로 점선은 변화 없음', fontsize=11, color='#4a617d')
    for j, (metric, title) in enumerate(METRICS.items()):
        axes[0,j].set_title(title, fontsize=15, weight='bold', pad=9)
        bounds = [x for row in rows.values() for dim in DIMENSIONS
                  for x in row['dimensions'][dim][metric]['delta_ci95']]
        extent = 110 * max(map(abs, bounds))
        axes[0,j].set_xlim(-extent, extent)
        axes[-1,j].xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True, prune='both'))
        axes[-1,j].set_xlabel('차이 (%p)', fontsize=10, color='#425a76')
    for i, (field, label) in enumerate(FIELDS.items()):
        row = rows[field]
        left = axes[i,0]
        left.text(-.53,.67,label,transform=left.transAxes,fontsize=13,weight='bold',ha='left')
        left.text(-.53,.39,f"n = {row['n']:,}",transform=left.transAxes,fontsize=10,ha='left',color='#4a617d')
        for j, metric in enumerate(METRICS):
            ax = axes[i,j]
            ax.axvline(0, color='#6d819a', lw=.9, ls=(0,(2,3)), zorder=1)
            ax.set_axisbelow(True)
            ax.grid(axis='x',color='#e9eff6',lw=.7)
            ax.set_ylim(len(DIMENSIONS)-.4,-.6)
            ax.set_yticks(range(len(DIMENSIONS)), [str(d) if j==0 else '' for d in DIMENSIONS])
            ax.tick_params(axis='y',length=0,labelsize=8,pad=5,colors='#435c79')
            ax.tick_params(axis='x',length=2,labelsize=9,colors='#435c79',labelbottom=i==len(FIELDS)-1)
            for spine in ('top','left','right'):
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#dce5ef')
            for k,d in enumerate(DIMENSIONS):
                estimate,lo,hi = (100*v for v in row['dimensions'][d][metric]['delta_ci95'])
                color = '#174a82' if d==1024 else '#7897b7'
                ax.hlines(k,lo,hi,color=color,lw=1.6 if d==1024 else 1.15,zorder=2)
                ax.plot(estimate,k,'o',color=color,ms=4.4 if d==1024 else 3.4,zorder=3)
    fig.text(.045,.062,'수렴·확산·분기 · 교원별 판정의 짝차이와 보수적 정확 이항구간',
             color='#4a617d', fontsize=10)
    fig.text(.045,.038,'연간 병행 · 교원별 임용 후−전 병행률의 정정 전후 짝차이와 교원 재표집 구간',
             color='#4a617d', fontsize=10)
    fig.savefig(output,dpi=165,facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('before', 'after', 'summary'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--font', type=Path, default=Path('/Library/Fonts/NanumHumanRegular.ttf'))
    args = parser.parse_args()
    result = summarize(args.before, args.after)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if args.figure:
        render(result, args.figure, args.font)


if __name__ == '__main__':
    main()
