"""Measure annual parallel-topic sensitivity to clustering seeds on frozen data."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from numba import njit
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from koad_parallel import connect, work_links
from koad_sensitivity import FIELDS, common_results
from native_dag import DIMENSIONS
from native_flow import groups_in

ORIGINAL_SEED = 321110
SEED_GENERATOR = 6911126
N_CLUSTERS = 64
PHASES = ['pre_multi', 'post_multi']


def original_anchors(portfolio):
    with connect(portfolio) as db:
        n = db.execute('SELECT count(*) FROM dag_native_coordinates').fetchone()[0]
        x = np.empty((n, 64), dtype=np.float32)
        labels = np.empty(n, dtype=np.int32)
        works = []
        for i, (work, cluster, vector) in enumerate(db.execute(
            'SELECT work,cluster,vector FROM dag_native_coordinates ORDER BY work'
        )):
            works.append(work)
            x[i] = np.frombuffer(vector, dtype='<f4', count=64)
            labels[i] = cluster
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return pd.Index(works), x, labels


def fit(anchors, seed):
    return MiniBatchKMeans(n_clusters=N_CLUSTERS, n_init=3, random_state=int(seed),
                           batch_size=4096).fit_predict(anchors)


def prepare_cells(portfolio, works, gcc_works, events):
    frame = work_links(portfolio).reset_index()
    frame = frame[frame.uid.isin(events.index)&frame.work.isin(gcc_works)].copy()
    frame['faculty'] = events.index.get_indexer(frame.uid)
    frame['event'] = frame.uid.map(events)
    frame['ix'] = works.get_indexer(frame.work)
    if (frame.ix<0).any():
        raise ValueError('A GCC paper is missing from the original corpus')
    frame['phase'] = (frame.year>frame.event).astype(int)
    totals = np.zeros(2*len(events), dtype=np.int64)
    codes, cells = [], []
    for (_, year), group in frame.groupby(['uid','year'], sort=False):
        if len(group)<2 or year==group.event.iloc[0]:
            continue
        code = 2*group.faculty.iloc[0]+group.phase.iloc[0]
        totals[code] += 1
        if len(group)>=4:
            codes.append(code)
            cells.append(group.ix.to_numpy())
    if (totals==0).any():
        raise ValueError('A faculty row lacks an eligible year on one side of appointment')
    return np.asarray(codes), cells, totals


def distances_for_cells(root, works, cells):
    # Precompute fixed within-year squared distances for the fast scoring path.
    squared = np.concatenate(([0], np.cumsum([len(c)**2 for c in cells])))
    distances = np.empty((len(DIMENSIONS), squared[-1]), dtype=np.float64)
    vectors = []
    for j,d in enumerate(DIMENSIONS):
        folder = root/f'd{d:04}'
        with np.load(folder/'paper_axis.npz', allow_pickle=False) as axis:
            if not pd.Index(axis['work']).equals(works):
                raise ValueError(f'Original work order changed at dimension {d}')
        x = np.load(folder/'paper_vectors.npy', mmap_mode='r', allow_pickle=False)
        if x.shape != (len(works),d):
            raise ValueError(f'Tensor has an unexpected shape at dimension {d}')
        vectors.append(x)
        for i,idx in enumerate(cells):
            cloud = x[idx].astype(np.float64)
            norm = np.einsum('ij,ij->i', cloud, cloud)
            d2 = np.maximum(norm[:,None]+norm[None,:]-2*(cloud@cloud.T), 0)
            distances[j,squared[i]:squared[i+1]] = d2.ravel()
    return squared, distances, vectors


@njit
def classify_years(labels, flat_cells, cell_bounds, squared_bounds, distances):
    """Apply the existing cluster-size and separation rules from squared distances.

    Status 2 requests the original point-coordinate calculation near the cutoff.
    """
    flags = np.zeros((len(cell_bounds)-1, distances.shape[0]), dtype=np.uint8)
    for i in range(len(flags)):
        topics = labels[flat_cells[cell_bounds[i]:cell_bounds[i+1]]]
        n = len(topics)
        counts = np.bincount(topics, minlength=N_CLUSTERS)
        present = np.flatnonzero((counts>=2)&(counts>=.15*n))
        if len(present)<2:
            continue
        mapping = np.full(N_CLUSTERS,-1)
        for j,k in enumerate(present):
            mapping[k] = j
        group = mapping[topics]
        start = squared_bounds[i]
        for d in range(distances.shape[0]):
            sums = np.zeros((len(present),len(present)))
            for p in range(n):
                a = group[p]
                if a>=0:
                    for q in range(p+1,n):
                        b = group[q]
                        if b>=0:
                            sums[a,b] += distances[d,start+p*n+q]
            status = 0
            for a in range(len(present)):
                na = counts[present[a]]
                for b in range(a):
                    nb = counts[present[b]]
                    within = sums[a,a]/(na*na)+sums[b,b]/(nb*nb)
                    cross = (sums[a,b]+sums[b,a])/(na*nb)
                    margin = cross-within-max(within,1e-12)
                    if margin>2e-6:
                        status = 1
                        break
                    if abs(margin)<=2e-6:
                        status = 2
                if status==1:
                    break
            flags[i,d] = status
    return flags


def score(labels, cells, codes, totals, squared, distances, vectors):
    flat = np.concatenate(cells)
    bounds = np.concatenate(([0], np.cumsum([len(c) for c in cells])))
    flags = classify_years(labels, flat, bounds, squared, distances)
    for i,d in zip(*np.where(flags==2),strict=True):
        idx = cells[i]
        _, separation = groups_in(labels[idx], vectors[d][idx], N_CLUSTERS)
        flags[i,d] = separation>=1
    numerator = np.stack([np.bincount(codes, weights=flag, minlength=len(totals)) for flag in flags.T],axis=1)
    return (numerator/totals[:,None]).reshape(-1,2,len(DIMENSIONS))


def collect(portfolio, full_tensors, gcc_tensors, before, after, count):
    versions, fields, _ = common_results(before, after)
    n = len(fields)
    events = versions[DIMENSIONS[0]][0].loc[fields.index,'event']
    with np.load(gcc_tensors/'d0064/paper_axis.npz', allow_pickle=False) as axis:
        gcc_works = pd.Index(axis['work'])
        gcc_labels = axis['cluster']
    works, x, archived_labels = original_anchors(portfolio)
    gcc_index = works.get_indexer(gcc_works)
    if (gcc_index<0).any() or not np.array_equal(archived_labels[gcc_index],gcc_labels):
        raise ValueError('The archived original GCC labels disagree with the original corpus')
    codes, cells, totals = prepare_cells(portfolio, works, gcc_works, events)
    with threadpool_limits(limits=4, user_api='blas'):
        squared, dist, vectors = distances_for_cells(full_tensors,works,cells)
        baseline_labels = fit(x, ORIGINAL_SEED)
        if not np.array_equal(baseline_labels,archived_labels):
            raise ValueError('The original seed did not reproduce archived cluster labels')
        base = score(baseline_labels,cells,codes,totals,squared,dist,vectors)
        before_phases, after_phases = (
            np.stack([v[k].loc[fields.index,PHASES].to_numpy() for v in versions.values()],axis=2)
            for k in (0,1)
        )
        np.testing.assert_allclose(base,before_phases,rtol=0,atol=1e-12)
        print(f'Original seed reproduced all {n:,} paired faculty rates at six dimensions.',flush=True)
        baseline = base[:,1]-base[:,0]
        corrected = after_phases[:,1]-after_phases[:,0]
        masks = {f:np.ones(n,dtype=bool) if f=='all' else fields.eq(f).to_numpy() for f in FIELDS}
        rng = np.random.default_rng(SEED_GENERATOR)
        seeds = rng.choice(2**31-1,size=count+1,replace=False)
        seeds = seeds[seeds!=ORIGINAL_SEED][:count]
        samples = {f:[] for f in FIELDS}
        for i,seed in enumerate(seeds,1):
            labels = fit(x,seed)
            rates = score(labels,cells,codes,totals,squared,dist,vectors)
            change = (rates[:,1]-rates[:,0])-baseline
            for field,mask in masks.items():
                samples[field].append(change[mask].mean(0).tolist())
            if i%20==0 or i==count:
                print(f'Scored {i}/{count} alternative seeds.',flush=True)
    output = {}
    for field,mask in masks.items():
        empirical = np.asarray(samples[field])
        observed = (corrected-baseline)[mask].mean(0)
        quantiles = np.quantile(empirical,[.025,.5,.975],axis=0)
        output[field] = {'n':int(mask.sum()),'dimensions':{
            d:{'koad':float(observed[j]),'seed_quantiles':quantiles[:,j].tolist(),
               'seed_min_max':[float(empirical[:,j].min()),float(empirical[:,j].max())],
               'seed_at_least_koad':int((np.abs(empirical[:,j])+1e-12>=abs(observed[j])).sum())}
            for j,d in enumerate(DIMENSIONS)
        },'seed_differences':samples[field]}
    return {'fields':output,'seeds':seeds.tolist(),'original_seed':ORIGINAL_SEED,
            'seed_generator':SEED_GENERATOR,'training_papers':len(works),'gcc_papers':len(gcc_works),
            'faculty_years':int(totals.sum()),'candidate_years':len(cells),
            'packages':{p:version(p) for p in ('numpy','pandas','scikit-learn','numba')}}


def render(summary, figure, font):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D

    fontManager.addfont(font)
    plt.rcParams.update({'font.family':[FontProperties(fname=font).get_name(),'DejaVu Sans'],
                         'text.color':'#172d47'})
    rows = summary['fields']
    fig,axes = plt.subplots(1,len(FIELDS),figsize=(13,4.5),sharey=True,sharex=True)
    fig.subplots_adjust(left=.074,right=.97,top=.68,bottom=.22,wspace=.18)
    fig.text(.045,.945,'군집화 시드만 바꿔도 연간 병행률 차이가 생기는가',fontsize=21,weight='bold')
    fig.text(.045,.869,f"공통 교원 {rows['all']['n']:,}행 · 좌표와 논문 고정 · 대체 시드 {len(summary['seeds']):,}개",
             fontsize=12,color='#425a76')
    handles = [Line2D([0],[0],marker='o',color='#6683a3',label='시드 차이의 중앙값과 가운데 95%'),
               Line2D([0],[0],marker='D',color='none',markerfacecolor='#a84f20',markeredgecolor='#a84f20',
                      label='실제 KOAD 정정 전후 차이')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.039,.829),ncol=2,frameon=False,fontsize=11)
    for ax,(field,label) in zip(axes,FIELDS.items(),strict=True):
        row = rows[field]
        ax.set_title(f"{label}  ·  {row['n']:,}행",fontsize=12,weight='bold',pad=9)
        ax.axvline(0,color='#8396ab',lw=.9,ls=(0,(2,3)),zorder=1)
        ax.grid(axis='x',color='#e8eff6')
        ax.set_ylim(len(DIMENSIONS)-.45,-.55)
        ax.set_yticks(range(len(DIMENSIONS)),[str(d) for d in DIMENSIONS])
        ax.tick_params(axis='y',length=0,labelsize=9,colors='#435c79')
        ax.tick_params(axis='x',length=2,labelsize=9,colors='#435c79')
        ax.set_xlabel('차이 (%p)',fontsize=9)
        for spine in ('top','left','right'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#dce5ef')
        for i,d in enumerate(DIMENSIONS):
            cell = row['dimensions'][d]
            lo,mid,hi = (100*v for v in cell['seed_quantiles'])
            ax.hlines(i,lo,hi,color='#6683a3',lw=1.7,zorder=2)
            ax.plot(mid,i,'o',ms=3.8,color='#6683a3',zorder=3)
            ax.plot(100*cell['koad'],i,'D',ms=4.5,color='#a84f20',markeredgewidth=.5,
                    markeredgecolor='white',zorder=4)
    axes[0].set_ylabel('임베딩 차원',fontsize=9)
    fig.text(.045,.068,'모든 차이는 정정 전 원래 시드를 기준으로 계산. 선은 대체 시드 분포의 경험적 분위수.',
             fontsize=10,color='#425a76')
    fig.savefig(figure,dpi=170,facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original_portfolio','full_tensors','gcc_tensors','before_rates','after_rates','summary'):
        p.add_argument(name,type=Path)
    p.add_argument('--seed-count',type=int,default=200)
    p.add_argument('--figure',type=Path)
    p.add_argument('--font',type=Path,default=Path('/Library/Fonts/NanumHumanRegular.ttf'))
    a = p.parse_args()
    if a.seed_count<1:
        p.error('Use at least one alternate seed.')
    result = collect(a.original_portfolio,a.full_tensors,a.gcc_tensors,a.before_rates,a.after_rates,a.seed_count)
    a.summary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if a.figure:
        render(result,a.figure,a.font)


if __name__=='__main__':
    main()
