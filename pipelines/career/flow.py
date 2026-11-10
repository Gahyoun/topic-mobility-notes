"""Estimate field, corpus and topic-conditioned background advection."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from motion import bootstrap

MODES=('all_corpus','field','local')
ADVECTION=('excess_dispersion','excess_rate','relative_con','relative_div')
METRICS=('observed_speed','background_speed','residual_speed','parallel','transverse','cosine')
FIELDS=('all','mathematics','physics','chemistry','biology')
MIN_PEERS=20
MIN_MASS=.50


def summarize_registry(frame,x,origin,span,k):
    """Cumulative peer sufficient statistics for all calendar-year ranges."""
    # Slot k aggregates the entire population of this registry.
    n=np.zeros((span,k+1),dtype=np.int64)
    sums=np.zeros((span,k+1,x.shape[1]),dtype=np.float64)
    yr=frame.year.to_numpy()-origin
    cluster=frame.cluster.to_numpy(dtype=int)
    vectors=x[frame.ix.to_numpy(dtype=int)]
    np.add.at(n,(yr,cluster),1)
    np.add.at(sums,(yr,cluster),vectors)
    n[:,k]=n[:,:k].sum(1)
    sums[:,k]=sums[:,:k].sum(1)
    return (np.concatenate((np.zeros_like(n[:1]),np.cumsum(n,axis=0))),
            np.concatenate((np.zeros_like(sums[:1]),np.cumsum(sums,axis=0))))


def excluding(registry,own,x,start,end,origin,k):
    before,after=start-origin,end-origin+1
    n=registry[0][after]-registry[0][before]
    s=registry[1][after]-registry[1][before]
    work=own[own.year.between(start,end)]
    v=x[work.ix.to_numpy(dtype=int)]
    cluster=work.cluster.to_numpy(dtype=int)
    np.add.at(n,cluster,-1)
    np.add.at(s,cluster,-v)
    n[k]-=len(work)
    s[k]-=v.sum(0)
    assert (n>=0).all()
    return n,s


def field_local(a,b,prior_mass,k):
    na,sa=a
    nb,sb=b
    support=(na[:k]>=MIN_PEERS)&(nb[:k]>=MIN_PEERS)&(prior_mass>0)
    mass=float(prior_mass[support].sum())
    global_vector=sb[k]/nb[k]-sa[k]/na[k] if min(na[k],nb[k])>=MIN_PEERS else None
    local_vector=None
    shifts=None
    if mass>=MIN_MASS and global_vector is not None:
        shifts=np.broadcast_to(global_vector,(k,len(global_vector))).copy()
        shifts[support]=sb[:k][support]/nb[:k][support,None]-sa[:k][support]/na[:k][support,None]
        local_vector=prior_mass@shifts
    return global_vector,local_vector,mass,shifts


def components(velocity,bg):
    speed=np.linalg.norm(velocity)
    b=np.linalg.norm(bg)
    if b<1e-12 or speed<1e-12:
        return None
    parallel=float(velocity@bg/b)
    transverse=float(np.linalg.norm(velocity-parallel*bg/b))
    residual=float(np.linalg.norm(velocity-bg))
    return np.array((speed,b,residual,parallel,transverse,np.clip(parallel/speed,-1.,1.)))


def mean_q(x):
    centered=x-x.mean(0)
    return 2*np.square(centered).sum()/(len(x)-1)


def run(db_file,output,coordinate='reference'):
    vector_table='work_coordinates' if coordinate=='reference' else 'dag_coordinates'
    flow_table='background_flow' if coordinate=='reference' else 'dag_background_flow'
    with sqlite3.connect(db_file.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        df=pd.read_sql_query('''SELECT uid,work,year,field,cluster,geometry
             FROM publication_metadata WHERE has_coordinate=1 ORDER BY uid,year,work''',db)
        works=pd.read_sql_query(f'SELECT work,vector FROM {vector_table} ORDER BY work',db)
        w=pd.read_sql_query('''SELECT t.uid,t.field,t.year_start,
                     p.year_start AS year_a,p.year_end AS end_a,c.year_end AS end_b
              FROM portfolio_transitions t JOIN portfolio_windows c USING(uid,year_start)
              JOIN portfolio_windows p ON p.uid=t.uid AND p.year_end+1=c.year_start
              WHERE t.geometry_complete=1 ORDER BY t.uid,t.year_start''',db)
    x=np.stack([np.frombuffer(v,dtype='<f4').astype(float) for v in works.vector])
    ix=pd.Series(np.arange(len(works)),index=works.work)
    df['ix']=df.work.map(ix).astype(int)
    df=df.sort_values(['year','work','uid'])
    unique_field=df.drop_duplicates(['field','work'])
    unique_all=df.drop_duplicates('work')
    origin=int(df.year.min())
    span=2025-origin+1
    k=int(df.cluster.max())+1
    global_reg=summarize_registry(unique_all,x,origin,span,k)
    field_regs={f:summarize_registry(g,x,origin,span,k) for f,g in unique_field.groupby('field')}
    field_works={f:g.set_index('work') for f,g in unique_field.groupby('field')}
    global_works=unique_all.set_index('work')
    links={u:g for u,g in df.groupby('uid')}
    stream=[]
    for start in (2001,2006,2011,2016):
        i=start-origin
        r=global_reg
        na,nb=r[0][i+5]-r[0][i],r[0][i+10]-r[0][i+5]
        sa,sb=r[1][i+5]-r[1][i],r[1][i+10]-r[1][i+5]
        keep=(na[:k]>=MIN_PEERS)&(nb[:k]>=MIN_PEERS)
        a,b=sa[:k][keep]/na[:k][keep,None],sb[:k][keep]/nb[:k][keep,None]
        stream.append({'start':start,'centers':a.tolist(),'velocity':((b-a)/5).tolist(),
                       'counts':na[:k][keep].tolist()})
    out=[]
    for uid,group in w.groupby('uid',sort=True):
        g=links[uid]
        f=g.field.iloc[0]
        focus=g.work.unique()
        own_all=global_works.loc[focus]
        own_field=field_works[f].loc[focus]
        for row in group.itertuples(index=False):
            a=g[g.geometry.eq(1)&g.year.between(row.year_a,row.end_a)]
            b=g[g.geometry.eq(1)&g.year.between(row.year_start,row.end_b)]
            assert len(a)>=2 and len(b)>=2
            dt=b.year.mean()-a.year.mean()
            observed=(x[b.ix].mean(0)-x[a.ix].mean(0))/dt
            prior=np.bincount(a.cluster.to_numpy(dtype=int),minlength=k)/len(a)
            field_a=excluding(field_regs[f],own_field,x,row.year_a,row.end_a,origin,k)
            field_b=excluding(field_regs[f],own_field,x,row.year_start,row.end_b,origin,k)
            all_a=excluding(global_reg,own_all,x,row.year_a,row.end_a,origin,k)
            all_b=excluding(global_reg,own_all,x,row.year_start,row.end_b,origin,k)
            fv,lv,mass,shifts=field_local(field_a,field_b,prior,k)
            av=(all_b[1][k]/all_b[0][k]-all_a[1][k]/all_a[0][k]) if min(all_a[0][k],all_b[0][k])>=MIN_PEERS else None
            record={'uid':uid,'field':f,'year_start':row.year_start,'dt':dt,'prior_mass_supported':mass}
            for name,v in zip(MODES,(av,fv,lv),strict=True):
                s=components(observed,v/dt) if v is not None else None
                record.update({name+'_'+m: None if s is None else s[i] for i,m in enumerate(METRICS)})
            record.update({m:None for m in ADVECTION})
            if shifts is not None:
                advected=x[a.ix]+shifts[a.cluster.to_numpy(dtype=int)]
                np.testing.assert_allclose((advected.mean(0)-x[a.ix].mean(0))/dt,lv/dt,atol=1e-12,rtol=0)
                q_pred,q_observed=mean_q(advected),mean_q(x[b.ix])
                if q_pred+q_observed>1e-12:
                    delta=(q_observed-q_pred)/(q_observed+q_pred)
                    record.update(excess_dispersion=delta,excess_rate=(q_observed-q_pred)/dt,
                                  relative_con=int(delta<=-.10),relative_div=int(delta>=.10))
            out.append(record)
    result=pd.DataFrame(out)
    assert len(result)==len(w)
    rng=np.random.default_rng(321110)
    stats={}
    for field in FIELDS:
        sub=result if field=='all' else result[result.field.eq(field)]
        common=sub.dropna(subset=[m+'_residual_speed' for m in MODES])
        means=common.groupby('uid')[[m+'_'+k for m in MODES for k in METRICS]].mean()
        interval=bootstrap(means.to_numpy(),rng).reshape(len(MODES),len(METRICS),3)
        row={'transitions':len(sub),'faculty':int(sub.uid.nunique()),'common_transitions':len(common),
             'common_faculty':len(means),'local_unsupported':int(sub.local_residual_speed.isna().sum()),
             'mass_75_transitions':int(common.prior_mass_supported.ge(.75).sum()),
             'supported_mass':float(common.prior_mass_supported.mean()),
             'modes':{m:dict(zip(METRICS,interval[i].tolist())) for i,m in enumerate(MODES)}}
        sample=common.dropna(subset=['excess_dispersion'])
        avg=sample.groupby('uid')[list(ADVECTION)].mean()
        row['advection_pairs']=len(sample)
        row['advection_faculty']=len(avg)
        row['advection']=dict(zip(ADVECTION,bootstrap(avg.to_numpy(),rng).tolist()))
        row['advection_con_n']=int(sample.relative_con.sum())
        row['advection_div_n']=int(sample.relative_div.sum())
        row['advection_75_estimate']=float(sample.loc[sample.prior_mass_supported.ge(.75)].groupby(
            'uid').excess_dispersion.mean().mean())
        stats[field]=row
    stats['stream']=stream
    with sqlite3.connect(db_file) as db:
        result.to_sql(flow_table,db,index=False,if_exists='fail')
        db.execute(f'CREATE UNIQUE INDEX {flow_table}_key ON {flow_table}(uid,year_start)')
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    output.write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(stats['all'],ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('database',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--coordinate', choices=('reference','dag'), default='reference')
    a=p.parse_args()
    run(a.database,a.output,a.coordinate)


if __name__=='__main__':
    main()
