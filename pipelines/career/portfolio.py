"""Roster-complete bibliographic trajectories relative to the field-period paper distribution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

import career
from transport import projection_components
from portfolio_views import create_views

CUTOFF = 2025
WIDTH = 5
MIN_PEERS = 20
K = 64
BAND = .10
FIELDS = ('mathematics','physics','chemistry','biology')
OBSERVATION = ('zero','unpositioned','singleton','multiple')


def fit_background(papers, coordinates):
    """One paper per field/period, with the focal teacher's works left out."""
    papers=papers.copy()
    papers['period']=papers.year//WIDTH*WIDTH
    z=np.full((len(papers),coordinates.shape[1]),np.nan)
    papers['row']=np.arange(len(papers))
    papers['peers']=0
    groups=papers.groupby(['field','period'],sort=True)
    for _,group in groups:
        unique=group.drop_duplicates('ix')
        overall=coordinates[unique.ix.to_numpy()]
        total=overall.sum(0)
        energy=np.square(overall).sum()
        for _,g in group.groupby('uid',sort=False):
            at=g.row.to_numpy()
            x=coordinates[g.ix.to_numpy()]
            n=len(overall)-len(x)
            papers.loc[at,'peers']=n
            if n<MIN_PEERS:
                continue
            mu=(total-x.sum(0))/n
            r2=(energy-np.square(x).sum())/n-mu@mu
            if r2>1e-12:
                z[at]=(x-mu)/np.sqrt(r2)
    return papers,z


def scatter(x):
    if not len(x):
        return None,None,None
    mu=x.mean(0)
    b=float(np.square(x-mu).sum()/len(x))
    return mu,b,2*b*len(x)/(len(x)-1) if len(x)>=2 else None


def group_stats(frame,z):
    supported={}
    for k,g in frame.groupby('cluster',sort=True):
        if len(g)>=2 and len(g)/len(frame)>=.15:
            mu,b,_=scatter(z[g.row.to_numpy(dtype=int)])
            supported[int(k)]=(g,mu,b)
    spread=None
    for i,(_,a,sa) in enumerate(supported.values()):
        for _,b,sb in list(supported.values())[i+1:]:
            q=float(np.linalg.norm(a-b)/np.sqrt(max(sa+sb,1e-12)))
            spread=q if spread is None else max(spread,q)
    if frame.empty:
        return supported,spread,None
    counts=frame.cluster.value_counts(normalize=True).to_numpy()
    return supported,spread,float(1/(counts@counts))


def transition(prev,curr,previous_groups,current_groups,seen,direct):
    uid=curr['uid']
    old_clusters=set(previous_groups)
    new_clusters=set(current_groups)-seen
    retained=old_clusters & set(current_groups)
    earlier=set().union(*(set(previous_groups[k][0].work) for k in retained))
    new_papers=[p for k in new_clusters for p in current_groups[k][0].work]
    branched=sum(bool(direct.get((uid,p),set()) & earlier) for p in new_papers)
    n=curr['n_geometry']
    con_div=None
    if prev['q'] is not None and curr['q'] is not None and prev['q']+curr['q']>1e-12:
        con_div=(curr['q']-prev['q'])/(curr['q']+prev['q'])
    raw_con_div=None
    if prev['raw_q'] is not None and curr['raw_q'] is not None and prev['raw_q']+curr['raw_q']>1e-12:
        raw_con_div=(curr['raw_q']-prev['raw_q'])/(curr['raw_q']+prev['raw_q'])
    parts=[None]*4
    if prev['n_geometry']>=2 and n>=2:
        a,b=prev['projection'],curr['projection']
        parts=(prev['dim']*projection_components(a,b,len(a),len(b)).mean(1)).tolist()
        np.testing.assert_allclose(sum(parts[:3]),parts[3],atol=1e-11,rtol=0)
    comparable=(prev['exposure']==curr['exposure']==WIDTH and con_div is not None)
    return {'uid':uid,'field':curr['field'],'year_start':curr['year_start'],'previous_state':prev['state'],
            'state':curr['state'],'geometry_complete':int(comparable),'con_div':con_div,'raw_con_div':raw_con_div,
            'con':int(comparable and con_div<=-BAND),'div':int(comparable and con_div>=BAND),
            'retained_groups':len(retained),'new_groups':len(new_clusters),
            'new_papers':len(new_papers),'branch_papers':branched,
            'branch_share':branched/n if n and prev['n_geometry']>=2 else None,
            'branch':int(comparable and branched>0),
            **dict(zip(('location','scale','shape','transport'),parts))}


def make_profile(cohort,roster,linked,papers,centers,dates,directory,metadata_db):
    with sqlite3.connect(cohort.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        citing=pd.read_sql_query('''SELECT a.professor_uid uid, a.work_id citing, b.work_id cited
          FROM faculty_publication a JOIN outgoing_reference r ON a.work_id=r.work_id
          JOIN faculty_publication b ON a.professor_uid=b.professor_uid AND b.work_id=r.reference_id
          WHERE a.review_only=0 AND b.review_only=0 AND a.year>b.year''',db)
    direct=defaultdict(set)
    for u,a,b in citing.itertuples(index=False,name=None):
        direct[u,a].add(b)
    unique=papers.drop_duplicates('ix')
    with threadpool_limits(limits=4):
        model=MiniBatchKMeans(n_clusters=K,random_state=41110,batch_size=4096,n_init=2,max_iter=60)
        model.fit(centers[unique.ix.to_numpy()])
        clusters=model.predict(centers)
    paper,z=fit_background(papers.reset_index(drop=True),centers)
    paper['cluster']=clusters[paper.ix.to_numpy()]
    paper['geometry']=np.isfinite(z).all(1)
    dim=centers.shape[1]
    rand=np.random.default_rng(251110)
    basis=np.concatenate([np.linalg.qr(rand.normal(size=(dim,dim)))[0] for _ in range(4)],axis=1)
    proj=z@basis
    linked=linked.copy()
    tagged=paper[['uid','work','cluster','period','peers','geometry','row','n_positioned_refs']]
    bibliog=linked.merge(tagged,on=['uid','work'],how='left',validate='one_to_one')
    with sqlite3.connect(cohort.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as source:
        coverage=pd.read_sql_query('SELECT work_id AS work,state AS reference_state,n_references FROM reference_coverage',source)
    bibliog=bibliog.merge(coverage,on='work',how='left',validate='many_to_one')
    with sqlite3.connect(metadata_db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as source:
        meta=pd.read_sql_query('''
          SELECT work AS work,title,doi,publication_date,topic_name,subfield_name FROM (
            SELECT work_id AS work,title,doi,publication_date,topic_name,subfield_name,
                   row_number() OVER (PARTITION BY work_id ORDER BY
                     NULLIF(title,'') IS NULL, NULLIF(doi,'') IS NULL, professor_uid) AS row_number
            FROM work_authorship_evidence
          ) WHERE row_number=1
        ''',source)
    bibliog=bibliog.merge(meta,on='work',how='left',validate='many_to_one')
    bibliog['geometry']=bibliog['geometry'].eq(True)
    bibliog['has_coordinate']=bibliog.row.notna()
    starts=linked.groupby('uid').year.min().to_dict()
    dates=dates.set_index('uid')
    roster=roster.copy()
    roster['appointment_year']=roster.uid.map(dates.anchor)
    roster['appointment_basis']=roster.uid.map(dates.basis)
    windows,transitions,annual,profiles=[],[],[],[]
    by_uid={u:g for u,g in bibliog.groupby('uid',sort=False)}
    for row in roster.itertuples(index=False):
        uid,field=row.uid,row.field
        first=starts.get(uid)
        if first is None:
            profiles.append((uid,field,None,None,None,0,0,0,0,0))
            continue
        pubs=by_uid[uid]
        last=int(pubs.year.max())
        years=pubs.groupby('year')
        for year in range(first,CUTOFF+1):
            g=years.get_group(year) if year in years.groups else None
            annual.append((uid,field,year,len(g) if g is not None else 0,
                           int(g.has_coordinate.sum()) if g is not None else 0,
                           int(g.geometry.sum()) if g is not None else 0))
        w=(pubs.year-first)//WIDTH
        groups={int(k):g for k,g in pubs.groupby(w)}
        seen=set()
        prev=None
        prev_groups={}
        for k,year in enumerate(range(first,CUTOFF+1,WIDTH)):
            g=groups.get(k,bibliog.iloc[:0])
            use=g[g.geometry]
            ids=use.row.to_numpy(dtype=int)
            n=len(use)
            _,b,q=scatter(z[ids])
            _,_,raw_q=scatter(centers[use.ix.to_numpy(dtype=int)])
            supported,separation,effective=group_stats(use,z)
            n_coord=int(g.has_coordinate.sum())
            state='zero' if g.empty else 'unpositioned' if n==0 else 'singleton' if n==1 else 'multiple'
            current={'uid':uid,'field':field,'year_start':year,'year_end':min(year+WIDTH-1,CUTOFF),
                     'exposure':min(WIDTH,CUTOFF-year+1),'n_papers':len(g),'n_coordinate':n_coord,
                     'n_geometry':n,'state':state,'radius2':b,'q':q,'raw_q':raw_q,'effective_groups':effective,
                     'supported_groups':len(supported),'separation':separation,'multi':int(separation is not None and separation>=1.),
                     'projection':proj[ids],'dim':dim}
            if prev is not None:
                transitions.append(transition(prev,current,prev_groups,supported,seen,direct))
            windows.append({k:v for k,v in current.items() if k not in {'projection','dim'}})
            seen.update(use.cluster.astype(int))
            prev,prev_groups=current,supported
        profiles.append((uid,field,first,last,CUTOFF-last,len(pubs),int(pubs.has_coordinate.sum()),
                         int(pubs.geometry.sum()),int(last<=CUTOFF-3),int(last<=CUTOFF-5)))
    profile=pd.DataFrame(profiles,columns=['uid','field','first_year','last_year','observed_tail_years',
                                         'n_papers','n_coordinate','n_geometry','zero_last_3','zero_last_5'])
    profile=profile.merge(roster[['uid','appointment_year','appointment_basis','n_supported_refs']],on='uid',
                          validate='one_to_one')
    annual=pd.DataFrame(annual,columns=['uid','field','year','n_papers','n_coordinate','n_geometry'])
    windows=pd.DataFrame(windows)
    trans=pd.DataFrame(transitions)
    assert len(profile)==3937 and profile.uid.is_unique
    assert len(bibliog)==int(profile.n_papers.sum())==int(annual.n_papers.sum())==int(windows.n_papers.sum())
    assert int(profile.n_coordinate.sum())==len(papers)
    assert int((profile.n_papers>0).sum())==3455
    metrics={}
    for field in ('all',*FIELDS):
        sel=lambda t: t if field=='all' else t[t.field.eq(field)]
        p,a,w,t=map(sel,(profile,annual,windows,trans))
        good=t[t.geometry_complete.eq(1)]
        comparable_raw=good[good.raw_con_div.notna()]
        r=np.sign(comparable_raw.raw_con_div.to_numpy())
        q=np.sign(comparable_raw.con_div.to_numpy())
        flip=int(np.count_nonzero(r*q<0))
        valid=w[w.exposure.eq(WIDTH) & w.state.eq('multiple')]
        parts=good[['location','scale','shape','transport']].mean()
        metrics[field]={
            'roster':len(p),'linked':int(p.n_papers.gt(0).sum()),'with_reference':int(p.n_supported_refs.gt(0).sum()),
            'with_coordinate':int(p.n_coordinate.gt(0).sum()),'with_geometry':int(p.n_geometry.gt(0).sum()),
            'with_geometry_2_total':int(p.n_geometry.ge(2).sum()),
            'papers_unique':int(sel(bibliog).work.nunique()),'papers_positioned_unique':int(sel(paper).work.nunique()),
            'links':int(p.n_papers.sum()),
            'title_unique':int(sel(bibliog).loc[lambda g:g.title.fillna('').ne(''),'work'].nunique()),
            'doi_unique':int(sel(bibliog).loc[lambda g:g.doi.fillna('').ne(''),'work'].nunique()),
            'year_rows':len(a),'zero_years':int(a.n_papers.eq(0).sum()),
            'zero_last_3':int(p.zero_last_3.sum()),'zero_last_5':int(p.zero_last_5.sum()),
            'window_states':{s:int(w.state.eq(s).sum()) for s in OBSERVATION},
            'zero_windows_teachers':int(w.loc[w.state.eq('zero'),'uid'].nunique()),
            'transitions_to_zero':int((t.previous_state.ne('zero') & t.state.eq('zero')).sum()),
            'transitions_from_zero':int((t.previous_state.eq('zero') & t.state.ne('zero')).sum()),
            'geometry_pairs_all':int(t.con_div.notna().sum()),
            'geometry_pairs_complete':len(good),'raw_comparable':len(comparable_raw),'raw_sign_flips':flip,'paired_faculty':int(good.uid.nunique()),
            'con':int(good.con.sum()),'div':int(good['div'].sum()),
            'multi_windows':int(valid.multi.sum()),'multi_denom':len(valid),
            'multi_teachers':int(valid.loc[valid.multi.eq(1),'uid'].nunique()),
            'branch':int(good.branch.sum()),'branch_teachers':int(good.loc[good.branch.eq(1),'uid'].nunique()),
            'branch_possible':int((good.new_groups.gt(0)&good.retained_groups.gt(0)).sum()),
            'transport_share':{k:float(parts[k]/parts['transport']) for k in ('location','scale','shape')},
            'threshold_counts':{str(h):[int(good.con_div.le(-h).sum()),int(good.con_div.ge(h).sum())]
                                for h in (.05,.10,.20)},
        }
    dbpath=directory/'portfolio.sqlite'
    assert not dbpath.exists()
    with sqlite3.connect(dbpath) as out:
        out.execute('PRAGMA foreign_keys=ON')
        profile.to_sql('faculty_observation',out,index=False)
        annual.to_sql('annual_activity',out,index=False)
        windows.to_sql('portfolio_windows',out,index=False)
        trans.to_sql('portfolio_transitions',out,index=False)
        bibliog.drop(columns='row').to_sql('publication_metadata',out,index=False)
        citing.to_sql('linked_earlier_citations',out,index=False)
        out.execute('CREATE TABLE work_coordinates(work TEXT PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID')
        out.executemany('INSERT INTO work_coordinates VALUES (?,?)',
                        ((work,centers[ix].astype('<f4').tobytes()) for work,ix in
                         paper[['work','ix']].drop_duplicates('work').itertuples(index=False,name=None)))
        out.execute('CREATE TABLE background_adjusted_coordinates(uid TEXT, work TEXT, vector BLOB NOT NULL, PRIMARY KEY(uid,work)) WITHOUT ROWID')
        out.executemany('INSERT INTO background_adjusted_coordinates VALUES (?,?,?)',
                        ((u,work,z[idx].astype('<f4').tobytes()) for u,work,idx in
                         paper.loc[paper.geometry,['uid','work','row']].itertuples(index=False,name=None)))
        out.execute('CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID')
        out.executemany('INSERT INTO settings VALUES (?,?)',[(k,str(v)) for k,v in
            {'cutoff_year':CUTOFF,'career_window_years':WIDTH,'background_min_papers':MIN_PEERS,
             'clusters':K,'con_div_band':BAND,'embedding_dim':dim,'vector_dtype':'little-endian float32'}.items()])
        out.executescript('''
          CREATE UNIQUE INDEX faculty_key ON faculty_observation(uid);
          CREATE UNIQUE INDEX year_key ON annual_activity(uid,year);
          CREATE UNIQUE INDEX window_key ON portfolio_windows(uid,year_start);
          CREATE UNIQUE INDEX transition_key ON portfolio_transitions(uid,year_start);
          CREATE UNIQUE INDEX publication_key ON publication_metadata(uid,work);
          CREATE INDEX citation_key ON linked_earlier_citations(uid,citing);
        ''')
        create_views(out)
        assert out.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    (directory/'results.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(metrics['all'],ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cohort','appointments','metadata_db','directory'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    career.main([str(a.cohort),str(a.appointments),str(a.directory/'unused.json')],
                collect_roster=lambda *inputs:make_profile(*inputs,a.directory,a.metadata_db))


if __name__=='__main__':
    main()
