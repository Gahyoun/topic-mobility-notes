"""Verify GCC coordinate restriction and compare the same faculty under both graph scopes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from appointment_dag import FIELDS, METRICS, SEED, SIGNALS, exact_difference
from motion import bootstrap
from native_dag import DIMENSIONS


def common_summary(original, gcc):
    a=original[original.comparison=='paired'].set_index('uid')
    b=gcc[gcc.comparison=='paired'].set_index('uid')
    common=a.index.intersection(b.index)
    a,b=a.loc[common],b.loc[common]
    assert a.field.equals(b.field)
    assert a.width.equals(b.width)
    rng=np.random.default_rng(SEED)
    result={}
    for f in FIELDS:
        keep=np.ones(len(common),dtype=bool) if f=='all' else a.field.eq(f).to_numpy()
        out={'n':int(keep.sum())}
        for m in METRICS:
            i,j=(f'{stage}_{m}' for stage in ('pre','post'))
            old=(a[j]-a[i]).to_numpy()[keep]
            new=(b[j]-b[i]).to_numpy()[keep]
            if len(new):
                d=exact_difference(int((new==1).sum()),int((new==-1).sum()),len(new)) if m in SIGNALS else bootstrap(new[:,None],rng)[0].tolist()
                out[m]={'original':float(old.mean()),'gcc':float(new.mean()),
                        'gcc_interval':d,'difference':float((new-old).mean())}
            else:
                out[m]=None
        result[f]=out
    return result


def verify_coordinate_identity(original, restricted, dim):
    with np.load(original/'paper_axis.npz',allow_pickle=False) as a, np.load(restricted/'paper_axis.npz',allow_pickle=False) as b:
        index=np.searchsorted(a['work'],b['work'])
        assert np.array_equal(a['work'][index],b['work'])
        assert np.array_equal(a['cluster'][index],b['cluster'])
    a=np.load(original/'paper_vectors.npy',mmap_mode='r',allow_pickle=False)
    b=np.load(restricted/'paper_vectors.npy',mmap_mode='r',allow_pickle=False)
    assert b.shape==(len(index),dim)
    for start in range(0,len(index),4096):
        block=slice(start,start+4096)
        assert np.array_equal(a[index[block]],b[block]),(dim,start)
    return len(a),len(b)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('baseline_tensors','baseline_appointment','gcc_tensors','gcc_appointment','output'):
        p.add_argument(x,type=Path)
    a=p.parse_args()
    out={}
    for d in DIMENSIONS:
        folder=f'd{d:04}'
        baseline=a.baseline_tensors/folder
        gcc=a.gcc_tensors/folder
        papers,gcc_papers=verify_coordinate_identity(baseline,gcc,d)
        pre=pd.read_csv(a.baseline_appointment/f'faculty-{folder}.csv')
        post=pd.read_csv(a.gcc_appointment/f'faculty-{folder}.csv')
        s0=json.loads((baseline/'summary.json').read_text())['all']
        s1=json.loads((gcc/'summary.json').read_text())['all']
        metrics=('transitions','local_pairs','local_faculty')
        out[d]={'positioned_core':[papers,gcc_papers],
                'stream':{m:[s0[m],s1[m]] for m in metrics},
                'paired_counts':[int(pre.comparison.eq('paired').sum()),int(post.comparison.eq('paired').sum())],
                'common':common_summary(pre,post)}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
