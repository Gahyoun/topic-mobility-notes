"""Estimate ordered subject-space motion with fixed papers and a common early anchor."""
import argparse
import json
from pathlib import Path

import numpy as np

import aggregation
from analyze import FIELDS, BOOTSTRAPS
from window_support import block_labels, fixed_papers

WIDTHS = (2, 3, 5)
METRICS = ('speed', 'radial', 'adjusted_squared_speed', 'alignment')


def centroid(coordinates, years):
    n = len(coordinates)
    center = coordinates.mean(axis=0)
    # Trace of sample covariance divided by n, for the noise adjustment only.
    noise = np.square(coordinates - center).sum() / (n * (n - 1))
    return center, years.mean(), noise


def velocity(x, years, anchor, width):
    bins = block_labels(years, width, 'separate_early')
    windows = {}
    for j in np.unique(bins[bins >= 0]):
        mask = bins == j
        if mask.sum() >= 2:
            windows[j] = centroid(x[mask], years[mask])
    distances, radial, elapsed, squared_adjusted = [], [], [], []
    for j in sorted(windows):
        if j+1 not in windows:
            continue
        a, t_a, noise_a = windows[j]
        b, t_b, noise_b = windows[j+1]
        distance = np.linalg.norm(b-a)
        dr = np.linalg.norm(b-anchor) - np.linalg.norm(a-anchor)
        dt = t_b-t_a
        assert dt > 0 and abs(dr) <= distance + 1e-12
        distances.append(distance)
        radial.append(dr)
        elapsed.append(dt)
        squared_adjusted.append((distance**2-noise_a-noise_b)/dt**2)
    if not elapsed:
        return np.full(4, np.nan), 0, np.nan
    total_time = sum(elapsed)
    speed = sum(distances)/total_time
    radial_speed = sum(radial)/total_time
    alignment = sum(radial)/sum(distances) if sum(distances) > 1e-12 else 0.
    return (speed, radial_speed, np.average(squared_adjusted, weights=elapsed), alignment), len(elapsed), np.mean(elapsed)


def bootstrap(values, rng):
    n = len(values)
    means = np.concatenate([values[rng.integers(n, size=(50,n))].mean(axis=1)
                            for _ in range(BOOTSTRAPS//50)], axis=0)
    return np.stack([values.mean(axis=0),*np.quantile(means,[.025,.975],axis=0)],axis=-1)


def measure(fixed, centers, time_col, previous):
    fixed = fixed.sort_values(['uid','work_id'])
    groups = list(fixed.groupby(['uid','field'],sort=True))
    fields = np.array([label[1] for label,_ in groups])
    years = [works[time_col].to_numpy() for _,works in groups]
    values = np.full((len(groups),len(WIDTHS),len(METRICS)),np.nan)
    counts = np.zeros((len(groups),len(WIDTHS)),int)
    gaps = np.full((len(groups),len(WIDTHS)),np.nan)
    early_n = np.array([(y<3).sum() for y in years])
    for i,((_,works),year) in enumerate(zip(groups,years,strict=True)):
        if early_n[i] < 2:
            continue
        x = centers[works.ix.to_numpy()]
        anchor = x[year<3].mean(axis=0)
        for w,width in enumerate(WIDTHS):
            values[i,w], counts[i,w], gaps[i,w] = velocity(x,year,anchor,width)
    common = (counts > 0).all(axis=1)
    rng = np.random.default_rng(20261110)
    output = {}
    for f in FIELDS:
        selected = np.ones(len(groups),dtype=bool) if f=='all' else fields==f
        original = previous[f]
        retained_works = sum(len(y) for y,s in zip(years,selected,strict=True) if s)
        assert selected.sum() == original['n']
        assert retained_works == original['paper_links_early']+original['paper_links_later']
        assert (selected & (early_n>=2)).sum() == original['early_teachers']
        matched = selected & common
        m = int(matched.sum())
        assert m > 1
        v = values[matched]
        ci = bootstrap(v,rng)
        delta = bootstrap(v[:,1:]-v[:,:1],rng)
        out = {'base':int(selected.sum()), 'early':int((selected & (early_n>=2)).sum()),
               'common':m,'widths':{}}
        for w,width in enumerate(WIDTHS):
            available = selected & (counts[:,w] > 0)
            out['widths'][str(width)] = {
                'n_available':int(available.sum()),
                'n_common':m,'pairs_common':int(counts[matched,w].sum()),
                'mean_gap_common':float(gaps[matched,w].mean()),
                'estimate':dict(zip(METRICS,ci[w].tolist())),
                'estimate_all_available':dict(zip(METRICS,bootstrap(values[available,w],rng).tolist())),
            }
            if w:
                out['widths'][str(width)]['minus_2yr'] = dict(zip(METRICS,delta[w-1].tolist()))
        output[f]=out
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('archive','appointments','pilot_reference','pilot_rebin','cohort','appointment_evidence',
                'career_reference','width_reference','out'):
        p.add_argument(arg,type=Path)
    args = p.parse_args()
    width_ref = json.loads(args.width_reference.read_text())
    data = {}

    def collect(sample, rows, windows, centers, time_col):
        data[sample] = measure(fixed_papers(rows,windows),centers,time_col,width_ref[sample])

    sources = ('archive','appointments','pilot_reference','pilot_rebin','cohort','appointment_evidence',
               'career_reference')
    aggregation.main([str(getattr(args,k)) for k in sources]+
                     [str(args.out.with_name('aggregation-reproduced.json'))],collect=collect)
    args.out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for sample,fields in data.items():
        for field,row in fields.items():
            print(sample,field,row['base'],row['early'],row['common'])
            for w,r in row['widths'].items():
                print(w,r['n_available'],r['pairs_common'],round(r['mean_gap_common'],3),
                      {k:[round(x,6) for x in v] for k,v in r['estimate'].items()})


if __name__ == '__main__':
    main()
