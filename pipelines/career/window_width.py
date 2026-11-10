"""Change temporal resolution while preserving the earlier matched paper links."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import aggregation
from analyze import FIELDS, BOOTSTRAPS

WIDTHS = (2, 3, 5)
METRICS = ('equal_within', 'within', 'between', 'pool')
GRIDS = ('fixed_start', 'separate_early')


def block_labels(year, width, grid):
    if grid == 'fixed_start':
        return (year - 1) // width
    # Keep the omitted pair of years as its own block in the alternative grid.
    return np.where(year < 3, -1, (year - 3) // width)


def components(x, bins):
    mu = x.mean(axis=0)
    squared = np.square(x - mu).sum(axis=1)
    pooled = float(squared.mean())
    counts, center_ss, in_bin = [], [], []
    for bin_id in np.unique(bins):
        part = x[bins == bin_id]
        center = part.mean(axis=0)
        counts.append(len(part))
        center_ss.append(np.square(center - mu).sum())
        in_bin.append(np.square(part - center).sum(axis=1).mean())
    within = float(np.average(in_bin, weights=counts))
    between = float(np.average(center_ss, weights=counts))
    if not np.isclose(within + between, pooled, rtol=0, atol=1e-12):
        raise AssertionError("Within/between decomposition failed")
    return (float(np.mean(in_bin)), within, between, pooled), (len(counts), counts.count(1))


def compare(rows, eligible_windows, centers, time_column):
    keys = ['uid', 'field', 'win']
    eligible = eligible_windows[['uid', 'field', 'win', 'n']].copy()
    keep = eligible.loc[eligible.win.ge(3), 'uid']
    eligible = eligible[eligible.uid.isin(keep)]
    fixed = rows.merge(eligible[keys], on=keys, how='inner', validate='many_to_one')
    assert not fixed.duplicated(['uid', 'work_id']).any()
    actual = fixed.groupby(keys).size().sort_index()
    expected = eligible.set_index(keys).n.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    results = []
    for (uid, field), works in fixed.groupby(['uid', 'field'], sort=True):
        x = centers[works.ix.to_numpy()]
        year = works[time_column].to_numpy()
        later = year >= 3
        assert later.sum() >= 2 and (year >= 1).all()
        phases = np.empty((len(GRIDS), len(WIDTHS), 2, len(METRICS)))
        counts = np.empty((len(GRIDS), len(WIDTHS), 2, 2), dtype=int)
        for g, grid in enumerate(GRIDS):
            for w, width in enumerate(WIDTHS):
                bins = block_labels(year, width, grid)
                phases[g, w, 0], counts[g, w, 0] = components(x, bins)
                phases[g, w, 1], counts[g, w, 1] = components(x[later], bins[later])
        # Paper membership and geometry are fixed across all resolutions.
        np.testing.assert_allclose(phases[:, :, :, 3], np.broadcast_to(phases[0, 0, :, 3],
                                                                    (len(GRIDS), len(WIDTHS), 2)),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(phases[0, 0], phases[1, 0], rtol=0, atol=1e-12)
        results.append((field, int(later.sum()), int((~later).sum()), phases, counts))
    return results


def summarize(teachers):
    fields = np.array([t[0] for t in teachers])
    papers = np.array([[t[1], t[2]] for t in teachers])
    phases = np.stack([t[3] for t in teachers])
    counts = np.stack([t[4] for t in teachers])
    delta = phases[:, :, :, 1] - phases[:, :, :, 0]
    relative = delta - delta[:, :, :1]
    means = np.stack((phases[:, :, :, 0], phases[:, :, :, 1], delta, relative), axis=3)
    rng = np.random.default_rng(20261110)
    out = {}
    for field in FIELDS:
        select = np.ones(len(fields), dtype=bool) if field == 'all' else fields == field
        selected = means[select]
        block_count = counts[select]
        work_count = papers[select]
        n = len(selected)
        draws = np.concatenate([selected[rng.integers(n, size=(50, n))].mean(1)
                                for _ in range(BOOTSTRAPS // 50)], axis=0)
        lo, hi = np.quantile(draws, [.025, .975], axis=0)
        summaries = np.stack((selected.mean(0), lo, hi), axis=-1)
        field_row = {'n': n, 'early_teachers': int((work_count[:, 1] > 0).sum()),
                     'paper_links_later': int(work_count[:, 0].sum()),
                     'paper_links_early': int(work_count[:, 1].sum())}
        for g, grid in enumerate(GRIDS):
            by_width = {}
            for w, width in enumerate(WIDTHS):
                row = {}
                for p, phase in enumerate(('include', 'later', 'delta', 'delta_minus_2yr')):
                    row[phase] = dict(zip(METRICS, summaries[g, w, p].tolist()))
                row['include_singleton_windows'] = int(block_count[:, g, w, 0, 1].sum())
                row['later_singleton_windows'] = int(block_count[:, g, w, 1, 1].sum())
                row['later_one_window_teachers'] = int((block_count[:, g, w, 1, 0] == 1).sum())
                by_width[str(width)] = row
            field_row[grid] = by_width
        out[field] = field_row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('archive', 'appointments', 'pilot_reference', 'pilot_rebin', 'cohort',
                'appointment_evidence', 'career_reference', 'aggregation_reference', 'out'):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    data = {}
    previous = json.loads(args.aggregation_reference.read_text())

    def collect(sample, rows, windows, centers, time_column):
        summary = summarize(compare(rows, windows, centers, time_column))
        for f, row in summary.items():
            earlier = previous[sample][f]
            for key in ('n', 'paper_links_later', 'paper_links_early'):
                assert row[key] == earlier[key], (sample, f, key)
            for k1, k2 in zip(METRICS, ('windows', 'weighted_windows', 'between_windows', 'pooled')):
                np.testing.assert_allclose(row['fixed_start']['2']['delta'][k1][0],
                                           earlier['delta'][k2][0], atol=1e-12, rtol=0)
        data[sample] = summary

    archive = args.out.with_name('aggregation-reproduced.json')
    aggregation.main([str(getattr(args, arg)) for arg in ('archive', 'appointments', 'pilot_reference',
                        'pilot_rebin', 'cohort', 'appointment_evidence', 'career_reference')] + [str(archive)],
                     collect=collect)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for sample, results in data.items():
        for field, row in results.items():
            print(sample, field, row['n'], 'papers', row['paper_links_early'], row['paper_links_later'])
            for grid in GRIDS:
                print(grid, [[tuple(round(q, 5) for q in row[grid][str(w)]['delta'][k])
                              for k in METRICS] for w in WIDTHS])


if __name__ == '__main__':
    main()
