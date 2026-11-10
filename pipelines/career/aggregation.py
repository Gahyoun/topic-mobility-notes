"""Compare B under different aggregation rules on the identical works and embedding."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze
import career

COLUMNS = ('windows', 'weighted_windows', 'between_windows', 'pooled',
           'windows_corrected', 'pooled_corrected')


def window_components(rows, centers, moments):
    out = []
    for (uid, field, start), works in rows.groupby(['uid', 'field', 'win'], sort=True):
        if len(works) < 2:
            continue
        ix = works.ix.to_numpy()
        x = centers[ix]
        mu = x.mean(axis=0)
        b = float(np.square(x - mu).sum(axis=1).mean())
        if moments[ix].mean() - mu @ mu <= 1e-10:
            continue
        assert works.work_id.is_unique
        out.append((uid, field, start, len(ix), mu, b, ix))
    return pd.DataFrame(out, columns=['uid', 'field', 'win', 'n', 'center', 'B', 'indices'])


def combine(windows, centers):
    weights = windows.n.to_numpy()
    values = windows.B.to_numpy()
    mus = np.stack(windows.center.to_list())
    mu = np.average(mus, axis=0, weights=weights)
    weighted = np.average(values, weights=weights)
    between = np.average(np.square(mus - mu).sum(axis=1), weights=weights)
    x = centers[np.concatenate(windows.indices.to_list())]
    direct = np.square(x - x.mean(axis=0)).sum(axis=1).mean()
    np.testing.assert_allclose(weighted + between, direct, atol=1e-12, rtol=0)
    return np.array([values.mean(), weighted, between, direct,
                     (values * weights / (weights - 1)).mean(),
                     direct * len(x) / (len(x) - 1)])


def summaries(windows, centers):
    result = []
    for (uid, field), all_windows in windows.groupby(['uid', 'field'], sort=True):
        later = all_windows[all_windows.win >= 3]
        if later.empty:
            continue
        result.append({'uid': uid, 'field': field, 'early': bool(all_windows.win.eq(1).any()),
                       'one_late_window': len(later) == 1,
                       'n_late': int(later.n.sum()), 'n_early': int(all_windows.n.sum() - later.n.sum()),
                       'include': combine(all_windows, centers), 'later': combine(later, centers)})
    return pd.DataFrame(result)


def interval(values, draws):
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    return np.column_stack((values.mean(0), lo, hi)).tolist()


def report(rows, seed=20261110):
    output = {}
    rng = np.random.default_rng(seed)
    for field in analyze.FIELDS:
        g = rows if field == 'all' else rows[rows.field == field]
        before, after = np.stack(g.include.to_list()), np.stack(g.later.to_list())
        delta = after - before
        aggregation = delta[:, 3] - delta[:, 0]
        weights = delta[:, 1] - delta[:, 0]
        across = delta[:, 2]
        np.testing.assert_allclose(aggregation, weights + across, atol=1e-12, rtol=0)
        keys = [*COLUMNS, 'aggregation', 'weights']
        x = np.column_stack((delta, aggregation, weights))
        draws = np.concatenate([x[rng.integers(len(g), size=(100, len(g)))].mean(1)
                                for _ in range(analyze.BOOTSTRAPS // 100)])
        output[field] = {
            'n': len(g), 'early': int(g.early.sum()), 'one_late_window': int(g.one_late_window.sum()),
            'paper_links_later': int(g.n_late.sum()), 'paper_links_early': int(g.n_early.sum()),
            'include': dict(zip(COLUMNS, before.mean(0))),
            'later': dict(zip(COLUMNS, after.mean(0))),
            'delta': dict(zip(keys, interval(x, draws))),
        }
    return output


def main(argv=None, collect=None):
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('archive', 'appointments', 'pilot_reference', 'pilot_rebin', 'cohort',
                'appointment_evidence', 'career_reference', 'out'):
        p.add_argument(arg, type=Path)
    args = p.parse_args(argv)
    root = args.out.parent
    pilot_ref = json.loads(args.pilot_rebin.read_text())['results']
    career_ref = json.loads(args.career_reference.read_text())['event']['database']
    data = {}

    def pilot(rows, centers, moments, _):
        windows = window_components(rows, centers, moments)
        stats = report(summaries(windows, centers))
        for f, v in stats.items():
            old = pilot_ref[f]
            assert v['n'] == old['n'] and v['early'] == old['early']
            np.testing.assert_allclose(v['delta']['windows'][0], old['delta']['B'][0], atol=1e-9, rtol=0)
        data['pilot'] = stats
        if collect is not None:
            collect('pilot', rows, windows, centers, 'relative_year')

    def full(rows, centers, moments, old):
        keys = ['uid', 'field']
        # The broad baseline used all linked papers. Preserve it as the starting point.
        original = old['pre'][keys].merge(old['inclusive'][keys+['B']], on=keys).merge(
            old['later'][keys+['B']], on=keys, suffixes=('_include', '_later'), validate='one_to_one')
        selected = rows[rows.uid.isin(original.uid) & rows.event_year.ge(1)].copy()
        selected['win'] = 1 + 2 * ((selected.event_year - 1) // 2)
        selected = selected[selected.anchor + selected.win + 1 <= 2025]
        selected = selected.rename(columns={'work': 'work_id'})
        windows = window_components(selected, centers, moments)
        fixed = summaries(windows, centers)
        stats = report(fixed)
        for f, v in stats.items():
            pool = original if f == 'all' else original[original.field == f]
            same_teacher = pool[pool.uid.isin(fixed.uid)]
            original_delta = (pool.B_later - pool.B_include).mean()
            teacher_delta = (same_teacher.B_later - same_teacher.B_include).mean()
            assert len(pool) == career_ref[f]['n'] and len(same_teacher) == v['n']
            np.testing.assert_allclose(original_delta, career_ref[f]['delta']['B'][0], atol=1e-9, rtol=0)
            v['bridge'] = {
                'original_n': len(pool), 'original_pooled': original_delta,
                'same_teachers_all_papers_pooled': teacher_delta,
                'same_teachers_eligible_papers_pooled': v['delta']['pooled'][0],
                'same_teachers_eligible_papers_windows': v['delta']['windows'][0],
            }
        data['database'] = stats
        if collect is not None:
            collect('database', selected, windows, centers, 'event_year')

    analyze.main([str(args.archive), str(args.appointments), str(args.pilot_reference),
                  str(root/'pilot-reproduced.json')], collect=pilot)
    career.main([str(args.cohort), str(args.appointment_evidence), str(root/'career-reproduced.json')],
                collect=full)
    assert set(data) == {'pilot', 'database'}
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for sample, results in data.items():
        for f, v in results.items():
            print(sample, f, v['n'], v['early'],
                  {k: [round(x, 5) for x in v['delta'][k]]
                   for k in ('windows', 'pooled', 'aggregation', 'weights', 'between_windows',
                             'windows_corrected', 'pooled_corrected')},
                  v.get('bridge'))


if __name__ == '__main__':
    main()
