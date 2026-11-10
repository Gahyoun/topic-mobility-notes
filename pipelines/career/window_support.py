"""Original fixed paper support and time-block rule."""
import numpy as np
import pandas as pd

def block_labels(year, width, grid):
    if grid == 'fixed_start':
        return (year - 1) // width
    # Keep the omitted pair of years as its own block in the alternative grid.
    return np.where(year < 3, -1, (year - 3) // width)


def fixed_papers(rows, eligible_windows):
    keys = ['uid', 'field', 'win']
    eligible = eligible_windows[['uid', 'field', 'win', 'n']].copy()
    keep = eligible.loc[eligible.win.ge(3), 'uid']
    eligible = eligible[eligible.uid.isin(keep)]
    fixed = rows.merge(eligible[keys], on=keys, how='inner', validate='many_to_one')
    assert not fixed.duplicated(['uid', 'work_id']).any()
    actual = fixed.groupby(keys).size().sort_index()
    expected = eligible.set_index(keys).n.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    return fixed
