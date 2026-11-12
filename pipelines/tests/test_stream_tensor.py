"""Synthetic checks for tensor transport and persistent citation branches."""
import unittest

import numpy as np

from stream_tensor import branch_counts, residual


class StreamTensorTest(unittest.TestCase):
    def test_advection_and_shape(self):
        cloud = np.array([[-2., 0.], [2., 0.], [0., 1.], [0., -1.]])
        group = np.zeros(len(cloud), dtype=int)
        self.assertTrue(np.allclose(residual(cloud, group, cloud + [2, 1], group), 0))
        total, within, between, shape = residual(cloud, group, cloud[:, ::-1], group)
        np.testing.assert_allclose([total, within, between], 0)
        self.assertGreater(shape, 0)
        total, within, between, shape = residual(cloud, group, cloud / 2, group)
        np.testing.assert_allclose([total, within, between, shape], [-.6, -.6, 0., 0.])

    def test_persistent_branch_requires_direct_ancestry(self):
        years = np.repeat([2010, 2015, 2020], [2, 4, 4])
        groups = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
        cloud = np.column_stack([groups, np.arange(len(groups)) * .01])
        work = np.arange(len(groups))
        self.assertEqual(branch_counts(years, groups, cloud, work, {4: {0}}, 2010), (1, 1, 1))
        self.assertEqual(branch_counts(years, groups, cloud, work, {}, 2010), (1, 1, 0))
        self.assertEqual(branch_counts(years[:6], groups[:6], cloud[:6], work[:6], {4: {0}}, 2010),
                         (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
