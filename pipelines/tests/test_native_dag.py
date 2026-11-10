import unittest

import numpy as np

from native_dag import direct_sketch, propagate
from native_flow import groups_in


class CitationDAGTest(unittest.TestCase):
    def test_ancestors_propagate_before_descendants(self):
        direct = direct_sketch(3,np.arange(3),np.array([101,102,103]),
                               np.ones(3,dtype=np.int64))
        observed = direct.copy()
        propagate(observed,np.arange(3),np.array([0,0,1,2]),np.array([0,1]),
                  np.ones(3,dtype=np.int64))
        np.testing.assert_allclose(observed[0],direct[0])
        np.testing.assert_allclose(observed[1],.75*direct[1]+.25*direct[0])
        np.testing.assert_allclose(observed[2],.75*direct[2]+.25*observed[1])
        np.testing.assert_array_equal(direct, direct_sketch(
            3,np.arange(3),np.array([101,102,103]),np.ones(3,dtype=np.int64)))

    def test_two_persistent_groups_are_geometrically_separate(self):
        x = np.array([[0.,0.],[0.,.1],[1.,0.],[1.,.1]])
        groups, score = groups_in(np.array([0,0,1,1]),x,2)
        self.assertEqual(groups,{0,1})
        self.assertGreater(score,1)
        groups, score = groups_in(np.array([0,0,0,1]),x,2)
        self.assertEqual(groups,{0})
        self.assertEqual(score,0)
