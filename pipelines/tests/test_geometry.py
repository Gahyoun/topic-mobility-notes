import unittest

import numpy as np
import pandas as pd

from portfolio import fit_background, scatter, transition
from transport import projection_components


class GeometryTest(unittest.TestCase):
    def test_components(self):
        a=np.array([[-1.],[1.]])
        z=np.array([[-2.],[2.]])
        shifted=projection_components(a,a+2,2,2)[:,0]
        scaled=projection_components(a,z,2,2)[:,0]
        different=projection_components(a,np.array([[-np.sqrt(2.)],[0],[0],[np.sqrt(2.)]]),2,4)[:,0]
        np.testing.assert_allclose(shifted,[4,0,0,4],atol=1e-12)
        np.testing.assert_allclose(scaled,[0,1,0,1],atol=1e-12)
        np.testing.assert_allclose(different[:2],0,atol=1e-12)
        self.assertGreater(different[2],0)
        np.testing.assert_allclose(projection_components(a,np.repeat(a,3,axis=0),2,6),0,atol=1e-12)

    def test_background_excludes_focal_and_counts_shared_work_once(self):
        coords=np.arange(24.,dtype=float).reshape(-1,1)
        p=pd.DataFrame({'uid':['focal']+['peers']*24,'field':'physics','year':2020,
                        'ix':[0]+list(range(24))})
        result,z=fit_background(p,coords)
        self.assertEqual(int(result.iloc[0].peers),23)
        self.assertAlmostEqual(z[0,0],-coords[1:].mean()/coords[1:].std())
        self.assertTrue(np.isnan(z[1:]).all())

    def test_zero_activity_and_branch(self):
        old=pd.DataFrame({'work':['a','b']})
        new=pd.DataFrame({'work':['c','d']})
        a=np.array([[-1.],[1.]])
        b=a*2
        _,_,q1=scatter(a)
        _,_,q2=scatter(b)
        before={'uid':'f','field':'physics','year_start':2010,'state':'multiple','n_geometry':2,
                'q':q1,'raw_q':q1,'exposure':5,'projection':a,'dim':1}
        after=dict(before,year_start=2015,n_geometry=4,q=q2,raw_q=q2,
                   projection=np.repeat(b,2,axis=0))
        anchors={0:(old,None,None)}
        current={**anchors,1:(new,None,None)}
        res=transition(before,after,anchors,current,{0},{('f','c'):{'a'}})
        self.assertEqual((res['div'],res['branch'],res['branch_share']),(1,1,.25))
        zero=dict(after,state='zero',n_geometry=0,q=None,raw_q=None)
        empty=transition(before,zero,anchors,{}, {0},{})
        self.assertEqual(empty['geometry_complete'],0)
        self.assertEqual(empty['state'],'zero')
        self.assertIsNone(empty['con_div'])


if __name__=='__main__':
    unittest.main()
