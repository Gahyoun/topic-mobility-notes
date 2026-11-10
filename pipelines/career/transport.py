"""Exact one-dimensional quantile coupling and its squared-distance components."""
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=4096)
def quantile_pairs(n, m):
    # Integer breakpoints on the least common denominator n*m are exact.
    b = np.union1d(np.arange(n+1)*m, np.arange(m+1)*n)
    return b[:-1]//m, b[:-1]//n, np.diff(b)/(n*m)


def projection_components(a,b,na,nb):
    """Exact 1D empirical W2^2 on each projection, with a known additive decomposition."""
    ia,ib,mass=quantile_pairs(na,nb)
    qa,qb=np.sort(a,axis=0),np.sort(b,axis=0)
    mean=(qa.mean(0)-qb.mean(0))**2
    width=(qa.std(0)-qb.std(0))**2
    total=(np.square(qa[ia]-qb[ib])*mass[:,None]).sum(0)
    residual=total-mean-width
    assert residual.min()>=-1e-11
    return np.stack((mean,width,np.maximum(residual,0),total))
