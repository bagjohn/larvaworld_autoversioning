import numpy as np
import pytest
import itertools
from larvaworld.lib import aux
from larvaworld.lib.plot.aux import configure_subplot_grid


def xx_test_subplot_grid_configuration():
    vs=[None,1,2,3,4,5]
    for idx in itertools.product(vs,vs,vs):
        if not None in idx :
            continue
        N,Ncols,Nrows=idx
        if N is not None :
            if (Ncols is not None and Ncols>N) or  (Nrows is not None and Nrows>N):
                continue
        kws=aux.AttrDict(configure_subplot_grid(N=N, Ncols=Ncols,Nrows=Nrows))
        if Ncols :
            assert(kws.ncols==Ncols)
        if Nrows :
            assert(kws.nrows==Nrows)
        if N :
            assert(kws.ncols*kws.nrows>=N)
            assert((kws.ncols-1)*kws.nrows<=N)
            assert((kws.nrows-1)*kws.ncols<=N)

