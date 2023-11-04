import os
import matplotlib.pyplot as plt

from larvaworld.lib import reg, aux, plot


def xx_test_plots():
    gIDs = reg.conf.Ref.RefGroupIDs
    gID = gIDs[0]
    dcol = reg.conf.Ref.loadRefGroup(gID)
    assert dcol.dir is not None
    assert os.path.exists(dcol.dir)
    kws = {'save_to': f'{reg.ROOT_DIR}/../../tests/plots', 'show': True}

    graphIDs = ['trajectories', 'pathlength','dispersal','angular pars','endpoint box', 'epochs', 'freq powerspectrum', 'dispersal summary',
                'kinematic analysis',  'crawl pars', 'stride cycle',
                'stride cycle multi', 'ethogram'][:3]

    figs = dcol.plot(ids=graphIDs, **kws)
    for k in graphIDs:
        assert isinstance(figs[k], plt.Figure)
        plt.close(figs[k])
