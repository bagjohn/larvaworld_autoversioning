"""
Frequency-related plotting
"""

import numpy as np
from matplotlib import ticker, cm
from scipy.fft import fft, fftfreq

from .. import reg, aux, plot

__all__ = [
    'plot_fft_multi',
]


@reg.funcs.graph('freq powerspectrum', required={'ks': ['v', 'fov']})
def plot_fft_multi(ks=['v', 'fov'], name=f'frequency_powerspectrum', axx=None, **kwargs):
    P = plot.AutoPlot(ks=ks, name=name, build_kws={'w': 15, 'h': 12}, **kwargs)
    if axx is None:
        axx = P.axs[0].inset_axes([0.64, 0.65, 0.3, 0.25])
    palette = aux.AttrDict({
        'v': {
            'color': 'green',
            'fr_range': (1.0, 2.5),
            'label': 'forward speed',
        },
        'fov': {
            'color': 'purple',
            'fr_range': (0.1, 0.8),
            'label': 'angular speed',
        }
    })
    # kks=list(palette.keys())
    # lks = [Dk.label for Dk in palette.values()]
    # col0ks = [Dk.color for Dk in palette.values()]
    # fr_ranges = [Dk.fr_range for Dk in palette.values()]


    prob_kws = {
        'bins': np.linspace(0, 2, 40),
        'alpha': 0.5,
        'ax': axx,
        'labels': P.labels,
    }

    col0ks, lks = [], []
    for k,Dk in palette.items():
        colsk, fsk = [], []
        lk = Dk.label
        lks.append(lk)
        col0k = Dk.color
        col0ks.append(col0k)

        for l, d, col in P.data_palette:
            c = d.config
            xf = fftfreq(c.Nticks, c.dt)[:c.Nticks // 2]

            res = P.dkdict[l][k].groupby('AgentID').apply(aux.fft_max, dt=c.dt, fr_range=Dk.fr_range,
                                                          return_amps=True).values
            fk = [r[0] for r in res]
            fsk.append(fk)
            yk = np.array([r[1] for r in res])
            colk = aux.mix2colors(col, col0k)
            colsk.append(colk)
            plot.plot_quantiles(yk, x=xf, axis=P.axs[0], label=lk, color=colk)
        plot.prob_hist(vs=fsk, colors=colsk, **prob_kws)
    P.conf_ax(0, ylim=(0, 8), xlim=(0, 3.5), ylab='Amplitude (a.u.)', xlab='Frequency (Hz)',
              title='Fourier analysis', titlefontsize=25, yMaxN=5)
    P.data_leg(0, loc='upper left', labels=lks, colors=col0ks, fontsize=15)
    P.conf_ax(ax=axx, ylab='Probability', xlab='Dominant frequency (Hz)', yMaxN=2,
              major_ticklabelsize=10, minor_ticklabelsize=10)
    if P.Ndatasets > 1:
        P.data_leg(0, loc='upper center', fontsize=15)
    return P.get()
