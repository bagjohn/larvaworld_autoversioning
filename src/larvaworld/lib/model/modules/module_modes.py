from ... import aux
from . import crawler, turner,crawl_bend_interference,intermitter

__all__ = [
    'ModuleModeDict',
]

ModuleModeDict = aux.AttrDict({
    'Crawler': {
        'gaussian': crawler.GaussOscillator,
        'square': crawler.SquareOscillator,
        'realistic': crawler.PhaseOscillator,
        'constant': crawler.Crawler
    },
    'Interference': {
        'default': crawl_bend_interference.DefaultCoupling,
        'square': crawl_bend_interference.SquareCoupling,
        'phasic': crawl_bend_interference.PhasicCoupling
    },
    'Turner': {
        'neural': turner.NeuralOscillator,
        'sinusoidal': turner.SinTurner,
        'constant': turner.ConstantTurner
    },
    'Intermitter': {
        'default': intermitter.Intermitter,
        'nengo': intermitter.NengoIntermitter,
        'branch': intermitter.BranchIntermitter
    }
})
