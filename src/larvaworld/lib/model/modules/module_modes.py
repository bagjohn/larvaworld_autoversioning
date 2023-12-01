from ... import aux
from . import crawler, turner,crawl_bend_interference,intermitter, sensor

__all__ = [
    'ModuleModeDict',
    'ModuleColorDict',
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
    },
    'Olfactor': {
        'default': sensor.Olfactor,
        'osn': sensor.Olfactor,
    },
})

ModuleColorDict = aux.AttrDict({
            'body': 'lightskyblue',
            'physics': 'lightsteelblue',
            'energetics': 'lightskyblue',
            'Box2D_params': 'lightcoral',
            'crawler': 'lightcoral',
            'turner': 'indianred',
            'interference': 'lightsalmon',
            'intermitter': '#a55af4',
            'olfactor': 'palegreen',
            'windsensor': 'plum',
            'toucher': 'pink',
            'feeder': 'pink',
            'memory': 'pink',
            # 'locomotor': locomotor.DefaultLocomotor,
        })