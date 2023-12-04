from ... import aux
from . import crawler, turner, crawl_bend_interference, intermitter, sensor, feeder, memory

__all__ = [
    'ModuleModeDict',
    'ModuleColorDict',
]

ModuleModeDict = aux.AttrDict({
    'crawler': {
        'gaussian': crawler.GaussOscillator,
        'square': crawler.SquareOscillator,
        'realistic': crawler.PhaseOscillator,
        'constant': crawler.Crawler
    },
    'interference': {
        'default': crawl_bend_interference.DefaultCoupling,
        'square': crawl_bend_interference.SquareCoupling,
        'phasic': crawl_bend_interference.PhasicCoupling
    },
    'turner': {
        'neural': turner.NeuralOscillator,
        'sinusoidal': turner.SinTurner,
        'constant': turner.ConstantTurner
    },
    'intermitter': {
        'default': intermitter.Intermitter,
        'nengo': intermitter.NengoIntermitter,
        'branch': intermitter.BranchIntermitter
    },
    'feeder': {
        'default': feeder.Feeder,

    },
    'olfactor': {
        'default': sensor.Olfactor,
        'osn': sensor.Olfactor,
    },
    'toucher': {
        'default': sensor.Toucher,
    },
    'windsensor': {
        'default': sensor.Windsensor,
    },
    'thermosensor': {
        'default': sensor.Thermosensor,
    },
    'memory': {
        'RL': {'olfaction': memory.RLOlfMemory, 'touch': memory.RLTouchMemory},
        'MB': {'olfaction': memory.RemoteBrianModelMemory, 'touch': memory.RemoteBrianModelMemory}
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
