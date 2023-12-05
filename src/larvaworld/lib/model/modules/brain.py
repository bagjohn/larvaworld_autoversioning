import numpy as np

from ... import reg, aux
from .. import modules
from ...param import NestedConf, ClassAttr
from .module_modes import ModuleModeDict, mod_gen,mod_parent_class

__all__ = [
    'Brain',
    'DefaultBrain',
]


class Brain(NestedConf):
    olfactor = ClassAttr(class_=mod_parent_class('olfactor'), default=None, doc='The olfactory sensor')
    toucher = ClassAttr(class_=mod_parent_class('toucher'), default=None, doc='The tactile sensor')
    windsensor = ClassAttr(class_=mod_parent_class('windsensor'), default=None, doc='The wind sensor')
    thermosensor = ClassAttr(class_=mod_parent_class('thermosensor'), default=None, doc='The temperature sensor')

    def __init__(self, agent=None, dt=None, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.modalities = aux.AttrDict({
            'olfaction': {'sensor': self.olfactor,'func': self.sense_odors, 'A': 0.0, 'mem': None},
            'touch': {'sensor': self.toucher,'func': self.sense_food_multi, 'A': 0.0, 'mem': None},
            'thermosensation': {'sensor': self.thermosensor,'func': self.sense_thermo, 'A': 0.0, 'mem': None},
            'windsensation': {'sensor': self.windsensor,'func': self.sense_wind, 'A': 0.0, 'mem': None}
        })

        if dt is None:
            dt = self.agent.model.dt
        self.dt = dt

    def sense_odors(self, pos=None):
        try:
            a = self.agent
            if pos is None:
                pos = a.olfactor_pos
            return {id:l.get_value(pos) for id, l in a.model.odor_layers.items()}
        except:
            return {}

    def sense_food_multi(self, **kwargs):
        try:
            a = self.agent
            kws = {
                'sources': a.model.sources, 'grid': a.model.food_grid, 'radius': a.radius
            }
            return {s: int(aux.sense_food(pos=a.get_sensor_position(s), **kws) is not None) for s in a.touch_sensorIDs}
        except:
            return {}



    def sense_wind(self, **kwargs):
        try:
            a = self.agent
            return {'windsensor': a.model.windscape.get_value(a)}
        except:
            return {'windsensor': 0.0}


    def sense_thermo(self, pos=None):
        try:
            a = self.agent
            if pos is None:
                pos = a.pos
            ad = a.model.space.dims
            return a.model.thermoscape.get_value([(pos[0] + (ad[0] * 0.5)) / ad[0], (pos[1] + (ad[1] * 0.5)) / ad[1]])
        except AttributeError:
            return {'cool': 0, 'warm': 0}

    @property
    def A_in(self):
        return np.sum([M.A for m, M in self.modalities.items()])
        # return self.A_olf + self.A_touch + self.A_thermo + self.A_wind

    @property
    def A_olf(self):
        return self.modalities['olfaction'].A

    @property
    def A_touch(self):
        return self.modalities['touch'].A

    @property
    def A_thermo(self):
        return self.modalities['thermosensation'].A

    @property
    def A_wind(self):
        return self.modalities['windsensation'].A


class DefaultBrain(Brain):
    def __init__(self, conf, agent=None, **kwargs):
        kws={'dt':self.dt, 'brain':self}

        for k in self.param_keys:
            kwargs[k] = mod_gen(k, conf[k], **kws)
        super().__init__(agent=agent, **kwargs)
        self.locomotor = modules.DefaultLocomotor(conf=conf, dt=self.dt)
        m = conf['memory']
        if m is not None:
            M=self.modalities[m.modality]
            if M.sensor:
                m.gain = M.sensor.gain
                M.mem = ModuleModeDict['memory'][m.mode][m.modality](**kws, **m)

    def sense(self, pos=None, reward=False):
        kws={'pos':pos}
        for m, M in self.modalities.items():
            if M.sensor:
                M.sensor.update_gain_via_memory(mem=M.mem, reward=reward)
                M.A = M.sensor.step(M.func(**kws))



    def step(self, pos, on_food=False, **kwargs):
        self.sense(pos=pos, reward=on_food)
        return self.locomotor.step(A_in=self.A_in, on_food=on_food, **kwargs)
