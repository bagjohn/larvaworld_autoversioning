from ... import reg, aux
from .. import modules
from ...param import NestedConf, ClassAttr

__all__ = [
    'Brain',
    'DefaultBrain',
]


class Brain(NestedConf):
    olfactor = ClassAttr(class_=modules.Olfactor, default=None, doc='The olfactory sensor')
    toucher = ClassAttr(class_=modules.Toucher, default=None, doc='The tactile sensor')
    windsensor = ClassAttr(class_=modules.Windsensor, default=None, doc='The wind sensor')
    thermosensor = ClassAttr(class_=modules.Thermosensor, default=None, doc='The temperature sensor')

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
        if self.agent is None:
            return {}
        if pos is None:
            pos = self.agent.olfactor_pos

        cons = {}
        for id, layer in self.agent.model.odor_layers.items():
            cons[id] = layer.get_value(pos)

        return cons

    def sense_food_multi(self, **kwargs):
        a = self.agent
        if a is None:
            return {}
        kws = {
            'sources': a.model.sources, 'grid': a.model.food_grid, 'radius': a.radius
        }
        input = {s: int(aux.sense_food(pos=a.get_sensor_position(s), **kws) is not None) for s in a.touch_sensorIDs}
        return input

    def sense_wind(self, **kwargs):
        if self.agent is None:
            v = 0.0
        else:
            w = self.agent.model.windscape
            if w is None:
                v = 0.0
            else:
                v = w.get_value(self.agent)
        return {'windsensor': v}

    def sense_thermo(self, pos=None):
        a = self.agent
        if a is None:
            return {'cool': 0, 'warm': 0}
        if pos is None:
            pos = a.pos
        ad = a.model.space.dims
        pos_adj = [(pos[0] + (ad[0] * 0.5)) / ad[0], (pos[1] + (ad[1] * 0.5)) / ad[1]]
        try:
            cons = a.model.thermoscape.get_value(pos_adj)
        except AttributeError:
            return {'cool': 0, 'warm': 0}
        return cons

    @property
    def A_in(self):
        return self.A_olf + self.A_touch + self.A_thermo + self.A_wind

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
        ms=conf.modules
        kws={'dt':self.dt, 'brain':self}
        from module_modes import ModuleModeDict as MD
        for k in self.param_keys:
            if ms[k]:
                m = conf[f'{k}_params']
                if 'mode' not in m:
                    m.mode = 'default'
                kwargs[k] = MD[k][m.mode](**kws, **{k: m[k] for k in m if k != 'mode'})
        super().__init__(agent=agent, **kwargs)
        self.locomotor = modules.DefaultLocomotor(conf=conf, dt=self.dt)
        if ms['memory']:
            m = conf['memory_params']
            M=self.modalities[m.modality]
            if M.sensor:
                m.gain = M.sensor.gain
                M.mem = MD['memory'][m.mode][m.modality](**kws, **m)

    def sense(self, pos=None, reward=False):
        kws={'pos':pos}
        for m, M in self.modalities.items():
            if M.sensor:
                M.sensor.update_gain_via_memory(mem=M.mem, reward=reward)
                M.A = M.sensor.step(M.func(**kws))



    def step(self, pos, on_food=False, **kwargs):
        self.sense(pos=pos, reward=on_food)
        return self.locomotor.step(A_in=self.A_in, on_food=on_food, **kwargs)
