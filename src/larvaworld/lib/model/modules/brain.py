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
    windsensor = ClassAttr(class_=modules.WindSensor, default=None, doc='The wind sensor')
    thermosensor = ClassAttr(class_=modules.Thermosensor, default=None, doc='The temperature sensor')

    def __init__(self, agent=None, dt=None, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.A_olf = 0
        self.A_touch = 0
        self.A_thermo = 0
        self.A_wind = 0
        # self.olfactor, self.toucher, self.windsensor, self.thermosensor = [None]*4
        # A dictionary of the possibly existing sensors along with the sensing functions and the possibly existing memory modules
        self.sensor_dict = aux.AttrDict({
            'olfactor': {'func': self.sense_odors, 'A': 0.0, 'mem': 'memory'},
            'toucher': {'func': self.sense_food_multi, 'A': 0.0, 'mem': 'touch_memory'},
            'thermosensor': {'func': self.sense_thermo, 'A': 0.0, 'mem': None},
            'windsensor': {'func': self.sense_wind, 'A': 0.0, 'mem': None}
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


class DefaultBrain(Brain):
    def __init__(self, conf, agent=None, **kwargs):
        super().__init__(agent=agent, **kwargs)
        self.locomotor = modules.DefaultLocomotor(conf=conf, dt=self.dt)

        kws = {"brain": self, "dt": self.dt}
        self.touch_memory = None
        self.memory = None

        mods = conf.modules

        if mods.olfactor:
            self.olfactor = modules.Olfactor(**kws, **conf['olfactor_params'])
        if mods.toucher:
            self.toucher = modules.Toucher(**kws, **conf['toucher_params'])


        memory_modes = {
            'RL': {'olfaction': modules.RLOlfMemory, 'touch': modules.RLTouchMemory},
            'MB': {'olfaction': modules.RemoteBrianModelMemory, 'touch': modules.RemoteBrianModelMemory}
        }
        if mods['memory']:

            mm = conf['memory_params'].get_copy()
            # FIXME
            if 'mode' not in mm:
                mm['mode'] = 'RL'
            modality_classes = memory_modes[mm['mode']]
            modality = mm['modality']
            class_func = modality_classes[modality]
            mm.pop('mode')
            mm.pop('modality')

            if modality == 'olfaction' and self.olfactor:
                mm.gain = self.olfactor.gain
                self.memory = class_func(**mm, **kws)

            elif modality == 'touch' and self.toucher:
                mm.gain = self.toucher.gain
                self.touch_memory = class_func(**mm, **kws)
        if mods.windsensor:
            self.windsensor = modules.WindSensor(**kws, **conf['windsensor_params'])
        if mods.thermosensor:
            self.thermosensor = modules.Thermosensor(**kws, **conf['thermosensor_params'])

    def sense(self, pos=None, reward=False):

        if self.olfactor:
            if self.memory:
                dx = self.olfactor.get_dX()
                self.olfactor.gain = self.memory.step(dx, reward)
            self.A_olf = self.olfactor.step(self.sense_odors(pos))
        if self.toucher:
            if self.touch_memory:
                dx = self.toucher.get_dX()
                self.toucher.gain = self.touch_memory.step(dx, reward)
            self.A_touch = self.toucher.step(self.sense_food_multi())
        if self.thermosensor:
            self.A_thermo = self.thermosensor.step(self.sense_thermo(pos))
        if self.windsensor:
            self.A_wind = self.windsensor.step(self.sense_wind())

    def step(self, pos, on_food=False, **kwargs):
        self.sense(pos=pos, reward=on_food)
        return self.locomotor.step(A_in=self.A_in, on_food=on_food, **kwargs)
