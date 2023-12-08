import itertools
import random
import numpy as np
import param

from ... import aux, reg
from ...aux import AttrDict, SuperList
from . import crawler, turner, crawl_bend_interference, intermitter, sensor, feeder, memory, basic
from ...param import class_defaults as cd, NestedConf, class_objs, ModeSelector, ClassDict
from .. import deb, agents

__all__ = [
    'ModuleModes',
    'ModuleColorDict',
    'AuxModules',
    'AllModules',
    'BrainModuleDB',
    'brainConf',
    'larvaConf',
    'SpaceDict',
    'moduleDB',
    # 'autogenerate_confs',
]


class ModuleModes(NestedConf):



    mID = param.String(default=None, doc='The unoique ID of the module')
    color = param.Color(default=None, doc='The background color when plotting module tables')
    mode_dict = param.Dict(default=aux.AttrDict(), doc='A dictionary of implemented modes as classes')


    @property
    def parent_class(self):
        return aux.common_ancestor_class(list(self.mode_dict.values()))

    @property
    def modes(self):
        return self.mode_dict.keylist



    def mod_gen(self, m, **kwargs):
        if m is not None:
            if 'mode' in m and m.mode in self.mode_dict:
                return self.mode_dict[m.mode](**{k: m[k] for k in m if k != 'mode'}, **kwargs)
        return None

    def mod_class(self, mode=None):
        if mode is not None and mode in self.mode_dict:
            return self.mode_dict[mode]
        else:
            return None

    def mod_kws(self, mode=None, excluded=[basic.Effector, 'phi'], include_mode=True, **kwargs):
        C=self.mod_class(mode=mode)
        if C is not None:
            dic=cd(A=C, excluded=excluded, **kwargs)
            if include_mode:
                dic['mode']=mode
            return dic
        else:
            return None

    def mod_objs(self, mode=None, excluded=[basic.Effector, 'phi', 'name']):
        C = self.mod_class(mode=mode)
        if C is not None:
            return class_objs(A=C, excluded=excluded)

        else:
            return AttrDict()

    def mod_vars(self, **kwargs):
        return self.mod_objs(**kwargs).keylist

# class ModuleDB(NestedConf):


# crawler_modes=ModuleModes(mode_dict={
#         'constant': crawler.Crawler,
#         'gaussian': crawler.GaussOscillator,
#         'square': crawler.SquareOscillator,
#         'realistic': crawler.PhaseOscillator
#
#     }, color='lightcoral')








ModuleColorDict = AttrDict({
    'body': 'lightskyblue',
    'physics': 'lightsteelblue',
    'energetics': 'lightskyblue',
    'DEB': 'lightskyblue',
    'gut': 'lightskyblue',
    'Box2D': 'lightcoral',
    'crawler': 'lightcoral',
    'turner': 'indianred',
    'interference': 'lightsalmon',
    'intermitter': '#a55af4',
    'olfactor': 'palegreen',
    'windsensor': 'plum',
    'thermosensor': 'plum',
    'toucher': 'pink',
    'feeder': 'pink',
    'memory': 'pink',
})



class BrainModuleDB(NestedConf):
    ModeKDict = AttrDict({'realistic': 'RE', 'square': 'SQ', 'gaussian': 'GAU', 'constant': 'CON',
                          'default': 'DEF', 'neural': 'NEU', 'sinusoidal': 'SIN', 'nengo': 'NENGO',
                          'phasic': 'PHI', 'branch': 'BR', 'osn': 'OSN', 'RL': 'RL', 'MB': 'MB'})
    ModuleModeDict = AttrDict({
        'crawler': {
            'constant': crawler.Crawler,
            'gaussian': crawler.GaussOscillator,
            'square': crawler.SquareOscillator,
            'realistic': crawler.PhaseOscillator

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
            'osn': sensor.OSNOlfactor,
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

    def __init__(self, **kwargs):
        self.dict=AttrDict({k: ModuleModes(mID=k, mode_dict=dic, color=ModuleColorDict[k]) for k, dic in self.ModuleModeDict.items()})
        self.LocoModsBasic=SuperList(['crawler', 'turner', 'interference', 'intermitter'])
        self.LocoMods=SuperList(['crawler', 'turner', 'interference', 'intermitter', 'feeder'])
        self.SensorMods = SuperList(['olfactor', 'toucher', 'windsensor', 'thermosensor'])
        super().__init__(**kwargs)

    def mod_modes(self, k, short=False):
        if k not in self.ids :
            return None
        else:
            ms=self.dict[k].modes
            if short :
                return SuperList([self.ModeKDict[m] for m in ms])
            else:
                return ms


    @property
    def ids(self):
        return self.dict.keylist

    def mod_gen(self, k, m, **kwargs):
        return self.dict[k].mod_gen(m, **kwargs) if k in self.ids else None



    def mod_gen_multi(self, ks, conf, **kwargs):
        return AttrDict({k: self.mod_gen(k, conf[k] if k in conf else None, **kwargs) for k in ks})

    def mod_kws(self, k, mode=None, as_entry=True, **kwargs):
        C =self.dict[k].mod_kws(mode= mode, **kwargs) if k in self.ids else None
        return AttrDict({f'brain.{k}': C}) if as_entry else C

    def mod_objs(self, k, mode=None, as_entry=True, **kwargs):
        C = self.dict[k].mod_objs(mode=mode, **kwargs) if k in self.ids else AttrDict()
        return AttrDict({f'brain.{k}': C}) if as_entry else C

    def mod_objs_multi(self, ks, conf,as_entry=True, **kwargs):
        C= AttrDict({k: self.mod_objs(k, conf[k] if k in conf else AttrDict(), **kwargs) for k in ks})
        return AttrDict({f'brain.{k}': C[k] for k in C}).flatten() if as_entry else C

    def mod_vars(self, **kwargs):
        return self.mod_objs(**kwargs).keylist

    def mod_vars_multi(self, **kwargs):
        return self.mod_objs_multi(**kwargs).keylist

    def brainConf(self, ms={}, mkws={}):
        C = AttrDict({k:m.mod_kws(mode=ms[k] if k in ms else None, **mkws[k] if k in mkws else {}) for k,m in self.dict.items()})
        C.nengo = ms['intermitter'] == 'nengo'
        return C


    def mcolor(self, k):
        return self.dict[k].color if k in self.ids else None

    def mod_combs(self, ks, short=False,to_return='yield'):
        ks=ks.existing(self.ids)
        x=itertools.product(*[self.mod_modes(k, short=short) for k in ks])
        if to_return=='yield':
            return x
        elif to_return=='list':
            return list(x)

    def parent_class(self,k):
        return self.dict[k].parent_class if k in self.ids else None


moduleDB=BrainModuleDB()
AuxModules = SuperList(['physics', 'body', 'energetics', 'sensorimotor'])
AllModules = moduleDB.ids + AuxModules



def brainConf(ms={}, mkws={}):
    C = AttrDict()
    for k in moduleDB.ids:
        if k not in ms:
            ms[k] = None
        if k not in mkws:
            mkws[k] = {}
        C[k] = moduleDB.mod_kws(k, mode=ms[k], as_entry=False, **mkws[k])
    C.nengo = ms['intermitter'] == 'nengo'
    return C


def larvaConf(ms={}, mkws={}):
    C = AttrDict({'brain': brainConf(ms=ms, mkws=mkws)})
    for k in AuxModules:
        if k not in mkws:
            mkws[k] = {}
        if k == 'energetics':
            C[k] = None
        elif k == 'sensorimotor':
            continue
        elif k == 'physics':
            C[k] = cd(agents.BaseController, **mkws[k])
        elif k == 'body':
            C[k] = cd(agents.LarvaSegmented,
                      excluded=[agents.OrientedAgent, 'vertices', 'base_vertices', 'width', 'guide_points', 'segs'],
                      **mkws[k])
        else:
            raise

    #  TODO thsi
    C.Box2D = {
        'joint_types': {
            'friction': {'N': 0, 'args': {}},
            'revolute': {'N': 0, 'args': {}},
            'distance': {'N': 0, 'args': {}}
        }
    }
    return C




class SpaceDict(NestedConf):
    base_model = reg.conf.Model.confID_selector()
    space_mkeys = param.ListSelector(default=[], objects=AllModules,
                                     label='keys of modules to include in space search',
                                     doc='Keys of the modules where the optimization parameters are')
    Pmutation = param.Magnitude(default=0.3, step=0.01,label='mutation probability',
                                doc='Probability of mutation for each agent in the next generation')
    Cmutation = param.Magnitude(default=0.1, step=0.01,label='mutation coeficient',
                             doc='Fraction of allowed parameter range to mutate within')

    init_mode = param.Selector(default='random', objects=['random', 'model', 'default'],
                               label='mode of initial generation', doc='Mode of initial generation')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mConf0 = reg.conf.Model.getID(self.base_model)
        self.space_objs = self.build()
        self.space_ks = self.space_objs.keylist
        self.parclasses = AttrDict({p: self.parclass(p) for p in self.space_ks})
        # for p in self.space_ks:
        #     self.param.add_parameter(p, self.space_objs[p])

    def build(self):
        D = AttrDict()
        for k in self.space_mkeys:
            xx = self.mConf0.brain[k]
            if xx is not None:
                A = moduleDB.dict[k][xx.mode]
                Aobjs = class_objs(A, excluded=[basic.Effector, 'phi', 'name'])
                for p, obj in Aobjs.items():
                    if p in xx:
                        obj.default = xx[p]
                    D[f'brain.{k}.{p}'] = obj
        return D

    def obj_attr(self, k, flat=True):
        if flat:
            return AttrDict({p: getattr(obj,k) if hasattr(obj,k) else None for p, obj in self.space_objs.items()})
        else:
            return AttrDict({obj.name: getattr(obj, k) if hasattr(obj, k) else None for p, obj in self.space_objs.items()})

    def obj_min_max_value(self, p):
        obj = self.space_objs[p]
        v = obj.default
        if isinstance(v,tuple):
            if v[0] is None:
                v=v[1]
            elif v[1] is None:
                v=v[0]
            else:
                v=np.mean([v[0], v[1]])
        min, max=obj.bounds if hasattr(obj,'bounds') else (None,None)
        step=obj.step if hasattr(obj,'step') else None
        try:
            return param._get_min_max_value(min, max, value=v, step=step)
        except:
            return None


    @property
    def defaults(self):
        return self.obj_attr('default')


    def parclass(self, p):
        obj = self.space_objs[p]
        c = type(obj)

        def check(k):
            m = getattr(param, k)
            ms = [m] + m.__subclasses__()
            return c in ms or issubclass(c,m)

        valid = [k for k in ['Magnitude', 'Integer', 'Number', 'Selector', 'Boolean', 'Range', 'Dict'] if check(k)]
        return valid[0]

    def randomize(self):
        g=self.defaults
        for p in self.space_ks:
            v=g[p]
            obj = self.space_objs[p]
            cl = self.parclasses[p]
            if cl in ['Selector']:
                g[p] = random.choice(obj.objects)
            elif cl in ['Boolean']:
                g[p] = random.choice([True, False])
            else:
                vmin, vmax = obj.bounds
                if None in (vmin, vmax):
                    vmin0,vmax0,vv=self.obj_min_max_value(p)
                else:
                    vmin0, vmax0=vmin, vmax

                if cl in ['Range']:
                    vnew = random.uniform(vmin0, vmax0)
                    vnew2 = random.uniform(vmin0, vmax0)
                    g[p] = (np.clip(vnew, a_min=vmin, a_max=vmax),np.clip(vnew2, a_min=vmin, a_max=vmax))
                elif cl in ['Integer']:
                    g[p] =obj.crop_to_bounds(random.randint(vmin0, vmax0))
                else:
                    g[p] = obj.crop_to_bounds(random.uniform(vmin0, vmax0))





    def mutate(self, g):
        for p in self.space_ks:
            v = g[p]
            if random.random() < self.Pmutation:
                obj = self.space_objs[p]
                cl = self.parclasses[p]
                if cl in ['Selector']:
                    g[p] = random.choice(obj.objects)
                elif cl in ['Boolean']:
                    g[p] = random.choice([True, False])
                else :
                    if v is not None:
                        if hasattr(obj, 'step') and obj.step is not None:
                            vr = obj.step * 5
                        else:
                            vmin, vmax = obj.bounds
                            if None in (vmin, vmax):
                                vmin, vmax, vv = self.obj_min_max_value(p)
                            vr = np.abs(vmax - vmin) * 0.5
                        s = self.Cmutation * vr
                        if cl in ['Range']:
                            vmin, vmax = obj.bounds
                            g[p] = (np.clip(random.gauss(v[0], s), a_min=vmin, a_max=vmax),
                                  np.clip(random.gauss(v[1], s), a_min=vmin, a_max=vmax))

                        elif cl in ['Integer']:
                            g[p] = obj.crop_to_bounds(int(random.gauss(v, s)))
                        else:
                            g[p] = obj.crop_to_bounds(random.gauss(v, s))


        return g



    def create_first_generation(self, N):
        m = self.init_mode
        if m == 'default':
            return [self.defaults] * N
        elif m == 'model':
            return [AttrDict({k: self.mConf0.flatten()[k] for k in self.space_ks})] * N
        elif m == 'random':
            return [self.randomize() for i in range(N)]
        else:
            raise ValueError('Not implemented')












@reg.funcs.stored_conf("Model")
def Model_dict():
    MD = moduleDB
    LMs = MD.LocoModsBasic

    def olf_kws(g={'Odor': 150.0}, mode='default', **kwargs):
        return MD.mod_kws('olfactor', mode=mode, gain_dict=g, **kwargs)

    def mem_kws(mode='RL', modality='olfaction', **kwargs):
        return AttrDict({'brain.memory': cd(MD.dict.memory.mode_dict[mode][modality], excluded=['dt'],
                                            included={'mode': mode, 'modality': modality}, **kwargs)})


    E = {}

    def new(id, id0, kws={}):
        try:
            E[id] = E[id0].new_dict(kws)
        except:
            pass

    def extend(id0):
        def new0(id, kws={}):
            new(id=id, id0=id0, kws=kws)

        for sg, g in zip(['', '0', '_x2'], [{'Odor': 150.0}, {'Odor': 0.0}, {'CS': 150.0, 'UCS': 0.0}]):
            for sb, br in zip(['', '_brute'], [False, True]):
                idd = f'{id0}_nav{sg}{sb}'
                o = olf_kws(g=g, brute_force=br)
                new0(idd, o)
                for k in ['RL', 'MB']:
                    new0(f'{idd}_{k}', {**o, **mem_kws(k)})

        for ss, eeb in zip(['', '_max'], [0.5, 0.9]):
            f = AttrDict({**MD.mod_kws('feeder', mode='default'), 'brain.intermitter.feed_bouts': True, 'brain.intermitter.EEB': eeb})
            new0(f'{id0}{ss}_feeder', f)
            for sg, g in zip(['', '0', '_x2'], [{'Odor': 150.0}, {'Odor': 0.0}, {'CS': 150.0, 'UCS': 0.0}]):
                idd = f'{id0}{ss}_forager{sg}'
                o = olf_kws(g=g)
                new0(idd, {**o, **f})
                for k in ['RL', 'MB']:
                    new0(f'{idd}_{k}', {**o, **f, **mem_kws(k)})

        for mm in [f'{id0}_avg', f'{id0}_var', f'{id0}_var2']:
            if mm in reg.conf.Model.confIDs:
                E[mm] = reg.conf.Model.getID(mm)

    for id, (Tm, ImM) in zip(['Levy', 'NEU_Levy', 'NEU_Levy_continuous'],
                             [('sinusoidal', 'default'), ('neural', 'default'), ('neural', None)]):
        E[id] = larvaConf(ms=AttrDict(zip(LMs, ['constant', Tm, 'default', ImM])),
                          mkws={'interference': {'attenuation': 0.0}, 'intermitter': {'run_mode': 'exec'}})
        extend(id0=id)

    for ii in MD.mod_combs(LMs):

    # for ii in itertools.product(*[MD.dict[mk].modes for mk in LMs]):
        mms = [MD.ModeKDict[i] for i in ii]
        id = "_".join(mms)
        E[id] = larvaConf(ms=AttrDict(zip(LMs, ii)),
                          mkws={'interference': {'attenuation': 0.1, 'attenuation_max': 0.6}} if mms[
                                                                                                     2] != 'DEF' else {})
        if mms[0] == 'RE' and mms[3] == 'DEF':
            extend(id0=id)
            if mms[1] == 'NEU' and mms[2] == 'PHI':
                for idd in ['forager', 'forager0', 'forager_x2', 'max_forager', 'max_forager0',
                            'forager_RL', 'forager0_RL', 'max_forager_RL', 'max_forager0_RL',
                            'forager_MB', 'forager0_MB', 'max_forager_MB', 'max_forager0_MB',
                            'feeder', 'max_feeder']:
                    E[idd] = E[f'{id}_{idd}']
                E['explorer'] = E[id]
                E['navigator'] = E[f'{id}_nav']
                E['navigator_x2'] = E[f'{id}_nav_x2']
                E['RLnavigator'] = E[f'{id}_nav_RL']

    for id, dd in zip(['imitator', 'zebrafish', 'thermo_navigator', 'OSNnavigator', 'OSNnavigator_x2'],
                      [{'body.Nsegs': 11},
                       {'body.body_plan': 'zebrafish_larva', 'Box2D': {'joint_types': {
                           'revolute': {'N': 1, 'args': {'maxMotorTorque': 10 ** 5, 'motorSpeed': 1}}}}},
                       MD.mod_kws('thermosensor', mode='default'),
                       olf_kws(mode='osn'),
                       olf_kws({'CS': 150.0, 'UCS': 0.0}, mode='osn')]):
        new(id, 'explorer', dd)
    for ss, kkws in zip(['', '_2', '_brute'], [{}, {'touch_sensors': [0, 2]}, {'brute_force': True}]):
        new(f'toucher{ss}', 'explorer', MD.mod_kws('toucher', mode='default', **kkws))
        new(f'RLtoucher{ss}', f'toucher{ss}', mem_kws(modality='touch'))
    for id, gd in zip(['follower-R', 'follower-L', 'gamer', 'gamer-5x'], [{'Left_odor': 150.0, 'Right_odor': 0.0},
                                                                          {'Left_odor': 0.0, 'Right_odor': 150.0},
                                                                          {'Flag_odor': 150.0, 'Left_base_odor': 0.0,
                                                                           'Right_base_odor': 0.0},
                                                                          {'Flag_odor': 150.0, 'Left_base_odor': 0.0,
                                                                           'Right_base_odor': 0.0, 'Left_odor': 0.0,
                                                                           'Right_odor': 0.0}
                                                                          ]):
        new(id, 'forager', {'brain.olfactor.gain_dict': gd})

    new('immobile', 'navigator', {'brain.crawler': None, 'brain.turner': None,
                                  'brain.intermitter': None, 'brain.interference': None,
                                  **MD.mod_kws('toucher', mode='default')})
    new('obstacle_avoider', 'navigator', {'sensorimotor': cd(agents.ObstacleLarvaRobot, excluded=[agents.LarvaRobot])})

    for id in ['explorer', 'navigator', 'feeder', 'forager']:
        new(f'{id}_sample', id, {f'brain.crawler.{k}': 'sample' for k in
                                 ['stride_dst_mean', 'stride_dst_std', 'max_scaled_vel', 'max_vel_phase', 'freq']})

    for sp, k_abs, eeb in zip(['rover', 'sitter'], [0.8, 0.4], [0.67, 0.37]):
        en_ws = {'energetics': AttrDict({
            'DEB': cd(deb.DEB, excluded=[deb.DEB_model, 'substrate', 'id'], species=sp),
            'gut': cd(deb.Gut, k_abs=k_abs)
        })}
        en_ws2 = {**en_ws, 'brain.intermitter.EEB': eeb}
        new(f'{sp}_explorer', 'explorer', en_ws)
        new(f'{sp}_navigator', 'navigator', en_ws)
        new(f'{sp}_feeder', 'feeder', en_ws2)
        new(f'{sp}_forager', 'forager', en_ws2)
        new(sp, f'{sp}_feeder')

    return E
