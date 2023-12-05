import itertools

from ... import aux, reg
from ...aux import AttrDict
from . import crawler, turner, crawl_bend_interference, intermitter, sensor, feeder, memory, basic
from ...param import class_defaults as cd
from .. import deb, agents

__all__ = [
    'ModuleModeDict',
    'ModuleModeKDict',
    'ModuleColorDict',
    'LocoModules',
    'SensorModules',
    'BrainModules',
    'AuxModules',
    'AllModules',
    'brainConf',
    'larvaConf',
    'mod_kws',
    'mod_gen',
    'mod_parent_class',
    'olf_kws',
    'mem_kws',
    'feed_kws',
    # 'autogenerate_confs',
]

ModuleModeDict = AttrDict({
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

ModuleModeKDict = AttrDict({'realistic': 'RE', 'square': 'SQ', 'gaussian': 'GAU', 'constant': 'CON',
                            'default': 'DEF', 'neural': 'NEU', 'sinusoidal': 'SIN', 'nengo': 'NENGO',
                            'phasic': 'PHI', 'branch': 'BR'})

LocoModules = ['crawler', 'turner', 'interference', 'intermitter']
SensorModules = ['olfactor', 'toucher', 'windsensor', 'thermosensor']
BrainModules = LocoModules + SensorModules + ['feeder', 'memory']
AuxModules = ['physics', 'body', 'energetics', 'sensorimotor']
AllModules = BrainModules + AuxModules
ModuleColorDict = AttrDict({
    'body': 'lightskyblue',
    'physics': 'lightsteelblue',
    'energetics': 'lightskyblue',
    'Box2D': 'lightcoral',
    'crawler': 'lightcoral',
    'turner': 'indianred',
    'interference': 'lightsalmon',
    'intermitter': '#a55af4',
    'olfactor': 'palegreen',
    'windsensor': 'plum',
    'toucher': 'pink',
    'feeder': 'pink',
    'memory': 'pink',
})


def mod_parent_class(k):
    assert k in ModuleModeDict
    M = ModuleModeDict[k]
    return aux.common_ancestor_class(list(M.values()))


def brainConf(ms={}, mkws={}):
    C = AttrDict()
    for k in BrainModules:
        if k not in ms:
            ms[k] = None
        if k not in mkws:
            mkws[k] = {}
        C[k] = mod_kws(k, mode=ms[k], as_entry=False, **mkws[k])
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


def mod_gen(k, m, **kwargs):
    assert k in BrainModules
    M = ModuleModeDict[k]
    if m is None:
        C = None
    else:
        mode = m.mode
        if mode is None or mode not in M:
            C = None
        else:
            C = M[mode](**{k: m[k] for k in m if k != 'mode'}, **kwargs)
    return C


def mod_kws(k, mode=None, as_entry=True, **kwargs):
    assert k in BrainModules
    M = ModuleModeDict[k]
    if mode is None or mode not in M:
        C = None
    else:
        C = cd(M[mode], excluded=[basic.Effector, 'phi'], included={'mode': mode}, **kwargs)
    if as_entry:
        return AttrDict({f'brain.{k}': C})
    else:
        return C


def mem_kws(mode='RL', modality='olfaction', **kwargs):
    return AttrDict({'brain.memory': cd(ModuleModeDict.memory[mode][modality], excluded=['dt'],
                                        included={'mode': mode, 'modality': modality}, **kwargs)})


def olf_kws(g={'Odor': 150.0}, mode='default', **kwargs):
    return mod_kws('olfactor', mode=mode, gain_dict=g, **kwargs)


def feed_kws(v=0.5, **kwargs):
    return AttrDict({**mod_kws('feeder', mode='default', **kwargs), 'brain.intermitter.feed_bouts': True,
                     'brain.intermitter.EEB': v})


@reg.funcs.stored_conf("Model")
def Model_dict():
    MD = ModuleModeDict
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
            f = feed_kws(eeb)
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
        E[id] = larvaConf(ms=AttrDict(zip(LocoModules, ['constant', Tm, 'default', ImM])),
                          mkws={'interference': {'attenuation': 0.0}, 'intermitter': {'run_mode': 'exec'}})
        extend(id0=id)

    for ii in itertools.product(*[MD[mk].keylist for mk in LocoModules]):
        mms = [ModuleModeKDict[i] for i in ii]
        id = "_".join(mms)
        E[id] = larvaConf(ms=AttrDict(zip(LocoModules, ii)),
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
                       mod_kws('thermosensor', mode='default'),
                       olf_kws(mode='osn'),
                       olf_kws({'CS': 150.0, 'UCS': 0.0}, mode='osn')]):
        new(id, 'explorer', dd)
    for ss, kkws in zip(['', '_2', '_brute'], [{}, {'touch_sensors': [0, 2]}, {'brute_force': True}]):
        new(f'toucher{ss}', 'explorer', mod_kws('toucher', mode='default', **kkws))
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
                                  **mod_kws('toucher', mode='default')})
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
