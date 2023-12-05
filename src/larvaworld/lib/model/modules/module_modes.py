import itertools

from ... import aux, reg
from . import crawler, turner, crawl_bend_interference, intermitter, sensor, feeder, memory, basic
from ...param import class_defaults as cd
from .. import deb, agents

__all__ = [
    'ModuleModeDict',
    'ModuleModeKDict',
    'ModuleColorDict',
    'LocoModules',
    'SensorModules',
    'brainConf',
    'larvaConf',
    'loco_kws',
    'sensor_kws',
    'olf_kws',
    'mem_kws',
    'feed_kws',
    # 'autogenerate_confs',
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

ModuleModeKDict = aux.bidict(aux.AttrDict({'realistic': 'RE', 'square': 'SQ', 'gaussian': 'GAU', 'constant': 'CON',
                   'default': 'DEF', 'neural': 'NEU', 'sinusoidal': 'SIN', 'nengo': 'NENGO', 'phasic': 'PHI',
                   'branch': 'BR'}))

LocoModules = ['crawler', 'turner', 'interference', 'intermitter']
SensorModules = ['olfactor', 'toucher', 'windsensor', 'thermosensor']
BrainModules = LocoModules+SensorModules+['feeder', 'memory']
AuxModules = ['physics', 'body', 'energetics', 'sensorimotor']
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


def brainConf(ms=None, mkws={}):
    if ms is None:
        ms = {'crawler': 'realistic',
              'turner': 'neural',
              'interference': 'phasic',
              'intermitter': 'default'}

    C = aux.AttrDict()

    for k in BrainModules:
        kk = f'{k}_params'
        if k not in ms or ms[k] is None:
            C[kk] = None
        else:
            kws = mkws[k] if k in mkws else {}
            C[kk] = cd(ModuleModeDict[k][ms[k]], excluded=[basic.Effector, 'phi'], included={'mode': ms[k]}, **kws)

    C.nengo = True if ('intermitter' in ms and ms['intermitter'] == 'nengo') else False
    return C

def larvaConf(ms=None, mkws={}):
    C = aux.AttrDict()
    C.brain = brainConf(ms, mkws)

    for k in AuxModules:
        kws = mkws[k] if k in mkws else {}
        if k == 'energetics':
            C[k] = None
            # continue
        elif k == 'sensorimotor':
            continue
        elif k == 'physics':
            C[k] = cd(agents.BaseController, **kws)
        elif k == 'body':
            C[k] = cd(agents.LarvaSegmented, excluded=[agents.OrientedAgent, 'vertices', 'base_vertices', 'width','guide_points', 'segs'], **kws)
        else:
            raise

    #  TODO thsi

    C.Box2D_params = {
        'joint_types': {
            'friction': {'N': 0, 'args': {}},
            'revolute': {'N': 0, 'args': {}},
            'distance': {'N': 0, 'args': {}}
        }
    }

    return C

def loco_kws(module, mode, **kwargs):
    assert module in LocoModules+['feeder']
    return {
        f'brain.{module}_params': cd(ModuleModeDict[module][mode], excluded=[basic.Effector, 'phi'], included={'mode': mode},
                                     **kwargs)}

def sensor_kws(sensor, mode='default', **kwargs):
    assert sensor in SensorModules
    return {
        f'brain.{sensor}_params': cd(ModuleModeDict[sensor][mode], excluded=[basic.Effector], included={'mode': mode},
                                     **kwargs)}


def mem_kws(mode='RL', modality='olfaction', **kwargs):
    return {'brain.memory_params': cd(ModuleModeDict.memory[mode][modality], excluded=['dt'],
                                      included={'mode': mode, 'modality': modality}, **kwargs)}


def olf_kws(g={'Odor': 150.0}, mode='default', **kwargs):
    return sensor_kws(sensor='olfactor', mode=mode, gain_dict=g, **kwargs)


def feed_kws(v=0.5):
    return aux.AttrDict(
        {'brain.feeder_params': cd(ModuleModeDict.feeder.default, excluded=['dt', 'phi'], included={'mode': 'default'}),
         'brain.intermitter_params.feed_bouts': True, 'brain.intermitter_params.EEB': v})


@reg.funcs.stored_conf("Model")
def Model_dict():
    # lc = reg.model.larvaConf
    MD = ModuleModeDict

    E = {}

    def new(id, id0, kws={}):
        try:
            E[id] = E[id0].new_dict(kws)
        except:
            pass

    def newexp(id, kws={}):
        new(id=id, id0='explorer', kws=kws)

    def newnav(id, kws={}):
        new(id=id, id0='navigator', kws=kws)

    def newfor(id, kws={}):
        new(id=id, id0='forager', kws=kws)

    def OD(gain_dict):
        return {'brain.olfactor_params.gain_dict': gain_dict}

    def EEB(v=0.9):
        return {'brain.intermitter_params.EEB': v}

    def extend(id0):
        for sg, g in zip(['', '0', '_x2'], [{'Odor': 150.0}, {'Odor': 0.0}, {'CS': 150.0, 'UCS': 0.0}]):
            for sb, br in zip(['', '_brute'], [False, True]):
                idd = f'{id0}_nav{sg}{sb}'
                o = olf_kws(g=g, brute_force=br)
                new(id=idd, id0=id0, kws=o)
                for k in ['RL', 'MB']:
                    new(id=f'{idd}_{k}', id0=id0, kws={**o, **mem_kws(k)})

        for ss, eeb in zip(['', '_max'], [0.5, 0.9]):
            f = feed_kws(eeb)
            new(id=f'{id0}{ss}_feeder', id0=id0, kws=f)
            for sg, g in zip(['', '0', '_x2'], [{'Odor': 150.0}, {'Odor': 0.0}, {'CS': 150.0, 'UCS': 0.0}]):
                idd = f'{id0}{ss}_forager{sg}'
                o = olf_kws(g=g)
                new(id=idd, id0=id0, kws={**o, **f})
                for k in ['RL', 'MB']:
                    new(id=f'{idd}_{k}', id0=id0, kws={**o, **f, **mem_kws(k)})

        for mm in [f'{id0}_avg', f'{id0}_var', f'{id0}_var2']:
            if mm in reg.conf.Model.confIDs:
                E[mm] = reg.conf.Model.getID(mm)

    mkws0 = aux.AttrDict({'interference': {'attenuation': 0.1, 'attenuation_max': 0.6}})

    E['explorer'] = larvaConf(mkws=mkws0)

    for id, (Tm,ImM) in zip(['Levy', 'NEU_Levy', 'NEU_Levy_continuous'],
                            [('sinusoidal', 'default'),('neural', 'default'),('neural', None)]):
        E[id] = larvaConf(ms=aux.AttrDict(zip(LocoModules, ['constant', Tm, 'default', ImM])),
                   mkws=aux.AttrDict({'interference': {'attenuation': 0.0}, 'intermitter': {'run_mode': 'exec'}}))
        extend(id0=id)

    newexp('imitator', {'body.Nsegs': 11})
    newexp('zebrafish', {'body.body_plan': 'zebrafish_larva', 'Box2D_params': {'joint_types': {
        'revolute': {'N': 1, 'args': {'maxMotorTorque': 10 ** 5, 'motorSpeed': 1}}}}})
    newexp('thermo_navigator', sensor_kws('thermosensor'))

    for ii in itertools.product(*[MD[mk].keylist for mk in LocoModules]):
        mms=[ModuleModeKDict[i] for i in ii]
        id = "_".join(mms)
        E[id] = larvaConf(ms=aux.AttrDict(zip(LocoModules, ii)), mkws=mkws0 if mms[2] != 'DEF' else {})
        if mms[0] == 'RE' and mms[3] == 'DEF':
            extend(id0=id)
            if mms[1] == 'NEU' and mms[2] == 'PHI':
                for idd in ['forager', 'forager0','forager_x2', 'max_forager', 'max_forager0',
                            'forager_RL', 'forager0_RL', 'max_forager_RL', 'max_forager0_RL',
                            'forager_MB', 'forager0_MB', 'max_forager_MB', 'max_forager0_MB',
                            'feeder', 'max_feeder']:
                    E[idd] = E[f'{id}_{idd}']
                E['navigator'] = E[f'{id}_nav']
                E['navigator_x2'] = E[f'{id}_nav_x2']
                E['RLnavigator'] = E[f'{id}_nav_RL']



    newexp('OSNnavigator', olf_kws(mode='osn'))
    newexp('OSNnavigator_x2', olf_kws({'CS': 150.0, 'UCS': 0.0}, mode='osn'))
    newexp('toucher', sensor_kws('toucher'))
    newexp('toucher_2', sensor_kws('toucher', touch_sensors=[0, 2]))
    newexp('toucher_brute', sensor_kws('toucher', brute_force=True))
    new('RLtoucher', 'toucher', mem_kws(modality='touch'))
    new('RLtoucher_2', 'toucher_2', mem_kws(modality='touch'))
    newfor('follower-R', OD({'Left_odor': 150.0, 'Right_odor': 0.0}))
    newfor('follower-L', OD({'Left_odor': 0.0, 'Right_odor': 150.0}))
    newfor('gamer', OD({'Flag_odor': 150.0, 'Left_base_odor': 0.0, 'Right_base_odor': 0.0}))
    newfor('gamer-5x', OD({'Flag_odor': 150.0, 'Left_base_odor': 0.0, 'Right_base_odor': 0.0, 'Left_odor': 0.0,
                           'Right_odor': 0.0}))

    newnav('immobile', {'brain.crawler_params': None, 'brain.turner_params': None,
                        'brain.intermitter_params': None, 'brain.interference_params': None,
                        **sensor_kws('toucher')})
    newnav('obstacle_avoider', {'sensorimotor': cd(agents.ObstacleLarvaRobot, excluded=[agents.LarvaRobot])})

    for id in ['explorer', 'navigator', 'feeder', 'forager']:
        new(f'{id}_sample', id, {k: 'sample' for k in [
            'brain.crawler_params.stride_dst_mean',
            'brain.crawler_params.stride_dst_std',
            'brain.crawler_params.max_scaled_vel',
            'brain.crawler_params.max_vel_phase',
            'brain.crawler_params.freq',
        ]})

    for sp, k_abs, eeb in zip(['rover', 'sitter'], [0.8, 0.4], [0.67, 0.37]):
        en_ws = {'energetics': aux.AttrDict({
            'DEB': cd(deb.DEB, excluded=[deb.DEB_model, 'substrate', 'id'], species=sp),
            'gut': cd(deb.Gut, k_abs=k_abs)
        })}
        newexp(f'{sp}_explorer', en_ws)
        newnav(f'{sp}_navigator', en_ws)
        new(f'{sp}_feeder', 'feeder', {**en_ws, **EEB(eeb)})
        newfor(f'{sp}_forager', {**en_ws, **EEB(eeb)})
        new(sp, f'{sp}_feeder')

    return E
