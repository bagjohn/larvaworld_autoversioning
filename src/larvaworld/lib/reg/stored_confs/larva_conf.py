'''
The larva model parameters
'''
import copy

import numpy as np

from ... import reg, aux

__all__ = [
    'Model_dict',
]

OD1 = {'Odor': 150.0}
OD2 = {'CS': 150.0, 'UCS': 0.0}


def Im(EEB, **kwargs):
    conf = reg.par.get_null('intermitter', feed_bouts=EEB > 0, EEB=EEB, **kwargs)

    return conf


# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------


def brain(ks, nengo=False, OD=None, **kwargs):
    base_coupling = reg.par.get_null('interference', mode='square', crawler_phi_range=(np.pi / 2, np.pi),
                                  feeder_phi_range=(0.0, 0.0),
                                  attenuation=0.1, attenuation_max=0.6)

    RL_olf_memory = reg.par.get_null('memory', Delta=0.1, state_spacePerSide=1, modality='olfaction',
                                  gain_space=np.arange(-200.0, 200.0, 50.0).tolist())

    module_dict = {
        'T': 'turner',
        'C': 'crawler',
        'If': 'interference',
        'Im': 'intermitter',
        'O': 'olfactor',
        'To': 'toucher',
        'W': 'windsensor',
        'Th': 'thermosensor',
        'F': 'feeder',
        'M': 'memory',
    }
    if 'L' in ks:
        ks.remove('L')
        ks += ['T', 'C', 'If', 'Im']
    elif 'LOF' in ks:
        ks.remove('LOF')
        ks += ['T', 'C', 'If', 'Im', 'O', 'F']
    modules = [module_dict[k] for k in ks]

    modules = reg.par.get_null('modules', **{m: True for m in modules})
    d = {'modules': modules}
    for k, v in modules.items():
        p = f'{k}_params'
        if not v:
            d[p] = None
        elif k in kwargs:
            d[p] = kwargs[k]
        elif k == 'interference':
            d[p] = base_coupling
        elif k == 'memory':
            d[p] = RL_olf_memory
        else:
            d[p] = reg.par.get_null(k)
        if k == 'olfactor' and d[p] is not None:
            d[p]['gain_dict'] = OD
    d['nengo'] = nengo
    return aux.AttrDict(d)




def mod(brain=None, bod={}, energetics=None, phys={}, Box2D={}):
    if Box2D == {}:
        null_Box2D_params = {
            'joint_types': {
                'friction': {'N': 0, 'args': {}},
                'revolute': {'N': 0, 'args': {}},
                'distance': {'N': 0, 'args': {}}
            }
        }
        Box2D_params = null_Box2D_params
    else:
        Box2D_params = reg.par.get_null('Box2D_params', **Box2D)
    return reg.par.get_null('Model', brain=brain,
                         energetics=energetics,
                         body=reg.par.get_null('body', **bod),
                         physics=reg.par.get_null('physics', **phys),
                         Box2D_params=Box2D_params
                         )


def OD(ids: list, gains: list) -> dict:
    return aux.AttrDict({id:g for id, g in zip(ids, gains)})


def create_mod_dict(b):

    RL_touch_memory = reg.par.get_null('memory', Delta=0.5, state_spacePerSide=1, modality='touch', train_dur=30,
                                    update_dt=0.5,
                                    gain_space=np.round(np.arange(-10.0, 11.0, 5), 1).tolist(), state_specific_best=True)

    gRL_touch_memory = reg.par.get_null('memory', Delta=0.5, state_spacePerSide=1, modality='touch', train_dur=30,
                                     update_dt=0.5,
                                     gain_space=np.round(np.arange(-10.0, 11.0, 5), 1).tolist(), state_specific_best=False)

    M0 = mod()

    def add_brain(brain, M0=M0, bod={}, phys={}, Box2D={}):
        M1 = aux.AttrDict(copy.deepcopy(M0))
        M1.brain = brain
        M1.body.update(**bod)
        M1.physics.update(**phys)
        M1.Box2D_params.update(**Box2D)
        return M1

    LOF = brain(['LOF'])
    L = brain(['L'])
    LTo = brain(['L', 'To'], toucher=reg.par.get_null('toucher', touch_sensors=[]))
    LToM = brain(['L', 'To', 'M'], toucher=reg.par.get_null('toucher', touch_sensors=[]),
                 memory=RL_touch_memory)
    LToMg = brain(['L', 'To', 'M'], toucher=reg.par.get_null('toucher', touch_sensors=[]),
                  memory=gRL_touch_memory)
    LTo2M = brain(['L', 'To', 'M'], toucher=reg.par.get_null('toucher', touch_sensors=[0, 2]),
                  memory=RL_touch_memory)
    LTo2Mg = brain(['L', 'To', 'M'], toucher=reg.par.get_null('toucher', touch_sensors=[0, 2]),
                   memory=gRL_touch_memory)
    LTo_brute = brain(['L', 'To'], toucher=reg.par.get_null('toucher', touch_sensors=[], brute_force=True))
    LTh = brain(['L', 'Th'])

    def add_OD(OD, B0=LOF):
        B1 = aux.AttrDict(copy.deepcopy(B0))
        B1.olfactor_params.gain_dict = OD
        return B1



    touchers = {
        'toucher': add_brain(LTo),
        'toucher_brute': add_brain(LTo_brute),
        'RL_toucher_0': add_brain(LToM),
        'gRL_toucher_0': add_brain(LToMg),
        'RL_toucher_2': add_brain(LTo2M),
        'gRL_toucher_2': add_brain(LTo2Mg),
    }

    other = {
        'thermo_navigator': add_brain(LTh),
        'imitator': add_brain(L, bod={'Nsegs': 11}),
        'immobile': add_brain(brain(['T', 'O'], OD=OD1)),
    }

    odors3 = [f'{i}_odor' for i in ['Flag', 'Left_base', 'Right_base']]
    odors5 = [f'{i}_odor' for i in ['Flag', 'Left_base', 'Right_base', 'Left', 'Right']]
    odors2 = [f'{i}_odor' for i in ['Left', 'Right']]

    gamers = {
        'gamer': add_brain(add_OD(OD(odors3, [150.0, 0.0, 0.0]))),
        'gamer-5x': add_brain(add_OD(OD(odors5, [150.0, 0.0, 0.0, 0.0, 0.0]))),
        'follower-R': add_brain(add_OD(OD(odors2, [150.0, 0.0]))),
        'follower-L': add_brain(add_OD(OD(odors2, [0.0, 150.0]))),
    }
    zebrafish = {
        'zebrafish': add_brain(L,
                               bod={'shape': 'zebrafish_larva'},
                               Box2D={
                                   'joint_types': {'revolute': {'N': 1, 'args': {'maxMotorTorque':10 ** 5, 'motorSpeed':1}}}})
    }

    grouped_mod_dict = {
        'touchers': touchers,
        # 'foraging phenotypes': build_RvsS(b),
        'games': gamers,
        'zebrafish': zebrafish,
        'other': other,
    }

    return grouped_mod_dict


@reg.funcs.stored_conf("Model")
def Model_dict():
    dnew = reg.model.autostored_confs
    b = dnew['navigator'].brain
    d = create_mod_dict(b)
    dd = aux.merge_dicts(list(d.values()))
    dnew.update(dd)
    return dnew


