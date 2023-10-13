import numpy as np

from ... import reg, aux
from ...param import Epoch

__all__ = [
    'Trial_dict',
    # 'Life_dict',
    'Tree_dict',
    # 'Food_dict',
]


# def trial_conf(durs=[], qs=[]):
#     cumdurs = np.cumsum([0] + durs)
#     return aux.AttrDict(
#         {i: reg.par.get_null('epoch', start=t0, stop=t1, substrate=reg.par.get_null('substrate', quality=q)) for
#          i, (t0, t1, q) in
#          enumerate(zip(cumdurs[:-1], cumdurs[1:], qs))})

def trial_conf(durs=[], qs=[]):
    cumdurs = np.cumsum([0] + durs)
    return aux.ItemList(
        Epoch(age_range=(t0,t1), sub=[q, 'standard']).nestedConf for
         i, (t0, t1, q) in enumerate(zip(cumdurs[:-1], cumdurs[1:], qs)))


@reg.funcs.stored_conf("Trial")
def Trial_dict():
    d = aux.AttrDict({
        'default': aux.AttrDict({'epochs' : trial_conf()}),
        'odor_preference': aux.AttrDict({'epochs' : trial_conf(
            [5.0] * 8,
            [1.0, 0.0] * 4)}),
        'odor_preference_short': aux.AttrDict({'epochs' : trial_conf(
            [0.125] * 8,
            [1.0, 0.0] * 4)})
    })
    return d


# def life_conf(durs=[], qs=[], age=0.0):
#     return reg.par.get_null('Life', epochs=trial_conf(durs, qs), age=age)
#
#
# @reg.funcs.stored_conf("Life")
# def Life_dict():
#     d = aux.AttrDict({
#         'default': life_conf(durs=[0.0], qs=[1.0], age=0.0),
#         '72h_q50': life_conf(durs=[72.0], qs=[0.5], age=72.0),
#     })
#     return d


@reg.funcs.stored_conf("Tree")
def Tree_dict():
    return aux.AttrDict()


# @reg.funcs.stored_conf("Food")
# def Food_dict():
#     from ...reg import gen
#
#     return aux.AttrDict()
