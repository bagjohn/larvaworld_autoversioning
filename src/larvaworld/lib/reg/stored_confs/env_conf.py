import numpy as np
from matplotlib import colors

from ... import reg, aux
from ...param import Odor

__all__ = [
    'Env_dict',
]


@reg.funcs.stored_conf("Env")
def Env_dict():
    from ...reg import gen
    from ...reg.generators import FoodConf as FC

    def env(arenaXY, f=FC(), o=None, bl={}, w=None, th=None, torus=False):
        if type(arenaXY) == float:
            arena = gen.Arena(geometry='circular', dims=(arenaXY, arenaXY), torus=torus)
        elif type(arenaXY) == tuple:
            arena = gen.Arena(geometry='rectangular', dims=arenaXY, torus=torus)
        else:
            raise
        if o == 'D':
            o = gen.DiffusionValueLayer()
        elif o == 'G':
            o = gen.GaussianValueLayer()
        if w is not None:
            if 'puffs' in w:
                for id, args in w['puffs'].items():
                    w['puffs'][id] = reg.par.get_null('air_puff', **args)
            else:
                w['puffs'] = {}
            w = gen.WindScape(**w)
        if th is not None:
            th = gen.ThermoScape(**th)
        return gen.Env(arena=arena, food_params=f, odorscape=o, border_list=bl, windscape=w, thermoscape=th).nestedConf


    d = {
        'focus': env((0.01, 0.01)),
        'dish': env(0.1),
        'dish_40mm': env(0.04),
        'arena_200mm': env((0.2, 0.2)),
        'arena_500mm': env((0.5, 0.5)),
        'arena_1000mm': env((1.0, 1.0)),
        'odor_gradient': env((0.1, 0.06), FC.su(pos=(0.04, 0.0), odor=Odor.oG(2)), 'G'),
        'mid_odor_gaussian': env((0.1, 0.06), FC.su(odor=Odor.oG()), 'G'),
        'odor_gaussian_square': env((0.2, 0.2), FC.su(odor=Odor.oG()), 'G'),
        'mid_odor_diffusion': env((0.3, 0.3), FC.su(r=0.03, odor=Odor.oD()), 'D'),
        '4corners': gen.Env.foodNodor_4corners(),
        'food_at_bottom': env((0.2, 0.2), FC.sg(id='FoodLine', odor=Odor.oG(), a=0.002,
                                                r=0.001, N=20, shape='oval',
                                                scale=(0.01, 0.0), mode='periphery'), 'G'),
        'thermo_arena': env((0.3, 0.3), th={}),
        'windy_arena': env((0.3, 0.3), w={'wind_speed': 10.0}),
        'windy_blob_arena': env((0.5, 0.2),
                                FC.sgs(4, qs=np.ones(4), cs=aux.N_colors(4), N=1, scale=(0.04, 0.02), loc=(0.005, 0.0),
                                       mode='uniform', shape='rect', can_be_displaced=True, regeneration=True, os='D',
                                       regeneration_pos={'loc': (0.005, 0.0), 'scale': (0.0, 0.0)}),
                                w={'wind_speed': 100.0}, o='D'),
        'windy_arena_bordered': env((0.3, 0.3), w={'wind_speed': 10.0},
                                    bl={'Border': gen.Border(vertices=[(-0.03, -0.01), (-0.03, -0.06)], width=0.005)}),
        'puff_arena_bordered': env((0.3, 0.3), w={
            'puffs': {'PuffGroup': {'N': 100, 'duration': 300.0, 'start_time': 0, 'speed': 100}}},
                                   bl={'Border': gen.Border(vertices=[(-0.03, -0.01), (-0.03, -0.06)], width=0.005)}),
        'single_puff': env((0.3, 0.3),
                           w={'puffs': {'Puff': {'N': 1, 'duration': 30.0, 'start_time': 55, 'speed': 100}}}),

        'CS_UCS_on_food': env(0.1, FC.CS_UCS(grid=gen.FoodGrid()), 'G'),
        'CS_UCS_on_food_x2': env(0.1, FC.CS_UCS(grid=gen.FoodGrid(), N=2), 'G'),
        'CS_UCS_off_food': env(0.1, FC.CS_UCS(), 'G'),

        'patchy_food': env((0.2, 0.2), FC.sg(N=8, scale=(0.07, 0.07), mode='periphery', a=0.001, odor=Odor.oG(2)), 'G'),
        'random_food': env((0.1, 0.1), FC.sgs(4, N=1, scale=(0.04, 0.04), mode='uniform', shape='rect')),
        'uniform_food': env(0.05, FC.sg(N=2000, scale=(0.025, 0.025), a=0.01, r=0.0001)),
        'patch_grid': env((0.2, 0.2), FC.sg(N=5 * 5, scale=(0.2, 0.2), a=0.01, r=0.007, mode='grid', shape='rect',
                                            odor=Odor.oG(0.2)), 'G'),

        'food_grid': env((0.02, 0.02), FC(food_grid=gen.FoodGrid())),
        'single_odor_patch': env((0.1, 0.1), FC.patch(odor=Odor.oG()), 'G'),
        'single_patch': env((0.05, 0.05), FC.patch()),
        'multi_patch': env((0.02, 0.02), FC.sg(N=8, scale=(0.007, 0.007), mode='periphery', a=0.1, r=0.0015)),
        'double_patch': env((0.24, 0.24), FC.patch2(), 'G'),

        'maze': gen.Env.maze(),
        'game': gen.Env.game(),
        'arena_50mm_diffusion': env(0.05, o='D'),
    }
    return d
