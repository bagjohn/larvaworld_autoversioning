import numpy as np

from ... import reg, aux
from ...param import Odor

__all__ = [
    'Env_dict',
]


@reg.funcs.stored_conf("Env")
def Env_dict():
    from ...reg import gen
    E=reg.gen.Env
    FC=reg.gen.FoodConf



    d = {
        'focus': E.rect(0.01),
        'dish': E.dish(0.1),
        'dish_40mm': E.dish(0.04),
        'arena_200mm': E.rect(0.2),
        'arena_500mm': E.rect(0.5),
        'arena_1000mm': E.rect(1.0),
        'odor_gradient': E.odor_gradient((0.1, 0.06), pos=(0.04, 0.0),o='G', c=2),
        'mid_odor_gaussian': E.odor_gradient((0.1, 0.06), o='G'),
        'odor_gaussian_square': E.odor_gradient(0.2, o='G'),
        'mid_odor_diffusion': E.odor_gradient(0.3, r=0.03, o='D'),
        '4corners': E.foodNodor_4corners(o='G'),
        'food_at_bottom': E.rect(0.2,  f=FC.sg(id='FoodLine', odor=Odor.oO(o='G'), a=0.002,
                                                r=0.001, N=20, sh='oval',
                                                s=(0.01, 0.0), mode='periphery'), o='G'),
        'thermo_arena': E.rect(0.3, th={}),
        'windy_arena': E.rect(0.3, w={'wind_speed': 10.0}),
        'windy_blob_arena': E.rect((0.5, 0.2),
                                f= FC.sgs(Ngs=4, qs=np.ones(4), cs=aux.N_colors(4), N=1, s=(0.04, 0.02),
                                       loc=(0.005, 0.0),
                                       mode='uniform', sh='rect', can_be_displaced=True, regeneration=True, o='D',
                                       regeneration_pos={'loc': (0.005, 0.0), 'scale': (0.0, 0.0)}),
                                w={'wind_speed': 100.0}, o='D'),
        'windy_arena_bordered':  E.rect(0.3, w={'wind_speed': 10.0},
                                    bl={'Border': gen.Border(vertices=[(-0.03, -0.01), (-0.03, -0.06)], width=0.005)}),
        'puff_arena_bordered': E.rect(0.3, w={
            'puffs': {'PuffGroup': {'N': 100, 'duration': 300.0, 'start_time': 0, 'speed': 100}}},
                                   bl={'Border': gen.Border(vertices=[(-0.03, -0.01), (-0.03, -0.06)], width=0.005)}),
        'single_puff':E.rect(0.3,
                           w={'puffs': {'Puff': {'N': 1, 'duration': 30.0, 'start_time': 55, 'speed': 100}}}),

        'CS_UCS_on_food': E.CS_UCS(grid=gen.FoodGrid(), o='G'),
        'CS_UCS_on_food_x2': E.CS_UCS(grid=gen.FoodGrid(), N=2, o='G'),
        'CS_UCS_off_food': E.CS_UCS(o='G'),

        'patchy_food':E.rect(0.2, f= FC.sg(N=8, s=(0.07, 0.07), mode='periphery', a=0.001, odor=Odor.oO(o='G', c=2)), o='G'),
        'random_food': E.rect(0.1,  f=FC.sgs(Ngs=4, N=1, s=(0.04, 0.04), mode='uniform', sh='rect')),
        'uniform_food': E.dish(0.05,  f=FC.sg(N=2000, s=(0.025, 0.025), a=0.01, r=0.0001)),
        'patch_grid': E.rect(0.2, f= FC.sg(N=5 * 5, s=(0.2, 0.2), a=0.01, r=0.007, mode='grid', sh='rect',
                                           odor=Odor.oO(o='G', c=0.2)),o= 'G'),

        'food_grid': E.rect(0.02, f= FC(food_grid=gen.FoodGrid())),
        'single_odor_patch':  E.rect(0.1, f= FC.patch(odor=Odor.oO(o='G')),o= 'G'),
        'single_patch': E.rect(0.05,  f=FC.patch()),
        'multi_patch': E.rect(0.02,  f=FC.sg(N=8, s=(0.007, 0.007), mode='periphery', a=0.1, r=0.0015)),
        'double_patch': E.double_patch(dim=0.24,  o='G'),

        'maze': E.maze(),
        'game': E.game(),
        'arena_50mm_diffusion': E.dish(0.05, o='D'),
    }
    return aux.AttrDict({k:v.nestedConf for k,v in d.items()})
