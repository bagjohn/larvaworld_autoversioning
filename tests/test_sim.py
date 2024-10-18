from larvaworld.lib import reg, sim, aux
from larvaworld.lib.process.dataset import LarvaDataset
#reg.VERBOSE=1


def test_exp_run():
    #ids = reg.conf.Exp.confIDs
    ids=['tethered',
        'dish',
        'dispersion_x2',
        
        'chemorbit_x4',
        'chemotaxis_diffusion',
        'single_odor_patch_x4',
        
        'tactile_detection',
        'anemotaxis',
        'single_puff',
        'thermotaxis',
        'PItest_off',
        'PItrain',

        'prey_detection'
        ]

    for id in ids:
        r = sim.ExpRun.from_ID(id, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)

def test_games():
    ids=['capture_the_flag',
        'catch_me',
        'keep_the_flag',
        'maze']

    for id in ids:
        r = sim.ExpRun.from_ID(id, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)

def test_foraging_experiments():
    ids=['4corners',
        'double_patch',
        'food_grid',
        'random_food',
        'patch_grid']

    for id in ids:
        r = sim.ExpRun.from_ID(id, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)

def test_growth_experiments():
    ids=['RvsS_on',
        'growth']

    for id in ids:
        r = sim.ExpRun.from_ID(id, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


'''
def test_evaluation() :
    # refID = 'exploration.merged_dishes'
    # mIDs = ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF']
    parameters = reg.par.get_null('Eval', **{
        'refID': 'exploration.merged_dishes',
        'modelIDs': ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF'],
        # 'groupIDs': dIDs,
        'N': 3,
        # 'offline': False,
    })
    evrun = sim.EvalRun(parameters=parameters, id=id,show=False)

    # evrun = sim.EvalRun(refID=refID, modelIDs=mIDs, N=3, show=False)
    evrun.simulate()
    evrun.plot_results()
    evrun.plot_models()








def xtest_batch_run() :
    for exp in ['PItest_off'] :
        conf=reg.conf.Batch.expand(exp)
        batch_run = sim.BatchRun(id=f'test_{exp}',batch_type=exp,**conf)
        batch_run.simulate()
        
        
'''
