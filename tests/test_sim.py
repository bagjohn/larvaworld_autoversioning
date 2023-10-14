from larvaworld.lib import reg, sim, aux
from larvaworld.lib.process.dataset import LarvaDataset
reg.VERBOSE=1


def test_exp_run():
    ids = reg.conf.Exp.confIDs
    for id in ids:
        r=sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)





'''
def test_evaluation() :
    # refID = 'exploration.merged_dishes'
    # mIDs = ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF']
    parameters = reg.par.get_null('Eval', **{
        'refID': 'exploration.merged_dishes',
        'modelIDs': ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF'],
        # 'dataset_ids': dIDs,
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
