from larvaworld.lib import reg, sim, aux
from larvaworld.lib.process.dataset import LarvaDataset
reg.VERBOSE=1

# Currently added try because of this  :
'''
self = RemoteBrianModelMemory(dt=0.1, modality='olfaction', mode='MB', name='RemoteBrianModelMemory04260')
model_instance_id = 'max_forager0_MB_0', odor_id = 0, t_sim = 100, t_warmup = 0
concentration = 0.0, kwargs = {'reward': 0}

    def runRemoteModel(self, model_instance_id, odor_id, t_sim=100, t_warmup=0, concentration=1, **kwargs):
        # T: duration of remote model simulation in ms
        # warmup: duration of remote model warmup in ms
>       msg = Message(self.sim_id, model_instance_id, odor_id=odor_id, odor_concentration=concentration,
                           T=t_sim, warmup=t_warmup, step_id=self.step_id, **kwargs)
E       TypeError: Message() takes no arguments

src/larvaworld/lib/model/modules/memory.py:182: TypeError
'''
def test_exp_run():
    ids = reg.conf.Exp.confIDs
    for id in ids:
        #try:
        r=sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)
        #except :
        #    print(f'Experiment {id} FAILED')



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
