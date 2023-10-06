from larvaworld.lib import reg, sim, aux
from larvaworld.lib.process.dataset import LarvaDataset
reg.VERBOSE=1

def xx_test_replay():
    refIDs = reg.conf.Ref.confIDs
    refID = refIDs[-1]
    d = reg.loadRef(refID)
    replay_kws = {
        'normal': {
            'time_range': (10, 80)
        },
        'dispersal': {
            'transposition': 'origin',
            'time_range': (60, 120)
        },
        'fixed_point': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'time_range': (80, 100)
        },
        'fixed_segment': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'fix_segment': 'rear',
            'time_range': (100, 130)
        },
        'fixed_overlap': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'fix_segment': 'front',
            'overlap_mode': True
        },
        '2segs': {
            'draw_Nsegs': 2
        },
        'all_segs': {
            'draw_Nsegs': d.config.Npoints - 1
        },
    }

    for mode, kws in replay_kws.items():
        print(mode)
        parameters = reg.gen.Replay(**aux.AttrDict({
            'refID': refID,
            # 'dataset' : dataset,
            **kws
        })).nestedConf
        rep = sim.ReplayRun(parameters=parameters, dataset=d, id=f'{refID}_replay_{mode}', dir=f'./media/{mode}')
        output = rep.run()
        assert output.parameters.constants['id'] == rep.id
        # raise


def test_genetic_algorithm_simulation():
    exp = 'realism'
    ga1 = sim.GAlauncher(experiment=exp)
    ga1.selector.Ngenerations = 5
    best1 = ga1.run()
    print(best1)
    assert best1 is not None

    p = reg.conf.Ga.expand(exp)
    p.ga_select_kws.Ngenerations = 5
    ga2 = sim.GAlauncher(parameters=p, screen_kws={'show_display': True})
    best2 = ga2.run()
    print(best2)
    assert best2 is not None


'''

def test_exp_run():
    for exp in ['chemotaxis']:
        conf=reg.conf.Exp.expand(exp)
        # conf.sim_params.duration=1
        # exp_run = sim.ExpRun(experiment=exp, duration=1)
        exp_run = sim.ExpRun(parameters=conf, duration=1)
        # exp_run.run()
        exp_run.simulate()
        for d in exp_run.datasets:
            assert isinstance(d, LarvaDataset)






def test_evaluation() :
    # refID = 'exploration.merged_dishes'
    # mIDs = ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF']
    parameters = reg.get_null('Eval', **{
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
