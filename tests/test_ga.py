from larvaworld.lib import reg, sim, aux
reg.VERBOSE=1

## FIXME this currently fails
'''
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

