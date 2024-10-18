from larvaworld.lib import reg, sim, aux
from larvaworld.lib.process.dataset import LarvaDataset

# NOTE :    This test requires the box2d-py package

def test_box2d_experiments():
    try:
        import Box2D 
        ids = ['realistic_imitation']
        for id in ids:
            r = sim.ExpRun.from_ID(id, store_data=False)
            for d in r.datasets:
                assert isinstance(d, LarvaDataset)                          
    except ImportError:
        print('WARNING : Module box2d-py is not installed. Larvaworld Box2D extension tests aborted')

    










