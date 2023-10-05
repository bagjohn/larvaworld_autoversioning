import larvaworld
import pandas as pd
from larvaworld.lib import reg


def test_import_Schleyer():
    g = reg.conf.LabFormat.get('Schleyer')
    kws0 = {
        # 'labID': 'Schleyer',
        'group_id': 'exploration',
    }

    # Merged case
    N = 40
    kws1 = {
        'parent_dir': 'exploration',
        'merged': True,
        'N': N,
        'min_duration_in_sec': 60,
        'refID': f'exploration.{N}controls',
        **kws0
    }

    # Single dish case
    folder = 'dish01'
    kws2 = {
        'parent_dir': f'exploration/{folder}',
        'merged': False,
        'N': None,
        'min_duration_in_sec': 90,
        'id': folder,
        'refID': f'exploration.{folder}',
        **kws0
    }

    for kws in [kws1, kws2]:
        d = g.import_dataset(**kws)
        assert isinstance(d, larvaworld.lib.LarvaDataset)
        d.preprocess(**g.preprocess.nestedConf)
        d.comp_spatial()
        d.comp_angular()
        d.save()
        s = d.step_data
        assert isinstance(s, pd.DataFrame)


def xx_test_import_Jovanic():
    g = reg.conf.LabFormat.get('Jovanic')

    kws0 = {
        #   'labID': 'Jovanic',
        'merged': False
    }

    kws1 = {
        'parent_dir': 'ProteinDeprivation',
        'source_ids': ['Fed', 'Pd'],
        'colors': ['green', 'red'],
        **kws0
    }

    for kws in [kws1]:
        ds = g.import_datasets(**kws)
        for d in ds:
            assert isinstance(d, larvaworld.lib.LarvaDataset)
