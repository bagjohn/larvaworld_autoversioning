"""
This module sets up the larvaworld registry where most functions, classes and configurations are registered.
It is initialized automatically when importing the package and serves as an accessible database for all functionalities
"""

import os
from os.path import dirname, abspath
import warnings

warnings.simplefilter(action='ignore')

__all__ = [
    'VERBOSE',
    'vprint',
    'default_refID',
    'ROOT_DIR',
    'DATA_DIR',
    'SIM_DIR',
    'BATCH_DIR',
    'CONF_DIR',
    'SIMTYPES',
    'CONFTYPES',
    'GROUPTYPES',
    'units',
    'funcs',
    'controls',
    'distro_database',
    'par',
    'stored',
    'model',
    'graphs',
    'getPar',
    'get_null',
    'loadRef',
    'getRef',
    'loadRefGroup',
]

__displayname__ = 'Registry'

VERBOSE = 2


def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)


vprint("Initializing larvaworld registry", 2)

# default_refID = 'exploration.40controls'
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
DATA_DIR = f'{ROOT_DIR}/data'
SIM_DIR = f'{DATA_DIR}/SimGroup'
BATCH_DIR = f'{SIM_DIR}/batch_runs'
CONF_DIR = f'{ROOT_DIR}/lib/reg/confDicts'
TEST_DIR = f'{ROOT_DIR}/../../tests'

os.makedirs(CONF_DIR, exist_ok=True)

SIMTYPES = ['Exp', 'Batch', 'Ga', 'Eval', 'Replay']
CONFTYPES = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Batch', 'Ga', 'LabFormat', 'Trial', 'Life',
             'Tree', 'Food']
GROUPTYPES = ['LarvaGroup', 'FoodGroup', 'epoch']

vprint("Initializing function registry")

from pint import UnitRegistry

units = UnitRegistry()
units.default_format = "~P"
units.setup_matplotlib(True)

from . import facade, keymap, distro

funcs = facade.FunctionDict()
controls = keymap.ControlRegistry()
distro_database = distro.generate_distro_database()

vprint("Initializing parameter registry")
from . import parDB, parFunc, stored_confs

par = parDB.ParamRegistry()

vprint("Initializing configuration registry")
from .config import Path, StoredConfRegistry

stored = StoredConfRegistry()
from .generators import gen, conf, resetConfs

from . import config, generators, models, graph

model = models.ModelRegistry()
graphs = graph.GraphRegistry()


def getPar(k=None, p=None, d=None, to_return='d'):
    return par.getPar(k=k, d=d, p=p, to_return=to_return)


def get_null(name, **kwargs):
    return par.get_null(name=name, **kwargs)


def loadRef(id, **kwargs):
    return conf.Ref.loadRef(id=id, **kwargs)


def getRef(id, **kwargs):
    return conf.Ref.getRef(id=id, **kwargs)


def loadRefGroup(group_id, **kwargs):
    return conf.Ref.loadRefGroup(group_id=group_id, **kwargs)


vprint(f"Registry configured!", 2)


def define_default_refID_by_running_test():
    if len(conf.Ref.confIDs) == 0:
        filename = 'test_import.py'
        filepath = f'{TEST_DIR}/{filename}'
        import_method = 'test_import_Schleyer'
        vprint('No reference datasets are available.', 2)
        vprint(f'Automatically importing one by running the {import_method} method in {filename} file.', 2)
        import runpy
        runpy.run_path(filepath, run_name='__main__')[import_method]()
        assert len(conf.Ref.confIDs) > 0
    return conf.Ref.confIDs[0]

def define_default_refID():
    if len(conf.Ref.confIDs) == 0:

        vprint('No reference datasets are available.', 2)
        vprint(f'Automatically importing one from the raw experimental data folder.', 2)

        g = conf.LabFormat.get('Schleyer')
        # Merged case
        N = 30
        kws = {
            'group_id': 'exploration',
            'parent_dir': 'exploration',
            'merged': True,
            'color': 'blue',
            'N': N,
            'min_duration_in_sec': 60,
            'id': f'{N}controls',
            'refID': f'exploration.{N}controls',
        }
        d = g.import_dataset(**kws)
        d.process(is_last=True)
        assert len(conf.Ref.confIDs) ==1
    return conf.Ref.confIDs[0]

default_refID = define_default_refID()
