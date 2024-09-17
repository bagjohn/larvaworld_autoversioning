"""
This module sets up the larvaworld registry where most functions, classes and configurations are registered.
It is initialized automatically when importing the package and serves as an accessible database for all functionalities
"""

import os
from os.path import dirname, abspath
from pint import UnitRegistry
import warnings

warnings.simplefilter(action='ignore')

__all__ = [
    'VERBOSE',
    'vprint',
    'default_refID',
    'default_ref',
    'default_modelID',
    'default_model',
    'ROOT_DIR',
    'DATA_DIR',
    'SIM_DIR',
    'BATCH_DIR',
    'CONF_DIR',
    'SIMTYPES',
    'CONFTYPES',
    'units',
    'funcs',
    'controls',
    'par',
    'graphs',
    'getPar',
    'loadRef',
]

__displayname__ = 'Registry'
VERBOSE = 1


def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)


from .. import aux

vprint(f"Initializing larvaworld registry", 2)
# vprint(f"Initializing larvaworld v.{__version__} registry", 2)

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
DATA_DIR = f'{ROOT_DIR}/data'
SIM_DIR = f'{DATA_DIR}/SimGroup'
BATCH_DIR = f'{SIM_DIR}/batch_runs'
CONF_DIR = f'{ROOT_DIR}/lib/reg/confDicts'
TEST_DIR = f'{ROOT_DIR}/../../tests'

os.makedirs(CONF_DIR, exist_ok=True)

SIMTYPES = ['Exp', 'Batch', 'Ga', 'Eval', 'Replay']
CONFTYPES = ['Env', 'LabFormat', 'Ref', 'Model', 'Trial', 'Exp', 'Batch', 'Ga']
# GROUPTYPES = ['LarvaGroup', 'FoodGroup', 'epoch']


units = UnitRegistry()
units.default_format = "~P"
units.setup_matplotlib(True)


class FunctionDict:
    def __init__(self):
        self.graphs = aux.AttrDict()
        self.graph_required_data = aux.AttrDict()
        self.stored_confs = aux.AttrDict()
        self.param_computing = aux.AttrDict()

    def param(self, name):
        return self.register_func(name, "param_computing")

    def graph(self, name, required={}):
        self.graph_required_data[name] = aux.AttrDict(required)
        return self.register_func(name, "graphs")

    def stored_conf(self, name):
        return self.register_func(name, "stored_confs")

    def register_func(self, name, group):
        if not hasattr(self, group):
            raise
        d = getattr(self, group)

        def wrapper(func):
            d[name] = func
            return func

        return wrapper


funcs = FunctionDict()

from . import keymap
controls = keymap.ControlRegistry()

from .distro import *
from .data_aux import *

vprint(f"Function registry complete", 1)

from . import parDB, parFunc
par = parDB.ParamRegistry()

vprint(f"Parameter registry complete", 1)

from .config import conf
from .generators import gen
from . import config, generators, graph, stored_confs

# model = models.ModelRegistry()
graphs = graph.GraphRegistry()

vprint(f"Configuration registry complete", 1)

def getPar(k=None, p=None, d=None, to_return='d'):
    return par.getPar(k=k, d=d, p=p, to_return=to_return)


def loadRef(id, **kwargs):
    return conf.Ref.loadRef(id=id, **kwargs)


def loadRefs(ids, **kwargs):
    return conf.Ref.loadRefs(ids=ids, **kwargs)


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
    R=conf.Ref
    R.cleanRefIDs()
    if len(R.confIDs) == 0:
        vprint('No reference datasets available.Automatically importing one from the experimental data folder.', 2)
        if 'Schleyer' not in conf.LabFormat.confIDs:
            config.resetConfs(conftypes=['LabFormat'])
        g = conf.LabFormat.get('Schleyer')
        N = 30
        kws = {
            'parent_dir': 'exploration',
            'merged': True,
            'color': 'blue',
            'max_Nagents': N,
            'min_duration_in_sec': 60,
            'id': f'{N}controls',
            'refID': f'exploration.{N}controls',
        }
        d = g.import_dataset(**kws)
        d.process(is_last=False)
        d.annotate(is_last=True)
        assert len(R.confIDs) == 1
    return R.confIDs[0]


default_refID = define_default_refID()

vprint(f"Registry configured!", 2)


def default_ref():
    return loadRef(default_refID, load=True)


default_modelID = 'explorer'


def default_model():
    return conf.Model.getID(default_modelID)
