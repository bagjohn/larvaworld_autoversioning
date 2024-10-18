"""
This module sets up the larvaworld registry where most functions, classes, and configurations are registered.
It is initialized automatically when importing the package and serves as an accessible database for all functionalities.
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
VERBOSE = 2

def vprint(text='', verbose=0):
    """
    Print text if the verbosity level is greater than or equal to the global VERBOSE level.

    Parameters:
    text (str): The text to print.
    verbose (int): The verbosity level of the message.
    """
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
    """
    Class that manages different groups of functions.
    Each group is registered as a dictionary under a certain class attribute.
    The attribute can then be used as a decorator to register a function under the specified group.

    Attributes:
        graphs (aux.AttrDict): A dictionary to store graph-related functions.
        graph_required_data (aux.AttrDict): A dictionary to store required data for graphs.
        stored_confs (aux.AttrDict): A dictionary to store the default configuration-generating functions.
        param_computing (aux.AttrDict): A dictionary to store parameter computing functions.

    Methods:
        param(name):
            Registers a function under the 'param_computing' group.
            Args:
                name (str): The name of the function to register.
            Returns:
                function: A wrapper function that registers the given function.
        
        graph(name, required={}):
            Registers a function under the 'graphs' group and stores its required data.
            Args:
                name (str): The name of the function to register.
                required (dict, optional): A dictionary of required data for the graph. Defaults to an empty dictionary.
            Returns:
                function: A wrapper function that registers the given function.
        
        stored_conf(name):
            Registers a function under the 'stored_confs' group.
            Args:
                name (str): The name of the function to register.
            Returns:
                function: A wrapper function that registers the given function.
        
        register_func(name, group):
            Registers a function under the specified group.
            Args:
                name (str): The name of the function to register.
                group (str): The group under which to register the function.
            Returns:
                function: A wrapper function that registers the given function.
            Raises:
                AttributeError: If the specified group does not exist.
    """
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

graphs = graph.GraphRegistry()

vprint(f"Configuration registry complete", 1)

def getPar(k=None, p=None, d=None, to_return='d'):
    """
    Shortcut function for easy use directly via the registry.
    See 'par.getPar' for more information.
    """
    return par.getPar(k=k, d=d, p=p, to_return=to_return)


def loadRef(id, **kwargs):
    """
    Shortcut function for easy use directly via the registry.
    See 'conf.Ref.loadRef' for more information.
    """
    return conf.Ref.loadRef(id=id, **kwargs)


def loadRefs(ids, **kwargs):
    """
    Shortcut function for easy use directly via the registry.
    See 'conf.Ref.loadRefs' for more information.
    """
    return conf.Ref.loadRefs(ids=ids, **kwargs)


def define_default_refID_by_running_test():
    """
    Defines the default reference dataset ID by running a test if no reference datasets are available.

    This function checks if there are any reference datasets available in `conf.Ref.confIDs`.
    If none are available, it automatically imports a reference dataset by running a specified test method
    from a test file. The test method is executed using the `runpy.run_path` function.

    Returns:
        str: The first reference dataset ID from `conf.Ref.confIDs`.

    Raises:
        AssertionError: If no reference dataset IDs are available after running the test method.
    """
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
    """
    Defines the default reference dataset ID for the package.

    This function performs the following steps:
    1. Purges the existing reference dataset IDs, deleting the ones where the corresponding datasets are missing.
    2. Checks if there are no reference dataset IDs available.
    3. If no reference datasets are available, it imports one from the experimental data folder.
    4. If the respective configuration is not present in LabFormat, it resets the configurations.
    5. Imports the default dataset with specified parameters.
    6. Processes and annotates the dataset.
    7. Ensures that exactly one reference dataset ID is available after import.

    Returns:
        str: The first reference dataset ID from the list of reference dataset IDs.
    """
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
