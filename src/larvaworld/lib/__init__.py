"""
The core functionalities of the larvaworld platform
"""


from . import aux, param, reg, plot, model, process, screen, sim
# from .aux import *
# from .reg import *
# import aux, param
from .process.dataset import ParamLarvaDataset,BaseLarvaDataset,LarvaDataset, LarvaDatasetCollection

__displayname__ = 'Core library'