"""
Larvaworld : A Drosophila larva behavioral analysis and simulation platform
"""

from . import lib, cli, gui
# print('now')
lib.reg.config.resetConfs(init=True)

__author__ = 'Panagiotis Sakagiannis'
__license__ = 'GNU GENERAL PUBLIC LICENSE'
__copyright__ = '2023, Panagiotis Sakagiannis'
# __version__ = '0.0.150'
__displayname__ = 'larvaworld'
__name__ = 'larvaworld'


import importlib.metadata

__version__ = importlib.metadata.version("larvaworld")
# print('now')