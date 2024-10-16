"""
Larvaworld : A Drosophila larva behavioral analysis and simulation platform
"""

from . import lib, cli, gui
lib.reg.config.resetConfs(init=True)

__author__ = 'Panagiotis Sakagiannis'
__license__ = 'GNU GENERAL PUBLIC LICENSE'
__copyright__ = '2024, Panagiotis Sakagiannis'

# TODO : the automatic version naming requires the package itself. Woraround by simply naming it 0.1
# import importlib.metadata
# __version__ = importlib.metadata.version("larvaworld")

__version__ = '0.1'
__displayname__ = 'larvaworld'
__name__ = 'larvaworld'


