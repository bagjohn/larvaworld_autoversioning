"""
Collection of standalone methods, custom classes and other tools, all of them independent of the larvaworld registry
"""

from .ang import *
from .color import *
from .dictsNlists import *
from .nan_interpolation import *
from .stdout import suppress_stdout_stderr, suppress_stdout, remove_prefix, remove_suffix, rgetattr, rsetattr, storeH5
from .time_util import TimeUtil

from .shapely_aux import *
from .combining import combine_pdfs

from .naming import *

nam = NamingRegistry()



from .xy import *

__displayname__ = 'Auxilliary methods'
