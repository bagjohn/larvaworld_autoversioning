import copy
from typing import List
from argparse import ArgumentParser
import param
from ..lib import reg, aux, sim, screen
from ..lib.param import SimOps, RuntimeOps

# from ..lib.param.custom import ClassAttr, ClassDict

__all__ = [
    'SingleParserArgument',
    'ParserArgumentDict',
    'SimModeParser',
    # 'update_larva_groups',
]

__displayname__ = 'CLI argument parsing classes'


class SingleParserArgument:
    """
    Create a single parser argument.

    This class is used to populate a parser with arguments and get their values.

    Parameters
    ----------
    short : str
        The short argument name.
    key : str
        The argument key.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    key : str
        The argument key.
    args : list of str
        A list containing the short and long argument names.
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, short, key, **kwargs):
        self.key = key
        self.args = [f'-{short}', f'--{key}']
        self.kwargs = kwargs

    def add(self, p):
        """
        Add the argument to a parser.

        Parameters
        ----------
        p : argparse.ArgumentParser
            The argument parser.

        Returns
        -------
        argparse.ArgumentParser
            The modified parser.
        """
        p.add_argument(*self.args, **self.kwargs)
        return p

    def get(self, input):
        """
        Get the value of the argument from parsed input.

        Parameters
        ----------
        input : argparse.Namespace
            The parsed input.

        Returns
        -------
        Any
            The value of the argument.
        """
        return getattr(input, self.key)

    @classmethod
    def from_dict(cls, name, **kwargs):
        """
        Create a SingleParserArgument from a dictionary.

        Parameters
        ----------
        name : str
            The argument name.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        SingleParserArgument
            A SingleParserArgument instance.
        """
        return cls(**parser_entry_from_dict(name, **kwargs))

    @classmethod
    def from_param(cls, k, p):
        """
        Create a SingleParserArgument from a parameter.

        Parameters
        ----------
        k : str
            The parameter key.
        p : param.Parameter
            The parameter instance.

        Returns
        -------
        SingleParserArgument
            A SingleParserArgument instance.
        """
        return cls(**parser_entry_from_param(k, p))


def parser_entry_from_param(k, p):
    """
    Create a dictionary entry for a parser argument from a parameter.

    Parameters
    ----------
    k : str
        The parameter key.
    p : param.Parameter
        The parameter instance.

    Returns
    -------
    dict
        A dictionary entry for the parser argument.
    """
    c = p.__class__
    v = p.default
    d = aux.AttrDict({
        'key': k,
        'short': k,
        'help': p.doc,
        'default': p.default,
    })
    # if v is not None:
    #     d.default = v
    if c == param.Boolean:
        d.action = 'store_true' if not v else 'store_false'
    elif c == param.String:
        d.type = str
    elif c in param.Integer.__subclasses__():
        d.type = int
    elif c in param.Number.__subclasses__():
        d.type = float
    elif c in param.Tuple.__subclasses__():
        d.type = tuple

    if hasattr(p, 'objects'):
        d.choices = p.objects
        if c in param.List.__subclasses__():
            d.nargs = '+'
        if hasattr(p, 'item_type'):
            d.type = p.item_type
    return d


def parser_entry_from_dict(name, k=None, h='', dtype=float, v=None, vs=None, **kwargs):
    """
    Create a dictionary entry for a parser argument from a dictionary.

    Parameters
    ----------
    name : str
        The argument name.
    k : str, optional
        The argument key. If not provided, it defaults to the name.
    h : str, optional
        The help text.
    dtype : type, optional
        The data type.
    v : Any, optional
        The default value.
    vs : list of Any, optional
        List of choices.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    dict
        A dictionary entry for the parser argument.
    """
    if k is None:
        k = name
    d = aux.AttrDict({
        'key': name,
        'short': k,
        'help': h,
    })
    if dtype == bool:
        d.action = 'store_true' if not v else 'store_false'
    elif dtype == List[str]:
        d.type = str
        d.nargs = '+'
        if vs is not None:
            d.choices = vs
    elif dtype == List[int]:
        d.type = int
        d.nargs = '+'
        if vs is not None:
            d.choices = vs
    else:
        d.type = dtype
        if vs is not None:
            d.choices = vs
        if v is not None:
            d.default = v
            d.nargs = '?'
    return d


class ParserArgumentDict:
    """
    Create a dictionary of parser arguments.

    This class can be instantiated either by a dictionary of param.Parameters or by a dictionary existing in the registry parameter Database.
    """

    def __init__(self, parsargs):
        """
        Initialize a ParserArgumentDict.

        Parameters
        ----------
        parsargs : dict
            A dictionary of parser arguments.
        """
        self.parsargs = parsargs

    @classmethod
    def from_param(cls, d0):
        """
        Create a ParserArgumentDict from a parameter dictionary.

        Parameters
        ----------
        d0 : dict
            A dictionary of parameters.

        Returns
        -------
        ParserArgumentDict
            A ParserArgumentDict instance.
        """
        return cls(parser_dict_from_param(d0))

    @classmethod
    def from_dict(cls, d0):
        """
        Create a ParserArgumentDict from a dictionary.

        Parameters
        ----------
        d0 : dict
            A dictionary of parser arguments.

        Returns
        -------
        ParserArgumentDict
            A ParserArgumentDict instance.
        """
        return cls(parser_dict_from_dict(d0))

    def add(self, parser=None):
        """
        Add parser arguments to an ArgumentParser.

        Parameters
        ----------
        parser : argparse.ArgumentParser, optional
            The ArgumentParser to add the arguments to.

        Returns
        -------
        argparse.ArgumentParser
            The modified ArgumentParser.
        """
        if parser is None:
            parser = ArgumentParser()
        for k, v in self.parsargs.items():
            parser = v.add(parser)
        return parser

    def get(self, input):
        """
        Get parser argument values from parsed input.

        Parameters
        ----------
        input : argparse.Namespace
            The parsed input.

        Returns
        -------
        dict
            A dictionary of argument values.
        """
        dic = aux.AttrDict({k: v.get(input) for k, v in self.parsargs.items()})
        return dic.unflatten()


def parser_dict_from_param(d0):
    """
    Create a dictionary of parser arguments from a parameter dictionary.

    Parameters
    ----------
    d0 : dict
        A dictionary of parameters.

    Returns
    -------
    dict
        A dictionary of parser arguments.
    """

    # dv0 = aux.AttrDict(d0.param.values())

    d = aux.AttrDict()
    for k, p in d0.param.objects().items():
        # print(k)
        if k == 'name' or p.readonly:
            continue
        elif p.__class__ in param.Parameterized.__subclasses__():
            d[k] = parser_dict_from_param(p)
        # elif k in dv0 and dv0[k] is not None:
        # elif type(p) == ClassAttr:
        #     d[k] = parser_dict_from_param(p.class_)
        # elif type(p) == ClassDict:
        #     d[k] = parser_dict_from_param(p.item_type)
        else:
            d[k] = SingleParserArgument.from_param(k, p)
    return d.flatten()


def parser_dict_from_dict(d0):
    """
    Create a dictionary of parser arguments from a dictionary.

    Parameters
    ----------
    d0 : dict
        A dictionary of parser arguments.

    Returns
    -------
    dict
        A dictionary of parser arguments.
    """
    p = aux.AttrDict()
    for n, v in d0.items():
        if 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
            p[n] = SingleParserArgument.from_dict(n, **v)
        else:
            p[n] = parser_dict_from_dict(v)
    return p.flatten()


class SimModeParser:
    """
    Parser for simulation modes and arguments.
    """

    def __init__(self):
        """
        Initialize the SimModeParser.
        """
        self.dict = aux.AttrDict({
            'Batch': [],
            'Eval': ['Eval'],
            'Exp': [],
            'Ga': ['ga_select_kws', 'ga_eval_kws', 'reference_dataset'],
            'Replay': ['Replay']
        })

        """
        Initialize parsers for different simulation modes.
        """
        #ks = aux.SuperList(self.dict.values()).flatten.unique
        #self.parsers = aux.AttrDict({k: ParserArgumentDict.from_dict(reg.par.PI[k]) for k in ks})
        self.parsers = aux.AttrDict()
        self.parsers.screen_kws = ParserArgumentDict.from_param(d0=screen.ScreenOps)
        self.parsers.SimOps = ParserArgumentDict.from_param(d0=SimOps)
        self.parsers.RuntimeOps = ParserArgumentDict.from_param(d0=RuntimeOps)

        # TODO set the parsers per mode
        self.parsers.Replay = ParserArgumentDict.from_param(d0=reg.gen.Replay)
        self.parsers.Eval = ParserArgumentDict.from_param(d0=reg.gen.Eval)
        self.parsers.GAselector = ParserArgumentDict.from_param(d0=reg.gen.GAselector)
        self.parsers.GAevaluation = ParserArgumentDict.from_param(d0=reg.gen.GAevaluation)

        self.cli_parser = self.build_cli_parser()
        self.mode = None
        self.run = None
        self.args = aux.AttrDict()
        self.run_kws = aux.AttrDict()

    def parse_args(self):
        """
        Parse command line arguments.
        """
        self.args = aux.AttrDict(vars(self.cli_parser.parse_args()))

    def populate_mode_subparser(self, sp, m):
        """
        Populate a subparser with arguments for a specific simulation mode.

        :param sp: The subparser.
        :param m: The simulation mode.
        :return: The modified subparser.
        """
        sp = self.parsers.screen_kws.add(sp)
        sp = self.parsers.RuntimeOps.add(sp)
        if m not in ['Replay', 'Eval']:
            sp = self.parsers.SimOps.add(sp)
        # for k in self.dict[m]:
        #     sp = self.parsers[k].add(sp)
        return sp

    def init_mode_subparser(self, sp, m):
        """
        Initialize a subparser with common arguments for a specific simulation mode.

        :param sp: The subparser.
        :param m: The simulation mode.
        :return: The modified subparser.
        """
        # sp.add_argument('-id', '--id', type=str, help='The simulation ID. If not provided a default is generated')
        if m in ['Exp', 'Batch', 'Ga']:
            sp.add_argument('experiment', choices=reg.conf[m].confIDs, help='The experiment mode')
        if m in ['Exp', 'Batch']:
            sp.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae in each larva group')
            sp.add_argument('-mIDs', '--modelIDs', type=str, nargs='+',
                            help='The larva models to use for creating the simulation larva groups')
            sp.add_argument('-gIDs', '--groupIDs', type=str, nargs='+',
                            help='The displayed IDs of the simulation larva groups')
        if m in ['Exp', 'Batch', 'Eval']:
            sp.add_argument('-a', '--analysis', action='store_true', default=False,
                            help='Whether to run data-analysis after the simulation')
            sp.add_argument('-show', '--show', action='store_true', default=False,
                            help='Whether to show the plots generated during data-analysis')
        return sp

    def build_cli_parser(self):
        """
        Build the command line argument parser.
        """
        p = ArgumentParser()
        p.add_argument('-parsargs', '--show_parser_args', action='store_true', default=False,
                       help='Whether to show the parser argument namespace')
        subparsers = p.add_subparsers(dest='sim_mode', help='The simulation mode to launch')
        for m in reg.SIMTYPES:
            sp = subparsers.add_parser(m)
            sp = self.init_mode_subparser(sp, m)
            sp = self.populate_mode_subparser(sp, m)
        return p

    def eval_parser(self, p_key):
        """
        Evaluate a parser argument.

        :param p_key: The argument key.
        :return: The parsed value of the argument.
        """
        return self.parsers[p_key].get(self.args)

    def configure(self, show_args=False):
        """
        Configure the simulation run based on parsed arguments.

        :param show_args: Whether to show parsed arguments.
        :return: The configured simulation run.
        """
        a = self.args
        self.mode = m = a.sim_mode
        kw = aux.AttrDict({'screen_kws': self.eval_parser('screen_kws'), **self.eval_parser('RuntimeOps')})
        if m not in ['Replay', 'Eval']:
            kw.update(**self.eval_parser('SimOps'))
            kw.experiment = a.experiment
        if m == 'Batch':
            kw.mode = 'batch'
            kw.run_externally = False
            kw.experiment = a.experiment
            kw.conf = reg.conf.Batch.getID(a.experiment)
            kw.conf.N = a.Nagents
            kw.conf.modelIDs = a.modelIDs
            kw.conf.groupIDs = a.groupIDs
            self.run = sim.Exec(**kw)
        elif m == 'Exp':
            kw.N = a.Nagents
            kw.modelIDs = a.modelIDs
            kw.groupIDs = a.modelIDs
            self.run = sim.ExpRun(**kw)
        elif m == 'Ga':
            p = reg.conf.Ga.expand(kw.experiment)
            ev = self.eval_parser('GAevaluation')
            sel = self.eval_parser('GAselector')
            #ref=self.eval_parser('reference_dataset')

            for k in ['base_model', 'bestConfID', 'space_mkeys']:
                if sel[k] is None:
                    sel.pop(k)
            p.ga_select_kws.update(**sel)

            if ev.refID is not None:
                p.refID = ev.refID
            kw.parameters = p
            self.run = sim.GAlauncher(**kw)
        elif m == 'Eval':
            self.run = sim.EvalRun(**self.eval_parser('Eval'))
        elif m == 'Replay':
            kw.parameters = self.eval_parser('Replay')
            self.run = sim.ReplayRun(**kw)
        self.run_kws = kw
        if show_args:
            self.show_args()
        return self.run

    def show_args(self, nested=True, flat_nested=False, input=False, run_args=True):
        """
        Show parsed arguments.

        :param nested: Whether to show arguments as a nested dictionary.
        :param flat_nested: Whether to show arguments as a flattened nested dictionary.
        :param input: Whether to show input arguments.
        :param default: Whether to show default arguments.
        :param run_args: Whether to show run arguments.
        """
        print(f'Simulation mode : {self.mode}')
        if nested:
            print(f'Simulation args as nested dictionary: ')
            self.run_kws.print(flat=False)
        if flat_nested:
            print(f'Simulation args as flattened nested dictionary: ')
            self.run_kws.print(flat=True)
        if input:
            print(f'Input args : ')
            self.args.print(flat=True)
        if run_args:
            if hasattr(self.run, 'configuration_text'):
                print(self.run.configuration_text)

    def launch(self):
        """
        Launch the simulation run.
        """
        anal_kws = aux.AttrDict({'show': self.args.show})
        m = self.mode
        r = self.run
        if m == 'Batch':
            r.run()
        elif m == 'Exp':
            _ = r.simulate()
            if self.args.analysis:
                r.analyze(show=self.args.show)
        elif m == 'Ga':
            _ = r.simulate()
        elif m == 'Eval':
            _ = r.simulate()
            if self.args.analysis:
                r.plot_results(show=self.args.show)
                r.plot_models(show=self.args.show)
        elif m == 'Replay':
            r.run()
