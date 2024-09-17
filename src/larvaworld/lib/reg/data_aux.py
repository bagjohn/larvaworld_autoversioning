"""
Methods for fitting and sampling between simulated and experimental datasets
"""

import random
import numpy as np
from types import FunctionType
import typing
import param
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from .. import reg, aux
from ..aux import nam

__all__ = [
    'SAMPLING_PARS',
    'sample_ps',
    'init2mdict',
    # 'LarvaworldParam',
    'get_LarvaworldParam',
    'prepare_LarvaworldParam',
    'build_LarvaworldParam',
]


def param_to_arg(k, p):
    c = p.__class__
    v = p.default
    d = aux.AttrDict({
        'key': k,
        'short': k,
        'help': p.doc,
    })
    if v is not None:
        d.default = v
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


def gConf(mdict, **kwargs):
    if mdict is None:
        return None
    elif isinstance(mdict, param.Parameterized):
        return mdict.v
    elif isinstance(mdict, dict):
        conf = aux.AttrDict()
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                conf[d] = p.v
            else:
                conf[d] = gConf(mdict=p)
            conf.update_existingdict(kwargs)
        return conf
    else:
        return aux.AttrDict(mdict)


def init2mdict(d0):
    def check(D0):
        D = {}
        for kk, vv in D0.items():
            if not isinstance(vv, dict):
                pass
            elif 'dtype' in vv and vv['dtype'] == dict:
                mdict = check(vv)
                vv0 = {kkk: vvv for kkk, vvv in vv.items() if kkk not in mdict}
                if 'v0' not in vv0:
                    vv0['v0'] = gConf(mdict)
                D[kk] = build_LarvaworldParam(p=kk, mdict=mdict, **vv0)

            elif any([a in vv for a in ['symbol', 'h', 'label', 'disp', 'k']]):
                D[kk] = build_LarvaworldParam(p=kk, **vv)
            else:
                D[kk] = check(vv)
        return D

    return aux.AttrDict(check(d0))


class LarvaworldParam(param.Parameterized):
    p = param.String(default='', doc='Name of the parameter')
    d = param.String(default='', doc='Dataset name of the parameter')
    disp = param.String(default='', doc='Displayed name of the parameter')
    k = param.String(default='', doc='Key of the parameter')
    sym = param.String(default='', doc='Symbol of the parameter')
    codename = param.String(default='', doc='Name of the parameter in code')
    flatname = param.String(default=None, doc='Name of the parameter in model configuration')
    dtype = param.Parameter(default=float, doc='Data type of the parameter value')
    func = param.Callable(default=None, doc='Function to get the parameter from a dataset', allow_None=True)
    required_ks = param.List(default=[], doc='Keys of prerequired parameters for computation in a dataset')

    @property
    def s(self):
        return self.disp

    @property
    def l(self):
        return self.disp + '  ' + self.ulabel

    @property
    def symunit(self):
        return self.sym + '  ' + self.ulabel

    @property
    def ulabel(self):
        if self.u == reg.units.dimensionless:
            return ''
        else:
            return '(' + self.unit + ')'

    @property
    def unit(self):
        if self.u == reg.units.dimensionless:
            return '-'
        else:
            return fr'${self.u}$'



    @property
    def short(self):
        return self.k


    @property
    def v0(self):
        return self.param.v.default

    @property
    def initial_value(self):
        return self.param.v.default

    @property
    def value(self):
        return self.v

    @property
    def symbol(self):
        return self.sym

    @property
    def label(self):
        return self.param.v.label

    @property
    def parameter(self):
        return self.disp

    # @property
    # def lab(self):
    #     return self.l

    @property
    def tooltip(self):
        return self.param.v.doc

    @property
    def description(self):
        return self.param.v.doc

    @property
    def help(self):
        return self.param.v.doc

    @property
    def parclass(self):
        return type(self.param.v)

    @property
    def min(self):
        try:
            vmin, vmax = self.param.v.bounds
            return vmin
        except:
            return None

    @property
    def max(self):
        try:
            vmin, vmax = self.param.v.bounds
            return vmax
        except:
            return None

    @property
    def lim(self):
        try:
            lim = self.param.v.bounds
            return lim
        except:
            return None

    @property
    def step(self):
        if self.parclass in [param.Number, param.Range] and self.param.v.step is not None:
            return self.param.v.step
        elif self.parclass == param.Magnitude:
            return 0.01
        elif self.parclass in [param.NumericTuple]:
            return 0.01
        else:
            return None

    @property
    def Ndec(self):
        if self.step is not None:
            return str(self.step)[::-1].find('.')
        else:
            return None

    def exists(self, dataset):
        return aux.AttrDict({'step': self.d in dataset.step_ps, 'end': self.d in dataset.end_ps})

    def get(self, dataset, compute=True):
        res = self.exists(dataset)
        for key, exists in res.items():
            if exists:
                return dataset.get_par(key=key, par=self.d)

        if compute:
            self.compute(dataset)
            return self.get(dataset, compute=False)
        else:
            print(f'Parameter {self.disp} not found')

    def compute(self, dataset):
        if self.func is not None:
            self.func(dataset)
        else:
            print(f'Function to compute parameter {self.disp} is not defined')

    def randomize(self):
        p = self.parclass
        if p in [param.Number] + param.Number.__subclasses__():
            vmin, vmax = self.param.v.bounds
            self.v = self.param.v.crop_to_bounds(np.round(random.uniform(vmin, vmax), self.Ndec))
        elif p in [param.Integer] + param.Integer.__subclasses__():
            vmin, vmax = self.param.v.bounds
            self.v = random.randint(vmin, vmax)
        elif p in [param.Magnitude] + param.Magnitude.__subclasses__():
            self.v = np.round(random.uniform(0.0, 1.0), self.Ndec)
        elif p in [param.Selector] + param.Selector.__subclasses__():
            self.v = random.choice(self.param.v.objects)
        elif p == param.Boolean:
            self.v = random.choice([True, False])
        elif p in [param.Range] + param.Range.__subclasses__():
            vmin, vmax = self.param.v.bounds
            vv0 = np.round(random.uniform(vmin, vmax), self.Ndec)
            vv1 = np.round(random.uniform(vv0, vmax), self.Ndec)
            self.v = (vv0, vv1)

    def mutate(self, Pmut, Cmut):
        if random.random() < Pmut:
            if self.parclass in [param.Magnitude] + param.Magnitude.__subclasses__():
                v0 = self.v if self.v is not None else 0.5
                vv = random.gauss(v0, Cmut)
                self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
                # self.v = np.round(self.v, self.Ndec)
            elif self.parclass in [param.Integer] + param.Integer.__subclasses__():
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0 = self.v if self.v is not None else int(vmin + vr / 2)
                vv = random.gauss(v0, Cmut * vr)
                self.v = self.param.v.crop_to_bounds(int(vv))
            elif self.parclass in [param.Number] + param.Number.__subclasses__():

                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0 = self.v if self.v is not None else vmin + vr / 2
                vv = random.gauss(v0, Cmut * vr)
                self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
            elif self.parclass in [param.Selector] + param.Selector.__subclasses__():
                self.v = random.choice(self.param.v.objects)
            elif self.parclass == param.Boolean:
                self.v = random.choice([True, False])
            elif self.parclass in [param.Range] + param.Range.__subclasses__():
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0, v1 = self.v if self.v is not None else (vmin, vmax)
                vv0 = random.gauss(v0, Cmut * vr)
                vv1 = random.gauss(v1, Cmut * vr)
                vv0 = np.round(np.clip(vv0, a_min=vmin, a_max=vmax), self.Ndec)
                vv1 = np.round(np.clip(vv1, a_min=vv0, a_max=vmax), self.Ndec)
                self.v = (vv0, vv1)


def get_LarvaworldParam(vparfunc, v0=None, dv=None, **kws):
    class _LarvaworldParam(LarvaworldParam):
        v = vparfunc
        u = param.Parameter(default=reg.units.dimensionless, doc='Unit of the parameter values')

    par = _LarvaworldParam(**kws)
    return par


SAMPLING_PARS = aux.bidict(
    aux.AttrDict(
        {
            'length': 'body.length',
            nam.freq(nam.scal(nam.vel(''))): 'brain.crawler.freq',
            # nam.freq(nam.scal(nam.vel(''))): 'brain.intermitter.crawl_freq',
            nam.mean(nam.chunk_track('stride', nam.scal(nam.dst('')))): 'brain.crawler.stride_dst_mean',
            nam.std(nam.chunk_track('stride', nam.scal(nam.dst('')))): 'brain.crawler.stride_dst_std',
            nam.freq('feed'): 'brain.feeder.freq',
            nam.max(nam.chunk_track('stride', nam.scal(nam.vel('')))): 'brain.crawler.max_scaled_vel',
            nam.phi(nam.max(nam.scal(nam.vel('')))): 'brain.crawler.max_vel_phase',
            'attenuation': 'brain.interference.attenuation',
            nam.max('attenuation'): 'brain.interference.attenuation_max',
            nam.freq(nam.vel(nam.orient(('front')))): 'brain.turner.freq',
            nam.phi(nam.max('attenuation')): 'brain.interference.max_attenuation_phase',
        }
    )
)


def sample_ps(ps, e=None):
    Sinv = reg.SAMPLING_PARS.inverse
    ps = aux.SuperList([Sinv[k] for k in aux.existing_cols(Sinv, ps)]).flatten
    if e:
        ps = ps.existing(e)
    return ps


def get_vfunc(dtype, lim, vs):
    func_dic = {
        float: param.Number,
        int: param.Integer,
        str: param.String,
        bool: param.Boolean,
        dict: param.Dict,
        list: param.List,
        type: param.ClassSelector,
        typing.List[int]: param.List,
        typing.List[str]: param.List,
        typing.List[float]: param.List,
        typing.List[typing.Tuple[float]]: param.List,
        FunctionType: param.Callable,
        typing.Tuple[float]: param.Range,
        typing.Tuple[int]: param.NumericTuple,
        TypedDict: param.Dict
    }
    if dtype == float and lim == (0.0, 1.0):
        return param.Magnitude
    if type(vs) == list and dtype in [str, int]:
        return param.Selector
    elif dtype in func_dic.keys():
        return func_dic[dtype]
    else:
        return param.Parameter


def vpar(vfunc, v0, doc, lab, lim, dv, vs):
    f_kws = {
        'default': v0,
        'doc': doc,
        'label': lab,
        'allow_None': True
    }
    if vfunc in [param.List, param.Number, param.Range]:
        if lim is not None:
            f_kws['bounds'] = lim
    if vfunc in [param.Range, param.Number]:
        if dv is not None:
            f_kws['step'] = dv
    if vfunc in [param.Selector]:
        f_kws['objects'] = vs
    func = vfunc(**f_kws, instantiate=True)
    return func


def prepare_LarvaworldParam(p, k=None, dtype=float, d=None, disp=None, sym=None, codename=None, lab=None,
                            doc=None,flatname=None,
                            required_ks=[], u=reg.units.dimensionless, v0=None, v=None, lim=None, dv=None, vs=None,
                            vfunc=None, vparfunc=None, func=None, **kwargs):
    '''
    Method that formats the dictionary of attributes for a parameter in order to create a LarvaworldParam instance
    '''
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    k = k if k is not None else d
    v0 = v if v is not None else v0

    if flatname is None:
        if p in SAMPLING_PARS:
            flatname = SAMPLING_PARS[p]
        else:
            flatname = p

    if sym is None:
        sym = k

    if dv is None:
        if dtype in [float, typing.List[float], typing.List[typing.Tuple[float]], typing.Tuple[float]]:
            dv = 0.01
        elif dtype in [int]:
            dv = 1
        else:
            pass

    if vparfunc is None:
        if vfunc is None:
            vfunc = get_vfunc(dtype=dtype, lim=lim, vs=vs)
        if lab is None:
            if u == reg.units.dimensionless:
                lab = f'{disp}'
            else:
                ulab = fr'${u}$'
                lab = fr'{disp} ({ulab})'
        doc = lab if doc is None else doc
        vparfunc = vpar(vfunc, v0, doc, lab, lim, dv, vs)
    else:
        vparfunc = vparfunc()

    return aux.AttrDict({
        'name': p,
        'p': p,
        'd': d,
        'k': k,
        'disp': disp,
        'sym': sym,
        'codename': codename,
        'flatname': flatname,
        'dtype': dtype,
        'func': func,
        'u': u,
        'required_ks': required_ks,
        'vparfunc': vparfunc,
        'dv': dv,
        'v0': v0,

    })


def build_LarvaworldParam(p, **kwargs):
    pre_p = prepare_LarvaworldParam(p=p, **kwargs)
    return get_LarvaworldParam(**pre_p)
