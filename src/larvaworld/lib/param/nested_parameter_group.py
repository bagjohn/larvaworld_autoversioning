import param

from .. import aux
from .custom import ClassAttr, ClassDict

__all__ = [
    'NestedConf',
    'class_generator',
    'expand_kws_shortcuts',
    'class_defaults',
]


class NestedConf(param.Parameterized):
    """
    A class for managing nested configuration parameters.
    """

    def __init__(self, **kwargs):
        """
        Initializes a NestedConf instance.

        :param kwargs: Keyword arguments for configuring the instance.
        """

        param_classes = self.param.objects()
        for k, p in param_classes.items():
            try:
                if k in kwargs:
                    if type(p) == ClassAttr and not isinstance(kwargs[k], p.class_):
                        kwargs[k] = p.class_(**kwargs[k])
                    elif type(p) == ClassDict and not all(isinstance(vv, p.item_type) for kk, vv in kwargs[k].items()):
                        kwargs[k] = p.class_({kk: p.item_type(**vv) for kk, vv in kwargs[k].items()})
            except:
                pass
        super().__init__(**kwargs)

    @property
    def nestedConf(self):
        """
        Generates a nested configuration dictionary.

        :return: A nested configuration dictionary.
        """
        d = aux.AttrDict(self.param.values())
        d.pop('name')
        for k, p in self.param.objects().items():
            if k in d and p.readonly == True:
                d.pop(k)
            elif k in d and d[k] is not None:
                if type(p) == ClassAttr:
                    d[k] = d[k].nestedConf
                elif type(p) == ClassDict:
                    d[k] = aux.AttrDict({kk: vv.nestedConf for kk, vv in d[k].items()})
        return d

    def entry(self, id=None):
        """
        Creates an entry in the configuration.

        :param id: The identifier for the entry.
        :return: A dictionary containing the configuration entry.
        """
        d = self.nestedConf
        if 'distribution' in d.keys():
            if 'group' in d.keys():
                if id is not None:
                    d.group = id
                elif d.group is not None:
                    id = d.group
            if 'model' in d.keys():
                if id is None:
                    id = d.model
        elif 'unique_id' in d.keys():
            if id is None and d.unique_id is not None:
                id = d.unique_id
                d.pop('unique_id')
        assert id is not None
        return {id: d}

    @property
    def param_keys(self):
        """
        Retrieves a list of parameter keys.

        :return: A list of parameter keys excluding 'name'.
        """
        ks = list(self.param.objects().keys())
        return aux.SuperList([k for k in ks if k not in ['name']])

    def params_missing(self, d):
        """
        Checks for missing parameters in the configuration.

        :param d: The configuration dictionary to compare against.
        :return: A list of missing parameter keys.
        """
        ks = self.param_keys
        return aux.SuperList([k for k in ks if k not in d])




def class_generator(A0, mode='Unit'):
    class A(NestedConf):
        def __init__(self, **kwargs):
            if hasattr(A, 'distribution'):
                D = A.distribution.__class__
                ks = list(D.param.objects().keys())
                kwargs = self.shortcut(kdict={
                    'ors': 'orientation_range',
                    's': 'scale',
                    'sh': 'shape',
                }, kws=kwargs)

                existing = [k for k in ks if k in kwargs]
                if len(existing) > 0:
                    d = {}
                    for k in existing:
                        d[k] = kwargs[k]
                        kwargs.pop(k)
                    kwargs['distribution'] = D(**d)
            kwargs = self.shortcut(kdict={
                'mID': 'model',
                'c': 'color',
                'or': 'orientation',
                'r': 'radius',
                'a': 'amount',
            }, kws=kwargs)

            kwargs = expand_kws_shortcuts(kwargs)

            super().__init__(**kwargs)

        def shortcut(self, kdict, kws):
            for k, key in kdict.items():
                if k in kws:
                    assert key not in kws
                    kws[key] = kws[k]
                    kws.pop(k)
            return kws

        @classmethod
        def from_entries(cls, entries):
            all_confs = []
            for gid, dic in entries.items():
                Ainst = cls(**dic)
                gconf = aux.AttrDict(Ainst.param.values())
                gconf.pop('name')
                if hasattr(Ainst, 'distribution'):
                    ids = [f'{gid}_{i}' for i in range(Ainst.distribution.N)]
                    gconf.pop('distribution')
                    gconf.group = gid
                    try:
                        ps, ors = Ainst.distribution()
                        confs = [{'unique_id': id, 'pos': p, 'orientation': ori, **gconf} for id, p, ori in
                                 zip(ids, ps, ors)]
                    except:
                        ps = Ainst.distribution()
                        confs = [{'unique_id': id, 'pos': p, **gconf} for id, p in zip(ids, ps)]
                    all_confs += confs
                else:
                    gconf.unique_id = gid
                    all_confs.append(gconf)
            return all_confs

        @classmethod
        def agent_class(cls):
            return A0.__name__

        @classmethod
        def mode(cls):
            return mode

    A.__name__ = f'{A0.__name__}{mode}'
    invalid = ['name', 'closed', 'visible', 'selected', 'centered']
    if mode == 'Group':
        from .xy_distro import Larva_Distro, Spatial_Distro
        if not 'pos' in A0.param.objects():
            raise ValueError(f'No Group distribution for class {A0.__name__}. Change mode to Unit')
        distro = Larva_Distro if 'orientation' in A0.param.objects() else Spatial_Distro
        A.param._add_parameter('distribution', ClassAttr(distro, doc='The spatial distribution of the group agents'))
        invalid += ['unique_id', 'pos', 'orientation']
    elif mode == 'Unit':
        pass
    for k, p in A0.param.params().items():
        if k not in invalid:
            A.param._add_parameter(k, p)
    return A


def expand_kws_shortcuts(kwargs):
    if 'life' in kwargs.keys():
        assert 'life_history' not in kwargs.keys()
        assert len(kwargs['life']) == 2
        kwargs['life_history'] = dict(zip(['age', 'epochs'], kwargs['life']))
        kwargs.pop('life')
    if 'o' in kwargs.keys():
        assert 'odor' not in kwargs.keys()
        assert len(kwargs['o']) == 3
        kwargs['odor'] = dict(zip(['id', 'intensity', 'spread'], kwargs['o']))
        kwargs.pop('o')
    if 'sub' in kwargs.keys():
        assert 'substrate' not in kwargs.keys()
        assert len(kwargs['sub']) == 2
        kwargs['substrate'] = dict(zip(['quality', 'type'], kwargs['sub']))
        kwargs.pop('sub')
    return kwargs


def class_defaults(A, excluded=[], **kwargs):
    d= class_generator(A)().nestedConf
    if len(excluded)>0 :
        for exc_A in excluded :
            exc_d=class_generator(exc_A)().nestedConf
            for k in exc_d :
                if k in d :
                    d.pop(k)
    d.update_existingdict(kwargs)
    return d
