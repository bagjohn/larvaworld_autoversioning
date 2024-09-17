import os
import param

from .. import reg, aux
from ..param import ClassDict, OptionalSelector, ClassAttr

__all__ = [
    'next_idx',
    'ConfType',
    'RefType',
    'conf',
    'resetConfs',
]


def next_idx(id, conftype='Exp'):
    f = f'{reg.CONF_DIR}/SimIdx.txt'
    if not os.path.isfile(f):
        d = aux.AttrDict({k: {} for k in ['Exp', 'Batch', 'Essay', 'Eval', 'Ga']})
    else:
        d = aux.load_dict(f)
    if conftype not in d:
        d[conftype] = aux.AttrDict()
    if id not in d[conftype]:
        d[conftype][id] = 0
    d[conftype][id] += 1
    aux.save_dict(d, f)
    return d[conftype][id]


class ConfType(param.Parameterized):
    """Select among available configuration types"""
    conftype = param.Selector(objects=reg.CONFTYPES, doc='The configuration type')
    dict = ClassDict(default=aux.AttrDict(), item_type=None, doc='The configuration dictionary')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CONFTYPE_SUBKEYS = self.build_ConfTypeSubkeys()
        self.update_dict()

    def build_ConfTypeSubkeys(self):
        d0 = {k: {} for k in reg.CONFTYPES}
        d1 = {
            'Batch': {'exp': 'Exp'},
            'Ga': {'env_params': 'Env'},
            'Exp': {'env_params': 'Env',
                    'trials': 'Trial',
                    'larva_groups': 'Model',
                    }
        }
        d0.update(d1)
        return aux.AttrDict(d0)

    @property
    def path_to_dict(self):
        return f'{reg.CONF_DIR}/{self.conftype}.txt'

    @param.depends('conftype', watch=True)
    def update_dict(self):
        self.param.params('dict').item_type = self.dict_entry_type
        self.load()

    def getID(self, id):
        if isinstance(id,list):
            return [self.getID(i) for i in id]
        if id in self.dict:
            return self.dict[id]
        else:
            reg.vprint(f'{self.conftype} Configuration {id} does not exist', 1)
            raise ValueError()

    def get(self, id):
        if isinstance(id,list):
            return [self.get(i) for i in id]
        entry = self.getID(id)
        return self.conf_class(**entry, name=id)

    def load(self):
        self.dict = aux.load_dict(self.path_to_dict)

    def save(self):
        return aux.save_dict(self.dict, self.path_to_dict)

    def set_dict(self,d):
        self.param.params('dict').item_type = self.dict_entry_type
        self.dict = d
        self.save()

    def reset(self, init=False):
        if os.path.isfile(self.path_to_dict):
            if init:
                reg.vprint(f'{self.conftype} configuration dict exists with {len(self.dict)} entries', 1)
                return
            else:
                d = self.dict
                Ncur = len(d)
                d.update(self.stored_dict)
                self.set_dict(d)
                reg.vprint(f'{self.conftype} configuration dict of {Ncur} entries enriched to {len(self.dict)}', 1)
        else:
            self.set_dict(self.stored_dict)
            reg.vprint(f'{self.conftype} configuration dict initialized with {len(self.dict)} entries', 1)

    def selectIDs(self, dic={}):
        valid=aux.SuperList()
        for id in self.confIDs:
            c=self.getID(id).flatten()
            if all([(k in c and c[k]==v) for k,v in dic.items()]):
                valid.append(id)
        return valid


    def setID(self, id, conf, mode='overwrite'):
        if id in self.dict and mode == 'update':
            self.dict[id] = self.dict[id].update_nestdict(conf.flatten())
        else:
            self.dict[id] = conf
        self.save()
        # self.update_dict()
        # self.load()
        if self.conftype=='Model':
            from ..sim.genetic_algorithm import GAselector
            GAselector.param.objects()['base_model'].objects = self.confIDs
            reg.generators.LarvaGroupMutator.param.objects()['modelIDs'].objects = self.confIDs
            reg.generators.LarvaGroup.param.objects()['model'].objects = self.confIDs
        reg.vprint(f'{self.conftype} Configuration saved under the id : {id}', 1)

    def delete(self, id=None):
        if id is not None:
            if id in self.dict:
                self.dict.pop(id, None)
                self.save()
                reg.vprint(f'Deleted {self.conftype} configuration under the id : {id}', 1)

    def expand(self, id=None, conf=None):
        if conf is None:
            if id in self.dict:
                conf = self.dict[id]
            else:
                return None
        subks = self.CONFTYPE_SUBKEYS[self.conftype]
        if len(subks) > 0:
            for subID, subk in subks.items():
                ids = reg.conf[subk].confIDs
                if subID == 'larva_groups' and subk == 'Model':
                    for k, v in conf['larva_groups'].items():
                        if v.model in ids:
                            v.model = reg.conf[subk].getID(v.model)
                else:
                    if conf[subID] in ids:
                        conf[subID] = reg.conf[subk].getID(conf[subID])

        return conf

    # @param.depends('confIDs','dict', watch=True)
    def confID_selector(self, default=None, single=True):
        kws = {
            'default': default,
            'objects': self.confIDs,
            'label': f'{self.conftype} configuration ID',
            'doc': f'Selection among stored {self.conftype} configurations by ID'

        }
        if single:
            return OptionalSelector(**kws)
        else:
            return param.ListSelector(**kws)

    @property
    def reset_func(self):
        return reg.funcs.stored_confs[self.conftype]

    @property
    def stored_dict(self):
        return self.reset_func()

    @property
    def confIDs(self):
        return sorted(list(self.dict.keys()))

    @property
    def conf_class(self):
        c = self.conftype
        if c is None:
            return None
        elif c in reg.gen:
            return reg.gen[c]
        else:
            return aux.AttrDict

    @property
    def dict_entry_type(self):
        return aux.AttrDict


class RefType(ConfType):
    """Select a reference dataset by ID"""

    def __init__(self, **kwargs):
        super().__init__(conftype='Ref', **kwargs)

    def getRefDir(self, id):
        assert id is not None
        return self.getID(id)

    def getRef(self, id=None, dir=None):
        path = self.path_to_Ref(id=id, dir=dir)
        assert os.path.isfile(path)
        c = aux.load_dict(path)
        assert 'id' in c
        reg.vprint(f'Loaded existing conf {c.id}', 1)
        return c

    def setRef(self, c, id=None, dir=None):
        path = self.path_to_Ref(id=id, dir=dir)
        aux.save_dict(c, path)
        assert 'id' in c
        reg.vprint(f'Saved conf under ID {c.id}', 1)

    def path_to_Ref(self, id=None, dir=None):
        if dir is None:
            dir = self.getRefDir(id)
        return f'{dir}/data/conf.txt'

    def loadRef(self, id=None, dir=None, load=False, **kwargs):
        from ..process.dataset import LarvaDataset
        c = self.getRef(id=id, dir=dir)
        assert c is not None
        d = LarvaDataset(config=c, load_data=False)
        if load:
            d.load(**kwargs)
        reg.vprint(f'Loaded stored reference dataset : {id}', 1)
        return d

    def loadRefs(self, ids=None, dirs=None, **kwargs):
        if ids is None :
            assert dirs is not None
            ids= [None]*len(dirs)
        if dirs is None:
            assert ids is not None
            dirs= [None]*len(ids)
        return aux.ItemList([self.loadRef(id=id, dir=dir, **kwargs) for id, dir in zip(ids,dirs)])

    def retrieve_dataset(self, dataset=None, load=True, **kwargs):
        if dataset is None:
            dataset = self.loadRef(load=load, **kwargs)
        return dataset

    def cleanRefIDs(self):
        ids = self.confIDs
        for id in ids:
            try:
                self.loadRef(id)
            except:
                self.delete(id)

    @property
    def dict_entry_type(self):
        return str

    def getRefGroups(self):
        d = self.Refdict
        gd = aux.AttrDict({c.group_id: c for id, c in d.items()})
        gIDs = aux.unique_list(list(gd.keys()))
        return aux.AttrDict({gID: {c.id: c.dir for id, c in d.items() if c.group_id == gID} for gID in gIDs})

    @property
    def RefGroupIDs(self):
        d = self.Refdict
        gd = aux.AttrDict({c.group_id: c for id, c in d.items()})
        return aux.unique_list(list(gd.keys()))

    @property
    def Refdict(self):
        return aux.AttrDict({id: self.getRef(id) for id in self.confIDs})

    def getRefGroup(self, group_id):
        d = self.getRefGroups()[group_id]
        return aux.AttrDict({id: self.getRef(dir=dir) for id, dir in d.items()})

    def loadRefGroup(self, group_id, to_return='collection', **kwargs):
        d = self.getRefGroups()[group_id]
        if to_return == 'dict':
            return aux.AttrDict({id: self.loadRef(dir=dir, **kwargs) for id, dir in d.items()})
        elif to_return == 'list':
            return aux.ItemList([self.loadRef(dir=dir, **kwargs) for id, dir in d.items()])
        elif to_return == 'collection':
            from ..process.dataset import LarvaDatasetCollection
            return LarvaDatasetCollection(datasets=aux.ItemList([self.loadRef(dir=dir, **kwargs) for id, dir in d.items()]))


conf = aux.AttrDict({k: ConfType(conftype=k) for k in reg.CONFTYPES if k != 'Ref'})

conf.Ref = RefType()


def resetConfs(conftypes=None, **kwargs):
    if conftypes is None:
        conftypes = reg.CONFTYPES

    for conftype in conftypes:
        conf[conftype].reset(**kwargs)
