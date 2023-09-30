import os
import shutil

import param

from .. import reg, aux, util
from ..param import Area, NestedConf, Larva_Distro, ClassAttr, SimTimeOps, \
    SimMetricOps, ClassDict, EnrichConf, OptionalPositiveRange, OptionalSelector, OptionalPositiveInteger, \
    generate_xyNor_distro, Odor, Life, class_generator, SimOps, RuntimeOps, Epoch, RuntimeDataOps, RandomizedColor, \
    OptionalPositiveNumber, Filesystem, TrackerOps, PreprocessConf, Substrate

__all__ = [
    'ConfType',
    'RefType',
    'conf',
    'resetConfs',
    'gen',
    'SimConfiguration',
    'SimConfigurationParams',
    'FoodConf',
    'EnvConf',
    'LarvaGroup',
    'LabFormat',
    'ExpConf',
    'full_lg',
    'GTRvsS',
    'DatasetConfig',
]



def build_GroupTypeSubkeys():
    d0 = {k: {} for k in reg.GROUPTYPES}
    d1 = {
        'LarvaGroup': {'Model'}
    }
    d0.update(d1)
    return aux.AttrDict(d0)




class ConfType(param.Parameterized) :
    """Select among available configuration types"""
    conftype = param.Selector(objects=reg.CONFTYPES, doc= 'The configuration type')
    dict= ClassDict(default=aux.AttrDict(), item_type=None, doc='The configuration dictionary')


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



    def getID(self,id):
        if id in self.dict.keys():
            return self.dict[id]
        else:
            reg.vprint(f'{self.conftype} Configuration {id} does not exist', 1)
            raise ValueError()

    def get(self, id):
        entry=self.getID(id)
        return self.conf_class(**entry)



    def load(self):
        self.dict= aux.load_dict(self.path_to_dict)

    def save(self):
        return aux.save_dict(self.dict, self.path_to_dict)

    def reset(self,init=False):
        dd = reg.funcs.stored_confs[self.conftype]()

        if os.path.isfile(self.path_to_dict):
            if init:
                return
            else:
                d = self.dict
        else:
            d = {}

        N0, N1 = len(d), len(dd)

        d.update(dd)

        Ncur = len(d)
        Nnew = Ncur - N0
        Nup = N1 - Nnew

        self.param.params('dict').item_type = self.dict_entry_type
        self.dict = d
        self.save()
        reg.vprint(f'{self.conftype}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing', 1)

    def setID(self, id, conf, mode='overwrite'):
        if id in self.dict.keys() and mode == 'update':
            self.dict[id] = self.dict[id].update_nestdict(conf.flatten())
        else:
            self.dict[id] = conf
        self.save()
        reg.vprint(f'{self.conftype} Configuration saved under the id : {id}', 1)

    def delete(self, id=None):
        if id is not None:
            if id in self.dict.keys():
                self.dict.pop(id, None)
                self.save()
                reg.vprint(f'Deleted {self.conftype} configuration under the id : {id}', 1)


    def expand(self, id=None, conf=None):
        if conf is None:
            if id in self.dict.keys():
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

    def confID_selector(self, default=None, single=True):
        kws={
            'default':default,
            'objects':self.confIDs,
            'label':f'{self.conftype} configuration ID',
            'doc':f'Selection among stored {self.conftype} configurations by ID'

        }
        if single :
            return OptionalSelector(**kws)
        else :
            return param.ListSelector(**kws)

    def confIDorNew(self):
        return ClassAttr(class_=(self.confID_selector(), self.conf_class()),default=None, doc='Accepts either an existing ID or a new configuration')

    @property
    def confIDs(self):
        return sorted(list(self.dict.keys()))

    @property
    def conf_class(self):
        c=self.conftype
        if c is None :
            return None
        elif c in gen.keys():
            return gen[c]
        else :
            return aux.AttrDict

    @property
    def dict_entry_type(self):
        return aux.AttrDict


class RefType(ConfType):

    """Select a reference dataset by ID"""


    def __init__(self, **kwargs):
        super().__init__(conftype='Ref', **kwargs)


    def getRefDir(self,id):
        assert id is not None
        return self.getID(id)

    def getRef(self, id=None, dir=None):
        path = self.path_to_Ref(id=id, dir=dir)
        assert os.path.isfile(path)
        c = aux.load_dict(path)
        assert 'id' in c.keys()
        reg.vprint(f'Loaded existing conf {c.id}', 1)
        return c

    def setRef(self, c, id=None, dir=None):
        path = self.path_to_Ref(id=id, dir=dir)
        aux.save_dict(c, path)
        assert 'id' in c.keys()
        reg.vprint(f'Saved conf under ID {c.id}', 1)


    def path_to_Ref(self, id=None, dir=None):
        if dir is None:
            dir=self.getRefDir(id)
        return f'{dir}/data/conf.txt'


    def loadRef(self, id=None, dir=None, load=False, **kwargs):
        from ..process.dataset import LarvaDataset
        c=self.getRef(id=id, dir=dir)
        assert c is not None
        d = LarvaDataset(config=c, load_data=False)
        if load:
            d.load(**kwargs)
        reg.vprint(f'Loaded stored reference dataset : {id}', 1)
        return d

    def retrieve_dataset(self, dataset=None,load=True,**kwargs):
        if dataset is None :
            dataset=self.loadRef(load=False,**kwargs)
        if load:
            dataset.load()
        return dataset

    @property
    def cleanRefIDs(self):
        ids=self.confIDs
        for id in ids:
            try:
                self.loadRef(id)
            except:
                self.delete(id)



    @property
    def dict_entry_type(self):
        return str

    def getRefGroups(self):
        d=self.Refdict
        gd=aux.AttrDict({c.group_id:c for id,c in d.items()})
        gIDs=aux.unique_list(list(gd.keys()))
        return aux.AttrDict({gID: {c.id : c.dir for id,c in d.items() if c.group_id==gID} for gID in gIDs})

    @property
    def RefGroupIDs(self):
        d=self.Refdict
        gd=aux.AttrDict({c.group_id:c for id,c in d.items()})
        return aux.unique_list(list(gd.keys()))

    @property
    def Refdict(self):
        return aux.AttrDict({id:self.getRef(id) for id in self.confIDs})

    def getRefGroup(self, group_id):
        d= self.getRefGroups()[group_id]
        return aux.AttrDict({id: self.getRef(dir=dir) for id, dir in d.items()})

    def loadRefGroup(self, group_id,to_return='collection', **kwargs):
        d = self.getRefGroups()[group_id]
        if to_return=='dict':
            return aux.AttrDict({id: self.loadRef(dir=dir, **kwargs) for id, dir in d.items()})
        elif to_return=='list':
            return [self.loadRef(dir=dir, **kwargs) for id, dir in d.items()]
        elif to_return=='collection':
            from ..process.dataset import LarvaDatasetCollection
            return LarvaDatasetCollection(datasets=[self.loadRef(dir=dir, **kwargs) for id, dir in d.items()])


conf=aux.AttrDict({k: ConfType(conftype=k) for k in reg.CONFTYPES if k!='Ref'})


conf.Ref=RefType()


def resetConfs(conftypes=None, **kwargs):
    if conftypes is None:
        conftypes = reg.CONFTYPES

    for conftype in conftypes:
        conf[conftype].reset(**kwargs)


from ..model import Food, Border, WindScape, ThermoScape, FoodGrid, OdorScape, DiffusionValueLayer, GaussianValueLayer

gen=aux.AttrDict({
    'FoodGroup':class_generator(Food, mode='Group'),
    'Food':class_generator(Food),
    'Arena':class_generator(Area),
    'Border':class_generator(Border),
    'Odor':class_generator(Odor),
    'Epoch':class_generator(Epoch),
    'Life':class_generator(Life),
    'Substrate':class_generator(Substrate),
    'FoodGrid':class_generator(FoodGrid),
    'WindScape':class_generator(WindScape),
    'ThermoScape':class_generator(ThermoScape),
    'OdorScape':class_generator(OdorScape),
    'DiffusionValueLayer':class_generator(DiffusionValueLayer),
    'GaussianValueLayer':class_generator(GaussianValueLayer),
})

# How to load existing

class SimConfiguration(RuntimeOps,SimMetricOps, SimOps):
    runtype = param.Selector(objects=reg.SIMTYPES, doc='The simulation mode')

    def __init__(self,runtype,**kwargs):
        self.param.add_parameter('experiment', self.exp_selector_param(runtype))
        super().__init__(runtype=runtype,**kwargs)
        # raise
        if 'experiment' in kwargs and kwargs['experiment'] is not None :
            self.experiment=kwargs['experiment']

        if self.id is None or not type(self.id)==str:
            self.id = self.generate_id(self.runtype, self.experiment)
        if self.dir is None:
            save_to = f'{self.path_to_runtype_data}/{self.experiment}'
            self.dir = f'{save_to}/{self.id}'

    @property
    def path_to_runtype_data(self):
        return f'{reg.SIM_DIR}/{self.runtype.lower()}_runs'

    def generate_id(self, runtype, exp):
        idx = reg.config.next_idx(exp, conftype=runtype)
        return f'{exp}_{idx}'

    # @ staticmethod
    def exp_selector_param(self, runtype):
        # runtype = self.runtype
        defaults = {
            'Exp': 'dish',
            'Batch': 'PItest_off',
            'Ga': 'exploration',
            'Eval': 'dispersion',
            'Replay': 'replay'
        }
        kws = {
            'default': defaults[runtype],
            'doc': 'The experiment simulated'
        }
        if runtype in ['Exp', 'Batch', 'Ga']:
            ids = conf[runtype].confIDs
            return param.Selector(objects=ids, **kws)
        else:
            return param.String(**kws)


class SimConfigurationParams(SimConfiguration):
    parameters = param.Parameter(default=None)
    def __init__(self,runtype='Exp',experiment=None,parameters=None,**kwargs):
        if parameters is None :
            if experiment is None or experiment not in reg.conf[runtype].confIDs:
                raise ValueError('Either a parameter dictionary or the name of the experiment must be provided')
            else :
                parameters = reg.conf[runtype].expand(experiment)
        elif experiment is None and 'experiment' in parameters.keys():
            experiment = parameters['experiment']
        if parameters is not None:
            for k in set(parameters).intersection(set(SimOps().nestedConf)):
                kwargs[k] = parameters[k]
        super().__init__(runtype=runtype,experiment=experiment,parameters=parameters,**kwargs)





class FoodConf(NestedConf):
    source_groups = ClassDict(item_type=gen.FoodGroup,  doc='The groups of odor or food sources available in the arena')
    source_units = ClassDict(item_type=gen.Food,  doc='The individual sources  of odor or food in the arena')
    food_grid = ClassAttr(gen.FoodGrid, default=None, doc='The food grid in the arena')




gen.FoodConf=class_generator(FoodConf)
gen.EnrichConf=class_generator(EnrichConf)

class EnvConf(NestedConf):
    arena = ClassAttr(gen.Arena, doc='The arena configuration')
    food_params = ClassAttr(gen.FoodConf, doc='The food sources in the arena')
    border_list = ClassDict(item_type=gen.Border, doc='The obstacles in the arena')
    odorscape = ClassAttr(class_=(gen.GaussianValueLayer, gen.DiffusionValueLayer), default=None, doc='The sensory odor landscape in the arena')
    windscape = ClassAttr(gen.WindScape, default=None, doc='The wind landscape in the arena')
    thermoscape = ClassAttr(gen.ThermoScape, default=None, doc='The thermal landscape in the arena')

class LarvaGroup(NestedConf):
    model = conf.Model.confID_selector()
    default_color = param.Color('black', doc='The default color of the group')
    odor = ClassAttr(Odor, doc='The odor of the agent')
    distribution = ClassAttr(Larva_Distro,doc='The spatial distribution of the group agents')
    life_history = ClassAttr(Life, doc='The life history of the group agents')
    sample = conf.Ref.confID_selector()
    imitation = param.Boolean(default=False, doc='Whether to imitate the reference dataset.')



    def __init__(self,id=None,**kwargs):
        super().__init__(**kwargs)
        if id is None:
            if self.model is not None :
                id = self.model
            else :
                id = 'LarvaGroup'
        self.id=id


    def entry(self, expand=False, as_entry=True):
        conf = self.nestedConf
        if expand and conf.model is not None:
            conf.model = reg.conf.Model.getID(conf.model)
            # conf.model = reg.stored.getModel(conf.model)
        if as_entry :
            return aux.AttrDict({self.id: conf})
        else:
            return conf

    def __call__(self, parameter_dict={}):
        Nids=self.distribution.N
        if self.model is not None:
            m=reg.conf.Model.getID(self.model)
        else :
            m=None
        kws={
            'm' : m,
            'refID' : self.sample,
            'parameter_dict' : parameter_dict,
            'Nids' : Nids,
        }

        if not self.imitation:
            ps, ors = generate_xyNor_distro(self.distribution)
            ids = [f'{self.id}_{i}' for i in range(Nids)]
            all_pars, refID = util.sampleRef(**kws)
        else:
            ids, ps, ors, all_pars = util.imitateRef(**kws)
        confs = []
        for id, p, o, pars in zip(ids, ps, ors, all_pars):
            conf = {
                'pos': p,
                'orientation': o,
                'default_color': self.default_color,
                'unique_id': id,
                'group': self.id,
                'odor': self.odor,
                'life_history': self.life_history,
                **pars
            }
            confs.append(conf)
        return confs

gen.Env=class_generator(EnvConf)


class LabFormat(NestedConf) :
    labID=param.String(doc='The identifier ID of the lab')
    tracker = ClassAttr(TrackerOps, doc='The dataset metadata')
    filesystem = ClassAttr(Filesystem, doc='The import-relevant lab-format filesystem')
    env_params = ClassAttr(gen.Env, doc='The environment configuration')
    preprocess=ClassAttr(PreprocessConf, doc='The environment configuration')

    @property
    def path(self):
        return f'{reg.DATA_DIR}/{self.labID}Group'

    @property
    def raw_folder(self):
        return f'{self.path}/raw'

    @property
    def processed_folder(self):
        return f'{self.path}/processed'

    def get_source_dir(self, parent_dir, raw_folder=None, merged=False):
        if raw_folder is None:
            raw_folder = self.raw_folder
        source_dir = f'{raw_folder}/{parent_dir}'
        if merged:
            source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
        return source_dir

    @property
    def import_func(self):
        from ..process.importing import lab_specific_import_functions as d
        return d[self.labID]

    def import_data_to_dfs(self, parent_dir, raw_folder=None, merged=False,**kwargs):
        source_dir = self.get_source_dir(parent_dir, raw_folder, merged)
        return self.import_func(source_dir=source_dir, **kwargs)

    def build_dataset(self, step,end,parent_dir, proc_folder=None, group_id=None, N=None, id=None, sample=None, color='black', epochs={}, age=0.0,):
        if group_id is None:
            group_id = parent_dir
        if id is None:
            id = f'{self.labID}_{group_id}_dataset'
        if proc_folder is None:
            proc_folder = self.processed_folder
        dir = f'{proc_folder}/{group_id}/{id}'

        conf = {
            'load_data': False,
            'dir': dir,
            'id': id,
            'larva_groups': reg.config.lg(id=group_id, c=color, sample=sample, mID=None, N=N, epochs=epochs, age=age),
            'env_params': self.env_params.nestedConf,
            **self.tracker.nestedConf,
            'step': step,
            'end': end,
        }
        from ..process.dataset import LarvaDataset
        d = LarvaDataset(**conf)
        reg.vprint(f'***-- Dataset {d.id} created with {len(d.config.agent_ids)} larvae! -----', 1)
        return d

    def enrich_dataset(self, d, conf=None):
        if conf is None:
            conf = reg.gen.EnrichConf(proc_keys=[], anot_keys=[]).nestedConf
        conf.pre_kws = self.preprocess.nestedConf
        d = d.enrich(**conf, is_last=False)
        reg.vprint(f'****- Processed dataset {d.id} to derive secondary metrics -----', 1)
        return d


    def import_dataset(self, parent_dir, raw_folder=None, merged=False,
                       proc_folder=None, group_id=None, N=None, id=None, sample=None, color='black', epochs={}, age=0.0,
                       refID=None, enrich_conf=None, save_dataset=True, **kwargs):

        """
        Imports a single experimental dataset defined by their ID from a source folder.

        Parameters
        ----------
        parent_dir: string
            The parent directory where the raw files are located.

        raw_folder: string, optional
            The directory where the raw files are located.
            If not provided it is set as the subfolder 'raw' under the lab-specific group directory.
         merged: boolean
            Whether to merge all raw datasets in the source folder in a single imported dataset.
            Defaults to False.

        proc_folder: string, optional
            The directory where the imported dataset will be placed.
            If not provided it is set as the subfolder 'processed' under the lab-specific group directory.
        group_id: string, optional
            The group ID of the dataset to be imported.
            If not provided it is set as the parent_dir argument.
        id: string, optional
            The ID under which to store the imported dataset.
            If not provided it is set by default.

        N: integer, optional
            The number of larvae in the dataset.
        sample: string, optional
            The reference ID of the reference dataset from which the current is sampled.
        color: string
            The default color of the new dataset.
            Defaults to 'black'.
        epochs: dict
            Any discrete rearing epochs during the larvagroup's life history.
            Defaults to '{}'.
        age: float
            The post-hatch age of the larvae in hours.
            Defaults to '0.0'.

       refID: string, optional
            The reference IDs under which to store the imported dataset as reference dataset.
            If not provided the dataset is not stored in the reference database.
        save_dataset: boolean
            Whether to store the imported dataset to disc.
            Defaults to True.
        enrich_conf: dict, optional
            The configuration for enriching the imported dataset with secondary parameters.
        **kwargs: keyword arguments
            Additional keyword arguments to be passed to the lab_specific build-function.

        Returns
        -------
        lib.process.dataset.LarvaDataset
            The imported dataset in the common larvaworld format.
        """

        reg.vprint('', 1)
        reg.vprint(f'----- Importing experimental dataset by the {self.labID} lab-specific format. -----', 1)
        step, end=self.import_data_to_dfs(parent_dir, raw_folder=raw_folder, merged=merged,**kwargs)
        if step is None and end is None:
            reg.vprint(f'xxxxx Failed to create dataset! -----', 1)
            return None
        else:
            d=self.build_dataset(step, end, parent_dir, proc_folder=proc_folder, group_id=group_id, N=N,
                                   id=id, sample=sample, color=color, epochs=epochs, age=age)
            d = self.enrich_dataset(d, conf=enrich_conf)
            if save_dataset:
                shutil.rmtree(d.config.dir, ignore_errors=True)
                d.save(refID=refID)
                # reg.vprint(f'***** Dataset {d.id} stored -----', 1)
            return d



class ExpConf(SimOps):
    env_params = ClassAttr(gen.Env, doc='The environment configuration')
    # env_params = conf.Env.confIDorNew()
    # env_params = conf.Env.confID_selector()
    experiment = conf.Exp.confID_selector()
    trials = conf.Trial.confID_selector('default')
    collections = param.ListSelector(default=['pose'],objects=reg.parDB.output_keys, doc='The data to collect as output')
    larva_groups = ClassDict(item_type=LarvaGroup, doc='The larva groups')
    parameter_dict = param.Dict(default={},doc='Dictionary of parameters to pass to the agents')
    # sim_params = aux.ClassAttr(SimOps,doc='The simulation configuration')
    enrichment = ClassAttr(gen.EnrichConf, doc='The post-simulation processing')




    def __init__(self,id=None,**kwargs):
        super().__init__(**kwargs)

# class DatasetConf(NestedConf):
#     environment = ClassAttr(EnvConf, doc='The environment configuration')
#     sim_options = ClassAttr(SimOps, doc='The spatiotemporal resolution')
#     larva_groups = ClassDict(item_type=LarvaGroup, doc='The larva groups')

class ReplayConfGroup(NestedConf):
    agent_ids = param.List(item_type=int,doc='Whether to only display some larvae of the dataset, defined by their indexes.')
    transposition = OptionalSelector(objects=['origin', 'arena', 'center'], doc='Whether to transpose the dataset spatial coordinates.')
    track_point = param.Integer(default=-1,softbounds=(-1,12), doc='The midline point to use for defining the larva position.')
    # dynamic_color = OptionalSelector(objects=['lin_color', 'ang_color'], doc='Whether to display larva tracks according to the instantaneous forward or angular velocity.')
    env_params = conf.Env.confID_selector()

class ReplayConfUnit(NestedConf):
    close_view = param.Boolean(False, doc='Whether to visualize a small arena on close range.')
    fix_segment = OptionalSelector(objects=['rear', 'front'],
                                   doc='Whether to additionally fixate the above or below body segment.')
    fix_point = OptionalPositiveInteger(softmin=1, softmax=12,
                                        doc='Whether to fixate a specific midline point to the center of the screen. Relevant when replaying a single larva track.')


class ReplayConf(ReplayConfGroup, ReplayConfUnit):
    refID = conf.Ref.confID_selector()
    refDir = param.String(None)
    time_range = OptionalPositiveRange(default=None, doc='Whether to only replay a defined temporal slice of the dataset.')
    overlap_mode = param.Boolean(False,doc='Whether to draw overlapped image of the track.')
    draw_Nsegs = OptionalPositiveInteger(softmin=1, softmax=12,doc='Whether to artificially simplify the experimentally tracked larva body to a segmented virtual body of the given number of segments.')







gen.LarvaGroup=class_generator(LarvaGroup)
gen.Exp=ExpConf
gen.LabFormat=LabFormat
gen.Replay=class_generator(ReplayConf)






def full_lg(id=None, expand=False,as_entry=True,**conf):
    try :
        lg=LarvaGroup(id=id,**conf)
        return lg.entry(expand=expand, as_entry=as_entry)
    except :
        raise

def GTRvsS(N=1, age=72.0, q=1.0, h_starved=0.0, sample=reg.default_refID, substrate_type='standard', pref='',
           navigator=False, expand=False, **kwargs):
    if age == 0.0:
        epochs = {}
    else:
        if h_starved == 0:
            epochs = {
                0: {'age_range': (0.0, age), 'substrate': {'type': substrate_type, 'quality': q}}
            }
        else:
            epochs = {
                0: {'age_range': (0.0, age - h_starved), 'substrate': {'type': substrate_type, 'quality': q}},
                1: {'age_range': (age - h_starved, age), 'substrate': {'type': substrate_type, 'quality': 0}},
            }
    kws0 = {
        'distribution': {'N': N, 'scale': (0.005, 0.005)},
        'life_history': {'age': age, 'epochs': epochs},
        'sample': sample,
        'expand': expand,
    }

    mcols = ['blue', 'red']
    mID0s = ['rover', 'sitter']
    lgs = {}
    for mID0, mcol in zip(mID0s, mcols):
        id = f'{pref}{mID0.capitalize()}'

        if navigator:
            mID0 = f'navigator_{mID0}'

        kws = {
            'id': id,
            'default_color': mcol,
            'model': mID0,
            **kws0
        }

        lgs.update(full_lg(**kws))
    return aux.AttrDict(lgs)


class DatasetConfig(RuntimeDataOps,SimMetricOps, SimTimeOps):
    # duration = OptionalPositiveNumber(default=None)
    Nticks = OptionalPositiveInteger(default=None)
    refID = param.String(None, doc='The unique ID of the reference dataset')
    group_id = param.String(None, doc='The unique ID of the group')
    color = RandomizedColor(default='black', doc='The color of the dataset', instantiate=True)
    # larva_groups = ClassDict(item_type=LarvaGroup, doc='The larva groups')
    env_params = ClassAttr(gen.Env, doc='The environment configuration')
    agent_ids=param.List(item_type=None, doc='The unique IDs of the agents in the dataset')
    N = OptionalPositiveInteger(default=None, softmax=500, doc='The number of agents in the group')
    sample = conf.Ref.confID_selector()
    filtered_at = OptionalPositiveNumber(default=None)
    rescaled_by = OptionalPositiveNumber(default=None)

    @property
    def h5_kdic(self):
        from ..process.dataset import h5_kdic
        return h5_kdic(self.point, self.Npoints, self.Ncontour)

    @param.depends('agent_ids', watch=True)
    def update_Nagents(self):
        self.N=len(self.agent_ids)
