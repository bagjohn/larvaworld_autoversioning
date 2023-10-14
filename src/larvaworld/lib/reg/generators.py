import os
import shutil

import numpy as np
import pandas as pd
import param

from .. import reg, aux, util
from ..param import Area, BoundedArea, NestedConf, Larva_Distro, ClassAttr, SimTimeOps, \
    SimMetricOps, ClassDict, EnrichConf, OptionalPositiveRange, OptionalSelector, OptionalPositiveInteger, \
    generate_xyNor_distro, Odor, Life, class_generator, SimOps, RuntimeOps, Epoch, RuntimeDataOps, RandomizedColor, \
    OptionalPositiveNumber, Filesystem, TrackerOps, PreprocessConf, Substrate, AirPuff
from ..model import Food, Border, WindScape, ThermoScape, FoodGrid, OdorScape, DiffusionValueLayer, GaussianValueLayer

__all__ = [
    # 'ConfType',
    # 'RefType',
    # 'conf',
    # 'resetConfs',
    'gen',
    'SimConfiguration',
    'SimConfigurationParams',
    'FoodConf',
    'EnvConf',
    'LarvaGroup',
    'LabFormat',
    'ExpConf',
    'GTRvsS',
    'DatasetConfig',
]

gen = aux.AttrDict({
    'FoodGroup': class_generator(Food, mode='Group'),
    'Food': class_generator(Food),
    'Arena': class_generator(Area),
    'Border': class_generator(Border),
    'Odor': class_generator(Odor),
    'Epoch': class_generator(Epoch),
    'Life': class_generator(Life),
    'Substrate': class_generator(Substrate),
    'FoodGrid': class_generator(FoodGrid),
    'WindScape': class_generator(WindScape),
    'ThermoScape': class_generator(ThermoScape),
    'OdorScape': class_generator(OdorScape),
    'DiffusionValueLayer': class_generator(DiffusionValueLayer),
    'GaussianValueLayer': class_generator(GaussianValueLayer),
    'AirPuff': class_generator(AirPuff),
})


# How to load existing

class SimConfiguration(RuntimeOps, SimMetricOps, SimOps):
    runtype = param.Selector(objects=reg.SIMTYPES, doc='The simulation mode')

    def __init__(self, runtype, **kwargs):
        self.param.add_parameter('experiment', self.exp_selector_param(runtype))
        super().__init__(runtype=runtype, **kwargs)
        # raise
        if 'experiment' in kwargs and kwargs['experiment'] is not None:
            self.experiment = kwargs['experiment']

        if self.id is None or not type(self.id) == str:
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

    def exp_selector_param(self, runtype):
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
        if runtype in reg.CONFTYPES:
            return param.Selector(objects=reg.conf[runtype].confIDs, **kws)
        else:
            return param.Parameter(**kws)


class SimConfigurationParams(SimConfiguration):
    parameters = param.Parameter(default=None)

    def __init__(self, runtype='Exp', experiment=None, parameters=None, **kwargs):
        # print(experiment)
        if parameters is None:
            if runtype in reg.CONFTYPES:
                ct = reg.conf[runtype]
                if experiment is None:
                    raise ValueError(
                        f'Either a parameter dictionary or the ID of an available {runtype} configuration must be provided')
                elif experiment not in ct.confIDs:
                    raise ValueError(f'Experiment {experiment} not available in {runtype} configuration dictionary')
                else:
                    parameters = ct.expand(experiment)
            elif runtype in reg.gen.keys():
                parameters = reg.gen[runtype]().nestedConf
            else:
                pass
        elif experiment is None and 'experiment' in parameters.keys():
            experiment = parameters['experiment']
        if parameters is not None:
            for k in set(parameters).intersection(set(SimOps().nestedConf)):
                if k in kwargs.keys():
                    parameters[k] = kwargs[k]
                else:
                    kwargs[k] = parameters[k]
        super().__init__(runtype=runtype, experiment=experiment, parameters=parameters, **kwargs)


class FoodConf(NestedConf):
    source_groups = ClassDict(item_type=gen.FoodGroup, doc='The groups of odor or food sources available in the arena')
    source_units = ClassDict(item_type=gen.Food, doc='The individual sources  of odor or food in the arena')
    food_grid = ClassAttr(gen.FoodGrid, default=None, doc='The food grid in the arena')


# class FoodConf(NestedConf):
#     source_groups = ClassDict(item_type=Food.generator(mode='Group'), doc='The groups of odor or food sources available in the arena')
#     source_units = ClassDict(item_type=Food.generator(), doc='The individual sources  of odor or food in the arena')
#     food_grid = ClassAttr(FoodGrid.generator(), default=None, doc='The food grid in the arena')


gen.FoodConf = class_generator(FoodConf)
gen.EnrichConf = class_generator(EnrichConf)


class EnvConf(NestedConf):
    arena = ClassAttr(gen.Arena, doc='The arena configuration')
    food_params = ClassAttr(gen.FoodConf, doc='The food sources in the arena')
    border_list = ClassDict(item_type=gen.Border, doc='The obstacles in the arena')
    odorscape = ClassAttr(class_=(gen.GaussianValueLayer, gen.DiffusionValueLayer), default=None,
                          doc='The sensory odor landscape in the arena')
    windscape = ClassAttr(gen.WindScape, default=None, doc='The wind landscape in the arena')
    thermoscape = ClassAttr(gen.ThermoScape, default=None, doc='The thermal landscape in the arena')

    def __init__(self, odorscape=None, **kwargs):
        if odorscape is not None and isinstance(odorscape, aux.AttrDict):
            mode = odorscape.odorscape
            odorscape_classes = list(EnvConf.param.odorscape.class_)
            odorscape_modes = dict(zip(['Gaussian', 'Diffusion'], odorscape_classes))
            odorscape = odorscape_modes[mode](**odorscape)

        super().__init__(odorscape=odorscape, **kwargs)

    def visualize(self, **kwargs):
        """
        Visualize the environment by launching a simulation without agents
        """

        from ..sim.base_run import BaseRun
        BaseRun.visualize_Env(envConf=self.nestedConf, envID=self.name, **kwargs)


class LarvaGroup(NestedConf):
    model = reg.conf.Model.confID_selector()
    color = param.Color('black', doc='The default color of the group')
    odor = ClassAttr(Odor, doc='The odor of the agent')
    distribution = ClassAttr(Larva_Distro, doc='The spatial distribution of the group agents')
    life_history = ClassAttr(Life, doc='The life history of the group agents')
    sample = reg.conf.Ref.confID_selector()
    imitation = param.Boolean(default=False, doc='Whether to imitate the reference dataset.')

    def __init__(self, id=None, **kwargs):
        super().__init__(**kwargs)
        if id is None:
            if self.model is not None:
                id = self.model
            else:
                id = 'LarvaGroup'
        self.id = id

    def entry(self, expand=False, as_entry=True):
        conf = self.nestedConf
        if expand and conf.model is not None:
            conf.model = reg.conf.Model.getID(conf.model)
            # conf.model = reg.stored.getModel(conf.model)
        if as_entry:
            return aux.AttrDict({self.id: conf})
        else:
            return conf

    def __call__(self, parameter_dict={}):
        Nids = self.distribution.N
        if self.model is not None:
            m = reg.conf.Model.getID(self.model)
        else:
            m = None
        kws = {
            'm': m,
            'refID': self.sample,
            'parameter_dict': parameter_dict,
            'Nids': Nids,
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
                'color': self.color,
                'unique_id': id,
                'group': self.id,
                'odor': self.odor,
                'life_history': self.life_history,
                **pars
            }
            confs.append(conf)
        return confs


gen.LarvaGroup = class_generator(LarvaGroup)
# gen.Env = class_generator(EnvConf)
gen.Env = EnvConf


class LabFormat(NestedConf):
    labID = param.String(doc='The identifier ID of the lab')
    tracker = ClassAttr(TrackerOps, doc='The dataset metadata')
    filesystem = ClassAttr(Filesystem, doc='The import-relevant lab-format filesystem')
    env_params = ClassAttr(EnvConf, doc='The environment configuration')
    preprocess = ClassAttr(PreprocessConf, doc='The environment configuration')

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

    def get_store_sequence(self, mode='semifull'):
        if mode == 'full':
            return self.filesystem.read_sequence[1:]
        elif mode == 'minimal':
            return aux.nam.xy(self.tracker.point)
        elif mode == 'semifull':
            return aux.nam.midline_xy(self.tracker.Npoints, flat=True) + aux.nam.contour_xy(self.tracker.Ncontour,
                                                                                            flat=True) + [
                'collision_flag']
        elif mode == 'points':
            return aux.nam.xy(self.tracker.points, flat=True) + ['collision_flag']
        else:
            raise

    @property
    def import_func(self):
        from ..process.importing import lab_specific_import_functions as d
        return d[self.labID]

    def import_data_to_dfs(self, parent_dir, raw_folder=None, merged=False, save_mode='semifull', **kwargs):
        source_dir = self.get_source_dir(parent_dir, raw_folder, merged)
        if self.filesystem.structure == 'per_larva':
            read_sequence = self.filesystem.read_sequence
            store_sequence = self.get_store_sequence(save_mode)
        return self.import_func(source_dir=source_dir, tracker=self.tracker, filesystem=self.filesystem, **kwargs)

    def build_dataset(self, step, end, parent_dir, proc_folder=None, group_id=None, id=None, sample=None,
                      color='black', epochs=[], age=0.0,refID=None):
        if group_id is None:
            group_id = parent_dir
        if id is None:
            id = f'{self.labID}_{group_id}_dataset'
        if proc_folder is None:
            proc_folder = self.processed_folder
        dir = f'{proc_folder}/{group_id}/{id}'

        conf = {
            'initialize': True,
            'load_data': False,
            'dir': dir,
            'id': id,
            'refID': refID,
            'color': color,
            'larva_groups': gen.LarvaGroup(c=color, sample=sample, mID=None, N=end.index.values.shape[0], life=[age, epochs]).entry(
                id=group_id),
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
        d.preprocess(**self.preprocess.nestedConf)
        if conf is None:
            conf = reg.gen.EnrichConf(proc_keys=[], anot_keys=[]).nestedConf
        reg.vprint(f'****- Processed dataset {d.id} to derive secondary metrics -----', 1)
        return d

    def import_dataset(self, parent_dir, raw_folder=None, merged=False,
                       proc_folder=None, group_id=None, id=None, sample=None, color='black', epochs=[], age=0.0,
                       refID=None, enrich_conf=None, save_dataset=False, **kwargs):

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
        step, end = self.import_data_to_dfs(parent_dir, raw_folder=raw_folder, merged=merged, **kwargs)
        if step is None and end is None:
            reg.vprint(f'xxxxx Failed to create dataset! -----', 1)
            return None
        else:
            step = step.astype(float)
            d = self.build_dataset(step, end, parent_dir, proc_folder=proc_folder, group_id=group_id,
                                   id=id, sample=sample, color=color, epochs=epochs, age=age, refID=refID)
            d = self.enrich_dataset(d, conf=enrich_conf)
            if save_dataset:
                shutil.rmtree(d.config.dir, ignore_errors=True)
                d.save()
                # reg.vprint(f'***** Dataset {d.id} stored -----', 1)
            return d

    def import_datasets(self, source_ids, ids=None, colors=None, refIDs=None, **kwargs):
        """
        Imports multiple experimental datasets defined by their IDs.

        Parameters
        ----------
        source_ids: list of strings
            The IDs of the datasets to be imported as appearing in the source files.
        ids: list of strings, optional
            The IDs under which to store the datasets to be imported.
            The source_ids are used if not provided.
        refIDs: list of strings, optional
            The reference IDs under which to store the imported datasets as reference datasets.
             If not provided the datasets are not stored in the reference database.
        colors: list of strings, optional
            The colors of the datasets to be imported.
            Randomly selected if not provided.
        **kwargs: keyword arguments
            Additional keyword arguments to be passed to the import_dataset function.

        Returns
        -------
        list of lib.process.dataset.LarvaDataset
            The imported datasets in the common larvaworld format.
        """

        Nds = len(source_ids)
        if colors is None:
            colors = aux.N_colors(Nds)
        if ids is None:
            ids = source_ids
        if refIDs is None:
            refIDs = [None] * Nds

        assert len(ids) == Nds
        assert len(colors) == Nds
        assert len(refIDs) == Nds

        return [self.import_dataset(id=ids[i], color=colors[i], source_id=source_ids[i], refID=refIDs[i], **kwargs) for
                i in
                range(Nds)]

    def read_timeseries_from_raw_files_per_larva(self, files, read_sequence, store_sequence, inv_x=False):
        """
        Reads timeseries data stored in txt files of the lab-specific Jovanic format and returns them as a pd.Dataframe.

        Parameters
        ----------
        files : list
            List of the absolute filepaths of the data files.
        read_sequence : list of strings
            The sequence of parameters found in each file
        store_sequence : list of strings
            The sequence of parameters to store
        inv_x : boolean
            Whether to invert x axis.
            Defaults to False

        Returns
        -------
        list of pandas.DataFrame
        """

        dfs = []
        for f in files:
            df = pd.read_csv(f, header=None, index_col=0, names=read_sequence)

            # If indexing is in strings replace with ascending floats
            if all([type(ii) == str for ii in df.index.values]):
                df.reset_index(inplace=True, drop=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            if inv_x:
                for x_par in [p for p in read_sequence if p.endswith('x')]:
                    df[x_par] *= -1
            df = df[store_sequence]
            dfs.append(df)
        return dfs


class ExpConf(SimOps):
    env_params = ClassAttr(gen.Env, doc='The environment configuration')
    experiment = reg.conf.Exp.confID_selector()
    trials = param.Dict(default=aux.AttrDict({'epochs': aux.ItemList()}),
                        doc='Dictionary of temporal epochs of the experiment')
    # trials = reg.conf.Trial.confID_selector('default')
    collections = param.ListSelector(default=['pose'], objects=reg.parDB.output_keys,
                                     doc='The data to collect as output')
    larva_groups = ClassDict(item_type=gen.LarvaGroup, doc='The larva groups')
    parameter_dict = param.Dict(default={}, doc='Dictionary of parameters to pass to the agents')
    enrichment = ClassAttr(gen.EnrichConf, doc='The post-simulation processing')

    def __init__(self, id=None, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def imitation_exp(cls, refID, mID='loco_default', **kwargs):
        c = reg.conf.Ref.getRef(refID)
        kws = {
            # 'id': f'Imitation {refID}',
            'sample': refID,
            'model': mID,
            # 'model': reg.conf.Model.getID(mID),
            'color': c.color,
            'distribution': {'N': c.N},
            'imitation': True,

        }
        return cls(dt=c.dt, duration=c.duration, env_params=gen.Env(**c.env_params),
                   larva_groups=aux.AttrDict({f'Imitation {refID}': gen.LarvaGroup(**kws)}),
                   experiment='dish', **kwargs)

    @property
    def agent_confs(self):
        confs = []
        for gID, gConf in self.larva_groups.items():
            lg = LarvaGroup(**gConf, id=gID)
            confs += lg(parameter_dict=self.parameter_dict)
        return confs


gen.Exp = ExpConf


class ReplayConfGroup(NestedConf):
    agent_ids = param.List(item_type=int,
                           doc='Whether to only display some larvae of the dataset, defined by their indexes.')
    transposition = OptionalSelector(objects=['origin', 'arena', 'center'],
                                     doc='Whether to transpose the dataset spatial coordinates.')
    track_point = param.Integer(default=-1, softbounds=(-1, 12),
                                doc='The midline point to use for defining the larva position.')
    env_params = reg.conf.Env.confID_selector()


class ReplayConfUnit(NestedConf):
    close_view = param.Boolean(False, doc='Whether to visualize a small arena on close range.')
    fix_segment = OptionalSelector(objects=['rear', 'front'],
                                   doc='Whether to additionally fixate the above or below body segment.')
    fix_point = OptionalPositiveInteger(softmin=1, softmax=12,
                                        doc='Whether to fixate a specific midline point to the center of the screen. Relevant when replaying a single larva track.')


class ReplayConf(ReplayConfGroup, ReplayConfUnit):
    refID = reg.conf.Ref.confID_selector()
    refDir = param.String(None)
    time_range = OptionalPositiveRange(default=None,
                                       doc='Whether to only replay a defined temporal slice of the dataset.')
    overlap_mode = param.Boolean(False, doc='Whether to draw overlapped image of the track.')
    draw_Nsegs = OptionalPositiveInteger(softmin=1, softmax=12,
                                         doc='Whether to artificially simplify the experimentally tracked larva body to a segmented virtual body of the given number of segments.')


gen.LabFormat = LabFormat
gen.Replay = class_generator(ReplayConf)


def GTRvsS(N=1, age=72.0, q=1.0, h_starved=0.0, sample=None, substrate_type='standard', pref='', navigator=False,
           expand=False):
    kws0 = {
        'distribution': {'N': N, 'scale': (0.005, 0.005)},
        'life_history': Life.prestarved(age=age, h_starved=h_starved, rearing_quality=q, substrate_type=substrate_type),
        'sample': sample,
    }

    mcols = ['blue', 'red']
    mID0s = ['rover', 'sitter']
    lgs = {}
    for mID0, mcol in zip(mID0s, mcols):
        id = f'{pref}{mID0.capitalize()}'

        if navigator:
            mID0 = f'{mID0}_nav'

        kws = {
            'id': id,
            'color': mcol,
            'model': mID0,
            **kws0
        }

        lgs.update(LarvaGroup(**kws).entry(expand=expand))
    return aux.AttrDict(lgs)


class DatasetConfig(RuntimeDataOps, SimMetricOps, SimTimeOps):
    Nticks = OptionalPositiveInteger(default=None)
    refID = param.String(None, doc='The unique ID of the reference dataset')
    group_id = param.String(None, doc='The unique ID of the group')
    color = RandomizedColor(default='black', doc='The color of the dataset', instantiate=True)
    env_params = ClassAttr(gen.Env, doc='The environment configuration')
    agent_ids = param.List(item_type=None, doc='The unique IDs of the agents in the dataset')
    N = OptionalPositiveInteger(default=None, softmax=500, doc='The number of agents in the group')
    sample = reg.conf.Ref.confID_selector()
    filtered_at = OptionalPositiveNumber(default=None)
    rescaled_by = OptionalPositiveNumber(default=None)

    @property
    def h5_kdic(self):
        from ..process.dataset import h5_kdic
        return h5_kdic(self.point, self.Npoints, self.Ncontour)

    @param.depends('agent_ids', watch=True)
    def update_Nagents(self):
        self.N = len(self.agent_ids)

    @property
    def arena_vertices(self):
        a = self.env_params.arena
        vs = BoundedArea(dims=a.dims, geometry=a.geometry).vertices
        return np.array(vs)
