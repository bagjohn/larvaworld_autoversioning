import os
import os.path
import shutil
import warnings

from .. import reg, aux
from ..process.build_aux import constrain_selected_tracks, finalize_timeseries_dataframe, \
    read_timeseries_from_raw_files_per_parameter, match_larva_ids, init_endpoint_dataframe_from_timeseries, \
    read_timeseries_from_raw_files_per_larva, read_Schleyer_timeseries_from_raw_files_per_larva, generate_dataframes

__all__ = [
    'import_datasets',
    'import_dataset',
    # 'build_dataset',
    'import_Jovanic',
    'import_Schleyer',
    'import_Berni',
    'import_Arguello',
    # 'lab_specific_build_functions'
]


def import_datasets(source_ids, ids=None, colors=None, refIDs=None, **kwargs):
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

    return [import_dataset(id=ids[i], color=colors[i], source_id=source_ids[i], refID=refIDs[i], **kwargs) for i in
            range(Nds)]


def import_dataset(labID, parent_dir, raw_folder=None, merged=False,
                   proc_folder=None, group_id=None, N=None, id=None, sample=None, color='black', epochs={}, age=0.0,
                   refID=None, enrich_conf=None, save_dataset=True, **kwargs):
    """
    Imports a single experimental dataset defined by their ID from a source folder.

    Parameters
    ----------
    labID: string
        The ID of the lab-specific format of the raw files.
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
    reg.vprint(f'----- Importing experimental dataset by the {labID} lab-specific format. -----', 1)

    g = reg.conf.LabFormat.get(labID)
    if raw_folder is None:
        raw_folder = f'{g.path}/raw'
    source_dir = f'{raw_folder}/{parent_dir}'
    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    step, end = lab_specific_import_functions[labID](source_dir=source_dir, **kwargs)

    if step is None and end is None:
        reg.vprint(f'xxxxx Failed to create dataset! -----', 1)
        return None
    else:
        from ..process.dataset import LarvaDataset

        if group_id is None:
            group_id = parent_dir
        if id is None:
            id = f'{labID}_{group_id}_dataset'
        if proc_folder is None:
            proc_folder = f'{g.path}/processed'
        # target_dir = f'{proc_folder}/{group_id}/{id}'

        conf = {
            'load_data': False,
            'dir': f'{proc_folder}/{group_id}/{id}',
            'id': id,
            'larva_groups': reg.config.lg(id=group_id, c=color, sample=sample, mID=None, N=N, epochs=epochs, age=age),
            'env_params': g.env_params.nestedConf,
            **g.tracker.nestedConf,
            'step': step,
            'end': end,
        }
        d = LarvaDataset(**conf)
        reg.vprint(f'***-- Dataset {d.id} created with {len(d.config.agent_ids)} larvae! -----', 1)
        # kws = {
        #     'labID': labID,
        #     'group_id': group_id,
        #     'Ν': N,
        #     'target_dir': target_dir,
        #     # 'source_dir': source_dir,
        #     # 'max_Nagents': N,
        #     **kwargs
        # }

        # d = build_dataset(id=id, **kws)

        # if d is not None:

        if enrich_conf is None:
            enrich_conf = reg.gen.EnrichConf(proc_keys=[], anot_keys=[]).nestedConf
        enrich_conf['pre_kws'] = g.preprocess.nestedConf
        d = d.enrich(**enrich_conf, is_last=False)
        reg.vprint(f'****- Processed dataset {d.id} to derive secondary metrics -----', 1)
        if save_dataset:
            shutil.rmtree(d.config.dir, ignore_errors=True)
            d.save(refID=refID)
            reg.vprint(f'***** Dataset {d.id} stored -----', 1)
        return d


# def build_dataset(labID, id, group_id, target_dir, N=None, **kwargs):
#     """
#     Converts experimental data to a single larvaworld dataset according to a lab-specific data format.
#
#     Parameters
#     ----------
#     labID: string
#         The ID of the lab-specific format of the raw files.
#     id: string
#         The ID under which to store the imported dataset.
#     group_id: string
#         The group ID of the dataset to be imported.
#     target_dir: string
#         The directory where the new dataset will be placed.
#     N: integer, optional
#         The number of larvae in the dataset.
#         If provided it also sets the maximum number of larvae 'max_Nagents' allowed in the dataset.
#
#     **kwargs: keyword arguments
#         Additional keyword arguments to be passed to the lab-specific build function.
#
#     Returns
#     -------
#     lib.process.dataset.LarvaDataset
#         The imported dataset in the common larvaworld format.
#     """
#
#     print(f'*---- Building dataset {id} under the {labID} format. -----')
#     warnings.filterwarnings('ignore')
#
#     step, end = lab_specific_build_functions[labID](**kwargs)
#
#     shutil.rmtree(target_dir, ignore_errors=True)
#     g = reg.conf.LabFormat.getID(labID)
#
#     conf = {
#         'load_data': False,
#         'dir': target_dir,
#         'id': id,
#         'larva_groups': reg.config.lg(id=group_id, c=color, sample=sample, mID=None, N=N, epochs=epochs, age=age),
#         'env_params': g.env_params,
#         **g.tracker,
#         'step': step,
#         'end': end,
#     }
#     from ..process.dataset import LarvaDataset
#     d = LarvaDataset(**conf)
#
#     # d.set_data(step=step, end=end)
#     return d


def import_Jovanic(source_id, source_dir, match_ids=True, matchID_kws={}, interpolate_ticks=True, **kwargs):
    """
    Builds a larvaworld dataset from Jovanic-lab-specific raw data

    Parameters
    ----------
    source_id : string
        The ID of the imported dataset
    source_dir : string
        The folder containing the imported dataset
    match_ids : boolean
        Whether to use the match-ID algorithm
        Defaults to True
    matchID_kws : dict
        Additional keyword arguments to be passed to the match-ID algorithm.
    interpolate_ticks : boolean
        Whether to interpolate timeseries into a fixed timestep timeseries
        Defaults to True
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """

    g = reg.conf.LabFormat.get('Jovanic')
    dt = g.tracker.dt

    df = read_timeseries_from_raw_files_per_parameter(pref=f'{source_dir}/{source_id}')

    if match_ids:
        Npoints = g.tracker.Npoints
        df = match_larva_ids(df, Npoints=Npoints, dt=dt, **matchID_kws)
    df = constrain_selected_tracks(df, **kwargs)

    e = init_endpoint_dataframe_from_timeseries(df=df, dt=dt)
    s = finalize_timeseries_dataframe(df, complete_ticks=False, interpolate_ticks=interpolate_ticks)
    return s, e


def import_Schleyer(source_dir, save_mode='semifull', **kwargs):
    """
    Builds a larvaworld dataset from Schleyer-lab-specific raw data

    Parameters
    ----------
    source_dir : string
        The folder containing the imported dataset
    save_mode : string
        Mode to define the sequence of columns/parameters to store.
        Defaults to 'semi-full'
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the generate_dataframes function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """

    g = reg.conf.LabFormat.get('Schleyer')
    dt = g.tracker.dt

    if type(source_dir) == str:
        source_dir = [source_dir]

    dfs = []
    for f in source_dir:
        dfs += read_Schleyer_timeseries_from_raw_files_per_larva(dir=f, save_mode=save_mode)

    return generate_dataframes(dfs, dt, **kwargs)


def import_Berni(source_files, **kwargs):
    """
    Builds a larvaworld dataset from Berni-lab-specific raw data

    Parameters
    ----------
    source_files : list
        List of the absolute filepaths of the data files.
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the generate_dataframes function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """
    labID = 'Berni'

    g = reg.conf.LabFormat.get(labID)
    dt = g.tracker.dt
    dfs = read_timeseries_from_raw_files_per_larva(files=source_files, labID=labID)
    return generate_dataframes(dfs, dt, **kwargs)


def import_Arguello(source_files, **kwargs):
    """
    Builds a larvaworld dataset from Arguello-lab-specific raw data

    Parameters
    ----------
    source_files : list
        List of the absolute filepaths of the data files.
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the generate_dataframes function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """

    labID = 'Arguello'

    g = reg.conf.LabFormat.get(labID)
    dt = g.tracker.dt
    dfs = read_timeseries_from_raw_files_per_larva(files=source_files, labID=labID)
    return generate_dataframes(dfs, dt, **kwargs)


lab_specific_import_functions = {
    'Jovanic': import_Jovanic,
    'Berni': import_Berni,
    'Schleyer': import_Schleyer,
    'Arguello':import_Arguello,
}
