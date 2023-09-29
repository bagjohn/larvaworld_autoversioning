import os
import os.path
import shutil
import warnings

from .. import reg, aux

from ..process.build_aux import lab_specific_build_functions

__all__ = [
    'import_datasets',
    'import_dataset',
    'build_dataset',
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


def import_dataset(labID, parent_dir, group_id=None, raw_folder=None, proc_folder=None, N=None, id=None,
                   merged=False,
                   refID=None, enrich_conf=None, save_dataset=True, **kwargs):
    """
    Imports a single experimental dataset defined by their ID from a source folder.

    Parameters
    ----------
    labID: string
        The ID of the lab-specific format of the raw files.
    parent_dir: string
        The parent directory where the raw files are located.
    group_id: string, optional
        The group ID of the dataset to be imported.
        If not provided it is set as the parent_dir argument.
    id: string, optional
        The ID under which to store the imported dataset.
        If not provided it is set by default.
    raw_folder: string, optional
        The directory where the raw files are located.
        If not provided it is set as the subfolder 'raw' under the lab-specific group directory.
    proc_folder: string, optional
        The directory where the imported dataset will be placed.
        If not provided it is set as the subfolder 'processed' under the lab-specific group directory.
    refID: string, optional
        The reference IDs under which to store the imported dataset as reference dataset.
        If not provided the dataset is not stored in the reference database.
    N: integer, optional
        The number of larvae in the dataset.
        If provided it also sets the maximum number of larvae 'max_Nagents' allowed in the dataset.
    merged: boolean
        Whether to merge all raw datasets in the source folder in a single imported dataset.
        Defaults to False.
    save_dataset: boolean
        Whether to store the imported dataset to disc.
        Defaults to True.
    enrich_conf: dict, optional
        The configuration for enriching the imported dataset with secondary parameters.
    **kwargs: keyword arguments
        Additional keyword arguments to be passed to the build_dataset function.

    Returns
    -------
    lib.process.dataset.LarvaDataset
        The imported dataset in the common larvaworld format.
    """

    print()
    print(f'----- Initializing {labID} format-specific dataset import. -----')

    if group_id is None:
        group_id = parent_dir
    if id is None:
        id = f'{labID}_{group_id}_dataset'

    g = reg.conf.LabFormat.get(labID)
    if raw_folder is None:
        raw_folder = f'{g.path}/raw'
    source_dir = f'{raw_folder}/{parent_dir}'
    if proc_folder is None:
        proc_folder = f'{g.path}/processed'
    target_dir = f'{proc_folder}/{group_id}/{id}'

    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    kws = {
        'labID': labID,
        'group_id': group_id,
        'Ν': N,
        'target_dir': target_dir,
        'source_dir': source_dir,
        'max_Nagents': N,
        **kwargs
    }
    d = build_dataset(id=id, **kws)

    if d is not None:
        print(f'***-- Dataset {d.id} created with {len(d.config.agent_ids)} larvae! -----')
        print(f'****- Processing dataset {d.id} to derive secondary metrics -----')
        if enrich_conf is None:
            enrich_conf = reg.gen.EnrichConf(proc_keys=[], anot_keys=[]).nestedConf
        enrich_conf['pre_kws'] = g.preprocess.nestedConf
        d = d.enrich(**enrich_conf, is_last=False)
        if save_dataset:
            d.save(refID=refID)
            if refID is not None:
                print(f'***** Dataset stored under the reference ID : {refID} -----')
            else:
                print(f'***** Dataset stored -----')
    else:
        print(f'xxxxx Failed to create dataset {id}! -----')
    return d


def build_dataset(labID, id, group_id, target_dir, N=None, sample=None,
                  color='black', epochs={}, age=0.0, **kwargs):
    """
    Converts experimental data to a single larvaworld dataset according to a lab-specific data format.

    Parameters
    ----------
    labID: string
        The ID of the lab-specific format of the raw files.
    id: string
        The ID under which to store the imported dataset.
    group_id: string
        The group ID of the dataset to be imported.
    target_dir: string
        The directory where the new dataset will be placed.
    N: integer, optional
        The number of larvae in the dataset.
        If provided it also sets the maximum number of larvae 'max_Nagents' allowed in the dataset.
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
    **kwargs: keyword arguments
        Additional keyword arguments to be passed to the lab-specific build function.

    Returns
    -------
    lib.process.dataset.LarvaDataset
        The imported dataset in the common larvaworld format.
    """

    print(f'*---- Building dataset {id} under the {labID} format. -----')
    warnings.filterwarnings('ignore')

    shutil.rmtree(target_dir, ignore_errors=True)
    g = reg.conf.LabFormat.getID(labID)

    conf = {
        'load_data': False,
        'dir': target_dir,
        'id': id,
        'larva_groups': reg.config.lg(id=group_id, c=color, sample=sample, mID=None, N=N, epochs=epochs, age=age),
        'env_params': g.env_params,
        **g.tracker
    }
    from ..process.dataset import LarvaDataset
    d = LarvaDataset(**conf)
    step, end = lab_specific_build_functions[labID](dataset=d, **kwargs)
    d.set_data(step=step, end=end)
    return d
