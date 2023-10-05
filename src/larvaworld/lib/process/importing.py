"""
Methods for importing data in lab-specific formats
"""

import os
import os.path
import shutil
import warnings

from .. import reg, aux
from ..process.import_aux import constrain_selected_tracks, finalize_timeseries_dataframe, \
    read_timeseries_from_raw_files_per_parameter, match_larva_ids, init_endpoint_dataframe_from_timeseries, \
    read_timeseries_from_raw_files_per_larva, read_Schleyer_timeseries_from_raw_files_per_larva, generate_dataframes

__all__ = [
    'import_Jovanic',
    'import_Schleyer',
    'import_Berni',
    'import_Arguello',
    'lab_specific_import_functions'
]


# def import_datasets(labID, source_ids, **kwargs):
#     g = reg.conf.LabFormat.get(labID)
#     return g.import_datasets(source_ids=source_ids, **kwargs)
#
#
# def import_dataset(labID, **kwargs):
#     g = reg.conf.LabFormat.get(labID)
#     return g.import_dataset(**kwargs)


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
    'Arguello': import_Arguello,
}
