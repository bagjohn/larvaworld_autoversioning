import copy
from scipy import interpolate
import os
import os.path
import numpy as np
import pandas as pd

from .. import aux, reg
from ..aux import nam

__all__ = [
    'init_endpoint_dataframe_from_timeseries',
    'read_timeseries_from_raw_files_per_parameter',
    'read_timeseries_from_raw_files_per_larva',
    'read_Schleyer_timeseries_from_raw_files_per_larva',
    'constrain_selected_tracks',
    'match_larva_ids',
    'match_larva_ids_including_by_length',
    'get_column_sequences',
    'concatenate_larva_tracks',
    'complete_timeseries_with_nans',
    'empty_2index_timeseries_df',
    'interpolate_timeseries_dataframe',
    'finalize_timeseries_dataframe',
    'generate_dataframes',
    'build_Jovanic',
    'build_Schleyer',
    'build_Berni',
    'build_Arguello',
    'lab_specific_build_functions'
]


def init_endpoint_dataframe_from_timeseries(df, dt):
    """
    Initializes a dataframe for endpoint metrics from a timeseries dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The timeseries dataframe
    dt : float
        The timeseries timestep in seconds


    Returns
    -------
    pandas.DataFrame
    """

    g = df['t'].groupby(level='AgentID')
    t0, t1, Nts, cum_t = reg.getPar(['t0', 't_fin', 'N_ts', 'cum_t'])
    tick0, tick1, Nticks = reg.getPar(['tick0', 'tick_fin', 'N_ticks'])
    e = pd.concat(dict(zip([t0, t1, Nts], [g.first(), g.last(), g.count()])), axis=1)
    e[cum_t] = e[t1] - e[t0]
    e['dt'] = e[cum_t] / (e[Nts] - 1)
    e[tick1] = np.ceil(e[t1] / dt).astype(int)
    e[tick0] = np.floor(e[t0] / dt).astype(int)
    e[Nticks] = e[tick1] - e[tick0]
    e.sort_index(inplace=True)
    reg.vprint(f'**--- Endpoint dataframe initialized -----', 1)
    return e


def read_timeseries_from_raw_files_per_parameter(pref, labID='Jovanic', Npoints=None, Ncontour=None):
    """
    Reads timeseries data stored in txt files of the lab-specific Jovanic format and returns them as a pd.Dataframe.

    Parameters
    ----------
    pref : string
        The prefix used for detecting the txt files.
        This includes the absolute folder the files are located in plus any filename prefix unique to the specific dataset's files.
    Npoints : integer, optional
        The number of midline points tracked for each larva.
        If not provided it is set to the lab-format's default value
    Ncontour : integer, optional
        The number of contour points tracked for each larva.
        If not provided it is set to the lab-format's default value

    Returns
    -------
    pandas.DataFrame
    """

    # Retrieve lab-specific default number of midline and contour points if not provided
    if Npoints is None and labID is not None:
        g = reg.conf.LabFormat.getID(labID)
        Npoints = g.tracker.Npoints
    if Ncontour is None and labID is not None:
        g = reg.conf.LabFormat.getID(labID)
        Ncontour = g.tracker.Ncontour

    t = 't'
    aID = 'AgentID'

    # Read the txt files setting each as a column of the dataframe
    kws = {'header': None, 'sep': '\t'}
    par_list = [pd.read_csv(f'{pref}_{suf}.txt', **kws) for suf in ['larvaid', 't', 'x_spine', 'y_spine']]

    columns = [aID, t] + aux.nam.xy(aux.nam.midline(Npoints, type='point'), xsNys=True, flat=True)
    try:
        states = pd.read_csv(f'{pref}_state.txt', **kws)
        par_list.append(states)
        columns.append('state')
    except:
        pass

    if Ncontour > 0:
        try:
            xcs = pd.read_csv(f'{pref}_x_contour.txt', **kws)
            ycs = pd.read_csv(f'{pref}_y_contour.txt', **kws)
            xcs, ycs = aux.convex_hull(xs=xcs.values, ys=ycs.values, N=Ncontour)
            xcs = pd.DataFrame(xcs, index=None)
            ycs = pd.DataFrame(ycs, index=None)
            par_list += [xcs, ycs]
            columns += aux.nam.xy(aux.nam.contour(Ncontour), xsNys=True, flat=True)
        except:
            pass

    df = pd.concat(par_list, axis=1, sort=False)
    df.columns = columns
    df.set_index(keys=[aID], inplace=True, drop=True)
    return df


def read_timeseries_from_raw_files_per_larva(files, labID=None, save_mode='full', read_sequence=None,
                                             store_sequence=None, inv_x=False):
    """
    Reads timeseries data stored in txt files of the lab-specific Jovanic format and returns them as a pd.Dataframe.

    Parameters
    ----------
    files : list
        List of the absolute filepaths of the data files.
    labID : string, optional
        The ID of the lab-specific format to use
    save_mode : string
        The mode defining the columns to store
        Used if store_sequence is not provided
    read_sequence : list of strings, optional
        The sequence of parameters found in each file
        If not provided it is set to the lab-format's default value
    store_sequence : list of strings, optional
        The sequence of parameters to store
        If not provided it is set to the lab-format's default value
    inv_x : boolean
        Whether to invert x axis.
        Defaults to False

    Returns
    -------
    list of pandas.DataFrame
    """
    if (read_sequence is None or store_sequence is None) and labID is not None:
        read_sequence, store_sequence = get_column_sequences(labID, save_mode)

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


def read_Schleyer_timeseries_from_raw_files_per_larva(dir, save_mode='semifull'):
    labID = 'Schleyer'
    read_sequence, store_sequence = get_column_sequences(labID, save_mode)
    g = reg.conf.LabFormat.getID(labID)
    if g.filesystem.read_metadata:

        def read_Schleyer_metadata(dir):
            d = {}
            with open(os.path.join(dir, 'vidAndLogs/metadata.txt')) as f:
                for j, line in enumerate(f):
                    try:
                        nb, list = line.rstrip('\n').split('=')
                        d[nb] = list
                    except:
                        pass
            return d

        # def get_invert_x_array(meta_dict, Nfiles):
        #     try:
        #         odor_side = meta_dict['OdorA_Side']
        #         if odor_side == 'right':
        #             invert_x_array = [True for i in range(Nfiles)]
        #         elif odor_side == 'left':
        #             invert_x_array = [False for i in range(Nfiles)]
        #         else:
        #             raise ValueError(f'Odor side found in metadata is not consistent : {odor_side}')
        #         return invert_x_array
        #     except:
        #         print('Odor side not found in metadata. Assuming left side')
        #         invert_x_array = [False for i in range(Nfiles)]
        #         return invert_x_array

        def get_odor_pos(meta_dict, arena_dims):
            ar_x, ar_y = arena_dims
            try:
                odor_side = meta_dict['OdorA_Side']
                x, y = meta_dict['OdorALocation'].split(',')
                x, y = float(x), float(y)
                # meta_dict.pop('OdorALocation', None)
                # meta_dict['OdorPos']=[x,y]
                x, y = 2 * x / ar_x, 2 * y / ar_y
                if odor_side == 'left':
                    return [x, y]
                elif odor_side == 'right':
                    return [-x, y]
            except:
                return [-0.8, 0]

        try:
            odor_side = read_Schleyer_metadata(dir)['OdorA_Side']
            if odor_side == 'right':
                inv_x = True
            elif odor_side == 'left':
                inv_x = False
            else:
                raise ValueError(f'Odor side found in metadata is not consistent : {odor_side}')
        except:
            inv_x = False
    else:
        inv_x = False
    fs = [os.path.join(dir, n) for n in os.listdir(dir) if n.endswith('.csv')]
    return read_timeseries_from_raw_files_per_larva(files=fs, read_sequence=read_sequence,
                                                    store_sequence=store_sequence, inv_x=inv_x)


def constrain_selected_tracks(df, max_Nagents=None, time_slice=None, min_duration_in_sec=0.0, **kwargs):
    """
    Applies constraints to the tracks included in timeseries data.

    Parameters
    ----------
    df : pandas.DataFrame
        The timeseries dataframe
    max_Nagents : integer, optional
        The maximum number of larvae allowed in the dataset.
    time_slice : tuple, optional
        Use only a defined time slice of the tracs in seconds.
    min_duration_in_sec : float
        Only include tracks longer than a given duration in seconds.
        Defaults to 0.0


    Returns
    -------
    pandas.DataFrame
    """

    aID = 'AgentID'
    t = 't'

    if time_slice is not None:
        tmin, tmax = time_slice
        df = df[df[t] < tmax]
        df = df[df[t] >= tmin]
        reg.vprint(f'**--- Tracks constrained within {time_slice} seconds -----', 1)
    if min_duration_in_sec!=0 :
        df = df.loc[df[t].groupby(aID).last() - df[t].groupby(aID).first() > min_duration_in_sec]
        reg.vprint(f'**--- Tracks required to last at least {min_duration_in_sec} seconds -----', 1)
    if max_Nagents is not None:
        df = df.loc[df['head_x'].dropna().groupby(aID).count().nlargest(max_Nagents).index]
        reg.vprint(f'**--- Number of tracks limited to {max_Nagents} larvae -----', 1)
    df.sort_index(inplace=True)
    return df


def match_larva_ids_including_by_length(s, e, pars=['head_x', 'head_y'], wl=100, wt=0.1, ws=0.5, max_error=600, Nidx=20,
                                        verbose=1):
    """
    Applies a matching-ID algorithm to concatenate segmented tracs.

    Parameters
    ----------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    pars : list
        The spatial parameters to use for computing vincinity.
        Defaults to ['head_x', 'head_y']
    wl : float
        Coefficient for body-length similarity.
        Defaults to 100
    wt : float
        Coefficient for temporal vincinity.
        Defaults to 0.1
    max_error : float
        Maximum accepted error
        Defaults to 600
    Nidx : integer
        Closest number of neighboring tracs to chec.
        Defaults to 20


    Returns
    -------
    pandas.DataFrame
    """

    t = 't'
    aID = 'AgentID'
    # s.reset_index(level=t, drop=False, inplace=True)
    s[t] = s[t].values.astype(float)

    pairs = {}

    def common_member(a, b):
        a_set = set(a)
        b_set = set(b)
        return a_set & b_set

    def eval(t0, xy0, l0, t1, xy1, l1):
        tt = t1 - t0
        if tt <= 0:
            return max_error * 2
        ll = np.abs(l1 - l0)
        dd = np.sqrt(np.sum((xy1 - xy0) ** 2))
        # print(tt,ll,dd)
        return wt * tt + wl * ll + ws * dd

    def get_extrema(ss, pars):
        ids = ss.index.unique().tolist()

        mins = ss[t].groupby(aID).min()
        maxs = ss[t].groupby(aID).max()
        durs = ss[t].groupby(aID).count()
        first_xy, last_xy = {}, {}
        for id in ids:
            first_xy[id] = ss[pars].xs(id).dropna().values[0, :]
            last_xy[id] = ss[pars].xs(id).dropna().values[-1, :]
        return ids, mins, maxs, first_xy, last_xy, durs

    def update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy):
        mins[id1], first_xy[id1] = mins[id0], first_xy[id0]
        del mins[id0]
        del maxs[id0]
        del first_xy[id0]
        del last_xy[id0]
        ids.remove(id0)
        return ids, mins, maxs, first_xy, last_xy

    ls = e['length']

    ids, mins, maxs, first_xy, last_xy, durs = get_extrema(s, pars)
    Nids0 = len(ids)
    while Nidx <= len(ids):
        cur_er, id0, id1 = max_error, None, None
        t0s = maxs.nsmallest(Nidx)
        t1s = mins.loc[mins > t0s.min()].nsmallest(Nidx)
        if len(t1s) > 0:
            for i in range(Nidx):
                cur_id0, t0 = t0s.index[i], t0s.values[i]
                xy0, l0 = last_xy[cur_id0], ls[cur_id0]
                ee = [eval(t0, xy0, l0, mins[id], first_xy[id], ls[id]) for id in t1s.index]
                temp_err = np.min(ee)
                if temp_err < cur_er:
                    cur_er, id0, id1 = temp_err, cur_id0, t1s.index[np.argmin(ee)]
        if id0 is not None:
            pairs[id0] = id1
            ls[id1] = (ls[id0] * durs[id0] + ls[id1] * durs[id1]) / (durs[id0] + durs[id1])
            durs[id1] += durs[id0]
            del durs[id0]
            ls.drop([id0], inplace=True)
            ids, mins, maxs, first_xy, last_xy = update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy)
            if verbose >= 2:
                print(len(ids), int(cur_er))
        else:
            Nidx += 1
    Nids1 = len(ids)
    if verbose >= 2:
        print('Finalizing dataset')
    while len(common_member(list(pairs.keys()), list(pairs.values()))) > 0:
        for id0, id1 in pairs.items():
            if id1 in pairs.keys():
                pairs[id0] = pairs[id1]
                break
    s.rename(index=pairs, inplace=True)
    # s.reset_index(drop=False, inplace=True)
    # s.set_index(keys=[t, aID], inplace=True, drop=True)
    if verbose >= 1:
        print(f'**--- Track IDs reduced from {Nids0} to {Nids1} by the matchIDs algorithm -----')
    return s


def comp_length(df, e, Npoints):
    xys = aux.nam.xy(aux.nam.midline(Npoints, type='point'), flat=True)
    xy2 = df[xys].values.reshape(-1, Npoints, 2)
    xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
    df['length'] = np.sum(np.sqrt(xy3), axis=1)
    e['length'] = df['length'].groupby('AgentID').quantile(q=0.5)


def match_larva_ids(df, Npoints, dt, e=None, **kwargs):
    """
    Computes larval body-lengths and then applies a matching-ID algorithm to concatenate segmented tracks.

    Parameters
    ----------
    df : pandas.DataFrame
        The timeseries dataframe
    Npoints : integer
        The number of midline points tracked for each larva.
    dt : float
        The timeseries timestep in seconds
    e : pandas.DataFrame, optional
        The endpoint dataframe
        If not provided it is initialized from df
    **kwargs: keyword arguments
        Additional keyword arguments to be passed to the match_larva_ids_including_by_length function.


    Returns
    -------
    pandas.DataFrame
    """

    if e is None:
        e = init_endpoint_dataframe_from_timeseries(df=df, dt=dt)
    comp_length(df, e, Npoints=Npoints)
    df = match_larva_ids_including_by_length(s=df, e=e, **kwargs)
    return df


def get_column_sequences(labID, save_mode='semifull'):
    """
    Define the column sequence in the raw data to be imported and the imported data to be stored

    Parameters
    ----------
    labID : string
        The ID of the lab-specific format to use
    save_mode : string
        The mode defining the columns to store

    Returns
    -------
    c1 : list of strings
        The sequence of parameters found in each file
    c2 : list of strings
        The sequence of parameters to store
    """
    g = reg.conf.LabFormat.getID(labID)
    c1 = g.filesystem.read_sequence
    if save_mode == 'full':
        c2 = c1[1:]
    elif save_mode == 'minimal':
        c2 = nam.xy(g.tracker.point)
    elif save_mode == 'semifull':
        c2 = nam.midline_xy(g.tracker.Npoints, flat=True) + nam.contour_xy(g.tracker.Ncontour, flat=True) + [
            'collision_flag']
    elif save_mode == 'points':
        c2 = nam.xy(g.tracker.points, flat=True) + ['collision_flag']
    else:
        raise
    return c1, c2


def concatenate_larva_tracks(dfs, dt):
    """
    Concatenate multiple single tracks to a single dataframe

    Parameters
    ----------
    dfs : list of pd.DataFrame
        The single tracks
    dt : float
        The tracking timestep

    Returns
    -------
    pd.DataFrame

    """
    t = 't'
    step = 'Step'
    aID = 'AgentID'

    for i, df in enumerate(dfs):
        df[t] = df.index * dt
        df[step] = df.index
        df[aID] = f'Larva_{i}'
        df.set_index(keys=[aID], inplace=True)
    s = pd.concat(dfs, axis=0, sort=False)
    # I add this because some 'na' values were found
    s = s.mask(s == 'na', np.nan)
    return s


def complete_timeseries_with_nans(s0):
    """
    Fill the non-existing ticks with nans

    Parameters
    ----------
    s0 : pd.DataFrame
        The timeseries dataframe

    Returns
    -------
    pd.DataFrame

    """


    s = empty_2index_timeseries_df(s0)
    s.update(s0)
    return s


def empty_2index_timeseries_df(s0):
    """
    Generate an empty dataframe with complete ticks based on an existing

    Parameters
    ----------
    s0 : pd.DataFrame
        The timeseries dataframe

    Returns
    -------
    pd.DataFrame

    """

    step = 'Step'
    aID = 'AgentID'
    index_names = [step, aID]
    idx = s0.index

    ticks = idx.unique(step).values
    trange = np.arange(int(np.floor(np.min(ticks))), int(np.ceil(np.max(ticks))), 1).astype(int)
    my_index = pd.MultiIndex.from_product([trange, idx.unique(aID).values], names=index_names)

    s = pd.DataFrame(index=my_index, columns=s0.columns)
    return s


def finalize_timeseries_dataframe(s, complete_ticks=True, interpolate_ticks=False):
    """
    Finalize the timeseries dataframe setting the double-index

    Parameters
    ----------
    s : pd.DataFrame
        The timeseries dataframe
    complete_ticks : boolean
        Whether to complete timeseries missing ticks with nans
        Defaults to False
    interpolate_ticks : boolean
        Whether to interpolate timeseries into a fixed timestep timeseries
        Defaults to False

    Returns
    -------
    pd.DataFrame

    """

    step = 'Step'
    aID = 'AgentID'
    index_names = [step, aID]

    s.reset_index(drop=False, inplace=True)
    s.set_index(keys=index_names, inplace=True, drop=True, verify_integrity=False)

    if complete_ticks:
        s = complete_timeseries_with_nans(s)
    if interpolate_ticks:
        s = interpolate_timeseries_dataframe(s)

    s.sort_index(level=index_names, inplace=True)
    return s


def generate_dataframes(dfs, dt, complete_ticks=True, **kwargs):
    # """
    # Helper function that connects other
    #
    # Parameters
    # ----------
    # s : pd.DataFrame
    #     The timeseries dataframe
    # complete_ticks : boolean
    #     Whether to complete timeseries missing ticks with nans
    #     Defaults to False
    # interpolate_ticks : boolean
    #     Whether to interpolate timeseries into a fixed timestep timeseries
    #     Defaults to False
    #
    # Returns
    # -------
    # pd.DataFrame
    #
    # """

    if len(dfs) == 0:
        return None, None
    s0 = concatenate_larva_tracks(dfs, dt)
    s0 = constrain_selected_tracks(s0, **kwargs)

    e = init_endpoint_dataframe_from_timeseries(df=s0, dt=dt)

    # s0 = s0[[col for col in s0.columns if col != 't']]
    s = finalize_timeseries_dataframe(s0, complete_ticks=complete_ticks)
    return s, e


def interpolate_timeseries_dataframe(s0):
    """
    Interplolate irregular-timestep timeseries to regular-timestep

    Parameters
    ----------
    s0 : pd.DataFrame
        The timeseries dataframe

    Returns
    -------
    pd.DataFrame

    """


    s = empty_2index_timeseries_df(s0)
    aID = 'AgentID'
    ids=s.index.unique(aID).values
    Nids=ids.shape[0]
    ticks = s.index.unique('Step').values
    tick0, tick1=np.min(ticks), np.max(ticks)
    Nticks = ticks.shape[0]
    ps=s.columns
    Nps=len(ps)
    A=np.zeros([Nticks,Nids,Nps])*np.nan


    for i,id in enumerate(ids):
        dff = s0.xs(id, level=aID, drop_level=True)
        idx = dff.index.values
        t0, t1 = int(np.floor(np.min(idx))), int(np.ceil(np.max(idx)))
        if t0<tick0:
            t0=tick0
        if t1>tick1:
            t1=tick1
        ts = np.arange(t0, t1, 1)
        for j, p in enumerate(ps):
            f = interpolate.interp1d(x=idx, y=dff[p].values,
                                     fill_value='extrapolate',
                                     assume_sorted=True)
            A[ts-tick0,i,j] = f(ts)
            # s[p].loc[(ts, id)] = f(ts)
    A=A.reshape(-1, Nps)
    s=pd.DataFrame(A, index=s.index, columns=s.columns)
    reg.vprint(f'**--- Timeseries dataframe interpolated -----', 1)
    return s


def build_Jovanic(source_id, source_dir, match_ids=True, matchID_kws={},interpolate_ticks=True, **kwargs):
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
    Npoints = g.tracker.Npoints

    df = read_timeseries_from_raw_files_per_parameter(pref=f'{source_dir}/{source_id}')

    if match_ids:
        df = match_larva_ids(df, Npoints=Npoints, dt=dt, **matchID_kws)
    df = constrain_selected_tracks(df, **kwargs)
    df['Step'] = df['t'] / dt
    e = init_endpoint_dataframe_from_timeseries(df=df, dt=dt)

    s = finalize_timeseries_dataframe(df, complete_ticks=False, interpolate_ticks=interpolate_ticks)
    # s = interpolate_timeseries_dataframe(df=df, dt=dt)
    # print(f'----- Timeseries data for group "{d.id}" of experiment "{d.group_id}" generated.')
    return s, e


def build_Schleyer(source_dir, save_mode='semifull', **kwargs):
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


def build_Berni(source_files, **kwargs):
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


def build_Arguello(source_files, **kwargs):
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


lab_specific_build_functions = {
    'Jovanic': build_Jovanic,
    'Berni': build_Berni,
    'Schleyer': build_Schleyer,
    'Arguello': build_Arguello,
}
