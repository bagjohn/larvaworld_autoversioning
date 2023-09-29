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
    'interpolate_timeseries_dataframe',
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
    return e


def read_timeseries_from_raw_files_per_parameter(pref, Npoints=None, Ncontour=None):
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
    if Npoints is None:
        g = reg.conf.LabFormat.getID('Jovanic')
        Npoints = g.tracker.Npoints
    if Ncontour is None:
        g = reg.conf.LabFormat.getID('Jovanic')
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


def read_timeseries_from_raw_files_per_larva(files, read_sequence, store_sequence, inv_x=False):
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
    df = df.loc[df[t].groupby(aID).last() - df[t].groupby(aID).first() > min_duration_in_sec]
    if max_Nagents is not None:
        df = df.loc[df['head_x'].dropna().groupby(aID).count().nlargest(max_Nagents).index]
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


def interpolate_timeseries_dataframe(df, dt):
    """
    Interplolates timeseries data with irregular timestep

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

    t = 't'
    step = 'Step'
    aID = 'AgentID'

    s = copy.deepcopy(df)
    s[step] = s[t] / dt
    Nticks = int(np.ceil(s[step].max()))
    s.reset_index(drop=False, inplace=True)
    s.set_index(keys=[step, aID], inplace=True, drop=True, verify_integrity=False)
    s.sort_index(level=[step, aID], inplace=True)
    ids = aux.index_unique(s, level=aID)

    ticks = np.arange(0, Nticks, 1).astype(int)

    my_index = pd.MultiIndex.from_product([ticks, ids], names=[step, aID])
    ps = s.columns
    A = np.zeros([Nticks, ids.shape[0], len(ps)]) * np.nan

    for j, id in enumerate(ids):
        dff = s.xs(id, level=aID, drop_level=True)
        float_ticks_j = dff.index
        ticks_j = np.arange(int(np.floor(float_ticks_j.min())), int(np.ceil(float_ticks_j.max())), 1)
        for i, p in enumerate(ps):
            f = interpolate.interp1d(x=float_ticks_j.values, y=dff[p].loc[float_ticks_j].values,
                                     fill_value='extrapolate',
                                     assume_sorted=True)
            A[ticks_j, j, i] = f(ticks_j)
    A = A.reshape([-1, len(ps)])
    df_new = pd.DataFrame(A, index=my_index, columns=ps)
    df_new.sort_index(level=[step, aID], inplace=True)
    return df_new


def build_Jovanic(dataset, source_id, source_dir, match_ids=True, matchID_kws={}, **kwargs):
    """
    Builds a larvaworld dataset from Jovanic-lab-specific raw data

    Parameters
    ----------
    dataset : lib.process.dataset.LarvaDataset
        The initialized larvaworld dataset
    source_id : string
        The ID of the imported dataset
    source_dir : string
        The folder containing the imported dataset
    match_ids : boolean
        Whether to use the match-ID algorithm
    matchID_kws : dict
        Additional keyword arguments to be passed to the match-ID algorithm.
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """



    d = dataset
    Npoints = d.config.Npoints
    Ncontour = d.config.Ncontour
    print(f'*---- Buiding dataset {d.id} of group {d.group_id}!-----')

    df = read_timeseries_from_raw_files_per_parameter(pref=f'{source_dir}/{source_id}', Npoints=Npoints,
                                                      Ncontour=Ncontour)

    if match_ids:
        df = match_larva_ids(df, Npoints=Npoints, dt=d.dt, **matchID_kws)
    df = constrain_selected_tracks(df, **kwargs)
    e = init_endpoint_dataframe_from_timeseries(df=df, dt=d.dt)
    s = interpolate_timeseries_dataframe(df=df, dt=d.dt)
    print(f'----- Timeseries data for group "{d.id}" of experiment "{d.group_id}" generated.')
    return s, e


def build_Schleyer(dataset, source_dir, save_mode='semifull', **kwargs):
    """
    Builds a larvaworld dataset from Schleyer-lab-specific raw data

    Parameters
    ----------
    dataset : lib.process.dataset.LarvaDataset
        The initialized larvaworld dataset
    source_dir : string
        The folder containing the imported dataset
    save_mode : string
        Mode to define the sequence of columns/parameters to store.
        Defaults to 'semi-full'
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """


    d = dataset

    if type(source_dir) == str:
        source_dir = [source_dir]

    dfs = []
    for f in source_dir:
        dfs += read_Schleyer_timeseries_from_raw_files_per_larva(dir=f, save_mode=save_mode)

    if len(dfs) == 0:
        return None, None

    t0, t1 = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])

    t = 't'
    step = 'Step'
    aID = 'AgentID'
    index_names = [step, aID]

    for i, df in enumerate(dfs):
        df[t] = df.index * d.dt
        df[step] = df.index
        df[aID] = f'Larva_{i}'
        df.set_index(keys=[aID], inplace=True)
    s0 = pd.concat(dfs, axis=0, sort=False)

    s0 = constrain_selected_tracks(s0, **kwargs)

    e = init_endpoint_dataframe_from_timeseries(df=s0, dt=d.dt)
    my_index = pd.MultiIndex.from_product([np.arange(t0, t1 + 1).astype(int), e.index.values.tolist()],
                                          names=index_names)

    s0.reset_index(drop=False, inplace=True)
    s0.set_index(keys=index_names, inplace=True, drop=True, verify_integrity=False)

    ps = [col for col in s0.columns if col != t]

    s = pd.DataFrame(index=my_index, columns=ps)
    s.update(s0[ps])
    s.sort_index(level=index_names, inplace=True)
    # I add this because some 'na' values were found
    s = s.mask(s == 'na', np.nan)

    # t0, t1 = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])
    # df0 = pd.DataFrame(np.nan, index=np.arange(t0, t1 + 1).tolist(), columns=store_sequence)
    # df0.index.name = step
    #
    # for i, df in enumerate(dfs):
    #     ddf = df0.copy(deep=True)
    #     ddf.update(df)
    #     ddf = ddf.assign(AgentID=f'Larva_{i}').set_index(aID, append=True)
    #     if i == 0:
    #         s = ddf
    #     else:
    #         s = pd.concat([s, ddf])

    # s.reset_index(drop=False, inplace=True)
    # s[t] = s[step] * d.dt
    # s.set_index(keys=[t], inplace=True, drop=True, verify_integrity=False)

    # s.reset_index(drop=False, inplace=True)
    # s.set_index(keys=[step, aID], inplace=True, drop=True, verify_integrity=False)
    # s.sort_index(level=[step, aID], inplace=True)
    # for i, f in enumerate(source_dir):
    #     fs = [os.path.join(f, n) for n in os.listdir(f) if n.endswith('.csv')]
    #     raw_fs += fs
    #
    #     if g.filesystem.read_metadata:
    #         try:
    #             inv_xs += get_invert_x_array(read_Schleyer_metadata(f), len(fs))
    #         except:
    #             pass
    # if len(inv_xs) == 0:
    #     inv_xs = [False] * len(raw_fs)

    # Nvalid = 0
    # dfs = []
    # ids = []
    # for f, inv_x in zip(raw_fs, inv_xs):
    #     df = pd.read_csv(f, header=None, index_col=0, names=cols0)
    #
    #     # If indexing is in strings replace with ascending floats
    #     if all([type(ii) == str for ii in df.index.values]):
    #         df.reset_index(inplace=True, drop=True)
    #     if len(df) >= int(min_duration_in_sec / dt) and df.index.max() >= int(min_end_time_in_sec / dt):
    #         df = df[df.index >= int(start_time_in_sec / dt)]
    #         df = df[cols1]
    #         df = df.apply(pd.to_numeric, errors='coerce')
    #         if inv_x:
    #             for x_par in [p for p in cols1 if p.endswith('x')]:
    #                 df[x_par] *= -1
    #         Nvalid += 1
    #         dfs.append(df)
    #         ids.append(f'Larva_{Nvalid}')
    #         if max_Nagents is not None and Nvalid >= max_Nagents:
    #             break
    # if len(dfs) == 0:
    #     return None, None
    # t0, t1 = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])
    # df0 = pd.DataFrame(np.nan, index=np.arange(t0, t1 + 1).tolist(), columns=cols1)
    # df0.index.name = 'Step'
    #
    # end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    # for i, (df, id) in enumerate(zip(dfs, ids)):
    #     ddf = df0.copy(deep=True)
    #
    #     end.loc[i] = {'AgentID': id,
    #                   'num_ticks': len(df),
    #                   'cum_dur': len(df) * dt}
    #     ddf.update(df)
    #     ddf = ddf.assign(AgentID=id).set_index('AgentID', append=True)
    #     if i == 0:
    #         step = ddf
    #     else:
    #         step = pd.concat([step, ddf])
    #
    # end.set_index('AgentID', inplace=True)

    return s, e


def build_Berni(dataset, source_files, **kwargs):
    """
    Builds a larvaworld dataset from Berni-lab-specific raw data

    Parameters
    ----------
    dataset : lib.process.dataset.LarvaDataset
        The initialized larvaworld dataset
    source_files : list
        List of the absolute filepaths of the data files.
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """



    read_sequence, store_sequence = get_column_sequences(labID='Berni', save_mode='full')
    dfs=read_timeseries_from_raw_files_per_larva(files=source_files, read_sequence=read_sequence,
                                             store_sequence=store_sequence)
    if len(dfs) == 0:
        return None, None
    t0, t1 = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])

    t = 't'
    step = 'Step'
    aID = 'AgentID'
    index_names = [step, aID]

    d = dataset

    for i, df in enumerate(dfs):
        df[t] = df.index * d.dt
        df[aID] = f'Larva_{i}'
        df.set_index(keys=[aID], inplace=True, drop=False)
    s0 = pd.concat(dfs, axis=0, sort=False)

    s0 = constrain_selected_tracks(s0, **kwargs)

    e = init_endpoint_dataframe_from_timeseries(df=s0, dt=d.dt)
    my_index = pd.MultiIndex.from_product([np.arange(t0, t1 + 1).astype(int), e.index.values.tolist()],
                                          names=index_names)

    s0.reset_index(drop=False, inplace=True)
    s0.set_index(keys=index_names, inplace=True, drop=True, verify_integrity=False)

    ps = [col for col in s0.columns if col != t]

    s = pd.DataFrame(index=my_index, columns=ps)
    s.update(s0[ps])
    s.sort_index(level=index_names, inplace=True)

    # g = reg.conf.LabFormat.getID('Berni')
    # cols0 = g.filesystem.read_sequence
    # cols1 = cols0[1:]
    #
    # d = dataset
    # dt = d.dt
    # end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    # Nvalid = 0
    # dfs = []
    # ids = []
    # fs = source_files
    # # fs = [os.path.join(source_dir, n) for n in os.listdir(source_dir) if n.startswith(dataset.id)]
    # for f in fs:
    #     df = pd.read_csv(f, header=0, index_col=0, names=cols0)
    #     df.reset_index(drop=True, inplace=True)
    #     if len(df) >= int(min_duration_in_sec / dt) and len(df) >= int(min_end_time_in_sec / dt):
    #         # df = df[df.index >= int(start_time_in_sec / dt)]
    #         df = df[cols1]
    #         df = df.apply(pd.to_numeric, errors='coerce')
    #         Nvalid += 1
    #         dfs.append(df)
    #         ids.append(f'Larva_{Nvalid}')
    #         if max_Nagents is not None and Nvalid >= max_Nagents:
    #             break
    #     if len(dfs) == 0:
    #         return None, None
    # Nticks = np.max([len(df) for df in dfs])
    # df0 = pd.DataFrame(np.nan, index=np.arange(Nticks).tolist(), columns=cols1)
    # df0.index.name = 'Step'
    #
    # for i, (df, id) in enumerate(zip(dfs, ids)):
    #     ddf = df0.copy(deep=True)
    #     end = end.append({'AgentID': id,
    #                       'num_ticks': len(df),
    #                       'cum_dur': len(df) * dt}, ignore_index=True)
    #     ddf.update(df)
    #     ddf = ddf.assign(AgentID=id).set_index('AgentID', append=True)
    #     step = ddf if i == 0 else step.append(ddf)
    # end.set_index('AgentID', inplace=True)
    return s, e


def build_Arguello(dataset, source_files, **kwargs):
    """
    Builds a larvaworld dataset from Arguello-lab-specific raw data

    Parameters
    ----------
    dataset : lib.process.dataset.LarvaDataset
        The initialized larvaworld dataset
    source_files : list
        List of the absolute filepaths of the data files.
   **kwargs: keyword arguments
        Additional keyword arguments to be passed to the constrain_selected_tracks function.


    Returns
    -------
    s : pandas.DataFrame
        The timeseries dataframe
    e : pandas.DataFrame
        The endpoint dataframe
    """

    read_sequence, store_sequence = get_column_sequences(labID='Berni', save_mode='full')
    dfs = read_timeseries_from_raw_files_per_larva(files=source_files, read_sequence=read_sequence,
                                                   store_sequence=store_sequence)
    if len(dfs) == 0:
        return None, None
    t0, t1 = np.min([df.index.min() for df in dfs]), np.max([df.index.max() for df in dfs])

    t = 't'
    step = 'Step'
    aID = 'AgentID'
    index_names = [step, aID]

    d = dataset

    for i, df in enumerate(dfs):
        df[t] = df.index * d.dt
        df[aID] = f'Larva_{i}'
        df.set_index(keys=[aID], inplace=True, drop=False)
    s0 = pd.concat(dfs, axis=0, sort=False)

    s0 = constrain_selected_tracks(s0, **kwargs)

    e = init_endpoint_dataframe_from_timeseries(df=s0, dt=d.dt)
    my_index = pd.MultiIndex.from_product([np.arange(t0, t1 + 1).astype(int), e.index.values.tolist()],
                                          names=index_names)

    s0.reset_index(drop=False, inplace=True)
    s0.set_index(keys=index_names, inplace=True, drop=True, verify_integrity=False)

    ps = [col for col in s0.columns if col != t]

    s = pd.DataFrame(index=my_index, columns=ps)
    s.update(s0[ps])
    s.sort_index(level=index_names, inplace=True)

    # g = reg.conf.LabFormat.getID('Arguello')
    # cols0 = g.filesystem.read_sequence
    # cols1 = cols0[1:]
    # d = dataset
    # dt = d.dt
    # end = pd.DataFrame(columns=['AgentID', 'num_ticks', 'cum_dur'])
    # Nvalid = 0
    # dfs = []
    # ids = []
    #
    # fs = source_files
    # # fs = [os.path.join(source_dir, n) for n in os.listdir(source_dir) if n.startswith(dataset.id)]
    # for f in fs:
    #     df = pd.read_csv(f, header=0, index_col=0, names=cols0)
    #     df.reset_index(drop=True, inplace=True)
    #     if len(df) >= int(min_duration_in_sec / dt) and len(df) >= int(min_end_time_in_sec / dt):
    #         # df = df[df.index >= int(start_time_in_sec / dt)]
    #         df = df[cols1]
    #         df = df.apply(pd.to_numeric, errors='coerce')
    #         Nvalid += 1
    #         dfs.append(df)
    #         ids.append(f'Larva_{Nvalid}')
    #         if max_Nagents is not None and Nvalid >= max_Nagents:
    #             break
    #     if len(dfs) == 0:
    #         return None, None
    # Nticks = np.max([len(df) for df in dfs])
    # df0 = pd.DataFrame(np.nan, index=np.arange(Nticks).tolist(), columns=cols1)
    # df0.index.name = 'Step'
    #
    # for i, (df, id) in enumerate(zip(dfs, ids)):
    #     ddf = df0.copy(deep=True)
    #     # ddf = ddf.interpolate() # DEALING WITH MISSING DATA? DROP OR INTERPOLATE? DEFAULT IS LINEAR.
    #     end = end.append({'AgentID': id,
    #                       'num_ticks': len(df),
    #                       'cum_dur': len(df) * dt}, ignore_index=True)
    #     ddf.update(df)
    #     ddf = ddf.assign(AgentID=id).set_index('AgentID', append=True)
    #     step = ddf if i == 0 else step.append(ddf)
    # end.set_index('AgentID', inplace=True)
    return s, e


lab_specific_build_functions = {
    'Jovanic': build_Jovanic,
    'Berni': build_Berni,
    'Schleyer': build_Schleyer,
    'Arguello': build_Arguello,
}


# if __name__ == "__main__":
#     print('ss')
