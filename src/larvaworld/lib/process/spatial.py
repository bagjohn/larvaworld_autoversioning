"""
Methods for processing spatial metrics in timeseries
"""

import copy
import itertools

import numpy as np
import pandas as pd

from .. import reg, aux
from ..aux import nam

__all__ = [
    'comp_length',
    'align_trajectories',
    'fixate_larva',
    'comp_PI',
    # 'comp_dataPI',
    'scale_to_length',
]




# @reg.funcs.proc("length")
def comp_length(s, e, c, mode='minimal', recompute=False):
    if 'length' in e.columns.values and not recompute:
        reg.vprint('Length is already computed. If you want to recompute it, set recompute_length to True', 1)
        return
    if not c.midline_xy.exist_in(s):
        reg.vprint(f'XY coordinates not found for the {c.Npoints} midline points. Body length can not be computed.', 1)
        return
    xy = s[c.midline_xy].values

    if mode == 'full':
        segs = c.midline_segs
        t = len(s)
        S = np.zeros([c.Nsegs, t]) * np.nan
        L = np.zeros([1, t]) * np.nan
        reg.vprint(f'Computing lengths for {c.Nsegs} segments and total body length', 1)
        for j in range(t):
            for i, seg in enumerate(segs):
                S[i, j] = np.sqrt(np.nansum((xy[j, 2 * i:2 * i + 2] - xy[j, 2 * i + 2:2 * i + 4]) ** 2))
            L[:, j] = np.nansum(S[:, j])
        for i, seg in enumerate(segs):
            s[seg] = S[i, :].flatten()
    elif mode == 'minimal':
        reg.vprint(f'Computing body length')
        xy2 = xy.reshape(xy.shape[0], c.Npoints, 2)
        xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
        L = np.sum(np.sqrt(xy3), axis=1)
    s['length'] = L
    e['length'] = s['length'].groupby('AgentID').quantile(q=0.5)
    reg.vprint('All lengths computed.', 1)






# @reg.funcs.preproc("transposition")
def align_trajectories(s, c, d=None, track_point=None, arena_dims=None, transposition='origin', replace=True, **kwargs):
    if not isinstance(c, reg.generators.DatasetConfig):
        c = reg.generators.DatasetConfig(**c)

    if transposition in ['', None, np.nan]:
        return
    mode = transposition

    xy_flat = c.all_xy.existing(s)
    xy_pairs = xy_flat.in_pairs
    # xy_flat=np.unique(aux.flatten_list(xy_pairs))
    # xy_pairs = aux.group_list_by_n(xy_flat, 2)

    if replace:
        ss = s
    else:
        ss = copy.deepcopy(s[xy_flat])

    if mode == 'arena':
        reg.vprint('Centralizing trajectories in arena center')
        if arena_dims is None:
            arena_dims = c.env_params.arena.dims
        x0, y0 = arena_dims
        X, Y = x0 / 2, y0 / 2

        for x, y in xy_pairs:
            ss[x] -= X
            ss[y] -= Y
        return ss
    else:
        if track_point is None:
            track_point = c.point
        XY = nam.xy(track_point) if aux.cols_exist(nam.xy(track_point), s) else ['x', 'y']
        if not aux.cols_exist(XY, s):
            raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')
        ids = s.index.unique(level='AgentID').values
        Nticks = len(s.index.unique('Step'))
        if mode == 'origin':
            reg.vprint('Aligning trajectories to common origin')
            xy = [s[XY].xs(id, level='AgentID').dropna().values[0] for id in ids]
        elif mode == 'center':
            reg.vprint('Centralizing trajectories in trajectory center using min-max positions')
            xy_max = [s[XY].xs(id, level='AgentID').max().values for id in ids]
            xy_min = [s[XY].xs(id, level='AgentID').min().values for id in ids]
            xy = [(max + min) / 2 for max, min in zip(xy_max, xy_min)]
        else:
            raise ValueError('Supported modes are "arena", "origin" and "center"!')
        xs = np.array([x for x, y in xy] * Nticks)
        ys = np.array([y for x, y in xy] * Nticks)

        for jj, (x, y) in enumerate(xy_pairs):
            ss[x] = ss[x].values - xs
            ss[y] = ss[y].values - ys

        if d is not None:
            d.store(ss, f'traj.{mode}')
            reg.vprint(f'traj_aligned2{mode} stored')
        return ss


def fixate_larva(s, c, P1, P2=None):
    if not isinstance(c, reg.generators.DatasetConfig):
        c = reg.generators.DatasetConfig(**c)

    pars = c.all_xy.existing(s)
    if not nam.xy(P1).exist_in(s):
        raise ValueError(f" The requested {P1} is not part of the dataset")
    reg.vprint(f'Fixing {P1} to arena center')
    X, Y = c.env_params.arena.dims
    xy = s[nam.xy(P1)].values
    xy_start = s[nam.xy(P1)].dropna().values[0]
    bg_x = (xy[:, 0] - xy_start[0]) / X
    bg_y = (xy[:, 1] - xy_start[1]) / Y

    for x, y in pars.in_pairs:
        s[[x, y]] -= xy

    N = s.index.unique('Step').size
    if P2 is not None:
        if not nam.xy(P2).exist_in(s):
            raise ValueError(f" The requested secondary {P2} is not part of the dataset")
        reg.vprint(f'Fixing {P2} as secondary point on vertical axis')
        xy_sec = s[nam.xy(P2)].values
        bg_a = np.arctan2(xy_sec[:, 1], xy_sec[:, 0]) - np.pi / 2

        s[pars] = [
            aux.flatten_list(aux.rotate_points_around_point(points=np.reshape(s[pars].values[i, :], (-1, 2)),
                                                            radians=bg_a[i])) for i in range(N)]
    else:
        bg_a = np.zeros(N)

    bg = np.vstack((bg_x, bg_y, bg_a))
    reg.vprint('Fixed-point dataset generated')

    return s, bg




def comp_PI(arena_xdim, xs, return_num=False):
    N = len(xs)
    r = 0.2 * arena_xdim
    xs = np.array(xs)
    N_l = len(xs[xs <= -r / 2])
    N_r = len(xs[xs >= +r / 2])
    # N_m = len(xs[(xs <= +r / 2) & (xs >= -r / 2)])
    pI = np.round((N_l - N_r) / N, 3)
    if return_num:
        return pI, N
    else:
        return pI





def scale_to_length(s, e, c=None, pars=None, keys=None):
    l_par = 'length'
    if l_par not in e.columns:
        comp_length(s, e, c=c, mode='minimal', recompute=True)
    l = e[l_par]
    if pars is None:
        if keys is not None:
            pars = reg.getPar(keys)
        else:
            raise ValueError('No parameter names or keys provided.')
    s_pars = aux.existing_cols(pars, s)

    if len(s_pars) > 0:
        ids = s.index.get_level_values('AgentID').values
        ls = l.loc[ids].values
        s[nam.scal(s_pars)] = (s[s_pars].values.T / ls).T
    e_pars = aux.existing_cols(pars, e)
    if len(e_pars) > 0:
        e[nam.scal(e_pars)] = (e[e_pars].values.T / l.values).T
