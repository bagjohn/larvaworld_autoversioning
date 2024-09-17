"""
Methods for model calibration
"""

import heapq
import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


from .. import reg, aux
from ..aux import nam


__all__ = [
    'vel_definition',
    'comp_stride_variation',
    'fit_metric_definition',
    'comp_segmentation',
]

def comp_orientations(s, e, c, mode='minimal'):
    Np = c.Npoints
    if Np == 1:
        comp_orientation_1point(s, e)
        return

    xy_pars = c.midline_xy
    Axy = s[xy_pars].values

    reg.vprint(f'Computing front/rear body-vector and head/tail orientation angles')
    vector_idx = c.vector_dict

    if mode == 'full':
        reg.vprint(f'Computing additional orients for {c.Nsegs} spinesegments')
        for i, vec in enumerate(c.midline_segs):
            vector_idx[vec] = (i + 1, i)

    for vec, (idx1, idx2) in vector_idx.items():
        par = aux.nam.orient(vec)
        x, y = Axy[:, 2 * idx2] - Axy[:, 2 * idx1], Axy[:, 2 * idx2 + 1] - Axy[:, 2 * idx1 + 1]
        aa = np.arctan2(y, x)
        aa[aa < 0] += 2 * np.pi
        s[par] = aa
        e[aux.nam.initial(par)] = s[par].dropna().groupby('AgentID').first()
    reg.vprint('All orientations computed')


def comp_orientation_1point(s, e):
    def func(ss):
        x, y = ss[:, 0].values, ss[:, 1].values
        dx, dy = np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)
        aa = np.arctan2(dy, dx)
        aa[aa < 0] += 2 * np.pi
        return aa

    return aux.apply_per_level(s[['x', 'y']], func).flatten()

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

def comp_linear(s, e, c, mode='minimal'):
    assert isinstance(c, reg.generators.DatasetConfig)
    points = c.midline_points
    if mode == 'full':
        reg.vprint(f'Computing linear distances, velocities and accelerations for {c.Npoints - 1} points')
        points = points[1:]
        orientations = c.seg_orientations
    elif mode == 'minimal':
        if c.point == 'centroid' or c.point == points[0]:
            reg.vprint('Defined point is either centroid or head. Orientation of front segment not defined.')
            return
        else:
            reg.vprint(f'Computing linear distances, velocities and accelerations for a single spinepoint')
            points = [c.point]
            orientations = ['rear_orientation']

    if not aux.cols_exist(orientations, s):
        reg.vprint('Required orients not found. Component linear metrics not computed.')
        return

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in c.agent_ids]
    dsts = nam.lin(nam.dst(points))
    cum_dsts = nam.cum(nam.lin(dsts))
    vels = nam.lin(nam.vel(points))
    accs = nam.lin(nam.acc(points))

    for p, dst, cum_dst, vel, acc, orient in zip(points, dsts, cum_dsts, vels, accs, orientations):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, data in enumerate(all_d):
            v, d = aux.compute_component_velocity(xy=data[nam.xy(p)].values, angles=data[orient].values, dt=c.dt,
                                                  return_dst=True)
            a = np.diff(v) / c.dt
            cum_d = np.nancumsum(d)
            D[:, i] = d
            Dcum[:, i] = cum_d
            V[:, i] = v
            A[1:, i] = a

        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]
    pars = nam.xy(points) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    reg.vprint('All linear parameters computed')


def comp_spatial(s, e, c, mode='minimal'):
    if mode == 'full':
        reg.vprint(f'Computing distances, velocities and accelerations for {c.Npoints} points', 1)
        points = c.midline_points + ['centroid', '']
    elif mode == 'minimal':
        reg.vprint(f'Computing distances, velocities and accelerations for a single spinepoint', 1)
        points = [c.point, '']
    else:
        raise ValueError(f'{mode} not in supported modes : [minimal, full]')
    points = [p for p in aux.unique_list(points) if nam.xy(p).exist_in(s)]

    dsts = nam.dst(points)
    cum_dsts = nam.cum(dsts)
    vels = nam.vel(points)
    accs = nam.acc(points)

    for p, dst, cum_dst, vel, acc in zip(points, dsts, cum_dsts, vels, accs):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, id in enumerate(c.agent_ids):
            D[:, i] = aux.eudist(s[nam.xy(p)].xs(id, level='AgentID').values)
            Dcum[:, i] = np.nancumsum(D[:, i])
            V[:, i] = D[:, i] / c.dt
            A[1:, i] = np.diff(V[:, i]) / c.dt
        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = s[cum_dst].dropna().groupby('AgentID').last()

    pars = nam.xy(points) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    reg.vprint('All spatial parameters computed')



def process_epochs(a, epochs, dt, return_idx=False):
    if epochs.shape[0] == 0:
        stops = np.array([])
        durs = np.array([])
        slices = []
        amps = np.array([])
        idx = np.array([])
        maxs = np.array([])
        if return_idx:
            return stops, durs, slices, amps, idx, maxs
        else:
            return durs, amps, maxs

    else:
        if epochs.shape == (2,):
            epochs = np.array([epochs, ])
        durs = (np.diff(epochs).flatten()) * dt
        slices = [np.arange(r0, r1, 1) for r0, r1 in epochs]
        amps = np.array([np.trapz(a[p][~np.isnan(a[p])], dx=dt) for p in slices])
        maxs = np.array([np.max(a[p]) for p in slices])
        if return_idx:
            stops = epochs[:, 1]
            idx = np.concatenate(slices) if len(slices) > 1 else slices[0]
            return stops, durs, slices, amps, idx, maxs
        else:
            return durs, amps, maxs

def vel_definition(d) :
    s, e, c = d.data
    assert isinstance(c,reg.generators.DatasetConfig)
    res_v = comp_stride_variation(s, e, c)
    res_fov = comp_segmentation(s, e, c)
    fit_metric_definition(str_var=res_v['stride_variability'], df_corr=res_fov['bend2or_correlation'], c=c)
    dic = {**res_v, **res_fov}
    d.vel_definition=dic
    d.save_config()
    aux.storeH5(dic, key=None, path=f'{d.data_dir}/vel_definition.h5')
    reg.vprint(f'Velocity definition dataset stored.')
    return dic

def comp_stride_variation(s, e, c):


    N = c.Npoints
    points = c.midline_points
    vels = aux.nam.vel(points)
    cvel = aux.nam.vel('centroid')
    lvels = aux.nam.lin(aux.nam.vel(points[1:]))

    all_point_idx = np.arange(N).tolist() + [-1] + np.arange(N).tolist()[1:]
    all_points = points + ['centroid'] + points[1:]
    lin_flag = [False] * N + [False] + [True] * (N - 1)
    all_vels0 = vels + [cvel] + lvels
    all_vels = aux.nam.scal(all_vels0)

    vel_num_strings = ['{' + str(i + 1) + '}' for i in range(N)]
    lvel_num_strings = ['{' + str(i + 2) + '}' for i in range(N - 1)]
    symbols = [rf'$v_{i}$' for i in vel_num_strings] + [r'$v_{cen}$'] + [rf'$v^{"c"}_{i}$' for i in
                                                                         lvel_num_strings]

    markers = ['o' for i in range(len(vels))] + ['s'] + ['v' for i in range(len(lvels))]
    cnum = 1 + N
    cmap0 = plt.get_cmap('hsv')
    cmap0 = [cmap0(1. * i / cnum) for i in range(cnum)]
    cmap0 = cmap0[1:] + [cmap0[0]] + cmap0[2:]

    dic = {all_vels[ii]: {'symbol': symbols[ii], 'marker': markers[ii], 'color': cmap0[ii],
                          'idx': ii, 'par': all_vels0[ii], 'point': all_points[ii], 'point_idx': all_point_idx[ii],
                          'use_component_vel': lin_flag[ii]} for ii in
           range(len(all_vels))}



    if not aux.cols_exist(vels + [cvel],s):
        s[c.centroid_xy] = np.sum(s[c.contour_xy].values.reshape([-1, c.Ncontour, 2]), axis=1) / c.Ncontour
        comp_spatial(s, e, c, mode='full')

    if not aux.cols_exist(lvels,s):
        comp_orientations(s, e, c, mode='full')
        comp_linear(s, e, c, mode='full')

    if not aux.cols_exist(all_vels,s):
        scale_to_length(s, e, c, pars=all_vels0)

    svels = aux.existing_cols(all_vels,s)

    shorts = ['fsv', 'str_N', 'str_tr', 'str_t_mu', 'str_t_std', 'str_sd_mu', 'str_sd_std', 'str_t_var', 'str_sd_var']

    my_index = pd.MultiIndex.from_product([svels, c.agent_ids], names=['VelPar', 'AgentID'])
    df = pd.DataFrame(index=my_index, columns=reg.getPar(shorts))

    for ii in range(c.N):
        id = c.agent_ids[ii]
        ss, ee = s.xs(id, level='AgentID'), e.loc[id]
        for i, vv in enumerate(svels):
            cum_dur = ss[vv].dropna().values.shape[0] * c.dt
            a = ss[vv].values
            fr = aux.fft_max(a, c.dt, fr_range=(1, 2.5))
            strides = aux.detect_strides(a, fr=fr, dt=c.dt)
            if len(strides) == 0:
                row = [fr, 0, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan]
            else:
                strides[:, 1] = strides[:, 1] - 1
                durs, amps, maxs = process_epochs(a, strides, dt=c.dt)
                Nstr = strides.shape[0]

                t_cv = stats.variation(durs)
                s_cv = stats.variation(amps)
                row = [fr, Nstr, np.sum(durs) / cum_dur, np.mean(durs), np.std(durs),
                       np.mean(amps), np.std(amps), t_cv, s_cv]

            df.loc[(vv, id)] = row
    str_var = df[reg.getPar(['str_sd_var', 'str_t_var', 'str_tr'])].astype(float).groupby('VelPar').mean()
    for ii in ['symbol', 'color', 'marker','par', 'idx', 'point', 'point_idx', 'use_component_vel'] :
        str_var[ii]= [dic[jj][ii] for jj in str_var.index.values]
    dic = {'stride_data': df, 'stride_variability': str_var}


    reg.vprint('Stride variability analysis complete!')
    return dic

def fit_metric_definition(str_var, df_corr, c) :
    Nangles=0 if c.Npoints<3 else c.Npoints-2
    sNt_cv = str_var[reg.getPar(['str_sd_var', 'str_t_var'])].sum(axis=1)
    best_idx = sNt_cv.argmin()

    best_combo = df_corr.index.values[0]
    best_combo_max = np.max(best_combo)

    md=c.metric_definition
    if not 'spatial' in md:
        md.spatial=aux.AttrDict()
    idx=md.spatial.point_idx=int(str_var['point_idx'].iloc[best_idx])
    md.spatial.use_component_vel=bool(str_var['use_component_vel'].iloc[best_idx])
    try:
        p = aux.nam.midline(c.Npoints, type='point')[idx - 1]
    except:
        p = 'centroid'
    c.point = p
    if not 'angular' in md:
        md.angular=aux.AttrDict()
    md.angular.best_combo = str(best_combo)
    md.angular.front_body_ratio = best_combo_max / Nangles
    md.angular.bend = 'from_vectors'



def comp_segmentation(s, e, c):
    N = np.clip(c.Npoints - 2, a_min=0, a_max=None)
    angles=[f'angle{i}' for i in range(N)]
    avels = aux.nam.vel(angles)
    hov = aux.nam.vel(aux.nam.orient('front'))


    if not aux.cols_exist(avels,s):
        raise ValueError('Spineangle angular velocities do not exist in step')

    ss = s.loc[s[hov].dropna().index.values]
    y = ss[hov].values

    reg.vprint('Computing linear regression of angular velocity based on segmental bending velocities')
    df_reg = []
    for i in range(N):
        p0 = avels[i]
        X0 = ss[[p0]].values
        reg0 = LinearRegression().fit(X0, y)
        sc0 = reg0.score(X0, y)
        c0 = reg0.coef_[0]
        p1 = avels[:i + 1]
        X1 = ss[p1].values
        reg1 = LinearRegression().fit(X1, y)
        sc1 = reg1.score(X1, y)
        c1 = reg1.coef_[0]

        df_reg.append({'idx': i + 1,
                       'single_par': p0, 'single_score': sc0, 'single_coef': c0,
                       'cum_pars': p1, 'cum_score': sc1, 'cum_coef': c1,
                       })
    df_reg = pd.DataFrame(df_reg)
    df_reg.set_index('idx', inplace=True)

    reg.vprint('Computing correlation of angular velocity with combinations of segmental bending velocities')
    df_corr = []
    for i in range(int(N * 4 / 7)):
        for idx in itertools.combinations(np.arange(N), i + 1):
            if i == 0:
                idx = idx[0]
                idx0 = idx + 1
                ps = avels[idx]
                X = ss[ps].values
            else:
                idx = list(idx)
                idx0 = [ii + 1 for ii in idx]
                ps = [avels[ii] for ii in idx]
                X = ss[ps].sum(axis=1).values
            r, p = stats.pearsonr(y, X)

            df_corr.append({'idx': idx0, 'pars': ps, 'corr': r, 'p-value': p})

    df_corr = pd.DataFrame(df_corr)
    df_corr.set_index('idx', inplace=True)
    df_corr.sort_values('corr', ascending=False, inplace=True)
    dic = {'bend2or_regression': df_reg, 'bend2or_correlation': df_corr}
    reg.vprint('Angular velocity definition analysis complete!')
    return dic


