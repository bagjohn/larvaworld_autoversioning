"""
Basic classes for larvaworld-format datasets
"""

import copy
import itertools
import os
import shutil
import numpy as np
import pandas as pd
import warnings

import param

from .. import reg, aux
from ..aux import nam
from ..param import ClassAttr, StepDataFrame, EndpointDataFrame, ClassDict

__all__ = [
    'ParamLarvaDataset',
    'BaseLarvaDataset',
    'LarvaDataset',
    'LarvaDatasetCollection',
    'convert_group_output_to_dataset',
    'h5_kdic',
]


class ParamLarvaDataset(param.Parameterized):
    config = ClassAttr(reg.generators.DatasetConfig, doc='The dataset metadata')
    step_data = StepDataFrame(doc='The timeseries data')
    endpoint_data = EndpointDataFrame(doc='The endpoint data')
    config2 = ClassDict(default=aux.AttrDict(), item_type=None, doc='Additional dataset metadata')

    def __init__(self, **kwargs):
        if 'config' not in kwargs.keys():
            kws = aux.AttrDict()
            for k in reg.generators.DatasetConfig().param_keys:
                if k in kwargs.keys():
                    kws[k] = kwargs[k]
                    kwargs.pop(k)
            kwargs['config'] = reg.generators.DatasetConfig(**kws)
        assert 'config2' not in kwargs.keys()

        ks = list(kwargs.keys())
        kws2 = aux.AttrDict()
        for k in ks:
            if k not in self.param.objects().keys():
                kws2[k] = kwargs[k]
                kwargs.pop(k)
        kwargs['config2'] = aux.AttrDict(kws2)
        super().__init__(**kwargs)
        self.merge_configs()
        self.epoch_dict = aux.AttrDict({'pause': None, 'run': None})
        self.larva_dicts = {}
        self.__dict__.update(self.config.nestedConf)

    # @param.depends('step_data', 'endpoint_data', watch=True)
    def validate_IDs(self):
        if self.step_data is not None and self.endpoint_data is not None:
            s1 = self.step_data.index.unique('AgentID').tolist()
            s2 = self.endpoint_data.index.values.tolist()
            assert len(s1) == len(s2)
            assert set(s1) == set(s2)
            assert s1 == s2
            self.config.agent_ids = s1

    # @param.depends('config.agent_ids', watch=True)
    def update_ids_in_data(self):
        s, e = None, None
        if self.step_data is not None:
            s = self.step_data.loc[(slice(None), self.config.agent_ids), :]
        if self.endpoint_data is not None:
            e = self.endpoint_data.loc[self.config.agent_ids]
        self.set_data(step=s, end=e)

    @param.depends('step_data', watch=True)
    def update_Nticks(self):
        self.config.Nticks = self.step_data.index.unique('Step').size
        self.config.duration = self.config.dt * self.config.Nticks / 60

    @property
    def s(self):
        return self.step_data

    @property
    def e(self):
        return self.endpoint_data

    @property
    def end_ps(self):
        return aux.SuperList(self.endpoint_data.columns).sorted

    @property
    def step_ps(self):
        return aux.SuperList(self.step_data.columns).sorted

    @property
    def end_ks(self):
        return aux.SuperList(reg.getPar(self.end_ps, to_return='k')).sorted

    @property
    def step_ks(self):
        return aux.SuperList(reg.getPar(self.step_ps, to_return='k')).sorted

    @property
    def c(self):
        return self.config

    @property
    def min_tick(self):
        return self.step_data.index.unique('Step').min()

    def timeseries_slice(self, time_range=None, df=None):
        if df is None :
            df=self.step_data
        if time_range is None:
            return df
        else :

            t0, t1 = time_range
            s0 = int(t0 / self.config.dt)
            s1 = int(t1 / self.config.dt)
            df_slice = df.loc[(slice(s0, s1), slice(None)), :]
            return df_slice


    def interpolate_nan_values(self):
        s, e, c = self.data
        pars=c.all_xy.existing(s)
        Npars=len(pars)
        for id in c.agent_ids:
            A=np.zeros([c.Nticks, Npars])
            ss=s.xs(id, level='AgentID')
            for i,p in enumerate(pars):
                A[:,i]=aux.interpolate_nans(ss[p].values)
            s.loc[(slice(None), id), pars] = A
        reg.vprint('All parameters interpolated', 1)

    def filter(self, filter_f=2.0, recompute=False):
        s, e, c = self.data
        assert isinstance(filter_f, float)
        if c.filtered_at is not None and not recompute:
            reg.vprint(
                f'Dataset already filtered at {c.filtered_at}. To apply additional filter set recompute to True',
                1)
            return
        c.filtered_at = filter_f

        pars = c.all_xy.existing(s)
        data = np.dstack(list(s[pars].groupby('AgentID').apply(pd.DataFrame.to_numpy))).astype(None)
        f_array = aux.apply_filter_to_array_with_nans_multidim(data, freq=filter_f, fr=1 / c.dt)
        for j, p in enumerate(pars):
            s[p] = f_array[:, j, :].flatten()
        reg.vprint(f'All spatial parameters filtered at {filter_f} Hz', 1)

    def rescale(self, recompute=False, rescale_by=1.0):
        s, e, c = self.data
        assert isinstance(rescale_by, float)
        if c.rescaled_by is not None and not recompute:
            reg.vprint(
                f'Dataset already rescaled by {c.rescaled_by}. To rescale again set recompute to True', 1)
            return
        c.rescaled_by = rescale_by
        points = c.midline_points + ['centroid', '']
        pars = c.all_xy + nam.dst(points) + nam.vel(points) + nam.acc(points) + ['length']
        for p in aux.existing_cols(pars, s):
            s[p] = s[p].apply(lambda x: x * rescale_by)
        if 'length' in e.columns:
            e['length'] = e['length'].apply(lambda x: x * rescale_by)
        reg.vprint(f'Dataset rescaled by {rescale_by}.', 1)

    def exclude_rows(self, flag='collision_flag', accepted=[0], rejected=None):
        s, e, c = self.data
        if accepted is not None:
            s.loc[s[flag] != accepted[0]] = np.nan
        if rejected is not None:
            s.loc[s[flag] == rejected[0]] = np.nan
        for id in c.agent_ids:
            e.loc[id, 'cum_dur'] = len(s.xs(id, level='AgentID', drop_level=True).dropna()) * c.dt
        reg.vprint(f'Rows excluded according to {flag}.', 1)

    def align_trajectories(self, track_point=None, arena_dims=None, transposition='origin', replace=True):
        s, e, c = self.data

        assert transposition in ['arena', 'origin', 'center']
        mode = transposition

        xy_flat = c.all_xy.existing(s)
        xy_pairs = xy_flat.in_pairs

        if replace:
            ss = s
        else:
            ss = copy.deepcopy(s[xy_flat])

        if mode == 'arena':
            reg.vprint('Centralizing trajectories in arena center')
            if arena_dims is None:
                arena_dims = c.env_params.arena.dims
            for x, y in xy_pairs:
                ss[x] -= arena_dims[0] / 2
                ss[y] -= arena_dims[1] / 2
            return ss
        else:
            if track_point is None:
                track_point = c.point
            XY = nam.xy(track_point) if aux.cols_exist(nam.xy(track_point), s) else c.traj_xy
            if not aux.cols_exist(XY, s):
                raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')
            ids = c.agent_ids
            if mode == 'origin':
                reg.vprint('Aligning trajectories to common origin')
                xy = [s[XY].xs(id, level='AgentID').dropna().values[0] for id in ids]
            elif mode == 'center':
                reg.vprint('Centralizing trajectories in trajectory center using min-max positions')
                xy = [(s[XY].xs(id, level='AgentID').max().values - s[XY].xs(id, level='AgentID').min().values) / 2 for
                      id in ids]
            else:
                raise ValueError('Supported modes are "arena", "origin" and "center"!')
            xs = np.array([x for x, y in xy] * c.Nticks)
            ys = np.array([y for x, y in xy] * c.Nticks)

            for x, y in xy_pairs:
                ss[x] = ss[x].values - xs
                ss[y] = ss[y].values - ys

            # if d is not None:
            #     d.store(ss, f'traj.{mode}')
            #     reg.vprint(f'traj_aligned2{mode} stored')
            return ss

    def preprocess(self, drop_collisions=False, interpolate_nans=False, filter_f=None, rescale_by=None,
                   transposition=None, recompute=False):
        if drop_collisions:
            self.exclude_rows()
        if interpolate_nans:
            self.interpolate_nan_values()
        if filter_f is not None:
            self.filter(filter_f=filter_f, recompute=recompute)
        if rescale_by is not None:
            self.rescale(rescale_by=rescale_by, recompute=recompute)
        if transposition is not None:
            self.align_trajectories(transposition=transposition)

    def merge_configs(self):
        d = param.guess_param_types(**self.config2)
        for n, p in d.items():
            self.config.param.add_parameter(n, p)

    def set_data(self, step=None, end=None, agents=None, **kwargs):
        if step is not None:
            self.step_data = step.sort_index(level=self.param.step_data.levels)
        if end is not None:
            self.endpoint_data = end.sort_index()
        if agents is not None:
            self.larva_dicts = get_larva_dicts(agents, validIDs=self.config.agent_ids)
        self.validate_IDs()

    @property
    def data(self):
        if self.step_data is None or self.endpoint_data is None :
            self.load()
        return self.step_data, self.endpoint_data, self.config

    def path_to_file(self, file='data'):
        return f'{self.config.data_dir}/{file}.h5'

    @property
    def path_to_config(self):
        return f'{self.config.data_dir}/conf.txt'

    def store(self, df, key, file='data'):
        path = self.path_to_file(file)
        if not isinstance(df, pd.DataFrame):
            pd.DataFrame(df).to_hdf(path, key)
        else:
            df.to_hdf(path, key)

    def read(self, key, file='data'):
        path = self.path_to_file(file)
        try:
            return pd.read_hdf(path, key)
        except:
            return None

    def load(self, step=True, h5_ks=None):
        s = self._load_step(h5_ks=h5_ks) if step else None
        e = self.read('end')
        self.set_data(step=s, end=e)

    def _load_step(self, h5_ks=None):
        s = self.read('step')
        if h5_ks is None:
            h5_ks = list(self.config.h5_kdic.keys())
        for h5_k in h5_ks:
            ss = self.read(h5_k)
            if ss is not None:
                ps = aux.nonexisting_cols(ss.columns.values, s)
                if len(ps) > 0:
                    s = s.join(ss[ps])
        return s

    def _save_step(self, s):
        s = s.loc[:, ~s.columns.duplicated()]
        stored_ps = []
        for h5_k, ps in self.config.h5_kdic.items():
            pps = ps.unique.existing(s)
            if len(pps) > 0:
                self.store(s[pps], h5_k)
                stored_ps += pps

        self.store(s.drop(stored_ps, axis=1, errors='ignore'), 'step')

    def save(self, refID=None):
        if self.step_data is not None:
            self._save_step(s=self.step_data)
        if self.endpoint_data is not None:
            self.store(self.endpoint_data, 'end')
        self.save_config(refID=refID)
        reg.vprint(f'***** Dataset {self.config.id} stored.-----', 1)

    def save_config(self, refID=None):
        c = self.config
        if refID is not None:
            c.refID = refID
        if c.refID is not None:
            reg.conf.Ref.setID(c.refID, c.dir)
            reg.vprint(f'Saved reference dataset under : {c.refID}', 1)
        aux.save_dict(c.nestedConf, self.path_to_config)

    def load_traj(self, mode='default'):
        key = f'traj.{mode}'
        df = self.read(key)
        if df is None:
            if mode == 'default':
                df = self._load_step(h5_ks=[])[['x', 'y']]
            elif mode in ['origin', 'center']:
                s = self._load_step(h5_ks=['contour', 'midline'])
                df = reg.funcs.preprocessing["transposition"](s, c=self.config, replace=False, transposition=mode)[
                    ['x', 'y']]
            else:
                raise ValueError('Not implemented')
            self.store(df, key)
        return df

    def load_dicts(self, type, ids=None):
        if ids is None:
            ids = self.config.agent_ids
        ds0 = self.larva_dicts
        if type in ds0 and all([id in ds0[type].keys() for id in ids]):
            ds = [ds0[type][id] for id in ids]
        else:
            ds = aux.loadSoloDics(agent_ids=ids, path=f'{self.config.data_dir}/individuals/{type}.txt')
        return ds

    @property
    def contour_xy_data_byID(self):
        xy = self.config.contour_xy
        assert xy.exist_in(self.step_data)
        grouped = self.step_data[xy].groupby('AgentID')
        return aux.AttrDict({id: df.values.reshape([-1, self.config.Ncontour, 2]) for id, df in grouped})

    @property
    def midline_xy_data_byID(self):
        xy = self.config.midline_xy
        assert xy.exist_in(self.step_data)
        grouped = self.step_data[xy].groupby('AgentID')
        return aux.AttrDict({id: df.values.reshape([-1, self.config.Npoints, 2]) for id, df in grouped})

    @property
    def traj_xy_data_byID(self):
        s = self.step_data
        xy = self.config.traj_xy
        if not xy.exist_in(s):
            xy0 = self.config.point_xy
            assert xy0.exist_in(s)
            s[xy] = s[xy0]
        assert xy.exist_in(s)
        return self.data_by_ID(s[xy])

    def data_by_ID(self, data):
        grouped = data.groupby('AgentID')
        return aux.AttrDict({id: df.values for id, df in grouped})


    @property
    def midline_xy_data(self):
        return self.step_data[self.config.midline_xy].values.reshape([-1, self.config.Npoints, 2])

    @property
    def contour_xy_data(self):
        return self.step_data[self.config.contour_xy].values.reshape([-1, self.config.Ncontour, 2])

    def empty_df(self, dim3=1):
        c = self.config
        if dim3 == 1:
            return np.zeros([c.Nticks, c.N]) * np.nan
        elif dim3 > 1:
            return np.zeros([c.Nticks, c.N, dim3]) * np.nan

    def apply_per_agent(self, pars, func, time_range=None, **kwargs):
        """
        Apply a function to each subdataframe of a MultiIndex DataFrame after grouping by the agentID.

        Parameters:
        ----------
        s : pandas.DataFrame
            A MultiIndex DataFrame with levels ['Step', 'AgentID'].
        func : function
            The function to apply to each subdataframe.

        **kwargs : dict
            Additional keyword arguments to pass to the 'func' function.

        Returns:
        -------
        numpy.ndarray
            An array of dimensions [N_ticks, N_ids], where N_ticks is the number of unique 'Step' values,
            and N_ids is the number of unique 'AgentID' values.

        Notes:
        -----
        This function groups the DataFrame 's' by the specified 'level', applies 'func' to each subdataframe, and
        returns the results as a numpy array.
        """
        level = 'AgentID'
        s = self.timeseries_slice(time_range)[pars]
        Nt = s.index.unique('Step').size
        s0 = s.index.unique('Step').min()-self.min_tick

        A = None

        for i, (v, ss) in enumerate(s.groupby(level=level)):

            ss = ss.droplevel(level)
            Ai = func(ss, **kwargs)
            if A is None:
                A = self.empty_df(dim3=len(Ai.shape))
            A[s0:s0 + Nt, i] = Ai
        return A

    def midline_xy_1less(self, mid):
        mid2 = copy.deepcopy(mid[:, :-1, :])
        for i in range(mid.shape[1] - 1):
            mid2[:, i, :] = (mid[:, i, :] + mid[:, i + 1, :]) / 2
        return mid2

    @property
    def midline_seg_xy_data_byID(self):
        g = self.midline_xy_data_byID
        return aux.AttrDict({id: self.midline_xy_1less(mid) for id, mid in g.items()})

    @property
    def midline_seg_orients_data_byID(self):
        g = self.midline_xy_data_byID
        return aux.AttrDict({id: self.midline_seg_orients_from_mid(mid) for id, mid in g.items()})

    def midline_seg_orients_from_mid(self, mid):
        Ax, Ay = mid[:, :, 0], mid[:, :, 1]
        Adx = np.diff(Ax)
        Ady = np.diff(Ay)
        return np.arctan2(Ady, Adx) % (2 * np.pi)

    # def comp_seg_orientations(self):
    #     s, e, c = self.data
    #     mid = self.midline_xy_data
    #     seg_ors = self.midline_seg_orients_from_mid(mid)
    #     return seg_ors

    def comp_orientations(self):
        s, e, c = self.data
        mid = self.midline_xy_data
        s[c.seg_orientations] = self.midline_seg_orients_from_mid(mid)
        # or_pars=c.seg_orientations
        for vec, (idx1, idx2) in c.vector_dict.items():
            par = aux.nam.orient(vec)
            x, y = mid[:, idx2, 0] - mid[:, idx1, 0], mid[:, idx2, 1] - mid[:, idx1, 1]
            s[par] = np.arctan2(y, x) % 2 * np.pi
            # or_pars.append(par)
        # for p in or_pars:
        #     p_unw = aux.nam.unwrap(p)
        #     s[p_unw] = aux.apply_per_level(s[p], aux.unwrap_deg).flatten()

        # ss = s[p_unw]
        # e[nam.initial(par)] = s[par].dropna().groupby('AgentID').first()

    def comp_angular(self, is_last=False):
        self.comp_orientations()
        self.comp_bend()
        self.comp_ang_moments()
        if is_last:
            self.save()

    def comp_bend(self):
        s, e, c = self.data
        if c.bend == 'from_vectors':
            reg.vprint(f'Computing bending angle as the difference between front and rear orients')
            fo, ro = nam.orient(['front', 'rear'])
            a = np.remainder(s[fo] - s[ro], 2 * np.pi)
            a[a > np.pi] -= 2 * np.pi
        elif c.bend == 'from_angles':
            reg.vprint(f'Computing bending angle as the sum of the first {c.Nbend_angles} front angles')
            Ada = np.diff(s[c.seg_orientations]) % (2 * np.pi)
            Ada[Ada > np.pi] -= 2 * np.pi
            a = np.sum(Ada[:, :c.Nbend_angles], axis=1)
            s[c.angles] = Ada
        else:
            raise

        s['bend'] = np.degrees(a)

    def comp_ang_moments(self, pars=None):
        s, e, c = self.data
        if pars is None:
            ho, to, fo, ro = nam.orient(['head', 'tail', 'front', 'rear'])
            if c.Npoints > 1:
                base_pars = ['bend', ho, to, fo, ro]
                pars = base_pars + c.angles + c.seg_orientations
            else:
                pars = [ho]

        pars = aux.existing_cols(aux.unique_list(pars), s)

        for p in pars:
            vel = nam.vel(p)
            acc = nam.acc(p)
            # ss = s[p]
            if p.endswith('orientation'):
                p_unw = nam.unwrap(p)
                s[p_unw] = self.apply_per_agent(pars=p, func=aux.unwrap_deg).flatten()
                pp = p_unw
            else:
                pp = p
            s[vel] = self.apply_per_agent(pars=pp, func=aux.rate, dt=c.dt).flatten()
            s[acc] = self.apply_per_agent(pars=vel, func=aux.rate, dt=c.dt).flatten()

            self.comp_operators(pars=[p, vel, acc])

            # if p in ['bend', ho, to, fo, ro]:
            #     for pp in [p, vel, acc]:
            #         temp = s[pp].dropna().groupby('AgentID')
            #         e[aux.nam.mean(pp)] = temp.mean()
            #         e[aux.nam.std(pp)] = temp.std()
            #         e[aux.nam.initial(pp)] = temp.first()
            # s[[aux.nam.min(pp), aux.nam.max(pp)]] = comp_extrema_solo(sss, dt=dt, **kwargs).reshape(-1, 2)

    def comp_xy_moments(self, point=''):
        s, e, c = self.data
        xy = nam.xy(point)
        if not xy.exist_in(s) :
            return

        dst = nam.dst(point)
        vel = nam.vel(point)
        acc = nam.acc(point)
        cdst = nam.cum(dst)

        sdst = nam.scal(dst)
        svel = nam.scal(vel)
        csdst = nam.cum(sdst)


        s[dst] = self.apply_per_agent(pars=xy, func=aux.eudist).flatten()
        s[vel] = s[dst] / c.dt
        s[acc] = self.apply_per_agent(pars=vel, func=aux.rate, dt=c.dt).flatten()

        self.scale_to_length(pars=[dst, vel, acc])

        s[cdst] = s[dst].groupby('AgentID').cumsum()
        s[csdst] = s[sdst].groupby('AgentID').cumsum()

        e[cdst] = s[dst].dropna().groupby('AgentID').sum()
        e[nam.mean(vel)] = s[vel].dropna().groupby('AgentID').mean()

        e[csdst] = s[sdst].dropna().groupby('AgentID').sum()
        e[nam.mean(svel)] = s[svel].dropna().groupby('AgentID').mean()

    def comp_tortuosity(self, dur=20, **kwargs):
        from ..process.spatial import rolling_window, straightness_index
        s, e, c = self.data
        p = reg.getPar(f'tor{dur}')
        w = int(dur / c.dt / 2)
        ticks = np.arange(c.Nticks)
        s[p] = self.apply_per_agent(pars=['x', 'y', 'dst'], func=straightness_index,
                                    rolling_ticks=rolling_window(ticks, w), **kwargs).flatten()
        e[nam.mean(p)] = s[p].groupby('AgentID').mean()
        e[nam.std(p)] = s[p].groupby('AgentID').std()

    def comp_dispersal(self, t0=0, t1=60, **kwargs):
        s, e, c = self.data
        p = reg.getPar(f'dsp_{int(t0)}_{int(t1)}')
        s[p] = self.apply_per_agent(pars=c.traj_xy, func=aux.compute_dispersal_solo, time_range=(t0, t1),
                                    **kwargs).flatten()
        self.scale_to_length(pars=[p])
        sp = nam.scal(p)
        self.comp_operators(pars=[p, sp])

    def comp_operators(self, pars):
        s, e, c = self.data
        for p in pars:
            g = s[p].dropna().groupby('AgentID')
            e[nam.max(p)] = g.max()
            e[nam.mean(p)] = g.mean()
            e[nam.std(p)] = g.std()
            e[nam.initial(p)] = g.first()
            e[nam.final(p)] = g.last()
            e[nam.cum(p)] = g.sum()

    def comp_centroid(self):
        c = self.config
        if c.Ncontour>0 :
            self.step_data[c.centroid_xy] = np.sum(self.contour_xy_data, axis=1) / c.Ncontour

    def comp_length(self):
        self.step_data['length'] = np.sum(np.sum(np.diff(self.midline_xy_data, axis=1) ** 2, axis=2) ** (1 / 2), axis=1)
        self.endpoint_data['length'] = self.step_data['length'].groupby('AgentID').quantile(q=0.5)

    def comp_spatial(self):
        self.comp_centroid()
        self.comp_length()
        self.step_data[self.config.traj_xy] = self.step_data[self.config.point_xy]
        self.comp_operators(pars=self.config.traj_xy)
        for point in ['', 'centroid']:
            self.comp_xy_moments(point)

    def scale_to_length(self, pars=None, keys=None):
        s, e, c = self.data
        l_par = 'length'
        if l_par not in e.keys():
            self.comp_length()
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

    def comp_source_metrics(self):
        s, e, c = self.data
        fo = reg.getPar('fo')
        for n, pos in c.source_xy.items():
            reg.vprint(f'Computing bearing and distance to {n} based on xy position')
            o, d = nam.bearing_to(n), nam.dst_to(n)
            pabs = nam.abs(o)
            temp = np.array(pos) - s[c.traj_xy].values
            s[o] = (s[fo] + 180 - np.rad2deg(np.arctan2(temp[:, 1], temp[:, 0]))) % 360 - 180
            s[pabs] = s[o].abs()
            s[d] = aux.eudi5x(s[c.traj_xy].values, pos)
            self.comp_operators(pars=[d, pabs])
            if 'length' in e.columns:
                l = e['length']

                def rowIndex(row):
                    return row.name[1]

                def rowLength(row):
                    return l.loc[rowIndex(row)]

                def rowFunc(row):
                    return row[d] / rowLength(row)

                sd = nam.scal(d)
                s[sd] = s.apply(rowFunc, axis=1)
                self.comp_operators(pars=[sd])

            reg.vprint('Bearing and distance to source computed')

    def comp_wind(self):
        w = self.config.env_params.windscape
        if w is not None:
            wo=w.wind_direction
            woo = np.deg2rad(wo)
            try:
                self.comp_wind_metrics(woo,wo)
            except:
                self.comp_final_anemotaxis(woo)

    def comp_wind_metrics(self,woo,wo):
        s, e, c = self.data
        for id in c.agent_ids:
            xy = s[c.traj_xy].xs(id, level='AgentID', drop_level=True).values
            origin = e[[nam.initial('x'), nam.initial('y')]].loc[id]
            d = aux.eudi5x(xy, origin)
            dx = xy[:, 0] - origin[0]
            dy = xy[:, 1] - origin[1]
            angs = np.arctan2(dy, dx)
            a = np.array([aux.angle_dif(ang, woo) for ang in angs])
            s.loc[(slice(None), id), 'anemotaxis'] = d * np.cos(a)
        s[nam.bearing_to('wind')] = s.apply(lambda r: aux.angle_dif(r[nam.orient('front')], wo), axis=1)
        e['anemotaxis'] = s['anemotaxis'].groupby('AgentID').last()

    def comp_final_anemotaxis(self,woo):
        s, e, c = self.data
        xy0 = s[c.traj_xy].groupby('AgentID').first()
        xy1 = s[c.traj_xy].groupby('AgentID').last()
        dx = xy1.values[:, 0] - xy0.values[:, 0]
        dy = xy1.values[:, 1] - xy0.values[:, 1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        angs = np.arctan2(dy, dx)
        a = np.array([aux.angle_dif(ang, woo) for ang in angs])
        e['anemotaxis'] = d * np.cos(a)

    def comp_PI2(self, xys, x=0.04):
        Nticks = xys.index.unique('Step').size
        ids = xys.index.unique('AgentID').values
        N = len(ids)
        dLR = np.zeros([N, Nticks]) * np.nan
        for i, id in enumerate(ids):
            xy = xys.xs(id, level='AgentID').values
            dL = aux.eudi5x(xy, [-x, 0])
            dR = aux.eudi5x(xy, [x, 0])
            dLR[i, :] = (dR - dL) / (2 * x)
        dLR_mu = np.mean(dLR, axis=1)
        mu_dLR_mu = np.mean(dLR_mu)
        return mu_dLR_mu

    def comp_PI(self, arena_xdim, xs, return_num=False):
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

    def comp_dataPI(self):
        s, e, c = self.data
        if 'x' in e.keys():
            xs = e['x'].values
        elif nam.final('x') in e.keys():
            xs = e[nam.final('x')].values
        elif 'x' in s.keys():
            xs = s['x'].dropna().groupby('AgentID').last().values
        elif 'centroid_x' in s.keys():
            xs = s['centroid_x'].dropna().groupby('AgentID').last().values
        else:
            raise ValueError('No x coordinate found')
        PI, N = self.comp_PI(xs=xs, arena_xdim=c.env_params.arena.dims[0], return_num=True)
        c.PI = {'PI': PI, 'N': N}
        try:
            c.PI2 = self.comp_PI2(xys=s[nam.xy('')])
        except:
            pass

    def process(self, proc_keys=['angular', 'spatial'],
                dsp_starts=[0], dsp_stops=[40, 60], tor_durs=[5, 10, 20], is_last=False):
        if 'angular' in proc_keys:
            self.comp_angular()
        if 'spatial' in proc_keys:
            self.comp_spatial()
        for t0, t1 in itertools.product(dsp_starts, dsp_stops):
            self.comp_dispersal(t0, t1)
        for dur in tor_durs:
            self.comp_tortuosity(dur)
        if 'source' in proc_keys:
            self.comp_source_metrics()
        if 'wind' in proc_keys:
            self.comp_wind()
        if 'PI' in proc_keys:
            self.comp_dataPI()
        if is_last:
            self.save()

    def get_par(self, par=None, k=None, key='step'):
        s, e = self.step_data, self.endpoint_data
        if par is None and k is not None:
            par = reg.getPar(k)

        def read_key(key, par):
            res = self.read(key)[par]
            if res is not None:
                return res

        if key == 'end':
            if e is not None and par in e.columns:
                return e[par]
        if key == 'step':
            if s is not None and par in s.columns:
                return s[par]
            else:
                for h5_k, ps in self.config.h5_kdic.items():
                    if par in ps:
                        try:
                            return read_key(h5_k, par)
                        except:
                            pass
        try:
            return read_key(key, par)
        except:
            if k is None:
                k = reg.getPar(p=par, to_return='k')
            return reg.par.get(k=k, d=self, compute=True)


class BaseLarvaDataset(ParamLarvaDataset):

    @staticmethod
    def initGeo(to_Geo=False, **kwargs):
        if to_Geo:
            try:
                from ..process.dataset_geo import GeoLarvaDataset
                return GeoLarvaDataset(**kwargs)
            except:
                pass
            # from larvaworld.lib.process.dataset import LarvaDataset
        return LarvaDataset(**kwargs)

    def __init__(self, dir=None, refID=None, load_data=True, config=None, step=None, end=None, agents=None,
                 initialize=False, **kwargs):
        '''
        Dataset class that stores a single experiment, real or simulated.
        Metadata and configuration parameters are stored in the 'config' dictionary.
        This can be provided as an argument, retrieved from a stored experiment or generated for a new experiment.

        The default pipeline goes as follows :
        The dataset needs the config file to be initialized. If it is not provided as an argument there are two ways to retrieve it.
        First if "dir" is an existing directory of a stored dataset the config file will be loaded from the default location
        within the dataset's file structure, specifically from a "conf.txt" in the "data" directory under the root "dir".
        As it is not handy to provide an absolute directory as an argument, the root "dir" locations of a number of stored "reference" datasets
        are stored in a file and loaded as a dictionary where the keys are unique "refID" strings holding the root "dir" value.

        Accessing the reference path dictionary is extremely easy through the "reg.stored" registry class with the following methods :
        -   getRefDir(id) returns the "root" directory stored in the "larvaworld/lib/reg/confDicts/Ref.txt" under the unique id
        -   getRef(id=None, dir=None) returns the config dictionary stored at the "root" directory. Accepts either the "dir" path or the "refID" id
        -   loadRef(id) first retrieves the config dictionary and then initializes the dataset.
            By setting load_data=True there is an attempt to load the data from the disc if found at the provided root directory.

        In the case that none of the above attempts yielded a config dictionary, a novel one is generated using any additional keyword arguments are provided.
        This is the default way that a new dataset is initialized. The data content is set after initialization via the "set_data(step, end)"
        method with which we provide the both the step-wise timeseries and the endpoint single-per-agent measurements

        Endpoint measurements are loaded always as a pd.Dataframe 'endpoint_data' with 'AgentID' indexing

        The timeseries data though can be initialized and processed in two ways :
        -   in the default mode  a pd.Dataframe 'step_data' with a 2-level index : 'Step' for the timestep index and 'AgentID' for the agent unique ID.
            Data is stored as a single HDF5 file or as nested dictionaries. The core file is 'data.h5' with keys like 'step' for timeseries and 'end' for endpoint metrics.
        -   in the trajectory mode a "movingpandas.TrajectoryCollection" is adjusted to the needs of the larva-tracking data format via the
            "lib.process.GeoLarvaDataset" class

        Args:
            dir: Path to stored data. Ignored if 'config' is provided. Defaults to None for no storage to disc
            load_data: Whether to load stored data
            config: The metadata dictionary. Defaults to None for attempting to load it from disc or generate a new.
            **kwargs: Any arguments to store in a novel configuration dictionary
        '''
        if initialize:
            assert config is None
            kws = {
                'dir': dir,
                'refID': refID,
                # 'config':config,
                **kwargs
            }
        else:
            if config is None:
                try:
                    config = reg.conf.Ref.getRef(dir=dir, id=refID)
                    config.update(**kwargs)
                except:
                    config = self.generate_config(dir=dir, refID=refID, **kwargs)
            kws = config

        super().__init__(**kws)

        if load_data:
            self.load()
        elif step is not None or end is not None:
            self.set_data(step=step, end=end, agents=agents)

    # def set_data(self, step=None, end=None,**kwargs):
    #     pass

    def generate_config(self, **kwargs):
        c0 = aux.AttrDict({'id': 'unnamed',
                           'group_id': None,
                           'refID': None,
                           'dir': None,
                           'fr': None,
                           'dt': None,
                           'duration': None,
                           'Nsteps': None,
                           'Npoints': 3,
                           'Ncontour': 0,
                           'u': 'm',
                           'x': 'x',
                           'y': 'y',
                           'sample': None,
                           'color': None,
                           'metric_definition': None,
                           'env_params': {},
                           'larva_groups': {},
                           'source_xy': {},
                           'life_history': None,
                           })

        c0.update(kwargs)
        if c0.dt is not None:
            c0.fr = 1 / c0.dt
        if c0.fr is not None:
            c0.dt = 1 / c0.fr
        if c0.metric_definition is None:
            c0.metric_definition = reg.par.get_null('metric_definition')

        points = aux.nam.midline(c0.Npoints, type='point')

        try:
            c0.point = points[c0.metric_definition.spatial.point_idx - 1]
        except:
            c0.point = 'centroid'

        if len(c0.larva_groups) == 1:
            c0.group_id, gConf = list(c0.larva_groups.items())[0]
            c0.color = gConf['default_color']
            c0.sample = gConf['sample']
            c0.model = gConf['model']
            c0.life_history = gConf['life_history']

        reg.vprint(f'Generated new conf {c0.id}', 1)
        return c0

    # @property
    # def data_dir(self):
    #     return f'{self.config.dir}/data'
    #
    # @property
    # def plot_dir(self):
    #     return f'{self.config.dir}/plots'

    # def save_config(self, refID=None):
    #     c = self.config
    #     if refID is not None:
    #         c.refID = refID
    #     if c.refID is not None:
    #         reg.conf.Ref.setID(c.refID, c.dir)
    #         reg.vprint(f'Saved reference dataset under : {c.refID}', 1)
    #     try:
    #         for k, v in c.items():
    #             if isinstance(v, np.ndarray):
    #                 c[k] = v.tolist()
    #     except:
    #         pass
    #     try:
    #         aux.save_dict(c, f'{self.data_dir}/conf.txt')
    #     except:
    #         aux.save_dict(c.nestedConf, f'{self.data_dir}/conf.txt')

    # @property
    # def Nangles(self):
    #     return np.clip(self.config.Npoints - 2, a_min=0, a_max=None)

    # @property
    # def points(self):
    #     return aux.aux.nam.midline(self.config.Npoints, type='point')

    # @property
    # def contour(self):
    #     return aux.aux.nam.contour(self.config.Ncontour)

    def delete(self):
        shutil.rmtree(self.config.dir)
        reg.vprint(f'Dataset {self.id} deleted', 2)

    def set_id(self, id, save=True):
        self.id = id
        self.config.id = id
        if save:
            self.save_config()

    # def set_endpoint_data(self,end):
    #     self.endpoint_data = end.sort_index()
    #     self.agent_ids = self.endpoint_data.index.values
    #     self.config.agent_ids = list(self.agent_ids)
    #     self.config.N = len(self.agent_ids)

    # def load(self, **kwargs):
    #     pass


class LarvaDataset(BaseLarvaDataset):
    def __init__(self, **kwargs):
        '''
        This is the default dataset class. Timeseries are stored as a pd.Dataframe 'step_data' with a 2-level index : 'Step' for the timestep index and 'AgentID' for the agent unique ID.
        Data is stored as a single HDF5 file or as nested dictionaries. The core file is 'data.h5' with keys like 'step' for timeseries and 'end' for endpoint metrics.
        To lesser the burdain of loading and saving all timeseries parameters as columns in a single pd.Dataframe, the most common parameters have been split in a set of groupings,
         available via keys that access specific entries of the "data.h5". The keys of "self.h5_kdic" dictionary store the parameters that every "h5key" keeps :
        -   'contour': The contour xy coordinates,
        -   'midline': The midline xy coordinates,
        -   'epochs': The behavioral epoch detection and annotation,
        -   'base_spatial': The most basic spatial parameters,
        -   'angular': The angular parameters,
        -   'dspNtor':  Dispersal and tortuosity,

        All parameters not included in any of these groups stays with the original "step" key that is always saved and loaded
        '''
        super().__init__(**kwargs)

    # def set_data(self, step=None, end=None, agents=None):
    #     c=self.config
    #     if step is not None:
    #         assert step.index.names == ['Step', 'AgentID']
    #         s = step.sort_index(level=['Step', 'AgentID'])
    #         self.Nticks = s.index.unique('Step').size
    #         # c.t0 = int(s.index.unique('Step')[0])
    #         c.Nticks = self.Nticks
    #         c.duration = c.dt * c.Nticks/60
    #         # if 'quality' not in c.keys():
    #         #     try:
    #         #         df = s[aux.aux.nam.xy(c.point)[0]].values.flatten()
    #         #         valid = np.count_nonzero(~np.isnan(df))
    #         #         c.quality = np.round(valid / df.shape[0], 2)
    #         #     except:
    #         #         pass
    #
    #         self.step_data = s
    #
    #     if end is not None:
    #         self.set_endpoint_data(end)
    #
    #     if agents is not None :
    #         self.larva_dicts = aux.get_larva_dicts(agents, validIDs=self.agent_ids)

    # def _load_step(self, h5_ks=None):
    #     s = self.read('step')
    #     if h5_ks is None :
    #         h5_ks=list(self.h5_kdic.keys())
    #     for h5_k in h5_ks:
    #         ss = self.read(h5_k)
    #         if ss is not None :
    #             ps = aux.nonexisting_cols(ss.columns.values,s)
    #             if len(ps) > 0:
    #                 s = s.join(ss[ps])
    #     return s
    #
    #
    # def load(self, step=True, h5_ks=None):
    #     s = self._load_step(h5_ks=h5_ks) if step else None
    #     e = self.read('end')
    #     self.set_data(step=s, end=e)
    #
    #
    # def _save_step(self, s):
    #     s = s.loc[:, ~s.columns.duplicated()]
    #     stored_ps = []
    #     for h5_k,ps in self.h5_kdic.items():
    #         pps = aux.unique_list(aux.existing_cols(ps,s))
    #         if len(pps) > 0:
    #             self.store(s[pps], h5_k)
    #             stored_ps += pps
    #
    #     self.store(s.drop(stored_ps, axis=1, errors='ignore'), 'step')
    #
    # def save(self, refID=None):
    #     if hasattr(self, 'step_data'):
    #         self._save_step(s=self.step_data)
    #     if hasattr(self, 'endpoint_data'):
    #         self.store(self.endpoint_data, 'end')
    #     self.save_config(refID=refID)
    #     reg.vprint(f'***** Dataset {self.id} stored.-----', 1)
    #
    # def store(self, df, key, file='data'):
    #     path=f'{self.data_dir}/{file}.h5'
    #     if not isinstance(df, pd.DataFrame):
    #         pd.DataFrame(df).to_hdf(path, key)
    #     else :
    #         df.to_hdf(path, key)
    #
    #
    # def read(self, key, file='data'):
    #     path=f'{self.data_dir}/{file}.h5'
    #     try :
    #         return pd.read_hdf(path, key)
    #     except:
    #         return None
    #
    #
    #
    #
    # def load_traj(self, mode='default'):
    #     key=f'traj.{mode}'
    #     df = self.read(key)
    #     if df is None :
    #         if mode=='default':
    #             df = self._load_step(h5_ks=[])[['x', 'y']]
    #         elif mode in ['origin', 'center']:
    #             s = self._load_step(h5_ks=['contour', 'midline'])
    #             df=reg.funcs.preprocessing["transposition"](s, c=self.config, replace=False, transposition=mode)[['x', 'y']]
    #         else :
    #             raise ValueError('Not implemented')
    #         self.store(df,key)
    #     return df
    #
    #
    #
    # def load_dicts(self, type, ids=None):
    #     if ids is None:
    #         ids = self.agent_ids
    #     ds0 = self.larva_dicts
    #     if type in ds0 and all([id in ds0[type].keys() for id in ids]):
    #         ds = [ds0[type][id] for id in ids]
    #     else:
    #         ds= aux.loadSoloDics(agent_ids=ids, path=f'{self.data_dir}/individuals/{type}.txt')
    #     return ds

    def visualize(self, parameters={}, **kwargs):
        from ..sim.dataset_replay import ReplayRun
        kwargs['dataset'] = self
        rep = ReplayRun(parameters=parameters, **kwargs)
        rep.run()

    # @property
    # def data_path(self):
    #     return f'{self.data_dir}/data.h5'

    def enrich(self, pre_kws={}, proc_keys=[], anot_keys=[], is_last=True, **kwargs):
        cc = {
            'd': self,
            's': self.step_data,
            'e': self.endpoint_data,
            'c': self.config,
            **kwargs
        }

        warnings.filterwarnings('ignore')
        for k, v in pre_kws.items():
            if v:
                ccc = cc
                ccc[k] = v
                reg.funcs.preprocessing[k](**ccc)
        for k in proc_keys:
            reg.funcs.processing[k](**cc)
        for k in anot_keys:
            reg.funcs.annotating[k](**cc)

        if is_last:
            self.save()
        return self

    # def get_par(self, par=None, k=None, key='step'):
    #     if par is None and k is not None:
    #         par=reg.getPar(k)
    #
    #     if key == 'end':
    #         if not hasattr(self, 'endpoint_data'):
    #             self.load(step=False)
    #         df=self.endpoint_data
    #
    #     elif key == 'step':
    #         if not hasattr(self, 'step_data'):
    #             self.load()
    #         df=self.step_data
    #     else :
    #         raise
    #
    #     if par in df.columns :
    #         return df[par]
    #     else :
    #         if k is None :
    #             k = reg.getPar(p=par, to_return='k')
    #         return reg.par.get(k=k, d=self, compute=True)

    def get_chunk_par(self, chunk, k=None, par=None, min_dur=0, mode='distro'):
        chunk_idx = f'{chunk}_idx'
        chunk_dur = f'{chunk}_dur'
        if par is None:
            par = reg.getPar(k)

        dic0 = aux.AttrDict(self.read('chunk_dicts'))

        dics = [dic0[id] for id in self.config.agent_ids]
        sss = [self.step_data[par].xs(id, level='AgentID') for id in self.config.agent_ids]

        if mode == 'distro':

            vs = []
            for ss, dic in zip(sss, dics):
                if min_dur == 0:
                    idx = dic[chunk_idx] + 1
                else:
                    epochs = dic[chunk][dic[chunk_dur] >= min_dur]
                    Nepochs = epochs.shape[0]
                    if Nepochs == 0:
                        idx = []
                    elif Nepochs == 1:
                        idx = np.arange(epochs[0][0], epochs[0][1] + 1, 1)
                    else:
                        slices = [np.arange(r0, r1 + 1, 1) for r0, r1 in epochs]
                        idx = np.concatenate(slices)
                vs.append(ss.loc[idx].dropna().values)
            vs = np.concatenate(vs)
            return vs
        elif mode == 'extrema':
            cc0s, cc1s, cc01s = [], [], []
            for ss, dic in zip(sss, dics):
                epochs = dic[chunk]
                if min_dur != 0:
                    epochs = epochs[dic[chunk_dur] >= min_dur]
                Nepochs = epochs.shape[0]
                if Nepochs > 0:
                    c0s = ss.loc[epochs[:, 0]].values
                    c1s = ss.loc[epochs[:, 1]].values
                    cc0s.append(c0s)
                    cc1s.append(c1s)
            cc0s = np.concatenate(cc0s)
            cc1s = np.concatenate(cc1s)
            cc01s = cc1s - cc0s
            return cc0s, cc1s, cc01s

    # @property
    # def data(self):
    #     s=self.step_data if hasattr(self, 'step_data') else None
    #     e=self.endpoint_data if hasattr(self, 'endpoint_data') else None
    #     return s, e, self.config


class LarvaDatasetCollection:
    def __init__(self, labels=None, add_samples=False, config=None, **kwargs):
        datasets = self.get_datasets(**kwargs)

        for d in datasets:
            assert isinstance(d, BaseLarvaDataset)
        if labels is None:
            labels = [d.id for d in datasets]

        if add_samples:
            targetIDs = aux.unique_list([d.config.sample for d in datasets])
            targets = [reg.conf.Ref.loadRef(id) for id in targetIDs if id in reg.conf.Ref.confIDs]
            datasets += targets
            if labels is not None:
                labels += targetIDs
        self.config = config
        self.datasets = datasets
        self.labels = labels
        self.Ndatasets = len(datasets)
        self.colors = self.get_colors()
        assert self.Ndatasets == len(self.labels)

        self.group_ids = aux.unique_list([d.config.group_id for d in self.datasets])
        self.Ngroups = len(self.group_ids)
        self.dir = self.set_dir()

    def set_dir(self, dir=None):
        if dir is not None:
            return dir
        elif self.config and 'dir' in self.config:
            return self.config.dir
        elif self.Ndatasets > 1 and self.Ngroups == 1:
            dir0 = aux.unique_list([os.path.dirname(os.path.abspath(d.dir)) for d in self.datasets])
            if len(dir0) == 1:
                return dir0[0]
            else:
                raise

    @property
    def plot_dir(self):
        return f'{self.dir}/group_plots'

    def plot(self, ids=[], gIDs=[], **kwargs):
        kws = {
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            'show': False,
            'subfolder': None
        }
        kws.update(**kwargs)
        plots = aux.AttrDict()
        for id in ids:
            plots[id] = reg.graphs.run(id, **kws)
        for gID in gIDs:
            plots[gID] = reg.graphs.run_group(gID, **kws)
        return plots

    def get_datasets(self, datasets=None, refIDs=None, dirs=None, group_id=None):
        if datasets:
            pass
        elif refIDs:
            datasets = [reg.conf.Ref.loadRef(refID) for refID in refIDs]
        elif dirs:
            datasets = [LarvaDataset(dir=f'{reg.DATA_DIR}/{dir}', load_data=False) for dir in dirs]
        elif group_id:
            datasets = reg.conf.Ref.loadRefGroup(group_id, to_return='list')
        return datasets

    def get_colors(self):
        colors = []
        for d in self.datasets:
            color = d.config.color
            while color is None or color in colors:
                color = aux.random_colors(1)[0]
            colors.append(color)
        return colors

    @property
    def data_dict(self):
        return dict(zip(self.labels, self.datasets))

    @property
    def data_palette(self):
        return zip(self.labels, self.datasets, self.colors)

    @property
    def data_palette_with_N(self):
        return zip(self.labels_with_N, self.datasets, self.colors)

    @property
    def color_palette(self):
        return dict(zip(self.labels, self.colors))

    @property
    def Nticks(self):
        Nticks_list = [d.config.Nticks for d in self.datasets]
        return int(np.max(aux.unique_list(Nticks_list)))

    @property
    def N(self):
        N_list = [d.config.N for d in self.datasets]
        return int(np.max(aux.unique_list(N_list)))

    @property
    def labels_with_N(self):
        return [f'{l} (N={d.config.N})' for l, d in self.data_dict.items()]

    @property
    def fr(self):
        fr_list = [d.fr for d in self.datasets]
        return np.max(aux.unique_list(fr_list))

    @property
    def dt(self):
        dt_list = aux.unique_list([d.dt for d in self.datasets])
        return np.max(dt_list)

    @property
    def duration(self):
        return int(self.Nticks * self.dt)

    @property
    def tlim(self):
        return 0, self.duration

    def trange(self, unit='min'):
        if unit == 'min':
            T = 60
        elif unit == 'sec':
            T = 1
        t0, t1 = self.tlim
        x = np.linspace(t0 / T, t1 / T, self.Nticks)
        return x

    @property
    def arena_dims(self):
        dims = np.array([d.env_params.arena.dims for d in self.datasets])
        if self.Ndatasets > 1:
            dims = np.max(dims, axis=0)
        else:
            dims = dims[0]
        return tuple(dims)

    @property
    def arena_geometry(self):
        geos = aux.unique_list([d.env_params.arena.geometry for d in self.datasets])
        if len(geos) == 1:
            return geos[0]
        else:
            return None

    def concat_data(self, key):
        return aux.concat_datasets(dict(zip(self.labels, self.datasets)), key=key)

    @classmethod
    def from_agentpy_output(cls, output=None, agents=None, to_Geo=False):
        config0 = aux.AttrDict(output.parameters['constants'])
        ds = []
        for gID, df in output.variables.items():
            assert 'sample_id' not in df.index.names
            step, end = convert_group_output_to_dataset(df, config0['collectors'])
            config = config0.get_copy()
            kws = {
                # 'larva_groups': {gID: gConf},
                # 'df': df,
                'group_id': config0.id,
                'id': gID,
                'refID': None,
                # 'refID': f'{config0.id}/{gID}',
                'dir': None,
                # 'color': None,
                # 'sample': gConf.sample,
                # 'life_history': gConf.life_history,
                # 'model': gConf.model,

            }
            if 'larva_groups' in config0.keys():
                gConf = config0.larva_groups[gID]
                kws.update(**{
                    'larva_groups': {gID: gConf},
                    # 'df': df,
                    # 'group_id': config0.id,
                    # 'id': gID,
                    # 'refID': None,
                    # 'refID': f'{config0.id}/{gID}',
                    'dir': f'{config0.dir}/data/{gID}',
                    'color': gConf.color,
                    # 'sample': gConf.sample,
                    # 'life_history': gConf.life_history,
                    # 'model': gConf.model,

                })
            config.update(**kws)
            d = BaseLarvaDataset.initGeo(to_Geo=to_Geo, load_data=False, step=step, end=end,
                                         agents=agents, initialize=True,**config)

            ds.append(d)

        return cls(datasets=ds, config=config0)


def convert_group_output_to_dataset(df, collectors):
    df.index.set_names(['AgentID', 'Step'], inplace=True)
    df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
    df.sort_index(level=['Step', 'AgentID'], inplace=True)

    end = df[collectors['end']].xs(df.index.get_level_values('Step').max(), level='Step')
    step = df[collectors['step']]

    return step, end


def h5_kdic(p, N, Nc):
    def epochs_ps():
        cs = ['turn', 'Lturn', 'Rturn', 'pause', 'exec', 'stride', 'stridechain', 'run']
        pars = ['id', 'start', 'stop', 'dur', 'dst', aux.nam.scal('dst'), 'length', aux.nam.max('vel'), 'count']
        return aux.SuperList([aux.nam.chunk_track(c, pars) for c in cs]).flatten

    def dspNtor_ps():
        tor_ps = [f'tortuosity_{dur}' for dur in [1, 2, 5, 10, 20, 30, 60, 100, 120, 240, 300]]
        dsp_ps = [f'dispersion_{t0}_{t1}' for (t0, t1) in
                  itertools.product([0, 5, 10, 20, 30, 60], [30, 40, 60, 90, 120, 240, 300])]
        pars = aux.SuperList(tor_ps + dsp_ps + aux.nam.scal(dsp_ps))
        return pars

    def base_spatial_ps(p=''):
        d, v, a = ps = [aux.nam.dst(p), aux.nam.vel(p), aux.nam.acc(p)]
        ld, lv, la = lps = aux.nam.lin(ps)
        ps0 = aux.nam.xy(p) + ps + lps + aux.nam.cum([d, ld])
        return aux.SuperList(ps0 + aux.nam.scal(ps0))

    def ang_pars(angs):
        avels = aux.nam.vel(angs)
        aaccs = aux.nam.acc(angs)
        uangs = aux.nam.unwrap(angs)
        avels_min, avels_max = aux.nam.min(avels), aux.nam.max(avels)
        return aux.SuperList(avels + aaccs + uangs + avels_min + avels_max)

    def angular(N):
        Nangles = np.clip(N - 2, a_min=0, a_max=None)
        Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        ors = aux.nam.orient(['front', 'rear', 'head', 'tail'] + aux.nam.midline(Nsegs, type='seg'))
        ang = ors + [f'angle{i}' for i in range(Nangles)] + ['bend']
        return aux.SuperList(ang + ang_pars(ang)).unique

    dic = aux.AttrDict({
        'contour': aux.nam.contour_xy(Nc, flat=True),
        'midline': aux.nam.midline_xy(N, flat=True),
        'epochs': epochs_ps(),
        'base_spatial': base_spatial_ps(p),
        'angular': angular(N),
        'dspNtor': dspNtor_ps(),
    })
    return dic


def get_larva_dicts(ls, validIDs=None):
    deb_dicts = {}
    nengo_dicts = {}
    bout_dicts = {}
    for id, l in ls.items():
        if validIDs and id not in validIDs:
            continue
        if hasattr(l, 'deb') and l.deb is not None:
            deb_dicts[id] = l.deb.finalize_dict()
        try:
            from ..model.modules.nengobrain import NengoBrain
            if isinstance(l.brain, NengoBrain):
                if l.brain.dict is not None:
                    nengo_dicts[id] = l.brain.dict
        except:
            pass
        if l.brain.locomotor.intermitter is not None:
            bout_dicts[id] = l.brain.locomotor.intermitter.build_dict()

    dic0 = aux.AttrDict({'deb': deb_dicts,
                         'nengo': nengo_dicts,
                         'bouts': bout_dicts,
                         })

    return aux.AttrDict({k: v for k, v in dic0.items() if len(v) > 0})
