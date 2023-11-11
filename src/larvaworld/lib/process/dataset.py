"""
Basic classes for larvaworld-format datasets
"""

import copy
import itertools
import os
import random
import shutil
import numpy as np
import pandas as pd
import warnings

import param

from .. import reg, aux, process, util
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
        if 'config' not in kwargs:
            kws = aux.AttrDict()
            for k in reg.generators.DatasetConfig().param_keys:
                if k in kwargs:
                    kws[k] = kwargs[k]
                    kwargs.pop(k)
            kwargs['config'] = reg.generators.DatasetConfig(**kws)
        assert 'config2' not in kwargs

        ks = list(kwargs.keys())
        kws2 = aux.AttrDict()
        for k in ks:
            if k not in self.param.objects():
                kws2[k] = kwargs[k]
                kwargs.pop(k)
        kwargs['config2'] = aux.AttrDict(kws2)
        super().__init__(**kwargs)
        self.merge_configs()
        self.epoch_dict = aux.AttrDict({'pause': None, 'run': None})
        self.larva_dicts = {}
        self.__dict__.update(self.config.nestedConf)
        self._epoch_dicts = None
        self._chunk_dicts = None
        self._pooled_epochs = None
        self._fitted_epochs = None

        self._cycle_curves = None

    def required(**pars):
        def wrap(f):
            def wrapped_f(self, *args, **kwargs):
                if self.data_exists(**pars):
                    f(self, *args, **kwargs)
                # else:
                #     reg.vprint(f'Required columns {pars.nonexisting(s)} not found. Aborting method.', 3)

            return wrapped_f

        return wrap

    # def returned(**pars):
    #     def wrap(f):
    #         def wrapped_f(self, recompute, *args, **kwargs):
    #             if not self.data_exists(**pars) or recompute:
    #                 f(self, *args, **kwargs)
    #             # else:
    #             #     reg.vprint(f'Required columns {pars.nonexisting(s)} not found. Aborting method.', 3)
    #         return wrapped_f
    #     return wrap

    def valid(required=None, returned=None):
        _verbose = -3

        def wrap(f):
            def wrapped_f(self, *args, **kwargs):

                reg.vprint(f'_______________________________', _verbose)
                reg.vprint(f'Checking method {f.__name__}', _verbose)
                if required is not None:

                    if self.data_exists(**required):
                        reg.vprint(f'   Required columns exist. Proceeding ...', _verbose)
                    else:
                        reg.vprint(f'   Required columns not found. Aborting...', _verbose)
                        return wrapped_f
                if returned is not None:
                    #     reg.vprint(f'   Columns to be returned not provided. Executing ...', _verbose)
                    #     f(self, *args, **kwargs)
                    # else:
                    #     returned_exists = self.data_exists(**returned)
                    if not self.data_exists(**returned):
                        reg.vprint(f'   Columns to be returned do not exist. Executing ...', _verbose)

                    else:

                        if 'recompute' in kwargs and kwargs['recompute']:
                            reg.vprint(f'   Forced to recompute. Executing...', _verbose)
                            f(self, *args, **kwargs)
                        else:
                            reg.vprint(f'   Columns to be returned exist and not forced to recompute. Aborting...',
                                       _verbose)
                            return wrapped_f

                f(self, *args, **kwargs)

            return wrapped_f

        return wrap

    def data_exists(self, ks=[], ps=[], eks=[], eps=[], config_attrs=[], attrs=[]):
        if not all([hasattr(self, attr) for attr in attrs]):
            return False
        s, e, c = self.data
        spars = aux.SuperList(
            ps + reg.getPar(ks) + [getattr(c, attr) for attr in config_attrs]).flatten.unique
        if not spars.exist_in(s):
            return False
        epars = aux.SuperList(eps + reg.getPar(eks))
        return epars.exist_in(s)

    @property
    def chunk_dicts(self):
        try:
            assert self._chunk_dicts is not None
        except AssertionError:
            self._chunk_dicts = aux.AttrDict(self.read('chunk_dicts'))
        except KeyError:
            self.detect_bouts()
        finally:
            return self._chunk_dicts

    @chunk_dicts.setter
    def chunk_dicts(self, d):
        self._chunk_dicts = d
        self.store(d, 'chunk_dicts')

    @property
    def epoch_dicts(self):
        try:
            assert self._epoch_dicts is not None
        except AssertionError:
            self._epoch_dicts = aux.AttrDict(self.read('epoch_dicts'))
        except KeyError:
            self.comp_pooled_epochs()
        finally:
            return self._epoch_dicts

    @epoch_dicts.setter
    def epoch_dicts(self, d):
        self._epoch_dicts = d
        self.store(d, 'epoch_dicts')

    @property
    def fitted_epochs(self):
        try:
            assert self._fitted_epochs is not None
        except AssertionError:
            self._fitted_epochs = aux.AttrDict(self.read('fitted_epochs'))
        except KeyError:
            self.fit_pooled_epochs()
        finally:
            return self._fitted_epochs

    @fitted_epochs.setter
    def fitted_epochs(self, d):
        self._fitted_epochs = d
        self.store(d, 'fitted_epochs')

    @property
    def pooled_epochs(self):
        try:
            assert self._pooled_epochs is not None
        except AssertionError:
            self._pooled_epochs = aux.load_dict(f'{self.config.data_dir}/pooled_epochs.txt')
            # self._pooled_epochs = aux.AttrDict(self.read('pooled_epochs'))
        except KeyError:
            self.comp_pooled_epochs()

        finally:
            return self._pooled_epochs

    @pooled_epochs.setter
    def pooled_epochs(self, d):
        self._pooled_epochs = d
        aux.save_dict(d, f'{self.config.data_dir}/pooled_epochs.txt')
        # self.store(d, 'pooled_epochs')

    @property
    def cycle_curves(self):
        try:
            assert self._cycle_curves is not None
        except AssertionError:
            self._cycle_curves = aux.AttrDict(self.read('cycle_curves'))
        except KeyError:
            self.comp_interference()
        finally:
            return self._cycle_curves

    @cycle_curves.setter
    def cycle_curves(self, d):
        self._cycle_curves = d
        self.store(d, 'cycle_curves')

    @property
    def pooled_cycle_curves(self):
        try:
            assert self.config.pooled_cycle_curves is not None
        except AssertionError:
            self.comp_pooled_cycle_curves()
        finally:
            return self.config.pooled_cycle_curves

    @pooled_cycle_curves.setter
    def pooled_cycle_curves(self, d):
        self.config.pooled_cycle_curves = d

    def track_par_in_chunk(self, chunk, par):
        s, e, c = self.data
        A = self.empty_df(dim3=3)
        d0 = self.epoch_dicts[chunk]
        for i, id in enumerate(c.agent_ids):
            epochs = d0[id]
            ss = s[par].xs(id, level='AgentID')
            if epochs.shape[0] > 0:
                t0s, t1s = epochs[:, 0], epochs[:, 1]
                b0s = ss.loc[t0s].values
                b1s = ss.loc[t1s].values
                A[t0s, i, 0] = b0s
                A[t1s, i, 1] = b1s
                A[t1s, i, 2] = b1s - b0s
        s[aux.nam.atStartStopChunk(par, chunk)] = A.reshape([-1, 3])

    def crawl_annotation(self, strides_enabled=True, vel_thr=0.3):
        from ..process.annotation import detect_strides, detect_pauses, detect_runs, epoch_idx, epoch_durs, epoch_amps
        c = self.config
        dt = c.dt
        if c.Npoints <= 1:
            strides_enabled = False
        kws = {'dt': dt, 'vel_thr': vel_thr}
        l, v, sv, dst, fov = reg.getPar(['l', 'v', 'sv', 'd', 'fov'])
        str_d_mu, str_d_std, str_sd_mu, str_sd_std, run_tr, pau_tr, cum_run_t, cum_pau_t, cum_t = \
            reg.getPar(
                ['str_d_mu', 'str_d_std', 'str_sd_mu', 'str_sd_std', 'run_tr', 'pau_tr', 'cum_run_t', 'cum_pau_t',
                 'cum_t'])
        Sps = [str_d_mu, str_d_std]+reg.getPar(['str_sv_mu','str_N','run_v_mu', 'pau_v_mu'])+[cum_run_t, cum_pau_t]
        Svs = np.zeros([c.N, len(Sps)]) * np.nan
        DD = {}
        for jj, id in enumerate(c.agent_ids):
            D = aux.AttrDict()
            ss = self.s.xs(id, level="AgentID")
            a_v = ss[v].values
            a_fov = ss[fov].values
            if strides_enabled:
                a = ss[sv].values
                D.vel_minima, D.vel_maxima, D.stride, D.exec, D.run_count = detect_strides(a, return_extrema=True, **kws)
            else:
                D.vel_minima, D.vel_maxima, D.stride, D.run_count = np.array([]), np.array([]), np.array([]), np.array([])
                a = a_v
                D.exec = detect_runs(a, **kws)
            D.stride_Dor = np.array([np.trapz(a_fov[s0:s1 + 1]) for s0, s1 in D.stride])
            D.stride_dur = epoch_durs(D.stride, dt)
            D.stride_dst = epoch_amps(D.stride, a, dt)
            D.stride_idx = epoch_idx(D.stride)
            D.run_dur = epoch_durs(D.exec, dt)
            D.run_dst = epoch_amps(D.exec, a_v, dt)
            D.run_idx = epoch_idx(D.exec)
            D.pause = detect_pauses(a, runs=D.exec, **kws)
            D.pause_dur = epoch_durs(D.pause, dt)
            D.pause_idx = epoch_idx(D.pause)

            Svs[jj, :] = [np.nanmean(D.stride_dst),np.nanstd(D.stride_dst),
                          np.nanmean(a[D.stride_idx]),np.nansum(D.run_count),
                          np.mean(a_v[D.run_idx]),np.mean(a_v[D.pause_idx]),
                          np.sum(D.run_dur),np.sum(D.pause_dur)]
            DD[id] = D
        self.e[Sps] = Svs
        self.e[run_tr] = self.e[cum_run_t] / self.e[cum_t]
        self.e[pau_tr] = self.e[cum_pau_t] / self.e[cum_t]
        if l in self.end_ps:
            self.e[str_sd_mu] = self.e[str_d_mu] / self.e[l]
            self.e[str_sd_std] = self.e[str_d_std] / self.e[l]
        return DD

    def turn_annotation(self, min_dur=None):
        from ..process.annotation import detect_turns, process_epochs
        c = self.config
        dt=c.dt
        A = self.s[reg.getPar('fov')]

        ps = reg.getPar(['Ltur_N', 'Rtur_N', 'tur_N', 'tur_H'])
        vs = np.zeros([c.N, len(ps)]) * np.nan
        DD = {}

        for j, id in enumerate(c.agent_ids):
            D=aux.AttrDict()
            a = A.xs(id, level="AgentID")
            D.Lturn, D.Rturn = detect_turns(a, dt, min_dur=min_dur)
            D.Lturn_dur, D.Lturn_amp, Lmaxs = process_epochs(a.values, D.Lturn, dt)
            D.Rturn_dur, D.Rturn_amp, Rmaxs = process_epochs(a.values, D.Rturn, dt)
            D.turn_dur = np.concatenate([D.Lturn_dur, D.Rturn_dur])
            D.turn_amp = np.concatenate([D.Lturn_amp, D.Rturn_amp])
            D.turn_vel_max = np.concatenate([Lmaxs, Rmaxs])
            LN, RN = D.Lturn.shape[0], D.Rturn.shape[0]
            N = LN + RN
            H = LN / N if N != 0 else 0


            vs[j, :] = [LN, RN, N, H]
            DD[id]=D
        self.e[ps] = vs
        return DD

    def detect_bouts(self, vel_thr=0.3, strides_enabled=True, castsNweathervanes=True):
        s, e, c = self.data
        aux.fft_freqs(s, e, c)
        turn_dict = process.annotation.turn_annotation(s, e, c)
        crawl_dict = process.annotation.crawl_annotation(s, e, c, strides_enabled=strides_enabled, vel_thr=vel_thr)
        self.chunk_dicts = aux.AttrDict({id: {**turn_dict[id], **crawl_dict[id]} for id in c.agent_ids})
        if castsNweathervanes:
            process.annotation.turn_mode_annotation(e, self.chunk_dicts)
        reg.vprint(f'Completed bout detection.', 1)

    def comp_pooled_epochs(self):
        d0 = self.chunk_dicts
        epoch_ks = aux.SuperList([list(dic.keys()) for dic in d0.values()]).flatten.unique
        self.epoch_dicts = aux.AttrDict({k: {id: d0[id][k] for id in list(d0)} for k in epoch_ks})

        def get_vs(dic):
            l = aux.SuperList(dic.values())
            try:
                sh = [len(ll.shape) for ll in l]
                if sh.count(2) > sh.count(1):
                    l = aux.SuperList([ll for ll in l if len(ll.shape) == 2])
            except:
                pass
            return np.concatenate(l)

        self.pooled_epochs = aux.AttrDict(
            {k: get_vs(dic) for k, dic in self.epoch_dicts.items() if
             k not in ['turn_slice', 'pause_idx', 'run_idx', 'stride_idx']})

        reg.vprint(f'Completed bout detection.', 1)

    def fit_pooled_epochs(self):
        try:
            dic = self.pooled_epochs
            assert dic is not None
            fitted = {}
            for k, v in dic.items():
                try:
                    fitted[k] = util.fit_bout_distros(np.abs(v), bout=k, combine=False,
                                                      discrete=True if k == 'run_count' else False)
                except:
                    fitted[k] = None
            self.fitted_epochs = aux.AttrDict(fitted)
            reg.vprint(f'Fitted pooled epoch durations.', 1)
        except:
            reg.vprint(f'Failed to fit pooled epoch durations.', 1)

    def comp_bout_distros(self):
        c = self.config
        c.bout_distros = aux.AttrDict()
        for k, dic in self.fitted_epochs.items():
            try:
                c.bout_distros[k] = dic['best']
                reg.vprint(f'Completed {k} bout distribution analysis.', 1)
            except:
                c.bout_distros[k] = None
                reg.vprint(f'Failed to complete {k} bout distribution analysis.', 1)
        self.register_bout_distros()

    def register_bout_distros(self):
        s, e, c = self.data
        from ..model.modules.intermitter import get_EEB_poly1d
        try:
            c.intermitter = {
                nam.freq('crawl'): e[nam.freq(nam.scal(nam.vel('')))].mean(),
                nam.freq('feed'): e[nam.freq('feed')].mean() if nam.freq('feed') in self.end_ps else 2.0,
                'dt': c.dt,
                'feed_bouts': True,
                'stridechain_dist': c.bout_distros.run_count,
                'pause_dist': c.bout_distros.pause_dur,
                'run_dist': c.bout_distros.run_dur,
                'feeder_reoccurence_rate': None,
            }
            c.EEB_poly1d = get_EEB_poly1d(**c.intermitter).c.tolist()
        except:
            pass

    def comp_interference(self, **kwargs):
        s, e, c = self.data
        try:
            self.cycle_curves = process.annotation.compute_interference(s, e, c, chunk_dicts=self.chunk_dicts, **kwargs)
            self.comp_pooled_cycle_curves()
            reg.vprint(f'Completed stridecycle interference analysis.', 1)
        except:
            reg.vprint(f'Failed to complete stridecycle interference analysis.', 1)

    def comp_pooled_cycle_curves(self):
        try:
            self.pooled_cycle_curves = aux.AttrDict({
                k: {mode: np.nanquantile(vs, q=0.5, axis=0).tolist() for mode, vs in dic.items()} for k, dic in
                self.cycle_curves.items()})
            reg.vprint(f'Computed average curves during stridecycle for diverse parameters.', 1)
        except:
            reg.vprint(f'Failed to compute average curves during stridecycle for diverse parameters.', 1)

    def annotate(self, anot_keys=["bout_detection", "bout_distribution", "interference"], is_last=False, **kwargs):
        if 'bout_detection' in anot_keys:
            self.detect_bouts()
            self.comp_pooled_epochs()
        if 'bout_distribution' in anot_keys:
            self.fit_pooled_epochs()
            self.comp_bout_distros()
        if 'interference' in anot_keys:
            self.comp_interference()
        if 'source_attraction' in anot_keys:
            s, e, c = self.data
            process.patch.comp_bearing_to_source(s, e, c)
        if 'patch_residency' in anot_keys:
            s, e, c = self.data
            process.patch.comp_time_on_patch(s, e, c)
        if is_last:
            self.save()

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
        if self.step_data is None:
            self.load()
        return self.step_data

    @property
    def e(self):
        if self.endpoint_data is None:
            self.load(step=False)
        return self.endpoint_data

    @property
    def end_ps(self):
        return aux.SuperList(self.e.columns).sorted

    @property
    def step_ps(self):
        return aux.SuperList(self.s.columns).sorted

    @property
    def end_ks(self):
        return aux.SuperList(reg.getPar(d=self.end_ps, to_return='k')).sorted

    @property
    def step_ks(self):
        return aux.SuperList(reg.getPar(d=self.step_ps, to_return='k')).sorted

    @property
    def c(self):
        return self.config

    @property
    def min_tick(self):
        return self.step_data.index.unique('Step').min()

    def timeseries_slice(self, time_range=None, df=None):
        if df is None:
            df = self.step_data
        if time_range is None:
            return df
        else:

            t0, t1 = time_range
            s0 = int(t0 / self.config.dt)
            s1 = int(t1 / self.config.dt)
            df_slice = df.loc[(slice(s0, s1), slice(None)), :]
            return df_slice

    def interpolate_nan_values(self):
        s, e, c = self.data
        pars = c.all_xy.existing(s)
        Npars = len(pars)
        for id in c.agent_ids:
            A = np.zeros([c.Nticks, Npars])
            ss = s.xs(id, level='AgentID')
            for i, p in enumerate(pars):
                A[:, i] = aux.interpolate_nans(ss[p].values)
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
        return self.s, self.e, self.config

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
        if type in ds0 and all([id in ds0[type] for id in ids]):
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
        # assert xy.exist_in(self.step_data)
        grouped = self.step_data[xy].groupby('AgentID')
        return aux.AttrDict({id: df.values.reshape([-1, self.config.Npoints, 2]) for id, df in grouped})

    @property
    def traj_xy_data_byID(self):
        s = self.step_data
        xy = self.config.traj_xy
        # if not xy.exist_in(s):
        #     xy0 = self.config.point_xy
        #     assert xy0.exist_in(s)
        #     s[xy] = s[xy0]
        # assert xy.exist_in(s)
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
        s0 = s.index.unique('Step').min() - self.min_tick

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

    def comp_freq(self, par, fr_range=(0.0, +np.inf)):
        s, e, c = self.data
        e[nam.freq(par)] = s[par].groupby("AgentID").apply(aux.fft_max, dt=c.dt, fr_range=fr_range)

    def comp_freqs(self):
        v = reg.getPar('v')
        if v in self.step_ps:
            self.comp_freq(par=v, fr_range=(1.0, 2.5))
        sv = nam.scal(v)
        if sv in self.step_ps:
            self.comp_freq(par=sv, fr_range=(1.0, 2.5))
        fov = reg.getPar('fov')
        if fov in self.step_ps:
            self.comp_freq(par=fov, fr_range=(0.1, 0.8))

    @valid(required={'config_attrs': ['midline_xy']}, returned={'ks': ['fo', 'ro']})
    def comp_orientations(self, mode='minimal', recompute=False):
        s, e, c = self.data
        all_vecs = list(c.vector_dict.keys())
        vecs = all_vecs[:2] if mode == 'minimal' else all_vecs
        pars = aux.nam.orient(vecs)
        if pars.exist_in(s) and not recompute:
            reg.vprint(
                'Vector orientations are already computed. If you want to recompute them, set recompute to True', 1)
        else:
            mid = self.midline_xy_data
            for vec, par in zip(vecs, pars):
                (idx1, idx2) = c.vector_dict[vec]
                x, y = mid[:, idx2, 0] - mid[:, idx1, 0], mid[:, idx2, 1] - mid[:, idx1, 1]
                s[par] = np.arctan2(y, x) % 2 * np.pi

        if mode == 'full':
            mid = self.midline_xy_data
            s[c.seg_orientations] = self.midline_seg_orients_from_mid(mid)

    def comp_angular(self, is_last=False, **kwargs):
        self.comp_orientations(**kwargs)
        self.comp_bend(**kwargs)
        self.comp_ang_moments(**kwargs)
        if is_last:
            self.save()
        reg.vprint('Angular analysis complete.', 1)

    def comp_bend(self, mode='minimal', recompute=False):

        if 'bend' in self.step_ps and not recompute:
            reg.vprint(
                'Vector orientations are already computed. If you want to recompute them, set recompute to True', 1)
        else:
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
                if mode == 'full':
                    s[c.angles] = Ada
            else:
                raise

            s['bend'] = a

    def comp_ang_moments(self, pars=None, mode='minimal', recompute=False):
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

    def comp_xy_moments(self, point='', **kwargs):
        s, e, c = self.data
        xy = nam.xy(point)
        if not xy.exist_in(s):
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

    @valid(required={'ps': ['x', 'y', 'dst']})
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

    @valid(required={'config_attrs': ['traj_xy']})
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

    @valid(required={'config_attrs': ['contour_xy']}, returned={'config_attrs': ['centroid_xy']})
    def comp_centroid(self, **kwargs):
        c = self.config
        if c.Ncontour > 0:
            self.step_data[c.centroid_xy] = np.sum(self.contour_xy_data, axis=1) / c.Ncontour

    @valid(required={'config_attrs': ['midline_xy']}, returned={'eks': ['l']})
    def comp_length(self, mode='minimal', recompute=False):
        if 'length' in self.end_ps and not recompute:
            reg.vprint('Length is already computed. If you want to recompute it, set recompute_length to True', 1)
        else:
            self.step_data['length'] = np.sum(np.sum(np.diff(self.midline_xy_data, axis=1) ** 2, axis=2) ** (1 / 2),
                                              axis=1)
            self.endpoint_data['length'] = self.step_data['length'].groupby('AgentID').quantile(q=0.5)

    def comp_spatial(self, **kwargs):
        s, e, c = self.data
        self.comp_centroid(**kwargs)
        self.comp_length(**kwargs)
        if not c.traj_xy.exist_in(s) and c.point_xy.exist_in(s):
            s[c.traj_xy] = s[c.point_xy]
        self.comp_operators(pars=c.traj_xy)
        for point in ['', 'centroid']:
            self.comp_xy_moments(point, **kwargs)
        reg.vprint('Spatial analysis complete.', 1)

    def scale_to_length(self, pars=None, keys=None):
        s, e, c = self.data
        if 'length' not in self.end_ps:
            self.comp_length()
        l = e['length']
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

    @required(ks=['fo'], config_attrs=['traj_xy'])
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

            reg.vprint('Bearing and distance to source computed', 1)

    def comp_wind(self):
        w = self.config.env_params.windscape
        if w is not None:
            wo = w.wind_direction
            woo = np.deg2rad(wo)
            try:
                self.comp_wind_metrics(woo, wo)
            except:
                self.comp_final_anemotaxis(woo)

    def comp_wind_metrics(self, woo, wo):
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

    def comp_final_anemotaxis(self, woo):
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
        if 'x' in self.end_ps:
            xs = e['x'].values
        elif nam.final('x') in self.end_ps:
            xs = e[nam.final('x')].values
        elif 'x' in self.step_ps:
            xs = s['x'].dropna().groupby('AgentID').last().values
        elif 'centroid_x' in self.step_ps:
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
                dsp_starts=[0], dsp_stops=[40, 60], tor_durs=[5, 10, 20], is_last=False, **kwargs):
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
        if par is None and k is not None:
            par = reg.getPar(k)

        def read_key(key, par):
            res = self.read(key)[par]
            if res is not None:
                return res

        if key == 'end' and par in self.end_ps:
            return self.e[par]
        if key == 'step' and par in self.step_ps:
            return self.s[par]
        try:
            return read_key(key, par)
        except:
            if k is None:
                k = reg.getPar(p=par, to_return='k')
            return reg.par.get(k=k, d=self, compute=True)

    def sample_larvagroup(self, N=1, ps=[]):
        e = self.endpoint_data
        ps = aux.existing_cols(aux.unique_list(ps), e)
        means = [e[p].mean() for p in ps]
        if len(ps) >= 2:
            base = e[ps].dropna().values.T
            cov = np.cov(base)
            vs = np.random.multivariate_normal(means, cov, N).T
        elif len(ps) == 1:
            std = np.std(e[ps].values)
            vs = np.atleast_2d(np.random.normal(means[0], std, N))
        else:
            return {}
        flatnames = reg.getPar(d=ps, to_return='flatname')
        # codenames = [codename_dict[p] if p in codename_dict else p for p in ps]
        dic = {p: v for p, v in zip(flatnames, vs)}
        return dic

    def imitate_larvagroup(self, N=None, ps=None):
        if N is None:
            N = self.config.N
        e = self.endpoint_data
        ids = random.sample(e.index.values.tolist(), N)
        poss = [tuple(e[reg.getPar(['x0', 'y0'])].loc[id].values) for id in ids]
        try:
            ors = [e[reg.getPar('fo0')].loc[id] for id in ids]
        except:
            ors = np.random.uniform(low=0, high=2 * np.pi, size=len(ids)).tolist()

        if ps is None:
            ps = list(util.SAMPLING_PARS.keys())
        ps = aux.existing_cols(aux.unique_list(ps), e)
        flatnames = reg.getPar(p=ps, to_return='flatname')
        # codenames = [codename_dict[p] if p in codename_dict else p for p in ps]
        dic = aux.AttrDict(
            {codename: [e[p].loc[id] if not np.isnan(e[p].loc[id]) else e[p].mean() for id in ids] for p, codename in
             zip(ps, flatnames)})
        return ids, poss, ors, dic


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

    def delete(self):
        shutil.rmtree(self.config.dir)
        reg.vprint(f'Dataset {self.id} deleted', 2)

    def set_id(self, id, save=True):
        self.id = id
        self.config.id = id
        if save:
            self.save_config()


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

    def visualize(self, parameters={}, **kwargs):
        from ..sim.dataset_replay import ReplayRun
        kwargs['dataset'] = self
        rep = ReplayRun(parameters=parameters, **kwargs)
        rep.run()

    def enrich(self, pre_kws={}, proc_keys=[], anot_keys=[], is_last=True, mode='minimal', recompute=False, **kwargs):

        warnings.filterwarnings('ignore')
        self.preprocess(**pre_kws, recompute=recompute)
        self.process(proc_keys=proc_keys, is_last=False, mode=mode, recompute=recompute, **kwargs)
        self.annotate(anot_keys=anot_keys, is_last=False, recompute=recompute, **kwargs)

        if is_last:
            self.save()
        return self

    @property
    def epoch_bound_dicts(self):
        d = aux.AttrDict()
        for k, dic in self.epoch_dicts.items():
            try:
                if all([vs.shape.__len__() == 2 for id, vs in dic.items()]):
                    d[k] = dic
            except:
                pass
        return d

    def get_chunk_par(self, chunk, k=None, par=None, min_dur=0, mode='distro'):
        s, e, c = self.data
        epochs = self.epoch_dicts[chunk]
        if min_dur != 0:
            epoch_durs = self.epoch_dicts[f'{chunk}_dur']
            epochs = aux.AttrDict({id: epochs[id][epoch_durs[id] >= min_dur] for id in c.agent_ids})
        if par is None:
            par = reg.getPar(k)
        grouped = s[par].groupby('AgentID')
        if mode == 'extrema':
            c01s = [[df.loc[epochs[id][:, 0]].values, df.loc[epochs[id][:, 1]].values] for id, df in grouped if
                    epochs[id].shape > 0]
            c0s = np.concatenate([c01[0] for c01 in c01s])
            c1s = np.concatenate([c01[1] for c01 in c01s])
            dc01s = c1s - c0s
            return c0s, c1s, dc01s
        elif mode == 'distro':

            def get_idx(eps):
                Nepochs = eps.shape[0]
                if Nepochs == 0:
                    idx = []
                elif Nepochs == 1:
                    idx = np.arange(epochs[0][0], epochs[0][1] + 1, 1)
                else:
                    slices = [np.arange(r0, r1 + 1, 1) for r0, r1 in eps]
                    idx = np.concatenate(slices)
                return idx

            return np.concatenate([df.loc[get_idx(epochs[id])].dropna().values for id, df in grouped])


class LarvaDatasetCollection:
    def __init__(self, labels=None, add_samples=False, config=None, **kwargs):
        ds = self.get_datasets(**kwargs)

        for d in ds:
            assert isinstance(d, BaseLarvaDataset)
        if labels is None:
            labels = ds.id

        if add_samples:
            targetIDs = aux.SuperList(ds.config.sample).unique.existing(reg.conf.Ref.confIDs)
            ds += reg.conf.Ref.loadRefs(ids=targetIDs)
            labels += targetIDs
        self.config = config
        self.datasets = ds
        self.labels = labels
        self.Ndatasets = len(ds)
        self.colors = self.get_colors()
        assert self.Ndatasets == len(self.labels)

        self.group_ids = aux.SuperList(ds.config.group_id).unique
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
            datasets = reg.conf.Ref.loadRefs(refIDs)
        elif dirs:
            datasets = [LarvaDataset(dir=f'{reg.DATA_DIR}/{dir}', load_data=False) for dir in dirs]
        elif group_id:
            datasets = reg.conf.Ref.loadRefGroup(group_id, to_return='list')
        return aux.ItemList(datasets)

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
            if 'larva_groups' in config0:
                gConf = config0.larva_groups[gID]
                kws.update(**{
                    'larva_group': gConf,
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
                                         agents=agents, initialize=True, **config)

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
