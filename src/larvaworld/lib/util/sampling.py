import random
import numpy as np
import pandas as pd

from .. import reg, aux
from ..aux import nam

__all__ = [
    'SAMPLING_PARS',
    'generate_larvae',
    # 'sample_group',
    'sampleRef',
    'imitateRef',
    'generate_agentGroup',
    'generate_agentConfs',
    'sim_model',
    'sim_models',
]

SAMPLING_PARS = aux.bidict(
    aux.AttrDict(
        {
            'length': 'body.length',
            nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.freq',
            'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
            nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
            nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
            nam.freq('feed'): 'brain.feeder_params.freq',
            nam.max(nam.chunk_track('stride', nam.scal(nam.vel('')))): 'brain.crawler_params.max_scaled_vel',
            'phi_scaled_velocity_max': 'brain.crawler_params.max_vel_phase',
            'attenuation': 'brain.interference_params.attenuation',
            'attenuation_max': 'brain.interference_params.attenuation_max',
            nam.freq(nam.vel(nam.orient(('front')))): 'brain.turner_params.freq',
            nam.max('phi_attenuation'): 'brain.interference_params.max_attenuation_phase',
        }
    )
)


def generate_larvae(base_model, N=None, sample_dict={}):
    Npars=len(sample_dict)
    if Npars > 0:
        if N is None:
            vs_per_par=[vs for p, vs in sample_dict.items()]
            Nvs_per_par=[len(vs) for vs in vs_per_par]
            Ns= aux.SuperList(Nvs_per_par).unique
            assert len(Ns)==1
            N=Ns[0]

        all_pars = []
        for i in range(N):
            dic = aux.AttrDict({p: vs[i] for p, vs in sample_dict.items()})
            all_pars.append(base_model.get_copy().update_nestdict(dic))
    else:
        assert N is not None
        all_pars = [base_model] * N
    return all_pars


def sampleRef(mID=None, m=None, refID=None, refDataset=None, sample_ks=[], Nids=1, parameter_dict={}):
    sample_dict = {}
    if m is None:
        m = reg.conf.Model.getID(mID)
    sample_ks += [k for k in m.flatten() if m.flatten()[k] == 'sample']
    Sinv = SAMPLING_PARS.inverse
    sample_ps = aux.SuperList([Sinv[k] for k in aux.existing_cols(Sinv, sample_ks)]).flatten

    d = refDataset
    if d is None and refID is not None:
        d = reg.conf.Ref.loadRef(refID, load=True, step=False)
    if d is not None:
        m = d.config.get_sample_bout_distros(m.get_copy())
        sample_dict = d.sample_larvagroup(N=Nids, ps=sample_ps, codename_dict=SAMPLING_PARS)

    sample_dict.update(parameter_dict)
    return generate_larvae(m, Nids, sample_dict)


def imitateRef(mID=None, m=None, refID=None, refDataset=None, sample_ks=[], Nids=1, parameter_dict={}):
    d = refDataset
    if d is None:
        if refID is not None:
            d = reg.conf.Ref.loadRef(refID, load=True, step=False)
        else:
            raise
    ids, ps, ors, sample_dict = d.imitate_larvagroup(N=Nids, codename_dict=SAMPLING_PARS)
    sample_dict.update(parameter_dict)

    if m is None:
        m = reg.conf.Model.getID(mID)
    m = d.config.get_sample_bout_distros(m.get_copy())
    ms = generate_larvae(m, N=None, sample_dict=sample_dict)
    return ids, ps, ors, ms


def generate_agentGroup(gID, Nids, imitation=False, distribution=None, **kwargs):
    from ..param import generate_xyNor_distro

    if not imitation:
        if distribution is not None:
            ps, ors = generate_xyNor_distro(distribution)
        else:
            ps = [(0.0, 0.0) for j in range(Nids)]
            ors = [0.0 for j in range(Nids)]
        ids = [f'{gID}_{i}' for i in range(Nids)]
        all_pars = sampleRef(Nids=Nids, **kwargs)
    else:
        ids, ps, ors, all_pars = imitateRef(Nids=Nids, **kwargs)
    return ids, ps, ors, all_pars


def generate_agentConfs(larva_groups, parameter_dict={}):
    agent_confs = []
    for gID, gConf in larva_groups.items():
        d = gConf.distribution
        ids, ps, ors, all_pars = generate_agentGroup(gID=gID, Nids=d.N,
                                                     m=gConf.model, refID=gConf.sample,
                                                     imitation=gConf.imitation,
                                                     distribution=d,
                                                     parameter_dict=parameter_dict)

        gConf.ids = ids
        for id, p, o, pars in zip(ids, ps, ors, all_pars):
            conf = {
                'pos': p,
                'orientation': o,
                'color': gConf.color,
                'unique_id': id,
                'group': gID,
                'odor': gConf.odor,

                'life_history': gConf.life_history,
                **pars
            }

            agent_confs.append(conf)
    return agent_confs


def sim_models(mIDs, colors=None, dataset_ids=None, lgs=None, data_dir=None, **kwargs):
    N = len(mIDs)
    if colors is None:
        colors = aux.N_colors(N)
    if dataset_ids is None:
        dataset_ids = mIDs
    if lgs is None:
        lgs = [None] * N
    if data_dir is None:
        dirs = [None] * N
    else:
        dirs = [f'{data_dir}/{dID}' for dID in dataset_ids]
    ds = [sim_model(mID=mIDs[i], color=colors[i], dataset_id=dataset_ids[i], lg=lgs[i], dir=dirs[i], **kwargs) for i in
          range(N)]
    return ds


def sim_model(mID, Nids=1, refID=None, refDataset=None, sample_ks=[], use_LarvaConfDict=False, imitation=False,
              tor_durs=[], dsp_starts=[0], dsp_stops=[40], enrichment=True,
              lg=None, env_params={}, dir=None, duration=3, dt=1 / 16, color='blue', dataset_id=None,
              **kwargs):
    if use_LarvaConfDict:
        pass
    if refID is None:
        refID = refDataset.refID
    if dataset_id is None:
        dataset_id = mID
    if lg is None:
        lg = reg.gen.LarvaGroup(c=color, sample=refID, mID=mID, N=Nids, expand=True, **kwargs).entry(id=dataset_id)

    Nticks = int(duration * 60 / dt)
    ids, p0s, fo0s, ms = generate_agentGroup(gID=mID, mID=mID, refID=refID, Nids=Nids,
                                             refDataset=refDataset, sample_ks=sample_ks,
                                             imitation=imitation)
    s, e = sim_multi_agents(Nticks, Nids, ms, dataset_id, dt=dt, ids=ids, p0s=p0s, fo0s=fo0s)

    c_kws = {
        'dir': dir,
        'id': dataset_id,
        'larva_groups': lg,
        'env_params': env_params,
        'Npoints': 3,
        'Ncontour': 0,
        'fr': 1 / dt,
    }
    from ..process.dataset import LarvaDataset
    d = LarvaDataset(**c_kws, load_data=False)
    d.set_data(step=s, end=e)
    if enrichment:
        d = d.enrich(proc_keys=['spatial', 'angular', 'dispersion', 'tortuosity'],
                     anot_keys=['bout_detection', 'bout_distribution', 'interference'],
                     dsp_starts=dsp_starts, dsp_stops=dsp_stops, tor_durs=tor_durs)

    return d


def sim_single_agent(m, Nticks=1000, dt=0.1, df_columns=None, p0=None, fo0=None):
    from ..model.modules.locomotor import DefaultLocomotor
    from ..model.agents._larva_sim import BaseController
    if fo0 is None:
        fo0 = 0.0
    if p0 is None:
        p0 = (0.0, 0.0)
    x0, y0 = p0
    if df_columns is None:
        df_columns = reg.getPar(['b', 'fo', 'ro', 'fov', 'I_T', 'x', 'y', 'd', 'v', 'A_T'])
    AA = np.ones([Nticks, len(df_columns)]) * np.nan

    controller = BaseController(**m.physics)
    l = m.body.length
    bend_errors = 0
    DL = DefaultLocomotor(dt=dt, conf=m.brain)
    for qq in range(100):
        if random.uniform(0, 1) < 0.5:
            DL.step(A_in=0, length=l)
    b, fo, ro, fov, x, y, dst, v = 0, fo0, 0, 0, x0, y0, 0, 0
    for i in range(Nticks):
        lin, ang, feed = DL.step(A_in=0, length=l)
        v = lin * controller.lin_vel_coef
        fov += (-controller.ang_damping * fov - controller.body_spring_k * b + ang * controller.torque_coef) * dt

        d_or = fov * dt
        if np.abs(d_or) > np.pi:
            bend_errors += 1
        dst = v * dt
        d_ro = controller.compute_delta_rear_angle(b, dst, l)
        b = aux.wrap_angle_to_0(b + d_or - d_ro)
        fo = (fo + d_or) % (2 * np.pi)
        ro = (ro + d_ro) % (2 * np.pi)
        x += dst * np.cos(fo)
        y += dst * np.sin(fo)

        AA[i, :] = [b, fo, ro, fov, DL.turner.input, x, y, dst, v, DL.turner.output]

    AA[:, :4] = np.rad2deg(AA[:, :4])
    return AA


def sim_multi_agents(Nticks, Nids, ms, group_id, dt=0.1, ids=None, p0s=None, fo0s=None):
    df_columns = reg.getPar(['b', 'fo', 'ro', 'fov', 'I_T', 'x', 'y', 'd', 'v', 'A_T'])
    if ids is None:
        ids = [f'{group_id}{j}' for j in range(Nids)]
    if p0s is None:
        p0s = [(0.0, 0.0) for j in range(Nids)]
    if fo0s is None:
        fo0s = [0.0 for j in range(Nids)]
    my_index = pd.MultiIndex.from_product([np.arange(Nticks), ids], names=['Step', 'AgentID'])
    AA = np.ones([Nticks, Nids, len(df_columns)]) * np.nan

    for j, id in enumerate(ids):
        m = ms[j]
        AA[:, j, :] = sim_single_agent(m, Nticks, dt=dt, df_columns=df_columns, p0=p0s[j], fo0=fo0s[j])

    AA = AA.reshape(Nticks * Nids, len(df_columns))
    s = pd.DataFrame(AA, index=my_index, columns=df_columns)
    s = s.astype(float)

    e = pd.DataFrame(index=ids)
    e['cum_dur'] = Nticks * dt
    e['num_ticks'] = Nticks
    e['length'] = [m.body.initial_length for m in ms]

    from ..process.spatial import scale_to_length
    scale_to_length(s, e, keys=['v'])
    return s, e
