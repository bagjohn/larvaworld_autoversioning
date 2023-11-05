import os
import warnings

import param

from ..param import NestedConf, PositiveInteger, class_generator
from ..process.evaluation import DataEvaluation

# from src.larvaworld.lib.reg.generators import update_larva_groups

warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import numpy as np
import pandas as pd

from .. import reg, aux, plot, util
from ..reg.generators import SimConfiguration, LarvaGroupMutator

__all__ = [
    'EvalRun',
    'eval_model_graphs',
    'modelConf_analysis',
]


class EvalConf(LarvaGroupMutator, DataEvaluation):

    def __init__(self, dataset=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.target.id = 'experiment'
        self.target.config.id = 'experiment'
        self.target.color = 'grey'
        self.target.config.color = 'grey'



class EvalRun(EvalConf, SimConfiguration):

    def __init__(self, enrichment=True, screen_kws={}, **kwargs):
        '''
        Simulation mode 'Eval' compares/evaluates different models against a reference dataset obtained by a real or simulated experiment.


        Args:
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            dataset: The stored dataset used as reference to evaluate the performance of the simulated models. If not specified it is retrieved using either the storage path (parameters.dir) or the respective unique reference ID (parameters.RefID)
            dur: Duration of the simulation. If not specifies defaults to the reference dataset duration.
            experiment: The type of experiment. Defaults to 'dispersion'
            **kwargs: Arguments passed to parent class
        '''
        EvalConf.__init__(self,runtype='Eval',**kwargs)
        kwargs['dt'] = self.target.config.dt
        kwargs['duration'] = self.target.config.Nticks * kwargs['dt'] / 60
        SimConfiguration.__init__(self,runtype='Eval', **kwargs)
        # super().__init__(runtype='Eval', **kwargs)
        self.screen_kws = screen_kws
        self.enrichment = enrichment
        self.figs = aux.AttrDict({'errors': {}, 'hist': {}, 'boxplot': {}, 'stride_cycle': {}, 'loco': {}, 'epochs': {},
                                  'models': {'table': {}, 'summary': {}}})
        self.error_plot_dir = f'{self.plot_dir}/errors'

    def simulate(self):
        kws = {
            'dt': self.dt,
            'duration': self.duration,
        }


        Nm = len(self.modelIDs)
        if self.offline is None:
            from ..model.agents.larva_offline import sim_models
            print(f'Simulating offline {Nm} models : {self.groupIDs} with {self.N} larvae each')
            temp = self.s_pars + self.e_pars
            tor_durs = np.unique([int(ii[len('tortuosity') + 1:]) for ii in temp if ii.startswith('tortuosity')])
            dsp = reg.getPar('dsp')
            dsp_temp = [ii[len(dsp) + 1:].split('_') for ii in temp if ii.startswith(f'{dsp}_')]
            dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
            dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()
            c = self.target.config
            lgs = c.larva_group.new_groups(Ns=self.N, modelIDs=self.modelIDs, groupIDs=self.groupIDs, sample=self.refID)
            self.datasets = sim_models(modelIDs=self.modelIDs, tor_durs=tor_durs,
                                       dsp_starts=dsp_starts, dsp_stops=dsp_stops,
                                       groupIDs=self.groupIDs, lgs=lgs,
                                       enrichment=self.enrichment,
                                       Nids=self.N, env_params=c.env_params,
                                       refDataset=self.target, data_dir=self.data_dir, **kws)
        else:
            from .single_run import ExpRun
            print(f'Simulating {Nm} models : {self.groupIDs} with {self.N} larvae each')
            kws0 = aux.AttrDict({
                'dir': self.dir,
                'store_data': self.store_data,
                'experiment': self.experiment,
                'id': self.id,
                'offline': self.offline,
                'modelIDs': self.modelIDs,
                'groupIDs': self.groupIDs,
                'N': self.N,
                'sample': self.refID,
                # 'parameters': conf,
                'screen_kws': self.screen_kws,
                **kws
            })
            run = ExpRun(**kws0)
            self.datasets = run.simulate()
        self.analyze()
        if self.store_data:
            self.store()
        return self.datasets

    def get_error_plots(self, error_dict, mode='pooled'):
        GD = reg.graphs.dict
        label_dic = {
            '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
            'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}
        }
        labels = label_dic[mode]
        dic = aux.AttrDict()
        for norm in self.norm_modes:
            d = self.norm_error_dict(error_dict, mode=norm)
            df0 = pd.DataFrame.from_dict({k: df.mean(axis=1) for i, (k, df) in enumerate(d.items())})
            kws = {
                'save_to': f'{self.error_plot_dir}/{norm}',
            }
            bars = {}
            tabs = {}
            for k, df in d.items():
                tabs[k] = GD['error table'](data=df, k=k, title=labels[k], **kws)
            tabs['mean'] = GD['error table'](data=df0, k='mean', title='average error', **kws)
            # print(d.step.keys())
            # print(d.end.keys())
            bars['full'] = GD['error barplot'](error_dict=d, evaluation=self.evaluation, labels=labels, **kws)
            # Summary figure with barplots and tables for both endpoint and timeseries metrics
            bars['summary'] = GD['error summary'](norm_mode=norm, eval_mode=mode, error_dict=d,
                                                  evaluation=self.evaluation, **kws)
            dic[norm] = {'tables': tabs, 'barplots': bars}
        return aux.AttrDict(dic)

    def analyze(self, **kwargs):
        reg.vprint('Evaluating all models',1)
        os.makedirs(self.plot_dir, exist_ok=True)

        for mode in self.eval_modes:
            d = self.eval_datasets(self.datasets, mode=mode, **kwargs)
            self.figs.errors[mode] = self.get_error_plots(d, mode)
            self.error_dicts[mode] = d

    def store(self):
        aux.save_dict(self.error_dicts, f'{self.data_dir}/error_dicts.txt')
        reg.vprint(f'Results saved at {self.data_dir}')

    def plot_models(self, **kwargs):
        GD = reg.graphs.dict
        save_to = self.plot_dir
        for mID in self.modelIDs:
            self.figs.models.table[mID] = GD['model table'](mID=mID, save_to=save_to, figsize=(14, 11), **kwargs)
            self.figs.models.summary[mID] = GD['model summary'](mID=mID, save_to=save_to, refID=self.refID, **kwargs)

    def plot_results(self, plots=['hists', 'trajectories', 'dispersion', 'bouts', 'fft', 'boxplots'], **kwargs):
        GD = reg.graphs.dict

        self.target.load(h5_ks=['epochs', 'angular', 'dspNtor'])
        kws = {
            'datasets': [self.target] + self.datasets,
            'save_to': self.plot_dir,
            **kwargs
        }
        kws1 = {
            'subfolder': None,
            **kws
        }

        kws2 = {
            'target': self.target,
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            **kwargs
        }
        self.figs.summary = GD['eval summary'](**kws2)
        self.figs.stride_cycle.norm = GD['stride cycle'](shorts=['sv', 'fov', 'rov', 'foa', 'b'],
                                                         individuals=True, **kws)
        if 'dispersion' in plots:
            for r0, r1 in itertools.product(self.dsp_starts, self.dsp_stops):
                self.figs.loco[f'dsp_{r0}_{r1}'] = aux.AttrDict({
                    'plot': GD['dispersal'](range=(r0, r1), **kws1),
                    'traj': GD['trajectories'](name=f'traj_{r0}_{r1}', range=(r0, r1), mode='origin', **kws1),
                    'summary': GD['dispersal summary'](range=(r0, r1), **kws2)
                })
        if 'bouts' in plots:
            self.figs.epochs.turn = GD['epochs'](turns=True, **kws)
            self.figs.epochs.runNpause = GD['epochs'](stridechain_duration=True, **kws)
        if 'fft' in plots:
            self.figs.loco.fft = GD['freq powerspectrum'](**kws)
        if 'hists' in plots:
            self.figs.hist.ang = GD['angular pars'](half_circles=False, absolute=False, Nbins=100, Npars=3,
                                                    include_rear=False, **kws1)
            self.figs.hist.crawl = GD['crawl pars'](pvalues=False, **kws1)
        if 'trajectories' in plots:
            self.figs.loco.trajectories = GD['trajectories'](**kws1)
        if 'boxplots' in plots:
            pass
            # self.figs.boxplot.end = self.plot_data(mode='end', type='box')
            # self.figs.boxplot.step = self.plot_data(mode='step', type='box')


reg.gen.Eval = class_generator(EvalConf)


def eval_model_graphs(refID, mIDs, groupIDs=None, id=None, dir=None, N=10,**kwargs):
    if id is None:
        id = f'{len(mIDs)}mIDs'
    if dir is None:
        dir = f'{reg.conf.Ref.getID(refID)}/model/evaluation'


    evrun = EvalRun(refID=refID, modelIDs=mIDs, groupIDs=groupIDs, N=N, id=id,dir=dir, **kwargs)
    evrun.simulate()
    evrun.plot_models()
    evrun.plot_results()
    return evrun


def modelConf_analysis(d):
    from collections import ChainMap
    warnings.filterwarnings('ignore')
    M=reg.model
    CM=reg.conf.Model

    mods = aux.AttrDict({
        'C': ['RE', 'SQ', 'GAU', 'CON'],
        'T': ['NEU', 'SIN', 'CON'],
        'If': ['PHI', 'SQ', 'DEF'],
    })

    fit_kws = {
        'eval_metrics': {
            'angular kinematics': ['b', 'fov', 'foa'],
            'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                     'dsp_0_40_max', 'dsp_0_60_max'],
            'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
        },
        'cycle_curve_metrics': ['fov', 'foa', 'b']
    }

    sample_kws = {k: 'sample' for k in [
        'brain.crawler_params.stride_dst_mean',
        'brain.crawler_params.stride_dst_std',
        'brain.crawler_params.max_scaled_vel',
        'brain.crawler_params.max_vel_phase',
        'brain.crawler_params.freq',
    ]}

    c = d.config
    kws = {'refID': c.refID}
    kws1 = {'N': 10, **kws}
    kws2 = {'dataset': d, **fit_kws}
    D = aux.AttrDict({'average': {}, 'variable': {}, 'individual': {}, '3modules': {}})
    ee = []
    for tt in mods.T[:2]:
        for ii in mods.If:
            ee.append(M.adapt_mID(mID0=f'RE_{tt}_{ii}_DEF', mID=f'{ii}on{tt}', space_mkeys=['turner', 'interference'], **kws2))
    D.average=dict(ChainMap(*ee))

    mIDs_avg = list(D.average)
    mIDs_var = [f'{mID0}_var' for mID0 in mIDs_avg]

    for mID0, mID in zip(mIDs_avg, mIDs_var):
        m0 = CM.getID(mID0).get_copy()
        D.variable[mID] = m0.update_existingnestdict(sample_kws)
        CM.setID(mID, D.variable[mID])



    ee=[]
    for cc in mods.C:
        for ii in mods.If:
            mIDs0x3=[f'{cc}_{tt}_{ii}_DEF' for tt in mods.T]
            mIDsx3 =[f'{mID0}_fit' for mID0 in mIDs0x3]
            ee+=[M.adapt_mID(mID0=mID0, mID=mID, space_mkeys=['crawler', 'turner', 'interference'],
                                **kws2) for mID0,mID in zip(mIDs0x3,mIDsx3)]
            eval_model_graphs(mIDs=mIDsx3, groupIDs=mods.T, id=f'Tmod_variable_Cmod_{cc}_Ifmod_{ii}', **kws1)
    D['3modules']=dict(ChainMap(*ee))
    mIDs_3m = list(D['3modules'])


    reg.graphs.store_model_graphs(mIDs_avg, d.dir)
    reg.graphs.store_model_graphs(mIDs_3m, d.dir)

    eval_model_graphs(mIDs=mIDs_avg, id='6mIDs_avg', **kws1)
    eval_model_graphs(mIDs=mIDs_var, id='6mIDs_var', **kws1)
    eval_model_graphs(mIDs=mIDs_avg[:3] + mIDs_var[:3], id='3mIDs_avgVSvar1', **kws1)
    eval_model_graphs(mIDs=mIDs_avg[3:] + mIDs_var[3:], id='3mIDs_avgVSvar2', **kws1)



    d.config.modelConfs = D
    d.save_config()
