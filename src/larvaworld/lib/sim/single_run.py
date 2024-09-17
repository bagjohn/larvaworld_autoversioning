import os
import time
import agentpy
import numpy as np
import pandas as pd

from .. import reg, aux, plot
from .conditions import get_exp_condition
from ..sim.base_run import BaseRun
from ..process.dataset import LarvaDatasetCollection

__all__ = [
    'ExpRun',
]


class ExpRun(BaseRun):
    def __init__(self, experiment=None, parameters=None, parameter_dict={}, **kwargs):
        '''
        Simulation mode 'Exp' launches a single simulation of a specified experiment type.

        Args:
            **kwargs: Arguments passed to the setup method

        '''

        super().__init__(runtype='Exp', experiment=experiment, parameters=parameters, **kwargs)
        self.parameter_dict = parameter_dict

    def setup(self):

        self.sim_epochs = self.p.trials.epochs
        for ep in self.sim_epochs:
            t1, t2 = ep.age_range
            ep['start'] = int(t1 * 60 / self.dt)
            ep['stop'] = int(t2 * 60 / self.dt)
        self.build_env(self.p.env_params)
        self.build_agents(self.p.larva_groups, self.parameter_dict)
        self.set_collectors(self.p.collections)
        self.accessible_sources = None
        if not self.larva_collisions:
            self.eliminate_overlap()
        k = get_exp_condition(self.experiment)
        self.exp_condition = k(self) if k is not None else None

    def step(self):
        """ Defines the models' events per simulation step. """
        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        if len(self.sources) > 10:
            self.space.accessible_sources_multi(self.agents)
        self.agents.step()
        if self.Box2D:
            self.space.Step(self.dt, 6, 2)
            self.agents.updated_by_Box2D()

    def update(self):
        """ Record a dynamic variable. """
        self.agents.nest_record(self.collectors['step'])

    def end(self):
        """ Repord an evaluation measure. """
        self.screen_manager.finalize()
        self.agents.nest_record(self.collectors['end'])

    def simulate(self, **kwargs):
        reg.vprint(f'--- Simulation {self.id} initialized!--- ', 1)
        start = time.time()
        self.run(**kwargs)
        self.data_collection = LarvaDatasetCollection.from_agentpy_output(self.output)
        self.datasets = self.data_collection.datasets
        end = time.time()
        dur = np.round(end - start).astype(int)
        reg.vprint(f'--- Simulation {self.id} completed in {dur} seconds!--- ', 1)
        if self.p.enrichment:
            for d in self.datasets:
                reg.vprint(f'--- Enriching dataset {d.id} ---', 1)
                d.enrich(**self.p.enrichment, is_last=False)
                reg.vprint(f'--- Dataset {d.id} enriched ---', 1)
                reg.vprint(f'--------------------------------', 1)
        if self.store_data:
            self.store()
        return self.datasets

    def build_agents(self, larva_groups, parameter_dict={}):
        reg.vprint(f'--- Simulation {self.id} : Generating agent groups!--- ', 1)
        confs = aux.SuperList([reg.generators.LarvaGroup(**v)(parameter_dict=parameter_dict) for v in larva_groups.values()]).flatten
        self.place_agents(confs)

    def eliminate_overlap(self):
        scale = 3.0
        while self.collisions_exist(scale=scale):
            self.larva_bodies = self.get_larva_bodies(scale=scale)
            for l in self.agents:
                dx, dy = np.random.randn(2) * l.sim_length / 10
                overlap = True
                while overlap:
                    ids = self.detect_collisions(l.unique_id)
                    if len(ids) > 0:
                        l.move_body(dx, dy)
                        self.larva_bodies[l.unique_id] = l.get_polygon(scale=scale)
                    else:
                        break

    def collisions_exist(self, scale=1.0):
        self.larva_bodies = self.get_larva_bodies(scale=scale)
        for l in self.agents:
            ids = self.detect_collisions(l.unique_id)
            if len(ids) > 0:
                return True
        return False

    def detect_collisions(self, id):
        ids = []
        for id0, body0 in self.larva_bodies.items():
            if id0 == id:
                continue
            if self.larva_bodies[id].intersects(body0):
                ids.append(id0)
        return ids

    def get_larva_bodies(self, scale=1.0):
        return {l.unique_id: l.get_shape(scale=scale) for l in self.agents}

    def analyze(self, **kwargs):
        os.makedirs(self.plot_dir, exist_ok=True)
        exp = self.experiment
        ds = self.datasets
        if ds is None or any([d is None for d in ds]):
            return

        if 'PI' in exp:
            PIs = {}
            PI2s = {}
            for d in ds:
                PIs[d.id] = d.config.PI["PI"]
                PI2s[d.id] = d.config.PI2
            self.results = {'PIs': PIs, 'PI2s': PI2s}
            return

        if 'disp' in exp:
            samples = aux.unique_list([d.config.sample for d in ds])
            ds += [reg.conf.Ref.loadRef(sd) for sd in samples if sd is not None]
        graphgroups = reg.graphs.get_analysis_graphgroups(exp, self.p.source_xy)
        self.figs = reg.graphs.eval_graphgroups(graphgroups, datasets=ds, save_to=self.plot_dir, **kwargs)

    def store(self):
        try:
            self.output.save(**self.p.agentpy_output_kws)
        except:
            pass
        os.makedirs(self.data_dir, exist_ok=True)
        for d in self.datasets:
            d.save()
            d.store_larva_dicts()

    def load_agentpy_output(self):
        df = agentpy.DataDict.load(**self.p.agentpy_output_kws)
        df1 = pd.concat(df.variables, axis=0).droplevel(1, axis=0)
        df1.index.rename('Model', inplace=True)
        return df1

    @classmethod
    def from_ID(cls, id, simulate=True, **kwargs):
        assert id in reg.conf.Exp.confIDs
        r = cls(experiment=id, **kwargs)
        if simulate:
            _ = r.simulate()
        return r
