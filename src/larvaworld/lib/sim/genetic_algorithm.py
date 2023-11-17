import random
import multiprocessing
import math
import threading
import warnings
import param
import pandas as pd
import progressbar
import numpy as np

import larvaworld
from .. import reg, aux, util
from ..param import NestedConf, ClassAttr, class_generator, SimOps, OptionalSelector
from ..process.evaluation import Evaluation
from .base_run import BaseRun

__all__ = [
    'GAevaluation',
    'GAselector',
    'GAlauncher',
    'optimize_mID',
]


def dst2source_evaluation(robot, source_xy):
    traj = np.array(robot.trajectory)
    dst = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst = np.sum(dst)
    l = []
    for label, pos in source_xy.items():
        l.append(aux.eudi5x(traj, pos))
    fitness = - np.mean(np.min(np.vstack(l), axis=0)) / cum_dst
    return fitness


def cum_dst(robot, **kwargs):
    return robot.cum_dst / robot.length


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 20:
        return True
    else:
        return False


fitness_funcs = aux.AttrDict({
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})

exclusion_funcs = aux.AttrDict({
    'bend_errors': bend_error_exclusion
})


class GAevaluation(Evaluation):
    exclusion_mode = param.Boolean(default=False, label='exclusion mode', doc='Whether to apply exclusion mode')
    exclude_func_name = OptionalSelector(default=None, objects=list(exclusion_funcs.keys()),
                                         label='name of exclusion function',
                                         doc='The function that evaluates exclusion', allow_None=True)
    fitness_func_name = OptionalSelector(default=None, objects=list(fitness_funcs.keys()),
                                         label='name of fitness function', doc='The function that evaluates fitness',
                                         allow_None=True)

    fit_kws = param.Dict(default={}, label='fitness metrics to evaluate', doc='The target metrics to optimize against')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.exclude_func = exclusion_funcs[self.exclude_func_name] if type(self.exclude_func_name) == str else None

        if self.exclusion_mode:
            self.fit_func = None
        elif self.fitness_func_name and self.fitness_func_name in fitness_funcs:

            def func(robot):
                return fitness_funcs[self.fitness_func_name](robot, **self.fit_kws)

            self.fit_func_arg = 'robot'
            self.fit_func = func
        elif self.target:
            self.fit_func_arg = 's'
            self.fit_func = self.fit_func_solo
        else:
            raise


class GAselector(NestedConf):
    Ngenerations = param.Integer(default=None, allow_None=True, label='# generations',
                                 doc='Number of generations to run for the genetic algorithm engine')
    Nagents = param.Integer(default=30, label='# agents per generation', doc='Number of agents per generation')
    Nelits = param.Integer(default=3, label='# best agents for next generation',
                           doc='Number of best agents to include in the next generation')

    selection_ratio = param.Magnitude(default=0.3, label='selection ratio',
                                      doc='Fraction of agent population to include in the next generation')
    Pmutation = param.Magnitude(default=0.3, label='mutation probability',
                                doc='Probability of mutation for each agent in the next generation')
    Cmutation = param.Number(default=0.1, label='mutation coeficient',
                             doc='Fraction of allowed parameter range to mutate within')
    bestConfID = param.String(default='best_model', label='model ID for optimized model',
                              doc='ID for the optimized model')

    init_mode = param.Selector(default='random', objects=['random', 'model', 'default'],
                               label='mode of initial generation', doc='Mode of initial generation')
    base_model = reg.conf.Model.confID_selector('explorer')

    space_mkeys = param.ListSelector(default=[], objects=reg.model.mkeys,
                                     label='keys of modules to include in space search',
                                     doc='Keys of the modules where the optimization parameters are')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.Nagents_min = round(self.Nagents * self.selection_ratio)
        if self.Nagents_min < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')

        self.mConf0 = reg.conf.Model.getID(self.base_model)
        self.space_dict = reg.model.space_dict(mkeys=self.space_mkeys, mConf0=self.mConf0)
        # self.space_columns = [p.name for k, p in self.space_dict.items()]
        self.gConf0 = reg.model.conf(self.space_dict)

    def create_first_generation(self):
        mode = self.init_mode
        N = self.Nagents
        d = self.space_dict
        if mode == 'default':
            gConf = reg.model.conf(d)
            gConfs = [gConf] * N
        elif mode == 'model':
            gConf = {k: self.mConf0.flatten()[k] for k, p in d.items()}
            gConfs = [gConf] * N
        elif mode == 'random':
            gConfs = []
            for i in range(N):
                reg.model.randomize(d)
                gConf = reg.model.conf(d)
                gConfs.append(gConf)
        else:
            raise ValueError('Not implemented')
        return gConfs

    def new_genome(self, gConf, mConf0):
        mConf = mConf0.update_nestdict(gConf)
        mConf.life_history = {'age': 0.0, 'epochs': {}}
        return aux.AttrDict({'fitness': None, 'fitness_dict': {}, 'gConf': gConf, 'mConf': mConf})

    def create_new_generation(self, sorted_gs):
        if len(sorted_gs) < self.Nagents_min:
            raise ValueError(
                f'The number of genomes ({len(sorted_gs)}) is lower than the minimum number required to breed a new generation ({self.Nagents_min})')
        gs0 = [sorted_gs[i].gConf for i in range(self.Nagents_min)]

        # elitism: keep the best genomes in the new generation
        gs = [gs0[i] for i in range(self.Nelits)]

        for i in range(self.Nagents - self.Nelits):
            g1, g2 = random.sample(gs0, 2)
            g0 = self.crossover(g1, g2)
            space_dict = reg.model.update_mdict(self.space_dict, g0)
            for d, p in space_dict.items():
                p.mutate(Pmut=self.Pmutation, Cmut=self.Cmutation)

            g = reg.model.generate_configuration(space_dict)
            gs.append(g)
        return gs

    def create_generation(self, sorted_gs=None):
        if sorted_gs is None:
            self.gConfs = self.create_first_generation()
        else:
            self.gConfs = self.create_new_generation(sorted_gs)
        return {i: self.new_genome(gConf, self.mConf0) for i, gConf in enumerate(self.gConfs)}

    def crossover(self, g1, g2):
        g = {}
        for k in g1:
            if np.random.uniform(0, 1, 1) >= 0.5:
                g[k] = g1[k]
            else:
                g[k] = g2[k]
        return g


class GAlauncher(BaseRun):

    def __init__(self, dataset=None, **kwargs):
        '''
        Simulation mode 'Ga' launches a genetic algorith optimization simulation of a specified agent model.

        Args:
            **kwargs: Arguments passed to the setup method

        '''

        super().__init__(runtype='Ga', **kwargs)
        self.evaluator = GAevaluation(dataset=dataset, **self.p.ga_eval_kws)
        self.selector = GAselector(**self.p.ga_select_kws)

    def setup(self):
        self.genome_dict = None
        self.best_genome = None
        self.best_fitness = None
        self.sorted_genomes = None
        self.all_genomes_dic = []

        self.generation_num = 0
        # self.start_total_time = aux.TimeUtil.current_time_millis()

        Ngens = self.selector.Ngenerations

        reg.vprint(f'--- Genetic Algorithm  "{self.id}" initialized!--- ', 2)
        if Ngens is not None:
            self.progress_bar = progressbar.ProgressBar(Ngens)
            self.progress_bar.start()
            temp = Ngens
        else:
            self.progress_bar = None
            temp = 'unlimited'
        # self.gen_progressbar=progressbar.ProgressBar(self.Nsteps)
        # self.gen_progressbar.start()
        reg.vprint(
            f'Launching {temp} generations of {self.duration} minutes, with {self.selector.Nagents} agents each!', 2)
        self.p.collections = ['pose']
        self.build_env(self.p.env_params)

        self.build_generation()

    def simulate(self):
        self.sim_setup()
        while self.running:
            self.sim_step()
        return self.best_genome

    def build_generation(self, sorted_genomes=None):
        self.prestart_generation_time = aux.TimeUtil.current_time_sec()
        self.generation_num += 1
        self.genome_dict = self.selector.create_generation(sorted_genomes)
        confs = [{'larva_pars': g.mConf, 'unique_id': str(id), 'genome': g, 'color': aux.Color.random_bright()} for
                 id, g in self.genome_dict.items()]
        self.place_agents(confs)
        self.set_collectors(self.p.collections)
        if self.multicore:
            self.threads = self.build_threads(self.agents)
        else:
            self.threads = None

        # self.generation_step_num = 0

        if self.progress_bar:
            self.progress_bar.update(self.generation_num)
        # self.gen_progressbar.start()
        self.start_generation_time = aux.TimeUtil.current_time_sec()
        gen_load_dur = np.round(self.start_generation_time - self.prestart_generation_time)
        reg.vprint(f'Generation {self.generation_num} started in {gen_load_dur} sec', 1)

    def eval_robots(self, ds, Ngen, genome_dict):
        reg.vprint(f'Evaluating generation {Ngen}', 1)
        assert self.evaluator.fit_func_arg == 's'
        for d in ds:
            d.enrich(proc_keys=['angular', 'spatial'],
                     dsp_starts=[], dsp_stops=[], tor_durs=[], is_last=False)
            valid_gs = {}
            for i, g in genome_dict.items():
                # for id in d.config.agent_ids:
                ss = d.step_data.xs(str(i), level='AgentID')
                g.fitness_dict = self.evaluator.fit_func(ss)
                mus = aux.AttrDict({k: -np.mean(list(dic.values())) for k, dic in g.fitness_dict.items()})
                if len(mus) == 1:
                    g.fitness = list(mus.values())[0]
                else:
                    coef_dict = {'KS': 10, 'RSS': 1}
                    g.fitness = np.sum([coef_dict[k] * mean for k, mean in mus.items()])
                if not np.isnan(g.fitness):
                    valid_gs[i] = g
            sorted_gs = [valid_gs[i] for i in
                         sorted(list(valid_gs.keys()), key=lambda i: valid_gs[i].fitness, reverse=True)]
            self.store(sorted_gs, Ngen)
            # reg.vprint(f'Generation {Ngen} evaluated', 1)
            return sorted_gs

    def store(self, sorted_gs, Ngen):
        if self.best_genome is None or sorted_gs[0].fitness > self.best_genome.fitness:
            self.best_genome = sorted_gs[0]
            self.best_fitness = self.best_genome.fitness
            reg.conf.Model.setID(self.selector.bestConfID, self.best_genome.mConf)
        reg.vprint(f'Generation {Ngen} best_fitness : {self.best_fitness}', 1)
        if self.store_data:
            self.all_genomes_dic += [
                {'generation': Ngen, **{p.name: g.gConf[k] for k, p in self.selector.space_dict.items()},
                 'fitness': g.fitness, **g.fitness_dict.flatten()}
                for g in sorted_gs if g.fitness_dict is not None]

    @property
    def generation_completed(self):
        return self.t >= self.Nsteps or len(self.agents) <= self.selector.Nagents_min

    @property
    def max_generation_completed(self):
        return self.selector.Ngenerations is not None and self.generation_num >= self.selector.Ngenerations

    def sim_step(self):
        self.t += 1
        self.step()
        self.screen_manager.step()
        self.update()
        # self.generation_step_num += 1
        if self.generation_completed:
            self.end()
            if not self.max_generation_completed:
                self.build_generation(self.sorted_genomes)
            else:
                self.finalize()

    def step(self):
        if self.threads:
            for thr in self.threads:
                thr.step()
        else:
            self.agents.step()
        if self.evaluator.exclude_func is not None:
            for robot in self.agents:
                if self.evaluator.exclude_func(robot):
                    robot.genome.fitness = -np.inf

    def end(self):
        self.end_generation_time = aux.TimeUtil.current_time_sec()
        gen_dur = self.end_generation_time - self.start_generation_time
        reg.vprint(f'Generation {self.generation_num} completed in {gen_dur} sec', 1)
        self.agents.nest_record(self.collectors['end'])
        self.create_output()
        self.data_collection = larvaworld.lib.LarvaDatasetCollection.from_agentpy_output(self.output)
        self.sorted_genomes = self.eval_robots(ds=self.data_collection.datasets, Ngen=self.generation_num,
                                               genome_dict=self.genome_dict)
        self.delete_agents()
        self._logs = {}
        self.t = 0
        # self.gen_progressbar.finish()
        self.end_generation_eval_time = aux.TimeUtil.current_time_sec()
        gen_eval_dur = self.end_generation_eval_time - self.prestart_generation_time
        reg.vprint(f'Generation {self.generation_num} evaluated in {gen_eval_dur} sec', 2)

    def update(self):
        self.agents.nest_record(self.collectors['step'])
        # self.gen_progressbar.update(self.t)

    def finalize(self):
        self.running = False
        self.screen_manager.finalize()
        if self.progress_bar:
            self.progress_bar.finish()
        reg.vprint(f'Best fittness: {self.best_genome.fitness}', 1)
        if self.store_data:
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.data_dir)

    def store_genomes(self, dic, save_to):
        df = pd.DataFrame.from_records(dic)
        df = df.round(3)
        df.sort_values(by='fitness', ascending=False, inplace=True)
        reg.graphs.dict['mpl'](data=df, font_size=18, save_to=save_to,
                               name=self.selector.bestConfID)
        df.to_csv(f'{save_to}/{self.selector.bestConfID}.csv')

        cols = [p.name for k, p in self.selector.space_dict.items()]

        self.corr_df = df[['fitness'] + cols].corr()
        try:
            self.diff_df, row_colors = reg.model.diff_df(mIDs=[self.selector.base_model, self.selector.bestConfID],
                                                         ms=[self.selector.mConf0, self.best_genome.mConf])
        except:
            pass

    def build_threads(self, robots):
        N = multiprocessing.cpu_count()
        threads = []
        N_per_cpu = math.floor(len(robots) / N)
        reg.vprint(f'num_robots_per_cpu: {N_per_cpu}', 0)

        for i in range(N - 1):
            p0 = i * N_per_cpu
            p1 = (i + 1) * N_per_cpu
            reg.vprint(f'core: {i + 1} positions: {p0} : {p1}', 0)
            thread = GA_thread(robots[p0:p1])
            thread.start()
            reg.vprint(f'thread {i + 1} started', 0)
            threads.append(thread)

        # last sublist of robots
        p0 = (N - 1) * N_per_cpu
        reg.vprint(f'last core, start_pos {p0}', 0)
        thread = GA_thread(robots[p0:])
        thread.start()
        reg.vprint(f'last thread started', 0)
        threads.append(thread)

        for t in threads:
            t.join()
        return threads


class GA_thread(threading.Thread):
    def __init__(self, robots):
        threading.Thread.__init__(self)
        self.robots = robots

    def step(self):
        for robot in self.robots:
            robot.step()


def optimize_mID(mID0, mID1=None, refID=None, space_mkeys=['turner', 'interference'],
                 experiment='exploration', dataset=None, multicore=False,
                 id=None, dt=1 / 16, dur=0.4, dir=None, Nagents=10, Nelits=2, Ngenerations=3,
                 **kwargs):
    warnings.filterwarnings('ignore')
    if mID1 is None:
        mID1 = mID0
    # print(mID0,mID0 in reg.conf.Model.confIDs)
    # gaconf = reg.gen.Ga(ga_select_kws=reg.gen.GAselector(Nagents=Nagents, Nelits=Nelits, Ngenerations=Ngenerations,
    #                                                      init_mode=init, space_mkeys=space_mkeys,
    #                                                      base_model=mID0, bestConfID=mID1),
    #                     ga_eval_kws=reg.gen.GAevaluation(refID=refID, **kwargs),
    #                     env_params='arena_200mm',
    #                     experiment=experiment).nestedConf

    gaconf = aux.AttrDict({'ga_select_kws': {'Nagents': Nagents, 'Nelits': Nelits, 'Ngenerations': Ngenerations,
                                             'init_mode': 'model', 'space_mkeys': space_mkeys,
                                             'base_model': mID0, 'bestConfID': mID1},
                           'ga_eval_kws': {'refID': refID, **kwargs},
                           'env_params': reg.conf.Env.getID('arena_200mm'),
                           'experiment': experiment})
    # gaconf.env_params = reg.conf.Env.getID(gaconf.env_params)
    GA = GAlauncher(parameters=gaconf, dir=dir, id=id, duration=dur, dt=dt, dataset=dataset, multicore=multicore)
    best_genome = GA.simulate()
    return {mID1: best_genome.mConf}


reg.gen.GAselector = class_generator(GAselector)
reg.gen.GAevaluation = class_generator(GAevaluation)


class GAconf(SimOps):
    env_params = reg.conf.Env.confID_selector()
    experiment = reg.conf.Ga.confID_selector()
    ga_eval_kws = ClassAttr(reg.gen.GAevaluation, doc='The GA evaluation configuration')
    ga_select_kws = ClassAttr(reg.gen.GAselector, doc='The GA selection configuration')

    scene = param.String('no_boxes', doc='The name of the scene to load')


# reg.gen.Ga=class_generator(GAlauncher)
reg.gen.Ga = class_generator(GAconf)
