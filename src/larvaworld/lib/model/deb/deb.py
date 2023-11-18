'''
DEB pipeline from literature
'''
import json
import os
import numpy as np
import param

from ... import reg, aux
from ...aux import nam

from ...model import deb
from ...param import Substrate, NestedConf, PositiveNumber, PositiveInteger, ClassAttr, substrate_dict, \
    Epoch, OptionalPositiveNumber

'''
Standard culture medium
50g Baker’s yeast; 100g sucrose; 16g agar; 0.1gKPO4; 8gKNaC4H4O6·4H2O; 0.5gNaCl; 0.5gMgCl2; and 0.5gFe2(SO4)3 per liter of tap water. 
Larvae were reared from egg-hatch to mid- third-instar (96±2h post-hatch) in 25°C at densities of 100 larvae per 35ml of medium in 100mm⫻15mm Petri dishes


[1] K. R. Kaun, M. Chakaborty-Chatterjee, and M. B. Sokolowski, “Natural variation in plasticity of glucose homeostasis and food intake,” J. Exp. Biol., vol. 211, no. 19, pp. 3160–3166, 2008.

--> 0.35 ml medium per larva for the 4 days
'''

__all__ = [
    'DEB_basic',
    'DEB',
    'deb_default',
    'DEB_runner',
    'get_best_EEB',
    'deb_sim',
]


class DEB_basic(NestedConf):
    F_m = PositiveNumber(6.5, doc='maximum surface-area specific searching rate (l cm**-2 d**-1)')
    kap_X = param.Magnitude(0.8, doc='assimilation efficiency')
    p_Am = PositiveNumber(229.0, doc='maximum surface-area specific assimilation rate (J cm**-2 d**-1)')
    E_G = PositiveNumber(4400.0, doc='volume-specific cost of structure (J/cm**3)')
    v = PositiveNumber(0.12, doc='energy conductance (cm/d)')
    p_M = PositiveNumber(210.0, doc='volume-specific somatic maintenance (J cm**-3 d**-1)')
    kap = param.Magnitude(0.99, doc='fraction of mobilized reserve allocated to soma')
    k_J = PositiveNumber(0.002, doc='maturity maintenance rate coefficient (d**-1)')
    E_Hb = PositiveNumber(0.0006, doc='maturity threshold from embryo to juvenile (J)')
    E_He = PositiveNumber(0.05, doc='maturity threshold from juvenile to adult (J)')
    z = PositiveNumber(0.5, doc='zoom factor')
    del_M = PositiveNumber(0.5, doc='shape correction coefficient')
    s_G = PositiveNumber(0.0001, doc='Gompertz stress coefficient')
    h_a = PositiveNumber(0.0001, doc='Weibull ageing acceleration (d**-2)')
    kap_R = param.Magnitude(0.95, doc='fraction of the reproduction buffer fixed into eggs')

    p_T = PositiveNumber(0.0, doc='??')
    kap_V = param.Magnitude(0.99, doc='??')
    kap_P = param.Magnitude(0.18, doc='??')

    eb = PositiveNumber(1.0, doc='scaled reserve density at birth')
    s_j = PositiveNumber(0.999, doc='??')

    T = PositiveNumber(298.15, doc='Temperature')
    T_ref = PositiveNumber(293.15, doc='Reference temperature')
    T_A = PositiveNumber(8000, doc='Arrhenius temperature')

    mu_E = PositiveNumber(550000.0, doc='specific chemical potential of compound E')
    mu_V = PositiveNumber(500000.0, doc='specific chemical potential of compound V')
    mu_X = PositiveNumber(525000.0, doc='specific chemical potential of compound X')
    mu_P = PositiveNumber(500000.0, doc='specific chemical potential of compound P')
    mu_C = PositiveNumber(480000.0, doc='specific chemical potential of compound C')
    mu_H = PositiveNumber(0.0, doc='specific chemical potential of compound H')
    mu_O = PositiveNumber(0.0, doc='specific chemical potential of compound O')
    mu_N = PositiveNumber(0.0, doc='specific chemical potential of compound N')
    d_V = PositiveNumber(0.17, doc='density of compound V')
    d_X = PositiveNumber(0.17, doc='density of compound X')
    d_E = PositiveNumber(0.17, doc='density of compound E')
    d_P = PositiveNumber(0.17, doc='density of compound P')
    w_V = PositiveNumber(23.9, doc='molar weight of compound V')
    w_E = PositiveNumber(23.9, doc='molar weight of compound E')
    w_X = PositiveNumber(23.9, doc='molar weight of compound X')
    w_P = PositiveNumber(23.9, doc='molar weight of compound P')
    y_E_X = PositiveNumber(0.7, doc='yield coefficient that couples mass flux E to mass flux X')
    y_P_X = PositiveNumber(0.2, doc='yield coefficient that couples mass flux P to mass flux X')

    def __init__(self, print_output=False, **kwargs):
        super().__init__(**kwargs)
        self.print_output = print_output

        self.L0 = 10 ** -10

        # self.p_Am = self.z*self.p_M/self.kap
        self.E_M = self.p_Am / self.v  # maximum reserve density
        self.k_M = self.p_M / self.E_G
        self.g = self.E_G / (self.kap * self.E_M)
        ii = self.g ** 2 * self.k_M ** 3 / ((1 - self.kap) * self.v ** 2)
        self.k = self.k_J / self.k_M
        self.U_Hb = self.E_Hb / self.p_Am
        self.vHb = self.U_Hb * ii
        self.U_He = self.E_He / self.p_Am
        self.vHe = self.U_He * ii
        self.Lm = self.v / (self.g * self.k_M)

        self.J_E_Am = self.p_Am / self.mu_E
        self.J_X_Am = self.J_E_Am / self.y_E_X
        self.K = self.J_X_Am / self.F_m
        self.E_V = self.mu_V * self.d_V / self.w_V

        self.T_factor = np.exp(self.T_A / self.T_ref - self.T_A / self.T)  # Arrhenius factor
        self.lb = self.get_lb()
        self.k_E = self.g * self.k_M / self.lb

        self.E0 = self.get_E0()

        self.Lb = self.lb * self.Lm
        self.Lwb = self.Lb / self.del_M

        self.E_Rm = (self.kap-1) * self.E_M * self.g *(1-self.lb)/ (1+self.lb)
        # self.E_Rm = (1 - self.kap) * self.g * self.E_M * (self.k_E + self.k_M) / (self.k_E - self.g * self.k_M)

        self.tau_b = self.get_tau_b()
        self.t_b = self.tau_b / self.k_M / self.T_factor

        # For the larva the volume specific max assimilation rate p_Amm is used instead of the surface-specific p_Am
        self.p_Amm = self.p_Am / self.Lb
        self.J_X_Amm = self.J_X_Am / self.Lb
        self.J_E_Amm = self.J_E_Am / self.Lb
        self.F_mm = self.F_m / self.Lb

        # DEB textbook p.91
        # self.y_VE = (self.d_V / self.w_V)*self.mu_E/E_G
        # self.J_E_Am = self.p_Am/self.mu_E

        # self.U0 = self.uE0 * v ** 2 / g ** 2 / k_M ** 3
        # self.E0 = self.U0 * p_Am
        self.Ww0 = self.E0 * self.w_E / self.mu_E  # g, initial wet weight

        self.v_Rm = (1 + self.lb / self.g) / (1 - self.lb)  # scaled max reprod buffer density
        self.v_Rj = self.s_j * self.v_Rm  # scaled reprod buffer density at pupation

        if self.print_output:
            print('------------------Egg------------------')
            print(f'Reserve energy  (mJ) :       {int(1000 * self.E0)}')
            print(f'Wet weight      (mg) :       {np.round(1000 * self.Ww0, 5)}')

        # Larva stage flags
        self.stage = 'embryo'
        self.alive = True

        self.E = self.E0
        self.E_H = 0
        self.E_R = 0
        self.V = self.L0 ** 3

    def get_lb(self):
        g = self.g
        n = 1000 + round(1000 * max(0, self.k - 1))
        xb = g / (g + self.eb)
        xb3 = xb ** (1 / 3)
        x = np.linspace(10 ** -5, xb, n)
        dx = xb / n
        x3 = x ** (1 / 3)

        b = deb.beta0(x, xb) / (3 * g)

        t0 = xb * g * self.vHb
        i = 0
        norm = 1
        ni = 100

        lb = self.vHb ** (1 / 3)

        while i < ni and norm > 1e-18:
            l = x3 / (xb3 / lb - b)
            s = (self.k - x) / (1 - x) * l / g / x
            vv = np.exp(- dx * np.cumsum(s))
            vb = vv[- 1]
            r = (g + l)
            rv = r / vv
            t = t0 / lb ** 3 / vb - dx * np.sum(rv)
            dl = xb3 / lb ** 2 * l ** 2. / x3
            dlnv = np.exp(- dx * np.cumsum(s * dl / l))
            dlnvb = dlnv[- 1]
            dt = - t0 / lb ** 3 / vb * (3 / lb + dlnvb) - dx * np.sum((dl / r - dlnv) * rv)
            lb -= t / dt  # Newton Raphson step
            norm = t ** 2
            i += 1
        return lb

    def get_tau_b(self):
        from scipy.integrate import quad
        def get_tb(x, ab, xb):
            return x ** (-2 / 3) / (1 - x) / (ab - deb.beta0(x, xb))

        xb = self.g / (self.eb + self.g)
        ab = 3 * self.g * xb ** (1 / 3) / self.lb
        return 3 * quad(func=get_tb, a=1e-15, b=xb, args=(ab, xb))[0]

    def get_E0(self):
        """
            This function calculates the maximum reserve density (E0) that an organism can achieve given its energy budget parameters.

            Parameters:
                kap (float): Fraction of assimilated energy that is used for somatic maintenance.
                v (float): Energy conductance.
                p_M (float): Specific somatic maintenance costs.
                p_Am (float): Maximum surface-specific assimilation rate.
                E_G (float): Energy investment ratio.
                eb (float, optional): Allocation fraction to reserve production. Defaults to 1.0.
                lb (float, optional): Length at birth. If not provided, it is calculated from the other parameters. Defaults to None.

            Returns:
                float: Maximum reserve density that an organism can achieve.
        """
        g = self.g
        xb = g / (g + self.eb)

        # Calculate uE0 using the equation in the Dynamic Energy Budget textbook
        uE0 = np.real((3 * g / (3 * g * xb ** (1 / 3) / self.lb - deb.beta0(0, xb))) ** 3)

        # Calculate U0 and E0 using the equations in the Dynamic Energy Budget textbook
        U0 = uE0 * self.v ** 2 / g ** 2 / self.k_M ** 3
        E0 = U0 * self.p_Am
        return E0


    def predict_larva_stage(self, f=1.0):
        g = self.g
        lb = self.lb
        c1 = f / g * (g + lb) / (f - lb)
        c2 = self.k * self.vHb / lb ** 3
        self.rho_j = (f / lb - 1) / (f / g + 1)  # scaled specific growth rate of larva

        def get_tj(tau_j):
            ert = np.exp(- tau_j * self.rho_j)
            return np.abs(self.v_Rj - c1 * (1 - ert) + c2 * tau_j * ert)

        self.tau_j = deb.simplex(get_tj, 1)
        self.lj = lb * np.exp(self.tau_j * self.rho_j / 3)
        self.t_j = self.tau_j / self.k_M / self.T_factor
        self.Lj = self.lj * self.Lm
        self.Lwj = self.Lj / self.del_M
        self.E_Rm = self.v_Rm * (1 - self.kap) * g * self.E_M * self.Lj ** 3
        self.E_Rj = self.E_Rm * self.s_j
        self.E_eggs = self.E_Rm * self.kap_R

    @property
    def Lw(self):
        return self.L / self.del_M

    @property
    def L(self):
        return self.V ** (1 / 3)

    def compute_Ww(self, V=None, E=None):
        if V is None:
            V = self.V
        if E is None:
            E = self.E + self.E_R
        return V * self.d_V + E * self.w_E / self.mu_E

    @property
    def Ww(self):
        return self.V * self.d_V + (self.E + self.E_R) * self.w_E / self.mu_E

    @property
    def e(self):
        return self.E / self.V / self.E_M

    @property
    def Vw(self):
        return self.V + self.w_E / self.d_E / self.mu_E * self.E

    @property
    def pupation_buffer(self):
        return self.E_R / self.E_Rj

    @classmethod
    def from_file(cls, species='default', **kwargs):
        # Drosophila model by default
        with open(f'{reg.ROOT_DIR}/lib/model/deb/models/deb_{species}.csv') as tfp:
            d = json.load(tfp)
        kwargs.update(**d)
        return cls(**kwargs)


class DEB(DEB_basic):
    id = param.String('DEB model', doc='The unique ID of the DEB model')
    species = param.Selector(objects=['default', 'rover', 'sitter'], label='phenotype',
                             doc='The phenotype/species-specific fitted DEB model to use.')  # Drosophila model by default
    assimilation_mode = param.Selector(objects=['gut', 'sim', 'deb'], label='assimilation mode',
                                       doc='The method used to calculate the DEB assimilation energy flow.')
    starvation_strategy = param.Boolean(False, doc='Whether starvation strategy is active')
    aging = param.Boolean(False, doc='Whether aging is active')
    hunger_as_EEB = param.Boolean(False,
                                  doc='Whether the DEB-generated hunger drive informs the exploration-exploitation balance.')
    use_gut = param.Boolean(True, doc='Whether to use the gut module.')
    hunger_gain = param.Magnitude(0.0, label='hunger sensitivity to reserve reduction',
                                  doc='The sensitivy of the hunger drive in deviations of the DEB reserve density.')
    dt = PositiveNumber(1 / (24 * 60), doc='The timestep of the DEB energetics module in days.')
    hours_as_larva = PositiveNumber(0.0, doc='The age since eclosion')
    substrate = ClassAttr(Substrate, doc='The substrate where the agent feeds')

    def __init__(self, species='default', save_dict=True, save_to=None, V_bite=0.0005, base_hunger=0.5,
                 simulation=True, intermitter=None, gut_params={}, **kwargs):

        # Drosophila model by default
        with open(f'{reg.ROOT_DIR}/lib/model/deb/models/deb_{species}.csv') as tfp:
            species_dict = json.load(tfp)
        kwargs.update(**species_dict)
        super().__init__(species=species, **kwargs)
        self.set_intermitter(intermitter, base_hunger)
        self.sim_start = self.hours_as_larva
        self.save_to = save_to
        self.simulation = simulation
        self.epochs = []
        self.epoch_qs = []

        # Stage duration parameters
        self.age = 0
        self.birth_time_in_hours = np.nan
        self.pupation_time_in_hours = np.nan
        # self.death_time_in_hours = np.nan

        self.deb_p_A = 0
        self.sim_p_A = 0

        self.base_f = self.substrate.get_f(K=self.K)
        self.f = self.base_f
        self.V_bite = V_bite

        self.gut = deb.Gut(deb=self, save_dict=save_dict, **gut_params) if self.use_gut else None
        self.scale_time()
        self.run_embryo_stage()
        self.predict_larva_stage(f=self.base_f)
        self.dict = self.init_dict() if save_dict else None

    def scale_time(self):
        dt = self.dt * self.T_factor
        self.F_m_dt = self.F_m * dt
        self.v_dt = self.v * dt
        self.p_M_dt = self.p_M * dt
        self.p_T_dt = self.p_T * dt if self.p_T != 0.0 else 0.0
        self.k_J_dt = self.k_J * dt
        self.h_a_dt = self.h_a * dt ** 2

        self.p_Am_dt = self.p_Am * dt
        self.p_Amm_dt = self.p_Amm * dt
        self.J_X_Amm_dt = self.J_X_Amm * dt
        self.J_E_Amm_dt = self.J_E_Amm * dt
        self.k_E_dt = self.k_E * dt

        if self.gut is not None:
            self.gut.get_residence_ticks(dt)
            self.J_X_A_array = np.ones(self.gut.residence_ticks) * self.J_X_A

    def set_intermitter(self, intermitter, base_hunger=0.5):
        self.intermitter = intermitter
        if self.hunger_as_EEB and self.intermitter is not None:
            base_hunger = self.intermitter.base_EEB
        self.base_hunger = base_hunger

    def hex_model(self):
        # p.161    [1] S. a. L. M. Kooijman, “Comments on Dynamic Energy Budget theory,” Changes, 2010.
        # For the larva stage
        # self.r = self.g * self.k_M * (self.e/self.lb -1)/(self.e+self.g) # growth rate at  constant food where e=f
        # self.k_E = self.v/self.Lb # Reserve turnover
        pass

    def predict_pupa_stage(self):
        from scipy.integrate import solve_ivp
        g = self.g
        k_M = self.k_M

        def emergence(t, luEvH, terminal=True, direction=0):
            return self.vHe - luEvH[2]

        def get_te(t, luEvH):
            l = luEvH[0]
            u_E = max(1e-6, luEvH[1])
            ii = u_E + l ** 3
            dl = (g * u_E - l ** 4) / ii / 3
            du_E = - u_E * l ** 2 * (g + l) / ii
            dv_H = - du_E - self.k * luEvH[2]
            return [dl, du_E, dv_H]  # pack output

        sol = solve_ivp(fun=get_te, t_span=(0, 1000), y0=[0, self.uEj, 0], events=emergence)
        self.tau_e = sol.t_events[0][0]
        self.le, self.uEe = sol.y_events[0][0][:2]
        self.t_e = self.tau_e / k_M / self.T_factor
        self.Le = self.le * self.Lm
        self.Lwe = self.Le / self.del_M
        self.Ue = self.uEe * self.v ** 2 / g ** 2 / k_M ** 3
        self.Ee = self.Ue * self.p_Am
        self.Wwe = self.compute_Ww(V=self.Le ** 3, E=self.Ee + self.E_Rj)  # g, wet weight at emergence

        self.V = self.Le ** 3
        self.E = self.Ee
        self.E_H = self.E_He
        # self.update()

        self.stage = 'imago'
        self.age = self.t_e
        if self.print_output:
            print('-------------Pupa stage-------------')
            print(f'Duration         (d) :      {np.round(self.t_e, 3)}')
            print('-------------Emergence--------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwe * 1000, 5)}')
            print(f'Physical length (mm) :      {np.round(self.Lwe * 10, 3)}')

    @property
    def emergence_time_in_hours(self):
        return self.pupation_time_in_hours + np.round(self.t_e * 24, 1)

    @property
    def death_time_in_hours(self):
        if not self.alive:
            return self.age * 24
        else:
            return np.nan

    @property
    def time_to_death_by_starvation(self):
        return self.v ** -1 * self.L * np.log(self.kap ** -1)

    def predict_imago_stage(self, f=1.0):
        # if np.abs(self.sG) < 1e-10:
        #     self.sG = 1e-10
        # self.uh_a =self.h_a/ self.k_M ** 2 # scaled Weibull aging coefficient
        self.lT = self.p_T / (self.p_M * self.Lm)  # scaled heating length {p_T}/[p_M]Lm
        self.li = f - self.lT
        # self.hW3 = self.ha * f * self.g/ 6/ self.li
        # self.hW = self.hW3**(1/3) # scaled Weibull aging rate
        # self.hG = self.sG * f * self.g * self.li**2
        # self.hG3 = self.hG**3;     # scaled Gompertz aging rate
        # self.tG = self.hG/ self.hW
        # self.tG3 = self.hG3/ self.hW3 # scaled Gompertz aging rate
        # # self.tau_m = sol.t_events[0][0]
        # # self.lm, self.uEm=sol.y_events[0][0][:2]
        # self.t_m = self.tau_m / self.k_M / self.T_factor
        self.Li = self.li * self.Lm
        self.Lwi = self.Li / self.del_M
        self.Ui = self.uEi * self.v ** 2 / self.g ** 2 / self.k_M ** 3
        self.Ei = self.Ui * self.p_Am
        self.Wwi = self.compute_Ww(V=self.Li ** 3, E=self.Ei + self.E_Rj)  # g, imago wet weight
        # self.age = self.t_m

        self.V = self.Li ** 3
        self.E = self.Ei

        if self.print_output:
            print('-------------Imago stage-------------')
            print(f'Duration         (d) :      {np.round(self.t_i_cor, 3)}')
            print('---------------Emergence---------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwi * 1000, 5)}')
            print(f'Physical length (mm) :      {np.round(self.Lwi * 10, 3)}')

    def run_embryo_stage(self, dt=None):
        if dt is None:
            dt = self.dt

        kap = self.kap
        E_G = self.E_G

        t = 0

        while self.E_H < self.E_Hb:
            # This is in e/t and below needs to be volume-specific
            p_S = self.p_M_dt * self.V + self.p_T_dt * self.V ** (2 / 3)
            p_C = self.E * (E_G * self.v_dt / self.V ** (1 / 3) + p_S / self.V) / (kap * self.E / self.V + E_G)
            p_G = kap * p_C - p_S
            p_J = self.k_J_dt * self.E_H
            p_R = (1 - kap) * p_C - p_J

            self.E -= p_C
            self.V += p_G / E_G
            self.E_H += p_R
            t += dt
        self.Eb = self.E
        L_b = self.V ** (1 / 3)
        Lw_b = L_b / self.del_M

        self.Wwb = self.compute_Ww(V=self.Lb ** 3, E=self.Eb)  # g, wet weight at birth
        self.birth_time_in_hours = np.round(self.t_b * 24, 2)
        self.sim_start += self.birth_time_in_hours
        self.stage = 'larva'
        self.age = self.t_b
        if self.print_output:
            print('-------------Embryo stage-------------')
            print(f'Duration         (d) :      predicted {np.round(self.t_b, 3)} VS computed {np.round(t, 3)}')
            print('----------------Birth----------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwb * 1000, 5)}')
            print(
                f'Physical length (mm) :      predicted {np.round(self.Lwb * 10, 3)} VS computed {np.round(Lw_b * 10, 3)}')

    def run_larva_stage(self, f=1.0, dt=None):
        if dt is None:
            dt = self.dt
        kap = self.kap
        E_G = self.E_G
        g = self.g
        del_M = self.del_M

        t = 0

        while self.E_R < self.E_Rj:
            p_A = self.p_Amm_dt * f * self.V
            p_S = self.p_M_dt * self.V
            p_C = self.E * (E_G * self.k_E_dt + p_S / self.V) / (kap * self.E / self.V + E_G)
            p_G = kap * p_C - p_S
            p_J = self.k_J_dt * self.E_Hb
            p_R = (1 - kap) * p_C - p_J

            self.E += (p_A - p_C)
            self.V += p_G / E_G
            self.E_R += p_R
            t += dt
        Lw_j = self.V ** (1 / 3) / del_M
        Ej = self.Ej = self.E
        self.Uj = Ej / self.p_Am
        self.uEj = self.lj ** 3 * (self.kap * self.kap_V + f / g)
        self.Wwj = self.compute_Ww(V=self.Lj ** 3,
                                   E=Ej + self.E_Rj)  # g, wet weight at pupation, including reprod buffer
        # self.Wwj = self.Lj**3 * (1 + f * self.w_V) # g, wet weight at pupation, excluding reprod buffer at pupation
        # self.Wwj += self.E_Rj * self.w_E/ self.mu_E/ self.d_E # g, wet weight including reprod buffer
        self.pupation_time_in_hours = self.birth_time_in_hours + np.round(t * 24, 1)
        self.stage = 'pupa'
        if self.print_output:
            print('-------------Larva stage-------------')
            print(f'Duration         (d) :      predicted {np.round(self.t_j, 3)} VS computed {np.round(t, 3)}')
            print('---------------Pupation---------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwj * 1000, 5)}')
            print(
                f'Physical length (mm) :      predicted {np.round(self.Lwj * 10, 3)} VS computed {np.round(Lw_j * 10, 3)}')

    def compute_hunger(self):
        h = np.clip(self.base_hunger + self.hunger_gain * (1 - self.e), a_min=0, a_max=1)

        return h

    def run(self, **kwargs):
        self.age += self.dt
        if self.E_R < self.E_Rj:
            k = self.kap
            E_G = self.E_G
            p_A = self.get_p_A(**kwargs)
            p_S = self.p_M_dt * self.V
            p_C = self.E * (E_G * self.k_E_dt + p_S / self.V) / (k * self.E / self.V + E_G)
            p_G = k * p_C - p_S
            p_J = self.k_J_dt * self.E_Hb
            p_R = (1 - k) * p_C - p_J
            self.E += (p_A - p_C)
            self.V += p_G / E_G
            self.E_R += p_R
            self.update_hunger()
        elif self.stage == 'larva':
            self.pupation_time_in_hours = np.round(self.age * 24, 2)
            self.stage = 'pupa'
        if self.dict is not None:
            self.update_dict()

    def update_hunger(self):
        self.hunger = self.compute_hunger()
        if self.hunger_as_EEB and self.intermitter is not None:
            self.intermitter.base_EEB = self.hunger

    def die(self):
        self.alive = False
        if self.print_output:
            print(f'Dead after {self.age} days')

    @property
    def J_X_A(self):
        return self.J_X_Amm * self.V * self.base_f

    @property
    def F(self):  # Vol specific filtering rate (cm**3/(d*cm**3) -> vol of environment/vol of individual*day
        # F = self.F_mm * self.K/(self.K + self.substrate.X)
        F = (self.F_mm ** -1 + self.substrate.X * self.J_X_Amm ** -1) ** -1
        # F = (self.F_mm ** -1 + self.substrate.X * self.J_X_Amm ** -1) ** -1
        return F

    @property
    def fr_feed(self):
        freq = self.F / self.V_bite * self.T_factor
        freq /= (24 * 60 * 60)
        return freq

    def grow_larva(self, epochs, **kwargs):
        for e in epochs:
            c = {'assimilation_mode': 'sim', 'f': e.substrate.get_f(K=self.K)}
            if e.end is None:
                while self.stage == 'larva':
                    self.run(**c)
            else:
                for i in range(e.ticks(self.dt)):
                    if self.stage == 'larva':
                        self.run(**c)
        tb = self.birth_time_in_hours
        tp = self.pupation_time_in_hours
        self.epochs = [[e.start + tb, e.end + tb if e.end is not None else tp] for e in epochs]
        self.epoch_qs = [e.substrate.quality for e in epochs]
        self.hours_as_larva = self.age * 24 - tb
        if self.gut is not None:
            self.gut.update()

    @property
    def EEB(self):
        if self.intermitter is None:
            return None
        else:
            return self.intermitter.EEB

    def init_dict(self):
        self.dict_keys = [
            'age',
            'mass',
            'length',
            'reserve',
            'reserve_density',
            'hunger',
            'pupation_buffer',
            'f',
            'deb_p_A',
            'sim_p_A',
            'EEB'
        ]
        d = aux.AttrDict({k: [] for k in self.dict_keys})
        return d

    def update_dict(self):
        dict_values = [
            self.age * 24,
            self.Ww * 1000,
            self.Lw * 10,
            self.E,
            self.e,
            self.hunger,
            self.pupation_buffer,
            self.f,
            self.deb_p_A / self.V,
            self.sim_p_A / self.V,
            self.EEB
        ]
        for k, v in zip(self.dict_keys, dict_values):
            self.dict[k].append(v)
        if self.gut is not None:
            self.gut.update_dict()

    def finalize_dict(self):
        if self.dict is not None:
            d = self.dict
            d['birth'] = self.birth_time_in_hours
            d['pupation'] = self.pupation_time_in_hours
            d['emergence'] = self.emergence_time_in_hours
            d['death'] = self.death_time_in_hours
            d['id'] = self.id
            d['simulation'] = self.simulation
            d['hours_as_larva'] = self.hours_as_larva
            d['sim_start'] = self.sim_start
            d['epochs'] = self.epochs
            d['epoch_qs'] = self.epoch_qs
            d['fr'] = 1 / (self.dt * 24 * 60 * 60)
            d['feed_freq_estimate'] = self.fr_feed
            d['f_mean'] = np.mean(d['f'])
            d['f_deviation_mean'] = np.mean(np.array(d['f']) - 1)

            if self.gut is not None:
                d['Nfeeds'] = self.gut.Nfeeds
                d['mean_feed_freq'] = self.gut.Nfeeds / (self.age - self.birth_time_in_hours) / (60 * 60)
                d['gut_residence_time'] = self.gut.residence_time
                d.update(self.gut.dict)
                # print(self.id, self.gut.Nfeeds, self.intermitter.base_EEB, self.base_hunger)

        # self.save_dict(path)
        return d

    def return_dict(self):
        if self.gut is None:
            return self.dict
        else:
            return {**self.dict, **self.gut.dict}

    def save_dict(self, path=None):
        if path is None:
            if self.save_to is not None:
                path = self.save_to
            else:
                return
                # raise ValueError ('No path to save DEB dict')
        if self.dict is not None:
            os.makedirs(path, exist_ok=True)
            if self.gut is not None:
                d = {**self.dict, **self.gut.dict}
            else:
                d = self.dict
            aux.save_dict(d, f'{path}/{self.id}.txt')

    def get_p_A(self, f=None, assimilation_mode=None, X_V=0.0):
        if f is None:
            f = self.base_f
        self.f = f
        self.deb_p_A = self.p_Amm_dt * self.base_f * self.V
        self.sim_p_A = self.p_Amm_dt * f * self.V
        if assimilation_mode is None:
            assimilation_mode = self.assimilation_mode
        if assimilation_mode == 'sim':
            return self.sim_p_A
        elif assimilation_mode == 'gut' and self.gut is not None:
            self.gut.update(X_V)
            return self.gut.p_A
        elif assimilation_mode == 'deb':
            return self.deb_p_A

    @property
    def steps_per_day(self):
        return int(1 / self.dt)

    @property
    def deb_f_mean(self):
        return np.mean(self.dict['f'])

    @property
    def ingested_body_mass_ratio(self):
        return self.gut.ingested_mass() / self.Ww * 100

    @property
    def ingested_body_volume_ratio(self):
        return self.gut.ingested_volume / self.V * 100

    @property
    def ingested_gut_volume_ratio(self):
        return self.gut.ingested_volume / (self.V * self.gut.V_gm) * 100

    @property
    def ingested_body_area_ratio(self):
        return (self.gut.ingested_volume / self.V) ** (1 / 2) * 100

    @property
    def amount_absorbed(self):
        return self.gut.absorbed_mass('mg')

    @property
    def volume_ingested(self):
        return self.gut.ingested_volume

    @property
    def deb_f_deviation(self):
        return self.f - 1

    @property
    def deb_f_deviation_mean(self):
        return np.mean(np.array(self.dict['f']) - 1)


class DEB_runner(DEB):
    f_decay = PositiveNumber(default=0.1, doc='The exponential decay coefficient of the DEB functional response.')

    def __init__(self, model=None, dt=None, life_history=None, **kwargs):
        if life_history is None:
            life_history = aux.AttrDict({'epochs': {}, 'age': None})
        self.model = model
        if self.model is not None:
            if dt is None:
                dt = self.model.dt / (24 * 60 * 60)
        super().__init__(dt=dt, **kwargs)
        self.grow_larva(**life_history)
        self.temp_cum_V_eaten = 0

    @property
    def valid(self):
        return self.model.Nticks % int(self.model.dt / self.dt) == 0

    def update(self, V_eaten=0):
        self.temp_cum_V_eaten += V_eaten
        if self.valid:
            if self.temp_cum_V_eaten > 0:
                self.f += self.gut.k_abs
            self.f *= np.exp(-self.f_decay * self.model.dt)
            self.run(X_V=self.temp_cum_V_eaten)
            self.temp_cum_V_eaten = 0


# p.257 in S. a. L. M. Kooijman, “Dynamic Energy Budget theory for metabolic organisation : Summary of concepts of the third edition,” Water, vol. 365, p. 68, 2010.


def deb_default(id='DEB model', epochs={}, age=None, **kwargs):
    deb = DEB(id=id, simulation=False, use_gut=False, **kwargs)
    deb.grow_larva(epochs=epochs)
    return deb.finalize_dict()


def get_best_EEB(deb, cRef):
    z = np.poly1d(cRef['EEB_poly1d'])
    if type(deb) == dict:
        s = deb['feed_freq_estimate']
    else:
        s = deb.fr_feed
    return np.clip(z(s), a_min=0, a_max=1)


def deb_sim(refID, id='DEB sim', EEB=None, deb_dt=None, dt=None, use_hunger=False, model_id=None, **kwargs):
    from ..modules.intermitter import OfflineIntermitter
    c = reg.conf.Ref.getRef(refID)
    kws2 = c['intermitter']
    if dt is not None:
        kws2['dt'] = dt
    if deb_dt is None:
        deb_dt = kws2['dt'] / (24 * 60 * 60)
    D = DEB(id=id, assimilation_mode='gut', dt=deb_dt, **kwargs)
    if EEB is None:
        EEB = get_best_EEB(D, c)
    D.base_hunger = EEB
    I = OfflineIntermitter(**kws2, EEB=EEB)
    cum_feeds = 0
    while (D.stage != 'pupa' and D.alive):
        I.step()
        if I.total_ticks % round(D.dt * (24 * 60 * 60) / I.dt) == 0:
            D.run(X_V=D.V_bite * D.V * (I.Nfeeds - cum_feeds))
            cum_feeds = I.Nfeeds
            if use_hunger:
                I.EEB = D.hunger
    D.finalize_dict()
    d_sim = D.return_dict()
    d_inter = I.build_dict()
    d_sim.update({
        'DEB model': D.species,
        'EEB': np.round(EEB, 2),
        **{f'{q} ratio': np.round(d_inter[nam.dur_ratio(p)], 2) for p, q in
           zip(['stridechain', 'pause', 'feedchain'], ['crawl', 'pause', 'feed'])},
        f"{nam.freq('feed')}_exp": np.round(I.mean_feed_freq, 2),
        f"{nam.freq('feed')}_est": np.round(D.fr_feed, 2)
    })
    if model_id is None:
        return d_sim
    else:
        d_mod = deb_default(id=model_id, **kwargs)
        return d_sim, d_mod
