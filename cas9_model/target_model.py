# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:17:05 2019

@authors: iasonas
"""

import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style('ticks')
from cas9_model import kinetic_parameters
from cas9_model import CRISPR_dCas9_binding_curve_Boyle as dCas9
from cas9_model.gillespie_for_TargetModel import Gillespie


class TargetModel(object):
    '''
    Constructs the kinetic model for a specific cas9 target.
    Model assumes solution state, PAM state and 20 internal states
    The following should be specified:

    - target name
    - concentration in nM
    - kon
    - internal forward rates
    - free energy differences for a match in each position
    - free energy penalties for a mismatch in each position
    - mismatch positions (if not full target)
    - k_clv (if not dcas9)
    - sequence (optional)
    '''
    def __init__(self, name, parameters, concentration=1, mismatches=[]):
        self.parameters = parameters
        self.name = name
        self.forward_rates = np.copy(self.parameters['rates'])  # The rates should be given as 1D numpy array in 1/s units, 20 internal and kon are expected

        self.epsilon = np.copy(self.parameters['epsilon'])  #  The energies should be given as 1D numpy array in KbT units. 21 energies expected (Pam & internal)
        self.penalties = np.copy(self.parameters['penalties'])
        self.concentration = concentration
        self.epsilon[0] -= np.log(self.concentration)  # correct PAM energy for concentration
        self.forward_rates[0] *= self.concentration # define concentration-dependent kon
        self.kon = self.forward_rates[0]  # assign kon attribute for easy access
        self.mismatches = mismatches
        self.k_clv = 0  # By default we have a dcas9
        self.guide_length = 20  # this can be changed if needed
        self.sequence = 'GACGCAUAAAGAUGAGACGC'  # Here, the λ-phage gRNA sequence is given as default

        # calculate the backward rates using detailed balance condition
        self.backward_rates = self.master_equation.backward_rates

    @property
    def landscape(self):
        return Landscape(self)

    @property
    def master_equation(self):
        return master_equation(self)

    @property
    def gillespie(self):
        return Gillespie(self)


class Landscape(object):
    def __init__(self, target):
        self.__target = target

    @property
    def energies(self):
        energies = -1*self.__target.epsilon  # convention: epsC>0 means downward slope
        energies[0] = self.__target.epsilon[0]  # convention: epsPAM>0 means upward slope

        if len(self.__target.mismatches) > 0:
            mismatches = np.array(self.__target.mismatches).flatten()
            energies[mismatches] += self.__target.penalties[mismatches - 1]

        return energies

    def get_landscape(self, zero_energy=0):
        return np.insert(np.cumsum(self.energies), 0, zero_energy)


    def plot(self, axis=None, line_kwargs={}):
        if line_kwargs == {}:
                line_kwargs = {'marker': 'o', 'markersize': 8,
                               'markeredgewidth': 2, 'markerfacecolor': 'white'}
        if axis is None:
            fig = plt.figure('Microscopic Landscape')

            line = plt.plot(range(-1, self.__target.guide_length+1),
                                   self.get_landscape(), label=self.__target.name,
                                   **line_kwargs)[0]

            plt.xlabel('targeting progression', fontsize=10)
            plt.ylabel(r'free-energy ($k_BT$)',fontsize=10)
            plt.xticks(range(-1, 21),
                       [ 'S','P',1,'', '', '', 5, '', '', '', '', 10,  # for now we assume 20 nt length
                        '', '', '', '', 15, '', '', '', '', 20],
                        rotation=0, fontsize=10);
            plt.yticks(fontsize=10)
            plt.grid(True)
            sns.despine()

        elif not issubclass(type(axis), matplotlib.axes.SubplotBase):
            raise ValueError('axis must be a matplotlib.axes.SubplotBase object or None')

        else:
            line = axis.plot(range(-1, self.__target.guide_length+1),
                                   self.landscape, **line_kwargs)

        return line, fig

class master_equation(object):
    def __init__(self, target):
        self.__target = target
        self.forward_rates = self.__target.forward_rates

    @property
    def energies(self):
        return self.__target.landscape.energies

    @property
    def backward_rates(self):
        '''
        Apply detailed balance condition to get the backward rates
        from the energies and forward rates

        '''
        # 0) Construct array containing backward rates
        backward_rates = np.zeros(self.__target.guide_length+2)
        # 1) Apply detailed balance condition:
        backward_rates[1:] = self.forward_rates[:-1] * np.exp(self.energies)
        # 2) No rate backward from solution state
        backward_rates[0] = 0.0
        return backward_rates

    @property
    def average_offtime(self):
        dist, times = self.offtime_dist()
        times1 = np.insert(times,0,0)
        dtimes = times1[1:] - times1[:-1]
        return np.sum(dist*times*dtimes)


    def offtime_dist(self, logbins=True, bins_per_decade=10,
                               tmin = 0.0001, tmax=10**(6),
                               t_resolution=0.1):

        '''
        Generates tau_off dwelltime distribution from solution to Master Equation(s)
        :param logbins:  Whether to have log bins in the x axis
        :param bins_per_decade:
        :param tMIN:
        :param tMAX:
        :param temporal_resolution:  The time interval every which the master
                            equation is solved
        :Returns: a list with dwelltime distribution and the time axis
        '''
        # -- 1. Construct Master Equations using the parameter set (use function as before) ----
        M = self.get_rate_matrix()  # SM experiment, rebinding is not allowed

        # -- 2. construct the bins/ timepoints at which we evaluate the distribution ----
        # Log-spaced bins:
        if  logbins:
            times = [tmin]
            t = tmin
            while t <= tmax:
                t = times[-1] * 10 ** ((1 / float(bins_per_decade)))
                times.append(t)
        else:
            # Linear bins:
            dt = t_resolution
            times = np.arange(tmin, tmax, dt)

        # -- 3.initial condition: Molecule enters at PAM state
        P = self.get_probability(0, P_init='PAM')

        # -- 4. For different timepoints, solve for the dwelltime distribution ----
        dwelltime_dist = []
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]

            P = self.get_probability(dt, P_init=P, rebinding=False)
            rate_to_sol = np.diag(M, k=1)[0]
            # the prob to unbind is equal to the unbinding rate times the probability to be at the PAM state
            p_unbind = P[1] * rate_to_sol
            dwelltime_dist.append(p_unbind)

        return np.array(dwelltime_dist), np.array(times[1:])


    def get_rate_matrix(self):
        '''
        build matrix in Master Equation

        '''
        diagonal1 = -(self.forward_rates + self.backward_rates)
        diagonal2 = self.backward_rates[1:]
        diagonal3 = self.forward_rates[:-1]

        rate_matrix = np.diag(diagonal1, k=0) + \
                      np.diag(diagonal2, k=1) + np.diag(diagonal3, k=-1)

        return rate_matrix


    def get_probability(self, t, P_init='sol', rebinding=True):
        '''
        solves the Master Equation for a given initial condition
        and desired time point
        :param initial_condition: vector with initial configuration
                                    or either of 'sol', 'PAM' or integer in [1,20]
        :param t: Evaluate solution at time t
        :return:
        '''
        psize = self.__target.guide_length + 2

        if type(P_init) == str and P_init in ['sol', 'PAM']:
            P0 = np.zeros(psize)
            P0[[0,1]] = np.array([1,0])*(P_init == 'sol') + np.array([0,1])*(P_init == 'PAM')


        elif type(P_init) == list and len(P_init) == psize:
            P0 = np.array(P_init)

        elif not isinstance(P_init, np.ndarray) or (len(P_init) != psize):
            raise ValueError('P_init must be a numpy array or list of size guidelength + 2')
        else:
            P0 = P_init

        M = self.get_rate_matrix()
        if not rebinding:  # Rebinding is not allowed. e.g to get tau_off distribution
            M[0][0] = 0.0
            M[1][0] = 0.0
        matrix_exponent = linalg.expm(M*t)
        return matrix_exponent.dot(P0)

# Helper functions
def generate_mismatches(length=20, guide_length=20, full_target=True):
    '''
    Generates a dictionary with the index of DNA as keys. e.g. DNA0 is PAM-only
    , DNA20 is the full target with [] as mismatches.
    '''
    constructs = []
    for mm1 in range(1, length+2):
        mismatch_positions = [i for i in range(mm1, guide_length+1)]
        constructs.append(mismatch_positions)
    if full_target:
        constructs.append([])

    constructs_dict = {}
    for i in range(len(constructs)):
        constructs_dict[f'{i}'] = constructs[i]
    return constructs

def load_parameters(fit_name):

    if fit_name == 'boyle':

        ID_dCas="init_limit_general_energies_v2"  # model ID
        filename = './best_fits/median_landscape_Boyle_2Dgrid.txt'
        boyle_params = np.loadtxt(filename,comments='#')

        epsilon, forward_rates = dCas9.unpack_parameters(boyle_params, ID_dCas, 20)

    if fit_name == 'cleavage':

        filename = './best_fits/fit_18_7_2019_sim_17.txt'
        Cas_params, dCas_params, epsilon, forward_rates, kon, kf, kcat \
                        = kinetic_parameters.kinetic_parameters(filename)
        forward_rates[-1] = 0  # for now we only care for dCas9

    params = {'rates': forward_rates, 'epsilon':epsilon[0:21], 'penalties': epsilon[21:]}
    return params


if __name__ == '__main__':

    parameters = load_parameters('cleavage')


    DNA = TargetModel('PAM_only', parameters, mismatches=list(range(10,21)),
                      concentration=10)

    g = DNA.gillespie
    transition_times, states, dwells, ontimes = g.find_events()

    dist, times = DNA.master_equation.offtime_dist(tmax=3600, t_resolution=1)
    plt.plot(times, dist)
    plt.xlim((0, 2000))
    print(f'average dwelltime: {DNA.master_equation.average_offtime:.1f}')






















