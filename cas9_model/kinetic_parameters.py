#########################################################
# code to extract model parameters from SA fit
#########################################################
import sys, os
path = os.getcwd()
sys.path.append(path)

import numpy as np
import pandas as pd
from cas9_model import read_model_ID

def kinetic_parameters(filename, ID_Cas="Clv_Saturated_general_energies_v2",
                       ID_dCas="general_energies_no_kPR",
                       concentration_nM=10.,
                       nmbr_fit_params=44):
    '''
    This is made based on the sequence averaged model.
    From the SA fit, we get the parameters, at the specified concentration

    :param filename: output file name from SA fit
    :param ID_Cas: model_id for active Cas
    :param ID_dCas: model_id for dead Cas
    :param concentration_nM: concentration in nM of originally stored parameters
    :param nmbr_fit_params: number of free-parameters in SAfit
    :return:
    '''

    # -- extract from output file SA fit  (use not the final solution perse, but whatever gave lowest chi2) ------
    SAfit = pd.read_csv(filename, delimiter='\t', index_col=39)  # might need to adjust "index_col=39" to make more general?
    SAfit.reset_index(inplace=True)
    best_solution = pd.Series.idxmin(SAfit.Potential)
    parameters = load_simm_anneal(filename, nmbr_fit_params, fatch_solution=best_solution)

    # --- split into parameters fitted using dCas9 and Cas9 (Nucleaseq is done under saturating conditions) ---
    # ---- might need to adjust this part to make more general ? ----
    Cas_params = np.append(parameters[1:41], parameters[42:44])
    dCas_params = np.array(parameters[0:43])

    # --- get epsilon and forward rates ----
    epsilon, forward_rates = read_model_ID.unpack_parameters(dCas_params, model_id=ID_dCas)

    # --- epsilon PAM at 1nM ---
    epsilon_1nM = epsilon.copy()
    epsilon_1nM[0] += np.log(concentration_nM)

    # --- binding rate at 1nM ---
    kon = forward_rates[0] * concentration_nM**(-1)

    # --- internal forward rate ---
    kf = forward_rates[1]

    # --- catalytic rate ----
    _, forward_rates = read_model_ID.unpack_parameters(Cas_params, model_id=ID_Cas)
    forward_rates[0] = kon
    kcat = forward_rates[-1]
    return Cas_params, dCas_params, epsilon_1nM, forward_rates, kon, kf, kcat



############################################
def load_simm_anneal(filename, Nparams, fatch_solution='final'):
    '''
    Load the parameter set from simmulated annealing.
    Fix indexing based on parameterisation to correctly import the table
    :param filename: filename output from SA fit
    :param Nparams: number of free-parameters in fit
    :param fatch_solution: allows to get intermediate solution. By default set to fatch the final solution
    :return:
    '''

    fit = pd.read_csv(filename, delimiter='\t', index_col=Nparams+2)
    fit = fit.reset_index()
    final_result = []
    for param in range(1, Nparams + 1):
        col = 'Parameter ' + str(param)

        if fatch_solution == 'final':
            final_result.append(fit[col].iloc[-1])
        else:
            final_result.append(fit[col].iloc[fatch_solution])

    sa_result = np.array(final_result)
    return sa_result
