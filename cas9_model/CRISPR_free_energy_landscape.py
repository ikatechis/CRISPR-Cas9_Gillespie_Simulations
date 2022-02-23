import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style('ticks')
import read_model_ID
#reload(read_model_ID);

def plot_free_energy_landscape(parameters,model_id,show_plot=True):
    '''
    Plot the approximate the free-energy landscape for the on-target

    Note: Technically, the system just has a single free-energy:
        F = -log( sum_i e^{-E_i})
    We will get F_n by  combining the Free-Energies for landscapes with mismatches from (n+1) on
    In this way, we may assume states after the block of mismatches to be inaccessible (the regime the data dictates)
    Hence, we can get an approximate free-energy landscape for the on-target in this manner.

    :param parameters:
    :param model_id:
    :return:
    '''
    epsilon, fwrd_rates = read_model_ID.unpack_parameters(parameters, model_id, guide_length=20)

    Epsilon = epsilon.copy()

    epsilon_C = Epsilon[:21]
    epsilon_C[1:] *= -1

    landscape = [0.0]
    for eps in epsilon_C:
        landscape.append(landscape[-1] + eps)
    landscape = np.array(landscape)

    if show_plot:
        plt.figure()
        plt.plot(range(-1,21),landscape, marker='s');
        plt.xlabel('targeting progression', fontsize=10)
        plt.ylabel(r'(approx.) free-energy ($k_BT$)',fontsize=10)
        plt.xticks(range(-1,21),
                   [ 'S','P',1,'', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20], rotation=0
                   ,fontsize=10);
        plt.yticks(fontsize=10)
        plt.grid('on')
        sns.despine()


    free_energy =  -1*np.log( np.cumsum( np.exp(-landscape[1:])   )  )

    # plotting:
    if show_plot:
        plt.figure()
        plt.plot(range(0,21),free_energy, marker='s');
        plt.xlabel('targeting progression', fontsize=10)
        plt.ylabel(r' free-energy bound molecule ($k_BT$)',fontsize=10)
        plt.xticks(range(0,21),
                   [ 'P',1,'', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20], rotation=0
                   ,fontsize=10);
        plt.yticks(fontsize=10)
        plt.grid('on')
        sns.despine()

    return landscape, free_energy



def plot_landscape(parameters, model_id, mismatch_positions, show_plot=True, axis=None, rel_concentration=1.):
    '''
    Plot the (free-)energy landscape of the on-target

    Added option to plot at different concentrations.
    Default is now set to 1 nM, the parameters are ASSUMED to be at 1 nM as well, hence concentration=0.1

    :param parameters:
    :param model_id:
    :return:
    '''

    # ---- retrieve model parameters from fit result -----
    epsilon, fwrd_rates = read_model_ID.unpack_parameters(parameters, model_id, guide_length=20)
    # epsilon[0] -= np.log(rel_concentration)
    # fwrd_rates[0] += np.log10(rel_concentration)

    # ---- Get (possibly) mismatched energy landscape ----
    energies = get_energies(epsilon, mismatch_positions, guide_length=20)
    energies[0] -= np.log(rel_concentration)

    # ---- Determine free-energy landscape ----
    landscape = [0.0] + list(np.cumsum(energies))
    landscape = np.array(landscape)


    if show_plot:
        if axis:
            axis.plot(range(-1, 21), landscape,
                 marker="o",
                 markersize=8,
                 markeredgewidth=2,
                 markerfacecolor="white");
            axis.set_xlabel('targeting progression', fontsize=10)
            axis.set_ylabel(r'free-energy ($k_BT$)', fontsize=10)
            axis.set_xticks(range(-1, 21))
            axis.set_xticklabels(['S', 'P', 1, '', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20],
                       rotation=0
                       , fontsize=10);
            axis.set_yticklabels(axis.get_yticks(),fontsize=10)
            plt.grid('on')
            sns.despine(ax=axis)
        else:
            plt.figure()
            axis = plt.plot(range(-1,21),landscape,
                            marker="o",
                            markersize=8,
                            markeredgewidth=2,
                            markerfacecolor="white");
            plt.xlabel('targeting progression', fontsize=10)
            plt.ylabel(r'free-energy ($k_BT$)',fontsize=10)
            plt.xticks(range(-1,21),
                       [ 'S','P',1,'', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20], rotation=0
                       ,fontsize=10);
            plt.yticks(fontsize=10)
            plt.grid('on')
            sns.despine()
    return landscape



def get_energies(epsilon,mismatch_positions, guide_length=20):
    '''
    For general (position dependent) model make a list with the energies at every bound state
    At positions with a mismatch incorporated: add mismatch penalty (epsI[mm_pos])

    So this function returns the minima in the energy lanscape (the actual energy at every state)

    :param epsilon: [epsPAM, epsC[state_1],....,epsC[state_N],epsI[state_1],...epsI[state_N] ]
    provide as np.array()
    :param mismatch_positions: each mismach position has a range [1,2, ... , 20]
    :return: vector with the minima in the landscape
    '''
    if type(mismatch_positions)==type([]):
        mismatch_positions = np.array(mismatch_positions)
    new_epsilon = epsilon.copy()
    epsI = new_epsilon[(guide_length+1):]
    energies = -1*new_epsilon[0:(guide_length+1)] # convention: epsC>0 means downward slope
    energies[0] = new_epsilon[0]                 # convention: epsPAM>0 means upward slope
    if len(mismatch_positions)>0:
        energies[mismatch_positions.astype(int)] += epsI[(mismatch_positions.astype(int)-1)]
    return energies









def plot_mismatch_penalties(parameters, model_id,axis=None, color=None):
    '''
    plot mismatch penalties VS position as a bar plot
    :param parameters:
    :param model_id:
    :return:
    '''
    epsilon, fwrd_rates = read_model_ID.unpack_parameters(parameters, model_id, guide_length=20)
    epsilon_I = epsilon[21:]

    if axis is None:
        ax = plt.gca()
    else:
        ax = axis
    if color:
        ax.bar([i+0.5 for i in range(1,21)], epsilon_I, color=color)
    else:
        ax.bar([i + 0.5 for i in range(1, 21)], epsilon_I)
    # window dressing:
    ax.set_xlabel('targeting progression', fontsize=10)
    ax.set_ylabel(r'mismatch penalties ($k_BT$)',fontsize=10)
    ax.set_xticks(np.arange(1,21)+0.5)
    ax.set_xticklabels([1, '', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20],
                       rotation=0,
                       fontsize=10);
    ax.set_yticklabels(ax.get_yticks(),fontsize=15)
    return epsilon_I