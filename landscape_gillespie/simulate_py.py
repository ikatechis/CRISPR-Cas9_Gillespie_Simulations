# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:03:12 2019

@author: iason
"""

import numpy as np
from cas9_model.target_model import TargetModel, generate_mismatches, load_parameters


def landscape_gillespie(parameters, concentration=1, N=10, length=10,
                        method='keep_same', guide_length=20):
    if length == guide_length:
        mismatches = generate_mismatches(length, full_target=True)
    elif length < guide_length and length >= 0:
        mismatches = generate_mismatches(length, full_target=False)

    else:
        raise ValueError('length should be from 0 to guide_length')

    DNA = {f'{i}':  TargetModel(f'DNA{i}', parameters, concentration, mis) \
           for i, mis in enumerate(mismatches)}

    off_times = np.ones(length+1, dtype=np.float)
    on_times = np.ones(length+1, dtype=np.float)

    F = np.zeros(length+2, dtype=np.float)  # F[0], the solution energy, is zero

    for i in range(0, length+1):
        t_off, t_on = DNA[f'{i}'].gillespie.find_events(max_unbind_events=N,
                                         print_exec_time=True, dwells_only=True)

        t_off = np.average(t_off)
        t_on = np.average(t_on)
        off_times[i] *= t_off
        on_times[i] *= t_on

    # calculate the average on_time. It should be the same for all constructs
    tau_on = np.average(on_times)
    # calculate first the PAM free energy
    F[1] += -np.log(off_times[0]/tau_on)
    # find the differences in dwelltimes ratio: (tau_off(n) - tau_off(n-1))/tau_on
    diff = np.diff(off_times)/tau_on
    negatives = np.where(diff < 0)[0]
    positives = np.where(diff > 0)[0]


    if negatives.size == 0:
        # if non-negative differences exist calculate the Free Energy
        F[2:] += -np.log(diff)
    else:
        if method == 'keep_same':
            F[positives+2] += - np.log(diff[positives])  # F has size +2 bigger than diff
            F[negatives+2] += F[negatives+1]
            zeros = np.where(F[2:] == 0.)[0] + 2
            F[zeros] = F[zeros - 1]

        elif method == 'discard':
            F[positives+2] += - np.log(diff[positives])
            F[negatives+2] += -np.log(off_times[negatives+1] - off_times[negatives-1]) + np.log(tau_on)

    return DNA, off_times, on_times, F

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')
    sns.set()
    parameters = load_parameters('boyle')
    DNA, off_times, on_times, F = landscape_gillespie(parameters, N=100, length=11)
    plt.plot(F)

