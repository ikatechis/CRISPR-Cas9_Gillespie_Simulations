# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:03:12 2019

@author: iason
"""
#  Use 'python setup.py build_ext --inplace' in a conda prompt to compile

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

from cas9_model.target_model import TargetModel, generate_mismatches, load_parameters


cpdef landscape_gillespie (dict parameters,
                           float concentration=1.,
                           int N=10,
                           int length=10,
                           str method='keep_same',
                           int guide_length=20):
    cdef list mismatches
    cdef np.ndarray[DTYPE_t, ndim=1] off_times, on_times, F, t_off, t_on

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

        tau_off = np.average(t_off)
        tau_on = np.average(t_on)
        off_times[i] *= tau_off
        on_times[i] *= tau_on

    # calculate the average on_time. It should be the same for all constructs
    tau_on = np.average(on_times)
    # calculate first the PAM free energy
    F[1] += -np.log(off_times[0]/tau_on)
    # find the differences in dwelltimes ratio: (tau_off(n) - tau_off(n-1))/tau_on
    cdef np.ndarray[DTYPE_t, ndim=1] diff = np.diff(off_times)/tau_on
    cdef np.ndarray[long long, ndim=1] negatives = np.where(diff < 0)[0]
    cdef np.ndarray[long long, ndim=1] positives = np.where(diff > 0)[0]
    cdef np.ndarray[long long, ndim=1] zeros

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

