# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:21:34 2019

@author: iason
"""
#  This is the cython version of the gillespie algorithm.
# Use 'python setup.py build_ext --inplace' in a conda prompt to compile

import time
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t


cpdef find_events (np.ndarray[DTYPE_t, ndim=1] for_rates,
                   np.ndarray[DTYPE_t, ndim=1] back_rates,
                   str name='',
                   int state_init=-1,
                   int state_monitored=-1,
                   int max_unbind_events=10,
                   float Tmin=-1,
                   dwells_only=False, print_exec_time=False):
    '''
    Performs gillespie algorithm to calculate the transition times and the
    corresponding states for the Cas9 model. It also outputs the tau off
    and on times for a selected state

    :param max_unbinding_events: The number of unbinding events in which
                                the algorithm stops
    :param state_init: The starting state. In default it starts from
                        solution
    :param state_unbound: The state for which the stop criterion is
                            implemented. For this state the t_off and
                            t_on are calculated
    :returns: transition times, states, off_times, on_times
    '''
    cdef double start, frate, brate, rate, transition_time, p

    start = time.time()  # start time of the simulation

    cdef np.ndarray[DTYPE_t, ndim=1] frates = np.copy(for_rates)
    cdef np.ndarray[DTYPE_t, ndim=1] brates = np.copy(back_rates)
    # -1: unbound
    #  0: PAM,
    #  1-20: number of bp in R-loop
    cdef list transition_times = [0]

    cdef np.ndarray[int, ndim=1] last_states = np.ones(3, dtype=int)
    last_states[0] = state_init

    cdef list states = [state_init]

    cdef np.ndarray[DTYPE_t, ndim=1] off_times = np.ones(max_unbind_events, dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=1] on_times = np.ones(max_unbind_events, dtype=np.float)
    cdef double off_time = 0
    cdef int state = state_init # starting state
    cdef int Ntransitions = np.int64(0)
    cdef int Nunbound = 0
    while True:
        frate = frates[state+1]
        brate = brates[state+1]

        rate = frate + brate
        # find the time for the next transition
        transition_time = np.random.exponential(scale=rate**(-1))

        # use Gillespie algorithm to switch state stochastically
        p = np.random.random()
        if p < frate/(brate + frate):
            state += 1
        else:
            state -= 1

        if dwells_only:
            last_states[-2] = last_states[-1]  # this brings the previous state in the second to last position
            last_states[-1] = state  # the last state is the last element
        else:
            transition_times.append(transition_times[-1] + transition_time)
            states.append(state)

        Ntransitions += 1

        # if new state is unbound_state (solution), append total bound time until now and start over
        if state == state_monitored:
            off_times[Nunbound] *= off_time
            Nunbound += 1
            off_time = 0
        # if the current state and previous state is a bound state add the:
        elif last_states[-1] != state_monitored and last_states[-2] != state_monitored:
            off_time += transition_time
        else:  # if the current state is unbound_state (solution), the only choice is to bind
            on_times[Nunbound] *= transition_time

        if Tmin == -1:
            b = (Nunbound >= max_unbind_events)
        else:
            b = (transition_times[-1] > Tmin)

        if b:
            break


    if print_exec_time:
        print(f'Gillespie done in {time.time() - start:.2f} sec'+\
                                   f' for {name}.'+\
                                   f' {Ntransitions} transitions')
        pass

    if dwells_only:
        return off_times, on_times
    else:
        states = states[:Ntransitions]
        return transition_times, states, off_times, on_times