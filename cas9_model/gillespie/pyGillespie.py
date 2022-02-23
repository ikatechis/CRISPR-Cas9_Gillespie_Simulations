# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:21:34 2019

@author: iason
"""
import numpy as np
import time
import random

def find_events(frates, brates, name='', state_init=-1, state_monitored=-1, max_unbind_events=10,
                Tmin=None, dwells_only=False, print_exec_time=False, tau_photobleach=200):
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
    start = time.time()  # start time of the simulation

    frates = np.copy(frates)
    brates = np.copy(brates)
    PBrate = 1./tau_photobleach
    # -1: unbound
    #  0: PAM,
    #  1-20: number of bp in R-loop
    transition_times = [0]

#    if dwells_only
#    states = np.ones(1_000_000, dtype=np.int8)
#    states[0] = state_init
    states = [state_init]

    off_times = np.ones(max_unbind_events)
    on_times = np.ones(max_unbind_events)
    off_time = 0
    state = state_init # starting state
    Ntransitions = 0
    Nunbound = 0
    while True:
        frate = frates[state+1]
        brate = brates[state+1]

        rate = frate + brate
        # find the time for the next transition
        transition_time = np.random.exponential(scale=rate**(-1))

        # use Gillespie algorithm to switch state stochastically
        p = random.random()
        if p < frate/(brate + frate):
            state += 1
        else:
            state -= 1
        if not dwells_only:
            transition_times.append(transition_time)
        states.append(state)
        if dwells_only and Ntransitions > 3:
            states = states[-3:]
        Ntransitions += 1

        # if new state is unbound_state (solution), append total bound time until now and start over
        if state == state_monitored:
            off_times[Nunbound] *= off_time
            Nunbound += 1
            off_time = 0
        # if the current state and previous state is a bound state add the:
        elif states[-1] != state_monitored and states[-2] != state_monitored:
            off_time += transition_time
        else:  # if the current state is unbound_state (solution), the only choice is to bind
            on_times[Nunbound] *= transition_time

        if Tmin is not None:
            b = (np.cumsum(transition_times)[-1] > Tmin)
        else:
            b = (Nunbound >= max_unbind_events)

        if b:
            break

    transition_times = np.array(transition_times)  # discard the first zero
    states = states[:Ntransitions]
    if print_exec_time:
        print(f'Gillespie done in {time.time() - start:.2f} sec'+\
                                   f' for {name}.'+\
                                   f' {Ntransitions} transitions')

    if dwells_only:
        return off_times, on_times

    return np.cumsum(transition_times), states, off_times, on_times