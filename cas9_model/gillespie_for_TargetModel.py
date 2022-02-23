# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 05:30:49 2019

@author: iason
"""

import numpy as np
#import random
#import time
import cas9_model.gillespie.cyGillespie as gill
import functools

class Gillespie(object):
    def __init__(self, target):
        self.__target = target

        self.find_events = functools.partial(gill.find_events,
                                             self.__target.forward_rates,
                                             self.__target.backward_rates,
                                             self.__target.name)

    def simulate_trace(self, time_resolution=0.1, Tmax=300, state_init=-1,
                       state_monitored=-1, high=100, low=0, all_states=False):
        '''
        Builds a time trace of certain duration and resolution. The trace shows
        how long a state is bound and unbound and report the tau_off time to
        reach the state and tau_on time to leave the state (counter-intuitive!)

        :param time_resolution: float or int. The step between consecutive
                                time-points
        :param Tmax: float or int. The time of measurement
        :param state_init: int in [-1, guide_length]. Starting state
        :param state_monitored: int in [-1, guide_length].
                              The state from which to bind or unbind

        :param high: int or float. The value which corresponds when we are on
                                   the state
        :param low: int or float. The value which corresponds to when we are
                                  off the state
        :param all_states: Boolean. if True, a trace showing all the states
                           will be returned.
        :returns: trace, time_axis, off_times, on_times
        '''

        transition_times, states, t_off, t_on = \
        self.find_events(state_init, state_monitored, Tmin=Tmax)

        frames = int(transition_times[-1]/time_resolution)
        trace = [state_monitored]
        indx = 0
        time = 0
        while len(trace) < frames:

            while time < transition_times[indx+1]:
                trace.append(states[indx])
                time += time_resolution
            indx += 1

        Nframes = int(Tmax/time_resolution)
        trace = np.array(trace[:Nframes])  # take only until Tmax
        time = np.arange(0, Nframes)*time_resolution

        return time, trace, t_off, t_on, transition_times, states


# supporting functions
from math import floor

def sample(array, sample_size):
    output = np.empty(sample_size)
    rel_size = float(array.size) / sample_size
    output = []
    for i in range(sample_size):
        output.append(array[int(floor(i * rel_size))])
    return np.array(output)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx




#    def find_events1(self, state_init=-1, state_monitored=-1, max_unbind_events=10,
#                    Tmin=None, dwells_only=False, print_exec_time=False):
#        '''
#        Performs gillespie algorithm to calculate the transition times and the
#        corresponding states for the Cas9 model. It also outputs the tau off
#        and on times for a selected state
#
#        :param max_unbinding_events: The number of unbinding events in which
#                                    the algorithm stops
#        :param state_init: The starting state. In default it starts from
#                            solution
#        :param state_unbound: The state for which the stop criterion is
#                                implemented. For this state the t_off and
#                                t_on are calculated
#        :returns: transition times, states, off_times, on_times
#        '''
#        start = time.time()  # start time of the simulation
#
#        frates = np.copy(self.__target.forward_rates)
#        brates = np.copy(self.__target.backward_rates)
#        # -1: unbound
#        #  0: PAM,
#        #  1-20: number of bp in R-loop
#
#        transition_times = [0]
#        off_times = []
#        on_times = []
#        states=[state_init]
#        off_time = 0
#        state = state_init # starting state
#        Ntransitions = 0
#        while True:
#            frate = frates[state+1]
#            brate = brates[state+1]
#
#            rate = frate + brate
#            # find the time for the next transition
#            transition_time = np.random.exponential(scale=rate**(-1))
#
#            # use Gillespie algorithm to switch state stochastically
#            p = random.random()
#            if p < frate/(brate + frate):
#                state += 1
#            else:
#                state -= 1
#            if not dwells_only:
#                transition_times.append(transition_time)
#            states.append(state)
#            Ntransitions += 1
#
#            # if new state is unbound_state (solution), append total bound time until now and start over
#            if state == state_monitored:
#                off_times.append(off_time)
#                off_time = 0
#            # if the current state and previous state is a bound state add the:
#            elif states[-1] != state_monitored and states[-2] != state_monitored:
#                off_time += transition_time
#            else:  # if the current state is unbound_state (solution), the only choice is to bind
#                on_times.append(transition_time)
#
#            if Tmin is not None:
#                b = (np.cumsum(transition_times)[-1] > Tmin)
#            else:
#                b = (len(off_times) >= max_unbind_events)
#
#            if b:
#                break
#
#        transition_times = np.array(transition_times)  # discard the first zero
#        states = np.array(states)
#        off_times = np.array(off_times)
#        on_times = np.array(on_times)
#        if print_exec_time:
#            print(f'Gillespie done in {time.time() - start:.2f} sec'+\
#                                       f' for {self.__target.name}.'+\
#                                       f' {Ntransitions} transitions')
#
#        if dwells_only:
#            return off_times, on_times
#
#        return np.cumsum(transition_times), states, off_times, on_times



