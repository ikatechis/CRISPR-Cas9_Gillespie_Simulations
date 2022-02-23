# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:14:39 2019

@author: iason
"""
import concurrent.futures
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cas9_model.target_model import load_parameters
from landscape_from_gillespie import landscape_gillespie
import seaborn as sns
sns.set_style('ticks')
sns.set()

start = time.time()

params = load_parameters('boyle')
length = 20
N = 100
repeats = 10

log = []
landscape = np.zeros((repeats, length+2))
tau_off = np.zeros((repeats, length+1))
tau_on = np.zeros((repeats, length+1))

if __name__ == '__main__':  # This is required for MP to work in Windows
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(landscape_gillespie, params, N=N, length=length)\
                   for _ in range(repeats)]

    for i, f in enumerate(concurrent.futures.as_completed(results)):
        tau_off[i, :], tau_on[i, :], landscape[i, :] = f.result()[1:]


    fname = f'landscape_simulations/landscapes_L{length}N{N}R{repeats}'
    np.savez(fname, off_times=tau_off, on_times=tau_on, landscape=landscape)
    #
    l = f'Total Simulation Time: {time.time() - start:.2f} sec\n'
    print(l)

    log_file = open(fname+'_MP.log', 'w')
    log_file.writelines(l)
    log_file.close()
