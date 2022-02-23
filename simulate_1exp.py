import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import seaborn as sns
sns.set(style="dark")
sns.set_color_codes()

start = time.time()
def simulate_1exp(t_on = 50, t_off = 20, measurement_time=300,
                        temporal_resolution=0.1, molecules=1, noise_amplitude=5):

    # Set (average) or ideal I values for bound and unbound configurations:
    high_I = 200  # state 1
    low_I = 0  # state 0

    data = np.zeros((molecules, int(measurement_time/temporal_resolution)))
    time = []
    for n in range(molecules):
        # At t=0 state from which to start:
        state = np.random.choice([0,1], p=[t_on/(t_off+t_on), t_off/(t_off+t_on)])
#        print(state)
        t = 0.1
        i = 0
        while t < measurement_time:
#            print (state)

            # ----- Gillespie Algorithm: Determine when next switch in I -----
            tau = t_on*(state == 0) + t_off*(state == 1)
#            print(state, tau)
            dwell_time = np.random.exponential(scale=tau)


            # ---- Adjust the I value start----
            if state == 1:
                I = high_I
            else:
                I = low_I

#             ----  Current event starts at:
            time_last_event = t
#            print(time_last_event + dwell_time)

            # ---- make a time trace with finite sampling resolution ----
            while t <= (time_last_event + dwell_time) and t < measurement_time:
#                print(state, tau)
                # ---- Add noise to I value ----
                I += np.random.normal(loc=0.0, scale=noise_amplitude)

                # ---- I values cannot fall below zero (negative fluoresence) ---
                I = np.maximum(I, 0.0)
                # ---- store the datapoint ----
                # ---- hold the (idealised) I value between two consequetive events ----
                data[n, i] = I
                time.append(t)
                i += 1
                t += temporal_resolution
            # ---- Now use Gillespie Algorithm to determine the nature of the next event
#            U = np.random.uniform()
#            if U < (1./tau) / (1./t_off + 1./t_on):
#                print('switching', tau, (1./tau) / (1./t_off + 1./t_on))
            if state == 0: state = 1
            elif state == 1: state = 0

    return data, time


t_on = 10
t_off = 50
T = 300
exposure_time = 0.1
N = 1000

#state = []
#for i in range(3000):
#    state.append(np.random.choice([0,1], p=[t_on/(t_off+t_on), t_off/(t_off+t_on)]))

red, time = simulate_1exp(t_on, t_off, measurement_time=T,
                        temporal_resolution=exposure_time, molecules=N)
green = np.zeros((N, red.shape[1]))
file = open(f'../simulations/hel{t_on}{t_off}.sim', 'wb')

data = {'red': red, 'green': green}
pickle.dump(data, file)
file.close()
print(f'Simulation time: {time.time() - start}')
#
plt.plot(time, red.flatten())
#plt.xticks(fontsize=15)
#plt.yticks([0,0.8,1.0],fontsize=15)
#plt.xlabel('time (s)', fontsize=15)
#plt.ylabel('I efficiency', fontsize=15)
#sns.despine()