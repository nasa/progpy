# This is an profiling tool to use to evaluate the performance of LSTM. 
# This tool can be used to measure the LSTM performance so we can evaluate any improvements

from matplotlib import pyplot as plt
import numpy as np
from progpy.data_models import LSTMStateTransitionModel
from progpy.loading import Piecewise
from progpy.models import BatteryElectroChemEOD
import time

N_TRAIN = 5
N_TEST = 3
DT = (0.2, 2) # min, max
LOAD = (0.5, 4) # min, max
SAVE_FREQ = 2
PROCESS_NOISE = 1e-4
N_FUTURE_LOAD_STEPS = 8
LAYERS = 2 # LSTM Layers
UNITS = 64 # LSTM Units in each layer
WINDOW = 16 # LSTM WINDOW SIZE
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def generate_data(m, N):
    times, inputs, outputs, event_states, t_met = [], [], [], [], []
    for _ in range(N):
        f_load_times = [np.random.uniform(0, 3500) for _ in range(N_FUTURE_LOAD_STEPS)]
        f_load_times.sort()
        f_load_times.append(np.inf)
        f_loads = {'i': [np.random.uniform(*LOAD) for _ in range(N_FUTURE_LOAD_STEPS)]}
        f_loads['i'].append(4)
        future_load = Piecewise(m.InputContainer, f_load_times, f_loads)
        dt = np.random.uniform(*DT)
        results = m.simulate_to_threshold(future_load, dt=1, save_freq=SAVE_FREQ)
        times.append(results.times)
        inputs.append(np.array([np.hstack((u_i.matrix[:][0].T, [dt])) for u_i in results.inputs], dtype=float))
        outputs.append(results.outputs)
        event_states.append(results.event_states)
        t_met.append([m.threshold_met(x)['EOD'] for x in results.states])
    return times, inputs, outputs, event_states, t_met

print("Configuration")
print("--------------------")
print(f"Training runs: {N_TRAIN}")
print(f"Test runs: {N_TEST}")
print(f"Save freq: {SAVE_FREQ}")
print(f"Process noise: {PROCESS_NOISE}")
print(f"Window size: {WINDOW}")
print(f"LSTM Layers {LAYERS}")
print(f"LSTM Units {UNITS}\n")

print('generating training data')
m = BatteryElectroChemEOD(process_noise=PROCESS_NOISE)
times_train, inputs_train, outputs_train, event_states_train, t_met_train = generate_data(m, N_TRAIN)

print('training model')
train_time = time.perf_counter()
m_lstm = LSTMStateTransitionModel.from_data(
    inputs=inputs_train,
    outputs=outputs_train,
    event_states=event_states_train,
    t_met=t_met_train,
    window=WINDOW,
    epochs=25,
    layers=LAYERS,
    units=UNITS,
    input_keys=['i', 'dt'],
    output_keys=['t', 'v'],
    event_keys=['EOD'])
train_time = time.perf_counter() - train_time

print('generating test data')
m.parameters['process_noise'] = 0
times_test, inputs_test, outputs_test, event_states_test, t_met_test = generate_data(m, N_TEST)

print("\nProfiling results")
print("--------------------")
print(f"Training time: {train_time} s")

class FutureLoad:
    def __init__(self, times, inputs, m, outputs):
        self.Container = m.InputContainer
        self.times = times
        self.dt = self.times[1]-self.times[0]
        self.inputs = [u[0] for u in inputs]
        self.m = m
        self.outputs = outputs

    def __call__(self, t, x=None):
        if x is None:
            x = self.m.initialize()
        if t > self.times[-1]:
            index = -1
        else:
            index = next(i for i, time in enumerate(self.times) if time >= t-self.dt/10)
        u = self.inputs[index]
        if x.matrix[0, 0] is None:
            index_z = index if index==0 else index-1
            z = self.outputs[index_z].matrix
        else:
            z = self.m.output(x).matrix
        return self.Container({
            'i': u,
            'dt': self.dt,
            't_t-1': z[0, 0],
            'v_t-1': z[1, 0]
        })

fig = plt.figure()
plt.ylabel('Voltage')
plt.xlabel('Time (s)')
sim_time = 0
toe_error = 0
for i, z in enumerate(outputs_test):
    plt.plot(times_test[i], [z_i['v'] for z_i in z], COLORS[i]+'-')
    future_load = FutureLoad(times_test[i], inputs_test[i], m_lstm, z)
    sim_time_i = time.perf_counter()
    results = m_lstm.simulate_to_threshold(future_load, dt=1, horizon=times_test[i][-1]+1000)
    sim_time += (time.perf_counter() - sim_time_i)/len(results.times)
    plt.plot(results.times, [z['v'] for z in results.outputs], COLORS[i]+'--')
    if results.times[-1] > times_test[i][-1]+1000:
        toe_error = np.inf
    else:
        toe_error += (times_test[i][-1] - results.times[-1]['EOD'])**2

plt.show()
print(f"Average Simulation Rate: {sim_time/(N_TEST*SAVE_FREQ)} (seconds clock / seconds sim)")

toe_error /= N_TEST
print(f"MSE in time of event: {toe_error} ")

z_error = m_lstm.calc_error(
    times=times_test,
    inputs=inputs_test,
    outputs=outputs_test
)
print(f"MSE error in Output: {z_error}")

# GOAL OF WORK
# 0. Verify approach
# 1. Improve performance, in order of importance (most->least)
#   a. ToE error
#   b. Output error
#   c. Simulation time
#   d. Training time
# 2. Simplify use (Reduce complexity of code above through helper fcns, etc)
# 3. Explore solutions for hyperparameter tuning