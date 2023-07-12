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

print("Configuration")
print("--------------------")
print(f"Training runs: {N_TRAIN}")
print(f"Test runs: {N_TEST}")
print(f"Save freq: {SAVE_FREQ}")
print(f"Process noise: {PROCESS_NOISE}")
print(f"Window size: {WINDOW}")
print(f"LSTM Layers {LAYERS}")
print(f"LSTM Units {UNITS}\n")

# Step 1: Generate data (with noise) from Battery Model
print('generating training data')
m = BatteryElectroChemEOD(process_noise=PROCESS_NOISE)

times, inputs, outputs, event_states, t_met = [], [], [], [], []
for _ in range(N_TRAIN):
    # Randomly Generate future loading profile
    f_load_times = [np.random.uniform(0, 3500) for _ in range(N_FUTURE_LOAD_STEPS)]
    f_load_times.sort()
    f_load_times.append(np.inf)
    f_loads = {'i': [np.random.uniform(*LOAD) for _ in range(N_FUTURE_LOAD_STEPS)]}
    f_loads['i'].append(4) # final load to ensure simulation doesn't run too long
    future_load = Piecewise(m.InputContainer, f_load_times, f_loads)

    # Randomly assign dt
    # Note: Different dt's are important for a dt-independent model
    dt = np.random.uniform(*DT)

    # Simulate
    results = m.simulate_to_threshold(future_load, dt=5, save_freq=SAVE_FREQ)

    # Save results
    times.append(results.times)
    # Add dt to inputs
    inputs.append(np.array([np.hstack((u_i.matrix[:][0].T, [dt])) for u_i in results.inputs], dtype=float))
    outputs.append(results.outputs)
    event_states.append(results.event_states)
    t_met.append([m.threshold_met(x)['EOD'] for x in results.states])

# Step 2: Train Model
print('training model')
train_time = time.perf_counter()
m_lstm = LSTMStateTransitionModel.from_data(
    inputs=inputs,
    outputs=outputs,
    event_states=event_states,
    t_met=t_met,
    window=WINDOW,
    layers=LAYERS,
    units=UNITS,
    epochs=1,
    input_keys=['i', 'dt'],
    output_keys=['t', 'v'],
    event_keys=['EOD'])
train_time = time.perf_counter() - train_time

# Step 3: Generate test data
print('generating test data')
times, inputs, outputs, event_states, t_met = [], [], [], [], []
m.parameters['process_noise'] = 0
for _ in range(N_TEST):
    # Randomly Generate future loading profile
    f_load_times = [np.random.uniform(0, 3500) for _ in range(N_FUTURE_LOAD_STEPS)]
    f_load_times.sort()
    f_load_times.append(np.inf)
    f_loads = {'i': [np.random.uniform(*LOAD) for _ in range(N_FUTURE_LOAD_STEPS)]}
    f_loads['i'].append(4) # final load to ensure simulation doesn't run too long
    future_load = Piecewise(m.InputContainer, f_load_times, f_loads)

    # Randomly assign dt
    # Note: Different dt's are important for a dt-independent model
    dt = np.random.uniform(*DT)

    # Simulate
    results = m.simulate_to_threshold(future_load, dt=5, save_freq=SAVE_FREQ)

    # Save results
    times.append(results.times)
    inputs.append(results.inputs)
    outputs.append(results.outputs)
    event_states.append(results.event_states)
    t_met.append([m.threshold_met(x)['EOD'] for x in results.states])

# Step 4: Evaluate performance
print("\nProfiling results")
print("--------------------")
print(f"Training time: {train_time} s")

# Future load function for model
class FutureLoad:
    def __init__(self, inputs, m, outputs):
        self.Container = m.InputContainer
        self.times = inputs.times
        self.dt = self.times[1]-self.times[0]
        self.inputs = [u['i'] for u in inputs]
        self.m = m
        self.outputs = outputs

    def __call__(self, t, x=None):
        if x is None:
            x = self.m.initialize()

        if t > self.times[-1]:
            index = -1
        else:
            index = next(i for i, time in enumerate(self.times) if time >= t-self.dt/10) # About equal

        u = self.inputs[index]
        if x.matrix[0, 0] is None:
            # Nominally use index-1, unless first spot
            index_z = index if index==0 else index-1
            z = self.outputs[index_z].matrix
        else:
            z = self.m.output(x).matrix

        return self.Container({
            'i': u,
            'dt': dt,
            't_t-1': z[0, 0],
            'v_t-1': z[1, 0]
        })

# Plot
# Dashed is lstm model, solid is test data, dashed is lstm

        # UNCOMMENT LATER 

# fig = plt.figure()
# plt.ylabel('Voltage')
# plt.xlabel('Time (s)')

# TESTING CALC_ERROR FUNCTION #
z_error = m_lstm.calc_error(
    times=times,
    inputs=inputs,
    outputs=outputs
)

#############         PLOTTING         #############

sim_time = 0
toe_error = 0
lstm_inputs = []
for i, z in enumerate(outputs):
    plt.plot(times[i], [z_i['v'] for z_i in z], COLORS[i]+'-', label=f'Test {i+1}')
    
    future_load = FutureLoad(inputs[i], m_lstm, z)
    
    # Record lstm input profile - needed for output profile
    lstm_inputs.append([future_load(t_i) for t_i in times[i]])

    # Simulate and plot
    sim_time_i = time.perf_counter()
    results = m_lstm.simulate_to_threshold(future_load, dt=5, save_freq=future_load.dt*10, horizon=times[i][-1]+1000)
    sim_time += (time.perf_counter() - sim_time_i)/len(results.times)
    plt.plot(results.times, [z['v'] for z in results.outputs], COLORS[i]+'--', label=f'LSTM {i+1}')

    # Record error in time of event for later metric
    if results.times[-1] >= times[i][-1]+1000:
        # Past horizon, didn't reach EOD with 1000 over real value
        toe_error = np.inf
    else:
        toe_error += (times[i][-1] - results.times[-1]['EOD'])**2

plt.legend()
plt.show()
print(f"Average Simulation Rate: {sim_time/(N_TEST*SAVE_FREQ)} (seconds clock / seconds sim)")

# Error in time of event
toe_error /= N_TEST
print(f"MSE in time of event: {toe_error} ")

# Error in output profile
z_error = m_lstm.calc_error(
    times=times,
    inputs=inputs,
    outputs=outputs
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
