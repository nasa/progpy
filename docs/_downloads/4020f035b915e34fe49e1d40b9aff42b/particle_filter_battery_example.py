# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
In this example the BatteryElectroChemEOD model is used with a particle filter to estimate the state of the battery
"""

import matplotlib.pyplot as plt
import numpy as np
from prog_algs import *
from prog_models.models import BatteryElectroChemEOD

def run_example():
    ## Setup
    # Save battery model
    # Time increment
    dt = 1
    # Process noise
    Q_vars = {
        'tb': 1,
        'Vo': 0.01,
        'Vsn': 0.01,
        'Vsp': 0.01,
        'qnB': 1,
        'qnS': 1,
        'qpB': 1,
        'qpS': 1
    }
    # Measurement noise
    R_vars = {
        't': 2, 
        'v': 0.02
    }
    battery = BatteryElectroChemEOD(process_noise= Q_vars,
                                    measurement_noise = R_vars, 
                                    dt = dt)
    load = battery.InputContainer({"i": 1})  # Optimization
    def future_loading(t, x=None):
        return load

    # Simulate data until EOD
    start_u = future_loading(0)
    start_x = battery.initialize(start_u)
    start_y = battery.output(start_x)
    sim_results = battery.simulate_to_threshold(future_loading, start_y, save_freq = 1)

    # Run particle filter
    all_particles = []
    n_times = int(np.round(np.random.uniform(len(sim_results.times)*.25,len(sim_results.times)*.45,1)))# Random current time

    for i in range(n_times):
        if i == 0:
            batt_pf = state_estimators.ParticleFilter(model = battery,  x0 = sim_results.states[i], num_particles = 250)
        else:
            batt_pf.estimate(t = sim_results.times[i], u = sim_results.inputs[i], z = sim_results.outputs[i])
        all_particles.append(batt_pf.particles)

    # Mean of the particles
    alpha = 0.05
    states_vsn = [s['tb'] for s in sim_results.states]
    pf_mean = [{key: np.mean(ps[key]) for key in battery.states} for ps in all_particles]
    pf_low = [{key: np.quantile(ps[key], alpha / 2.0) for key in battery.states} for ps in all_particles]
    pf_upp = [{key: np.quantile(ps[key], 1.0 - alpha / 2.0) for key in battery.states} for ps in all_particles]
    print("First State:", pf_mean[0])
    print("Current State:", pf_mean[-1])
    plt.plot(sim_results.times[:n_times],[p['tb'] for p in pf_mean],linewidth=0.7,color="blue")
    plt.plot(sim_results.times[:n_times], states_vsn[:n_times],"--",linewidth=0.7,color="red")
    plt.fill_between(sim_results.times[:n_times],[p['tb'] for p in pf_low],[p['tb'] for p in pf_upp],alpha=0.5,color="blue")
    plt.show()
    

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()