# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating approaches for adding and handling model noise
"""

import matplotlib.pyplot as plt
from prog_models.models.thrown_object import ThrownObject

def run_example():
    # Define future loading
    def future_load(t=None, x=None):  
        # The thrown object model has no inputs- you cannot load the system (i.e., affect it once it's in the air)
        # So we return an empty input container
        return m.InputContainer({})

    # Define configuration for simulation
    config = {
        'threshold_keys': 'impact', # Simulate until the thrown object has impacted the ground
        'dt': 0.005, # Time step (s)
        'save_freq': 0.5, # Frequency at which results are saved (s)
    }

    # Define a function to print the results - will be used later
    def print_results(simulated_results):
        # Print results
        print('states:')
        for (t,x) in zip(simulated_results.times, simulated_results.states):
            print('\t{:.2f}s: {}'.format(t, x))

        print('outputs:')
        for (t,x) in zip(simulated_results.times, simulated_results.outputs):
            print('\t{:.2f}s: {}'.format(t, x))

        print('\nimpact time: {:.2f}s'.format(simulated_results.times[-1]))
        # The simulation stopped at impact, so the last element of times is the impact time

        # Plot results
        simulated_results.states.plot()

    # Ex1: No noise
    m = ThrownObject(process_noise = False)
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex1: No noise')

    # Ex2: with noise - same noise applied to every state
    process_noise = 15
    m = ThrownObject(process_noise = process_noise)  # Noise with a std of 0.5 to every state
    print('\nExample without same noise for every state')
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex2: Basic Noise')

    # Ex3: noise- more noise on position than velocity
    process_noise = {'x': 30, 'v': 1}
    m = ThrownObject(process_noise = process_noise) 
    print('\nExample with more noise on position than velocity')
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex3: More noise on position')

    # Ex4: noise- Ex3 but uniform
    process_noise_dist = 'uniform'
    model_config = {'process_noise_dist': process_noise_dist, 'process_noise': process_noise}
    m = ThrownObject(**model_config) 
    print('\nExample with more uniform noise')
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex4: Ex3 with uniform dist')

    # Ex5: noise- Ex3 but triangle
    process_noise_dist = 'triangular'
    model_config = {'process_noise_dist': process_noise_dist, 'process_noise': process_noise}
    m = ThrownObject(**model_config) 
    print('\nExample with triangular process noise')
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex5: Ex3 with triangular dist')

    # Ex6: Measurement noise
    # Everything we've done with process noise, we can also do with measurement noise.
    # Just use 'measurement_noise' and 'measurement_noise_dist' 
    measurement_noise = {'x': 20}  # For each output
    measurement_noise_dist = 'uniform'
    model_config = {'measurement_noise_dist': measurement_noise_dist, 'measurement_noise': measurement_noise}
    m = ThrownObject(**model_config) 
    print('\nExample with measurement noise')
    print('- Note: outputs are different than state- this is the application of measurement noise')
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex6: Measurement noise')

    # Ex7: OK, now for something a little more complicated. Let's try proportional noise on v only (more variation when it's going faster)
    # This can be used to do custom or more complex noise distributions
    def apply_proportional_process_noise(self, x, dt = 1):
        x['v'] -= dt*0.5*x['v']
        return x
    model_config = {'process_noise': apply_proportional_process_noise}
    m = ThrownObject(**model_config)
    print('\nExample with proportional noise on velocity')
    simulated_results = m.simulate_to_threshold(future_load, **config)
    print_results(simulated_results)
    plt.title('Ex7: Proportional noise on velocity')

    print('\nNote: If you would like noise to be applied in a repeatable manner, set the numpy random seed to a fixed value')
    print('e.g., numpy.random.seed(42)')
    plt.show()

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
