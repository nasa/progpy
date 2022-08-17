# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating when and how to identify model state limits. 
"""

from prog_models.models.thrown_object import ThrownObject
from math import inf

def run_example():
    # Demo model
    # Step 1: Create instance of model (without drag)
    m = ThrownObject( cd = 0 )

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # add state limits
    m.state_limits = {
        # object may not go below ground height
        'x': (0, inf),

        # object may not exceed the speed of light
        'v': (-299792458, 299792458)
    }

    # Step 3: Simulate to impact
    event = 'impact'
    simulated_results = m.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=1)
    
    # Print states
    print('Example 1')
    for i, state in enumerate(simulated_results.states):
        print(f'State {i}: {state}')
    print()

    # Let's try setting x to a number outside of its bounds
    x0 = m.initialize(u = {}, z = {})
    x0['x'] = -1

    simulated_results = m.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=1, x = x0)

    # Print states
    print('Example 2')
    for i, state in enumerate(simulated_results.states):
        print('State ', i, ': ', state)
    print()

    # Let's see what happens when the objects speed aproaches its limit
    x0 = m.initialize(u = {}, z = {})
    x0['x'] = 1000000000
    x0['v'] = 0
    m.parameters['g'] = -50000000
    
    print('Example 3')
    simulated_results = m.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=0.3, x = x0, print = True, progress = False)

    # Note that the limits can also be applied manually using the apply_limits function
    print('limiting states')
    x = {'x': -5, 'v': 3e8}  # Too fast and below the ground
    print('\t Pre-limit: {}'.format(x))
    x = m.apply_limits(x)
    print('\t Post-limit: {}'.format(x))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()