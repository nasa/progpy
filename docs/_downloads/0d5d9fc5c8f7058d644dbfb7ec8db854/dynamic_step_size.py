# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating ways to use the dynamic step size feature. This feature allows users to define a time-step that changes with time or state. 
"""

import prog_models
from prog_models.models.thrown_object import ThrownObject

def run_example():
    print("EXAMPLE 1: dt of 1 until 8 sec, then 0.5\n\nSetting up...\n")
    # Step 1: Create instance of model
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # Step 3: Define dynamic step size function
    # This `next_time` function will specify what the next step of the simulation should be at any state and time. 
    # f(x, t) -> (t, dt)
    def next_time(t, x):
        # In this example dt is a function of time. We will use a dt of 1 for the first 8 seconds, then 0.5 
        if t < 8:
            return 1
        return 0.5

    # Step 4: Simulate to impact
    # Here we're printing every time step so we can see the step size change
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, save_freq=1e-99, print=True, dt=next_time, threshold_keys=['impact'])

    # Example 2
    print("EXAMPLE 2: dt of 1 until impact event state 0.5, then 0.25 \n\nSetting up...\n")

    # Step 3: Define dynamic step size function
    # This `next_time` function will specify what the next step of the simulation should be at any state and time. 
    # f(x, t) -> (t, dt)
    def next_time(t, x):
        # In this example dt is a function of state. Uses a dt of 1 until impact event state 0.5, then 0.25
        event_state = m.event_state(x)
        if event_state['impact'] < 0.5:
            return 0.25
        return 1

    # Step 4: Simulate to impact
    # Here we're printing every time step so we can see the step size change
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, save_freq=1e-99, print=True, dt=next_time, threshold_keys=['impact'])

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
