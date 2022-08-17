# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example using simulate_to_threshold with vectorized states. In this example we are using the thrown_object model to simulate multiple thrown objects
"""

from prog_models.models.thrown_object import ThrownObject
from numpy import array, all

def run_example():
    # Step 1: Setup object
    m = ThrownObject()
    def future_load(t, x=None):
        return {}  # No load for thrown objects

    # Step 2: Setup vectorized initial state
    # For this example we are saying there are 4 throwers of various strengths and heights
    first_state = {
        'x': array([1.75, 1.8, 1.85, 1.9]),
        'v': array([35, 39, 22, 47])
    }

    # Step 3: Simulate to threshold
    # Here we are simulating till impact using the first state defined above
    (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, x = first_state, threshold_keys=['impact'], print = True, dt=0.1, save_freq=2)

    # Now lets do the same thing but only stop when all hit the ground
    def thresholds_met_eqn(thresholds_met):
        return all(thresholds_met['impact'])  # Stop when all impact ground

    simulated_results = m.simulate_to_threshold(future_load, x = first_state, thresholds_met_eqn=thresholds_met_eqn, print = True, dt=0.1, save_freq=2)

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
