# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Simple example benchmarking the computational efficiency of models.
"""

from prog_models.models import BatteryCircuit
from timeit import timeit

def run_example():
    # Step 1: Create a model object
    batt = BatteryCircuit()
    
    # Step 2: Define future loading function 
    loading = batt.InputContainer({'i': 2})  # Constant loading
    def future_loading(t, x=None):
        # Constant Loading
        return loading

    # Step 3: Benchmark simulation of 600 seconds
    print('Benchmarking...')
    def sim():  
        batt.simulate_to(600, future_loading)
    time = timeit(sim, number=500)

    # Print results
    print('Simulation Time: {} ms/sim'.format(time))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
