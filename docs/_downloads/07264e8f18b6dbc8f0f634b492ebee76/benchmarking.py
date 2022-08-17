# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example benchmarking the computational efficiency of models. 
"""

from timeit import timeit
from prog_models.models import BatteryCircuit

def run_example():
    # Step 1: Create a model object
    batt = BatteryCircuit()
    
    # Step 2: Define future loading function 
    def future_loading(t, x=None):
        # Constant Loading
        return batt.InputContainer({'i': 2})

    # Step 3: Benchmark simulation of 600 seconds
    print('Benchmarking...')
    def sim():  
        results = batt.simulate_to(600, future_loading)
    time = timeit(sim, number=500)

    # Print results
    print('Simulation Time: {} ms/sim'.format(time*2))

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
