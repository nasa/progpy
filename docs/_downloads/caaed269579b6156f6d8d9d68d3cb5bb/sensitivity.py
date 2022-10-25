# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example performing a sensitivity analysis on a new model. 
"""

import numpy as np
# Deriv prog model was selected because the model can be described as x' = x + dx*dt
from prog_models.models.thrown_object import ThrownObject

def run_example():
    # Demo model
    # Step 1: Create instance of model
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return m.InputContainer({})

    # Step 3: Setup range on parameters considered
    thrower_height_range = np.arange(1.2, 2.1, 0.1)

    # Step 4: Sim for each 
    event = 'impact'
    eods = np.empty(len(thrower_height_range))
    for (i, thrower_height) in zip(range(len(thrower_height_range)), thrower_height_range):
        m.parameters['thrower_height'] = thrower_height
        simulated_results = m.simulate_to_threshold(future_load, threshold_keys=[event], dt =1e-3, save_freq =10)
        eods[i] = simulated_results.times[-1]

    # Step 5: Analysis
    print('For a reasonable range of heights, impact time is between {} and {}'.format(round(eods[0],3), round(eods[-1],3)))
    sensitivity = (eods[-1]-eods[0])/(thrower_height_range[-1] - thrower_height_range[0])
    print('  - Average sensitivity: {} s per cm height'.format(round(sensitivity/100, 6)))
    print("  - It seems impact time is not very sensitive to thrower's height")

    # Now lets repeat for throw speed
    throw_speed_range = np.arange(20, 40, 1)
    eods = np.empty(len(throw_speed_range))
    for (i, throw_speed) in zip(range(len(throw_speed_range)), throw_speed_range):
        m.parameters['throwing_speed'] = throw_speed
        simulated_results = m.simulate_to_threshold(future_load, threshold_keys=[event], options={'dt':1e-3, 'save_freq':10})
        eods[i] = simulated_results.times[-1]

    print('\nFor a reasonable range of throwing speeds, impact time is between {} and {}'.format(round(eods[0],3), round(eods[-1],3)))
    sensitivity = (eods[-1]-eods[0])/(throw_speed_range[-1] - throw_speed_range[0])
    print('  - Average sensitivity: {} s per m/s speed'.format(round(sensitivity/100, 6)))
    print("  - It seems impact time is much more dependent on throwing speed")

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
