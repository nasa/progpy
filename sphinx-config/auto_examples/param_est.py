# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating the model parameter estimation feature. 
"""

from prog_models.models.thrown_object import ThrownObject

def run_example():
    # Step 1: Build the model with your best guess in parameters
    # Here we're guessing that the thrower is 20 meters tall. Obviously not true!
    # Let's see if parameter estimation can fix this
    m = ThrownObject(thrower_height=20)

    # Step 2: Collect data from the use of the system. Let's pretend we threw the ball once, and collected position measurements 
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inputs = [{}]*9
    outputs = [
        {'x': 1.83},
        {'x': 36.95},
        {'x': 62.36},
        {'x': 77.81},
        {'x': 83.45},
        {'x': 79.28},
        {'x': 65.3},
        {'x': 41.51},
        {'x': 7.91},
    ]

    # Step 3: Identify the parameters to be estimated
    keys = ['thrower_height', 'throwing_speed']

    # Printing state before
    print('Model configuration before')
    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4))

    # Step 4: Run parameter estimation with data
    m.estimate_params([(times, inputs, outputs)], keys, dt=0.01)

    # Print result
    print('\nOptimized configuration')
    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4))
    
    # Sure enough- parameter estimation determined that the thrower's height wasn't 20 m, instead was closer to 1.9m, a much more reasonable height!

if __name__=='__main__':
    run_example()
