# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a battery being simulated for a set period of time and then till threshold is met.
"""

import matplotlib.pyplot as plt
from prog_models.models import BatteryElectroChem
from prog_models.models import BatteryCircuit as Battery

# VVV Uncomment this to use Electro Chemistry Model VVV
# Battery = BatteryElectroChem

def run_example(): 
    # Step 1: Create a model object
    batt = Battery()

    # Step 2: Define future loading function 
    def future_loading(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        elif (t < 3000):
            i = 2     
        else:
            i = 3
        return batt.InputContainer({'i': i})
    # simulate for 200 seconds
    print('\n\n------------------------------------------------')
    print('Simulating for 200 seconds\n\n')
    simulated_results = batt.simulate_to(200, future_loading, print = True, progress = True)
    # The result of the simulation is now stored in simulated_results.
    # You can access the results by accessing the individual variables:
    #   times, inputs, states, outputs, event_states
    # e.g., states = simulated_results.states

    # You can also plot the results, for example
    simulated_results.outputs.plot()

    # or, with configuration
    simulated_results.outputs.plot(compact = False, suptitle = 'Outputs', title = 'example title', xlabel = 'time', ylabel = 'output')

    # Simulate to threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    options = {
        'save_freq': 100, # Frequency at which results are saved
        'dt': 2, # Timestep
        'print': True,
        'progress': True
    }
    simulated_results = batt.simulate_to_threshold(future_loading, **options)

    # Alternately, you can set a max step size and allow step size to be adjusted automatically
    options['dt'] = ('auto', 2)  # set step size automatically, with a max of 2 seconds
    options['save_freq'] = 201  # Save every 201 seconds
    options['save_pts'] = [250, 772, 1023]  # Special points we sould like to see reported
    simulated_results = batt.simulate_to_threshold(future_loading, **options)
    # Note that even though the step size is 2, the odd points in the save frequency are met perfectly, dt is adjusted automatically to capture the save points

    simulated_results.outputs.plot()

    if isinstance(batt, BatteryElectroChem):
        # Plotting max current with time
        # This is the maximum sustainable current that can be drawn
        # from the battery at steady-state. It decreases with discharge
        # This information can be used to inform planning
        pm = [batt.performance_metrics(x)['max_i'][0] for x in simulated_results.states]
        plt.figure()
        plt.plot(simulated_results.times, pm)
        plt.xlabel('Time (s)')
        plt.ylabel('Maximum Sustainable Current Draw (amps)')

    # You can also change the integration method. For example:
    options['integration_method'] = 'rk4'  # Using Runge-Kutta 4th order
    simulated_results_rk4 = batt.simulate_to_threshold(future_loading, **options)

    simulated_results_rk4.outputs.plot(compact=False)

    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
