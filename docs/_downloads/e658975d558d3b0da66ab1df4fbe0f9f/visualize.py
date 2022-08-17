# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating the Visualization Module. 
"""

import matplotlib.pyplot as plt
from prog_models.visualize import plot_timeseries
from prog_models.models.thrown_object import ThrownObject

def run_example():
    print('Visualize Module Example')
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return {}

    # Step 3: Simulate to impact
    event = 'impact'
    options={'dt':0.005, 'save_freq':1}
    simulated_results = m.simulate_to_threshold(future_load,
                                                                             threshold_keys=[event], 
                                                                             **options)
    

    # Display states
    # ==============
    plot_timeseries(simulated_results.times, simulated_results.states, 
                          options = {'compact': False, 'suptitle': 'state evolution', 'title': True,
                                     'xlabel': 'time', 'ylabel': {'x': 'position', 'v': 'velocity'}, 'display_labels': 'minimal'},
                          legend  = {'display': True, 'display_at_subplot': 'all'} )
    plot_timeseries(simulated_results.times, simulated_results.states, options = {'compact': True, 'suptitle': 'state evolution', 'title': 'example title',
                                                    'xlabel': 'time', 'ylabel':'position'})
    plt.show()

if __name__ == '__main__':
    run_example()
