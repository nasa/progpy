# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating ways to use future loading. 
"""

from prog_models.models import BatteryCircuit
from statistics import mean
from numpy.random import normal

def run_example(): 
    m = BatteryCircuit()

    ## Example 1: Variable loading 
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
        return m.InputContainer({'i': i})
    
    # Simulate to threshold
    options = {
        'save_freq': 100,  # Frequency at which results are saved
        'dt': 2  # Timestep
    }
    simulated_results = m.simulate_to_threshold(future_loading, **options)

    # Now lets plot the inputs and event_states
    simulated_results.inputs.plot(ylabel = 'Variable Load Current (amps)')
    simulated_results.event_states.plot(ylabel = 'Variable Load Event State')

    ## Example 2: Moving Average loading 
    # This is useful in cases where you are running reoccuring simulations, and are measuring the actual load on the system, 
    # but dont have a good way of predicting it, and you expect loading to be steady

    def future_loading(t, x=None):
        return future_loading.load
    future_loading.load = m.InputContainer({key : 0 for key in m.inputs})

    # Lets define another function to handle the moving average logic
    window = 10 # Number of elements in window
    def moving_avg(i):
        for key in m.inputs:
            moving_avg.loads[key].append(i[key])
            if len(moving_avg.loads[key]) > window:
                del moving_avg.loads[key][0]  # Remove first item

        # Update future loading eqn
        future_loading.load = {key : mean(moving_avg.loads[key]) for key in m.inputs} 
    moving_avg.loads = {key : [] for key in m.inputs} 

    # OK, we've setup the logic of the moving average. 
    # Now lets say you have some measured loads to add
    measured_loads = [10, 11.5, 12.0, 8, 2.1, 1.8, 1.99, 2.0, 2.01, 1.89, 1.92, 2.01, 2.1, 2.2]
    
    # We're going to feed these into the future loading eqn
    for load in measured_loads:
        moving_avg({'i': load})
    
    # Now the future_loading eqn is setup to use the moving average of whats been seen
    # Simulate to threshold
    simulated_results = m.simulate_to_threshold(future_loading, **options)

    # Now lets plot the inputs and event_states
    simulated_results.inputs.plot(ylabel = 'Moving Average Current (amps)')
    simulated_results.event_states.plot(ylabel = 'Moving Average Event State')

    # In this case, this estimate is wrong because loading will not be steady, but at least it would give you an approximation.

    # If more measurements are received, the user could estimate the moving average here and then run a new simulation. 

    ## Example 3: Gaussian Distribution 
    # In this example we will still be doing a variable loading like the first option, but we are going to use a 
    # gaussian distribution for each input. 

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
        return m.InputContainer({'i': normal(i, future_loading.std)})
    future_loading.std = 0.2

    # Simulate to threshold
    simulated_results = m.simulate_to_threshold(future_loading, **options)

    # Now lets plot the inputs and event_states
    simulated_results.inputs.plot(ylabel = 'Variable Gaussian Current (amps)')
    simulated_results.event_states.plot(ylabel = 'Variable Gaussian Event State')

    # Example 4: Gaussian- increasing with time
    # For this we're using moving average. This is realistic because the further out from current time you get, 
    # the more uncertainty there is in your prediction. 

    def future_loading(t, x=None):
        std = future_loading.base_std + future_loading.std_slope * (t - future_loading.t)
        return {key : normal(future_loading.load[key], std) for key in future_loading.load.keys()}
    future_loading.load = {key : 0 for key in m.inputs} 
    future_loading.base_std = 0.001
    future_loading.std_slope = 1e-4
    future_loading.t = 0

    # Lets define another function to handle the moving average logic
    window = 10  # Number of elements in window
    def moving_avg(i):
        for key in m.inputs:
            moving_avg.loads[key].append(i[key])
            if len(moving_avg.loads[key]) > window:
                del moving_avg.loads[key][0]  # Remove first item

        # Update future loading eqn
        future_loading.load = {key : mean(moving_avg.loads[key]) for key in m.inputs} 
    moving_avg.loads = {key : [] for key in m.inputs} 

    # OK, we've setup the logic of the moving average. 
    # Now lets say you have some measured loads to add
    measured_loads = [10, 11.5, 12.0, 8, 2.1, 1.8, 1.99, 2.0, 2.01, 1.89, 1.92, 2.01, 2.1, 2.2]
    
    # We're going to feed these into the future loading eqn
    for load in measured_loads:
        moving_avg({'i': load})

    # Simulate to threshold
    simulated_results = m.simulate_to_threshold(future_loading, **options)

    # Now lets plot the inputs and event_states
    simulated_results.inputs.plot(ylabel = 'Moving Average Current (amps)')
    simulated_results.event_states.plot(ylabel = 'Moving Average Event State')
    
    # In this example future_loading.t has to be updated with current time before each prediction.
    
    # Example 5 Function of state
    # here we're pretending that input is a function of SOC. It increases as we approach SOC

    def future_loading(t, x=None):
        if x is not None:
            event_state = future_loading.event_state(x)
            return m.InputContainer({'i': future_loading.start + (1-event_state['EOD']) * future_loading.slope})  # default
        return m.InputContainer({'i': future_loading.start})
    future_loading.t = 0
    future_loading.event_state = m.event_state
    future_loading.slope = 2  # difference between input with EOD = 1 and 0. 
    future_loading.start = 0.5

    # Simulate to threshold
    simulated_results = m.simulate_to_threshold(future_loading, **options)

    # Now lets plot the inputs and event_states
    simulated_results.inputs.plot(ylabel = 'Moving Average Current (amps)')
    simulated_results.event_states.plot(ylabel = 'Moving Average Event State')

    # In this example future_loading.t has to be updated with current time before each prediction.

    # Show plots
    import matplotlib.pyplot as plt
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
