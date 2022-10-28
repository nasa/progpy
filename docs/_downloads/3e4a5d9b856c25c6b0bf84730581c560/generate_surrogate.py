# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a Dynamic Mode Decomposition surrogate model from a battery model.

.. dropdown:: More details

    In this example, an instance of a battery model is created. The DMD DataModel is used to generate a surrogate of this battery model for specific loading schemes. This surrogate can be used in place of the original model, approximating it's behavior. Frequently, surrogate models run faster than the original, at the cost of some accuracy. The performance of the two models are then compared. 

"""

import matplotlib.pyplot as plt
from prog_models.models import BatteryElectroChemEOD as Battery

def run_example(): 
    ### Example 1: Standard DMD Application 
    ## Step 1: Create a model object
    batt = Battery()

    ## Step 2: Define future loading functions for training data 
    # Here, we define two specific loading profiles. These could also be generated programmatically, for as many loading profiles as desired 
    def future_loading_1(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 500):
            i = 3
        elif (t < 1000):
            i = 2
        elif (t < 1500):
            i = 0.5
        else:
            i = 4.5
        return batt.InputContainer({'i': i})
    
    def future_loading_2(t, x=None):
        # Variable (piece-wise) future loading scheme 
        if (t < 300):
            i = 2
        elif (t < 800):
            i = 3.5
        elif (t < 1300):
            i = 4
        elif (t < 1600):
            i = 1.5
        else:
            i = 5
        return batt.InputContainer({'i': i})
    
    load_functions = [future_loading_1, future_loading_2]

    ## Step 3: generate surrogate model 
    # Simulation options for training data and surrogate model generation
    # Note: here dt is less than save_freq. This means the model will iterate forward multiple steps per saved point.
    # This is commonly done to ensure accuracy. 
    options_surrogate = {
        'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'trim_data_to': 0.7 # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
    }

    # Set noise in Prognostics Model, default for surrogate model is also this value
    batt.parameters['process_noise'] = 0

    # Generate surrogate model  
    surrogate = batt.generate_surrogate(load_functions,**options_surrogate)

    ## Step 4: Use surrogate model 
    # Simulation options for implementation of surrogate model
    options_sim = {
        'save_freq': 1 # Frequency at which results are saved, or equivalently time step in results
    }

    # Define loading profile 
    def future_loading(t, x=None):
        if (t < 600):
            i = 3
        elif (t < 1000):
            i = 2
        elif (t < 1500):
            i = 1.5
        else:
            i = 4
        return batt.InputContainer({'i': i})

    # Simulate to threshold using DMD approximation
    simulated_results = surrogate.simulate_to_threshold(future_loading,**options_sim)

    # Calculate Error
    MSE = batt.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs)
    print('Example 1 MSE:',MSE)
    # Not a very good approximation

    # Plot results
    simulated_results.inputs.plot(ylabel = 'Current (amps)',title='Example 1 Input')
    simulated_results.outputs.plot(ylabel = 'Predicted Outputs (temperature and voltage)',title='Example 1 Predicted Outputs')
    simulated_results.event_states.plot(ylabel = 'Predicted State of Charge', title='Example 1 Predicted SOC')

    # To visualize the accuracy of the approximation, run the high-fidelity model
    options_hf = {
        'dt': 0.1,
        'save_freq': 1,
    }
    high_fidelity_results = batt.simulate_to_threshold(future_loading,**options_hf)

    # Save voltage results to compare
    voltage_dmd = [simulated_results.outputs[iter1]['v'] for iter1 in range(len(simulated_results.times))]
    voltage_hf = [high_fidelity_results.outputs[iter2]['v'] for iter2 in range(len(high_fidelity_results.times))]

    plt.subplots()
    plt.plot(simulated_results.times,voltage_dmd,'-b',label='DMD approximation')
    plt.plot(high_fidelity_results.times, voltage_hf,'--r',label='High fidelity result')
    plt.legend()
    plt.title('Comparing DMD approximation to high-fidelity model results')

    ### Example 2: Add process_noise to the surrogate model 
        # Without re-generating the surrogate model, we can re-define the process_noise to be higher than the high-fidelity model (since the surrogate model is less accurate)
    surrogate.parameters['process_noise'] = 1e-04
    surrogate.parameters['process_noise_dist'] = 'normal'

    # Simulate to threshold using DMD approximation 
    simulated_results = surrogate.simulate_to_threshold(future_loading,**options_sim)

    # Plot results
    simulated_results.inputs.plot(ylabel = 'Current (amps)',title='Example 2 Input')
    simulated_results.outputs.plot(keys=['v'],ylabel = 'Predicted Voltage (volts)', title='Example 2 Predicted Outputs')
    simulated_results.event_states.plot(ylabel = 'Predicted State of Charge', title='Example 2 Predicted SOC')

    ### Example 3: Generate surrogate model with a subset of internal states, inputs, and/or outputs
        # Note: we use the same loading profiles as defined in Ex. 1

    ## Generate surrogate model 
    # Simulation options for training data and surrogate model generation
    options_surrogate = {
        'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
        'dt': 0.1, # For DMD, this value is the time step of the training data
        'trim_data': 1, # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
        'state_keys': ['Vsn','Vsp','tb'], # Define internal states to be included in surrogate model
        'output_keys': ['v'] # Define outputs to be included in surrogate model 
    }

    # Set noise in Prognostics Model, default for surrogate model is also this value
    batt.parameters['process_noise'] = 0

    # Generate surrogate model  
    surrogate = batt.generate_surrogate(load_functions,**options_surrogate)

    ## Use surrogate model 
    # The surrogate model can now be used anywhere the original model is used. It is interchangeable with the original model. 
    # The surrogate model results will be faster but less accurate than the original model. 

    # Simulation options for implementation of surrogate model
    options_sim = {
        'save_freq': 1 # Frequency at which results are saved, or equivalently time step in results
    }

    # Simulate to threshold using DMD approximation
    simulated_results = surrogate.simulate_to_threshold(future_loading,**options_sim)

    # Calculate Error
    MSE = batt.calc_error(simulated_results.times, simulated_results.inputs, simulated_results.outputs)
    print('Example 3 MSE:',MSE)

    # Plot results
    simulated_results.inputs.plot(ylabel = 'Current (amps)',title='Example 3 Input')
    simulated_results.outputs.plot(ylabel = 'Outputs (voltage)',title='Example 3 Predicted Output')
    simulated_results.event_states.plot(ylabel = 'State of Charge',title='Example 3 Predicted SOC')
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
