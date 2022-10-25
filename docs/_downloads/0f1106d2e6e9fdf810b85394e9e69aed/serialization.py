# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of serializing and de-serializing a battery model using JSON and pickling methods
"""

import matplotlib.pyplot as plt
import pickle
from prog_models.models import BatteryElectroChemEOD as Battery

def run_example():  
    ## Step 1: Create a model object
    batt = Battery()

    # Set process nosie to 0 to illustrate match between original and serialized versions
    batt.parameters['process_noise'] = 0 

    ### Step 2: serialize model for future use
    # Note: Model serialization has a lot of purposes, like saving a specific model to a file to be loaded later or sending a model to another machine over a network connection.
    
    # METHOD 1: Serialize with JSON 
    save_json = batt.to_json()

    # Model can be called directly with serialized result
    serial_1 = Battery.from_json(save_json)

    # Serialized result can also be saved to a text file and uploaded later using the following code:
    txtFile = open("model_save_json.txt", "w")
    txtFile.write(save_json)
    txtFile.close() 

    with open('model_save_json.txt') as infile: 
        load_json = infile.read()

    serial_2 = Battery.from_json(load_json)

    # METHOD 2: Serialize by pickling
    pickle.dump(batt, open('model_save_pkl.pkl','wb'))
    load_pkl = pickle.load(open('model_save_pkl.pkl','rb'))

    ## Step 3: Simulate to threshold and compare results
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

    # Simulate to threshold 
    results_orig = batt.simulate_to_threshold(future_loading,**options_sim)
    results_serial_1 = serial_1.simulate_to_threshold(future_loading, **options_sim)
    results_serial_2 = serial_2.simulate_to_threshold(future_loading, **options_sim)
    results_serial_3 = load_pkl.simulate_to_threshold(future_loading, **options_sim)

    # Plot results for comparison
    voltage_orig = [results_orig.outputs[iter]['v'] for iter in range(len(results_orig.times))]
    voltage_serial_1 = [results_serial_1.outputs[iter]['v'] for iter in range(len(results_serial_1.times))]
    voltage_serial_2 = [results_serial_2.outputs[iter]['v'] for iter in range(len(results_serial_2.times))]
    voltage_serial_3 = [results_serial_3.outputs[iter]['v'] for iter in range(len(results_serial_3.times))]

    plt.plot(results_orig.times,voltage_orig,'-b',label='Original surrogate') 
    plt.plot(results_serial_1.times,voltage_serial_1,'--r',label='First JSON serialized surrogate') 
    plt.plot(results_serial_2.times,voltage_serial_2,'-.g',label='Second JSON serialized surrogate') 
    plt.plot(results_serial_3.times, voltage_serial_3, '--y', label='Pickled serialized surrogate')
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (volts)')
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
