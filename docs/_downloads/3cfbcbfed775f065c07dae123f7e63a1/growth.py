# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
 Example demonstrating the Paris Law Crack Growth Equation
"""

from prog_models.models.paris_law import ParisLawCrackGrowth 
import matplotlib.pyplot as plt
import csv
import os

def run_example(): 
    # Step 1: Create a model object
    m = ParisLawCrackGrowth(process_noise = 0)
    
    # Step 2: Define future loading function 
    def future_loading(t, x=None):
        #variable (piece-wise) future loading scheme 
        #inputs are ['k_min', 'k_max']
        if (t < 500):
            k_min = 12
            k_max = 24
        elif (t < 750):
            k_min = 8
            k_max = 32
        else:
            k_min = 0
            k_max = 28
        return m.InputContainer({'k_min': k_min, 'k_max': k_max})

    # Step 3: Estimate parameters
    # We do not know the model parameters for this system, 
    # so we need to estimate it using data collected from the system
    # First we have to import some data from the real system
    # This is what we use to estimate parameters
    times = []
    inputs = []
    outputs = []

    #Finds file path
    csv_dir = os.path.join(os.path.dirname(__file__), 'growth.csv')

    #Reads csv file
    try:
        with open(csv_dir, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='|' , quoting=csv.QUOTE_NONNUMERIC)
            for row in data:
                times.append(row[0])
                inputs.append({'k_min': row[1], 'k_max': row[2]})
                outputs.append({'c_l': row[3]})
    except FileNotFoundError:
        print("No data file found")

    # Estimates the model parameters
    keys = ['c', 'm']

    print('Model configuration before')
    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=10))

    m.estimate_params([(times, inputs, outputs)], keys, dt=10)

    print('\nOptimized configuration')
    for key in keys:
        print("-", key, m.parameters[key])
    print(' Error: ', m.calc_error(times, inputs, outputs, dt=10))

    # Step 4: Simulate to threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    options = {
        'save_freq': 10, # Frequency at which results are saved
        'dt': 10, # Timestep
        'print': True,
        'horizon': 1e5, # Horizon
    }

    (times, inputs, _, outputs, event_states) = m.simulate_to_threshold(future_loading, **options)

    # Step 5: Plot Results
    # crack length
    # plot event state

    inputs.plot(ylabel='Stress Intensity')
    event_states.plot(ylabel= 'CGF')
    outputs.plot(ylabel= {'c_l': "Crack Length"}, compact= False)
    plt.show()

if __name__ == '__main__':
    run_example()
