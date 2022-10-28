# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs a state estimation with uncertainty given a Prognostics Model for a system in which not all output values are measured. 
 
Method: An instance of the BatteryCircuit model in prog_models is created. We assume that we are only measuring one of the output values, and we define a subclass to remove the other output value.  
        Estimation of the current state is performed at various time steps, using the defined state_estimator.

Results: 
    i) Estimate of the current state given various times
    ii) Display of results, such as prior and posterior state estimate values and SOC
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

from prog_algs import *

def run_example():
    # Step 1: Subclass model with measurement equation
    # In this case we're only measuring 'v' (i.e., removing temperature)
    # To do this we're creating a new class that's subclassed from the complete model.
    # To change the outputs we just have to override outputs (the list of keys)
    class MyBattery(Battery):
        outputs = ['v']

    # Step 2: Setup model & future loading
    batt = MyBattery()
    loads = [  # Define loads here to accelerate prediction
        batt.InputContainer({'i': 2}),
        batt.InputContainer({'i': 1}),
        batt.InputContainer({'i': 4}),
        batt.InputContainer({'i': 2}),
        batt.InputContainer({'i': 3})
    ]
    def future_loading(t, x = None):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            return loads[0]
        elif (t < 900):
            return loads[1]
        elif (t < 1800):
            return loads[2]
        elif (t < 3000):
            return loads[3]
        return loads[-1]

    x0 = batt.parameters['x0']

    # Step 3: Use the updated model
    filt = state_estimators.ParticleFilter(batt, x0)

    # Step 4: Run step and print results
    print('Running state estimation step with only one of 2 outputs measured')

    # Print Prior
    print("\nPrior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    # Estimate Step
    # Note, only voltage was needed in the measurement step, since that is the only output we're measuring
    t = 0.1
    load = future_loading(t)
    filt.estimate(t, load, {'v': 3.915})

    # Print Posterior
    print("\nPosterior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    # Another Estimate Step
    t = 0.2
    load = future_loading(t)
    filt.estimate(t, load, {'v': 3.91})

    # Print Posterior Again
    print("\nPosterior State (t={}):".format(t), filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

    # Note that the particle filter was still able to perform state estimation.
    # The updated outputs can be used for any case where the measurement doesn't match the model outputs
    # For example, when units are different, or when the measurement is some combination of the outputs
    # These are a little more complicated, since they require an instance of the parent class. For example:

    parent = Battery()


    class MyBattery(Battery):
        outputs = ['tv']  # output is temperature * voltage (for some reason)

        def output(self, x):
            parent.parameters = self.parameters  # only needed if you expect to change parameters
            z = parent.output(x)
            return self.OutputContainer({'tv': z['v'] * z['t']})

    batt = MyBattery()
    filt = state_estimators.ParticleFilter(batt, x0)

    print('-----------------\n\nExample 2')
    print("\nPrior State:", filt.x.mean)
    print("\toutput: ", batt.output(filt.x.mean))
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])
    t = 0.1
    load = future_loading(t)
    filt.estimate(t, load, {'tv': 80})
    print("\nPosterior State:", filt.x.mean)
    print("\toutput: ", batt.output(filt.x.mean))
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
