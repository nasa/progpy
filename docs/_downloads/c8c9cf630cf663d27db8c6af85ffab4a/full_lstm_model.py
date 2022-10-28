# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories

"""
Example building a full model with events and thresholds using LSTMStateTransitionModel. 

.. dropdown:: More details

    In this example, we generate fake data using the ThrownObject model. This is a case where we're generating a surrogate model from the physics-based model. For cases where you're generating a model from data (e.g., collected from a testbed or a real-world environment), you'll replace that generated data with your own. 

    We then create a subclass of the LSTMStateTransitionModel, defining the event_state and threshold equations as a function of output. We use the generated model and compare to the original model.
"""

import matplotlib.pyplot as plt
import numpy as np
from prog_models.data_models import LSTMStateTransitionModel
from prog_models.models import ThrownObject

def run_example():
    # -----------------------------------------------------
    # Method 1 - manual definition
    # In this example we complete the models by manually defining event_state 
    # and thresholds_met as function of output.
    # -----------------------------------------------------
    TIMESTEP = 0.01
    m = ThrownObject()
    def future_loading(t, x=None):
        return m.InputContainer({})  # No input for thrown object 

    # Step 1: Generate additional data
    # We will use data generated above, but we also want data at additional timesteps 
    print('Generating data...')
    data = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP, dt=TIMESTEP)
    data_half = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP/2, dt=TIMESTEP/2)
    data_quarter = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP/4, dt=TIMESTEP/4)
    data_twice = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP*2, dt=TIMESTEP*2)
    data_four = m.simulate_to_threshold(future_loading, threshold_keys='impact', save_freq=TIMESTEP*4, dt=TIMESTEP*4)

    # Step 2: Data Prep
    # We need to add the timestep as a input
    u = np.array([[TIMESTEP] for _ in data.inputs])
    u_half = np.array([[TIMESTEP/2] for _ in data_half.inputs])
    u_quarter = np.array([[TIMESTEP/4] for _ in data_quarter.inputs])
    u_twice = np.array([[TIMESTEP*2] for _ in data_twice.inputs])
    u_four = np.array([[TIMESTEP*4] for _ in data_four.inputs])

    # In this case we are saying that velocity is directly measurable, 
    # unlike the original model. This is necessary to calculate the events.
    # Since the outputs will then match the states, we pass in the states below

    u_data = [u, u_half, u_quarter, u_twice, u_four]
    z_data = [data.states, data_half.states, data_quarter.states, data_twice.states, data_four.states]

    # Step 3: Create model
    print('Creating model...')

    # Create a subclass of LSTMStateTransitionModel, 
    # overridding event-related methods and members
    class LSTMThrownObject(LSTMStateTransitionModel):
        events = [
            'falling', # Event- object is falling
            'impact' # Event- object has impacted ground
        ]

        def initialize(self, u=None, z=None):
            # Add logic required for thrown object
            self.max_x = 0.0
            return super().initialize(u, z)

        def event_state(self, x):
            # Using class name instead of self allows the class to be subclassed
            z = LSTMThrownObject.output(self, x)
            # Logic from ThrownObject.event_state, using output instead of state
            self.max_x = max(self.max_x, z['x'])  # Maximum altitude
            return {
                'falling': max(z['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
                'impact': max(z['x']/self.max_x,0)  # 1 until falling begins, then it's fraction of height
            }

        def threshold_met(self, x):
            z = LSTMThrownObject.output(self, x)
            # Logic from ThrownObject.threshold_met, using output instead of state
            return {
                'falling': z['v'] < 0,
                'impact': z['x'] <= 0
            }
    
    # Step 4: Generate Model
    print('Building model...')
    m2 = LSTMThrownObject.from_data(
        inputs=u_data,  
        outputs=z_data,
        window=4, 
        epochs=30, 
        input_keys = ['dt'],
        output_keys = m.states)
    m2.plot_history()

    # Step 5: Simulate with model
    t_counter = 0
    x_counter = m.initialize()
    def future_loading3(t, x = None):
        nonlocal t_counter, x_counter
        z = m2.InputContainer({'x_t-1': x_counter['x'], 'v_t-1': x_counter['v'], 'dt': t - t_counter})
        x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z

    # Use new dt, not used in training
    # Using a dt not used in training will demonstrate the model's 
    # ability to handle different timesteps not part of training set
    data = m.simulate_to_threshold(future_loading, threshold_keys='impact', dt=TIMESTEP*3, save_freq=TIMESTEP*3)
    results3 = m2.simulate_to_threshold(future_loading3, threshold_keys='impact', dt=TIMESTEP*3, save_freq=TIMESTEP*3)

    # Step 6: Compare Results
    print('Comparing results...')
    print('Predicted impact time:')
    print('\tOriginal: ', data.times[-1])
    print('\tLSTM: ', results3.times[-1])
    data.outputs.plot(title='original model')
    results3.outputs.plot(title='generated model')
    plt.show()

if __name__ == '__main__':
    run_example()
