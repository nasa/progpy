# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories

"""
Example building a custom model with LSTMStateTransitionModel.

For most cases, you will be able to use the standard LSTMStateTransitionModel.from_data class with configuration (see the LSTMStateTransitionModel class for more details). However, sometimes you might want to add custom layers, or other complex components. In that case, you will build a custom model and pass it into LSTMStateTransitionModel.

In this example, we generate fake data using the BatteryElectroChemEOD model. This is a case where we're generating a surrogate model from the physics-based model. For cases where you're generating a model from data (e.g., collected from a testbed or a real-world environment), you'll replace that generated data with your own. 

We build and fit a custom model using keras.layers. Finally, we compare performance to the standard format and the original model.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from prog_models.data_models import LSTMStateTransitionModel
from prog_models.models import BatteryElectroChemEOD

def run_example():
    print('Generating data...')
    batt = BatteryElectroChemEOD()
    future_loading_eqns = [lambda t, x=None: batt.InputContainer({'i': 1+1.5*load}) for load in range(6)]
    # Generate data with different loading and step sizes
    # Adding the step size as an element of the output
    training_data = []
    input_data = []
    output_data = []
    for i in range(9):
        dt = i/3+0.25
        for loading_eqn in future_loading_eqns:
            d = batt.simulate_to_threshold(loading_eqn, save_freq=dt, dt=dt) 
            u = np.array([np.hstack((u_i.matrix[:][0].T, [dt])) for u_i in d.inputs], dtype=float)
            z = d.outputs
            training_data.append((u, z))
            input_data.append(u)
            output_data.append(z)

    # Step 2: Build standard model
    print("Building standard model...")
    m_batt = LSTMStateTransitionModel.from_data(
        inputs = input_data,
        outputs = output_data,  
        window=12, 
        epochs=30, 
        units=64,  # Additional units given the increased complexity of the system
        input_keys = ['i', 'dt'],
        output_keys = ['t', 'v'])   

    # Step 3: Build custom model
    print('Building custom model...')
    (u_all, z_all) = LSTMStateTransitionModel.pre_process_data(training_data, window=12)
    
    # Normalize
    n_inputs = len(training_data[0][0][0])
    u_mean = np.mean(u_all[:,0,:n_inputs], axis=0)
    u_std = np.std(u_all[:,0,:n_inputs], axis=0)
    # If there's no variation- dont normalize 
    u_std[u_std == 0] = 1
    z_mean = np.mean(z_all, axis=0)
    z_std = np.std(z_all, axis=0)
    # If there's no variation- dont normalize 
    z_std[z_std == 0] = 1

    # Add output (since z_t-1 is last input)
    u_mean = np.hstack((u_mean, z_mean))
    u_std = np.hstack((u_std, z_std))

    u_all = (u_all - u_mean)/u_std
    z_all = (z_all - z_mean)/z_std

    # u_mean and u_std act on the column vector form (from inputcontainer)
    # so we need to transpose them to a column vector
    normalization = (u_mean[np.newaxis].T, u_std[np.newaxis].T, z_mean, z_std)

    callbacks = [
        keras.callbacks.ModelCheckpoint("jena_sense.keras", save_best_only=True)
    ]
    inputs = keras.Input(shape=u_all.shape[1:])
    x = layers.Bidirectional(layers.LSTM(128))(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(z_all.shape[1] if z_all.ndim == 2 else 1)(x)
    model = keras.Model(inputs, x)
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    model.fit(u_all, z_all, epochs=30, callbacks = callbacks, validation_split = 0.1)

    # Step 4: Build LSTMStateTransitionModel
    m_custom = LSTMStateTransitionModel(model, 
        normalization=normalization, 
        input_keys = ['i', 'dt'],
        output_keys = ['t', 'v']
    )

    # Step 5: Simulate
    print('Simulating...')
    t_counter = 0
    x_counter = batt.initialize()
    def future_loading(t, x=None):
        return batt.InputContainer({'i': 3})

    def future_loading2(t, x = None):
        nonlocal t_counter, x_counter
        z = batt.output(x_counter)
        z = m_batt.InputContainer({'i': 3, 't_t-1': z['t'], 'v_t-1': z['v'], 'dt': t - t_counter})
        x_counter = batt.next_state(x_counter, future_loading(t), t - t_counter)
        t_counter = t
        return z
    data = batt.simulate_to_threshold(future_loading, dt=1, save_freq=1)
    results = m_batt.simulate_to(data.times[-1], future_loading2, dt=1, save_freq=1)
    results_custom = m_custom.simulate_to(data.times[-1], future_loading2, dt=1, save_freq=1)

    # Step 6: Compare performance
    print('Comparing performance...')
    data.outputs.plot(title='original model', compact=False)
    results.outputs.plot(title='generated model', compact=False)
    results_custom.outputs.plot(title='custom model', compact=False)
    plt.show()

if __name__ == '__main__':
    run_example()
