# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example shows how to use the PaaS Client and Server for online prognostics. Prior to running the example start the server in a terminal window with the command:
    python -m prog_server

This example creates a session with the server to run prognostics for a Thrown Object, a simplified model of an object thrown into the air. Data is then sent to the server and a prediction is requested. The prediction is then displayed.
"""

import prog_client
from pprint import pprint
from time import sleep

def run_example():
    # Step 1: Open a session with the server for a thrown object. 
    # Use all default configuration options.
    # Except for the save frequency, which we'll set to 1 second.
    session = prog_client.Session('ThrownObject', pred_cfg =  {'save_freq': 1})
    print(session)  # Printing the Session Information

    # Step 2: Prepare data to send to server
    # The data is a dictionary of values. The keys are the names of the inputs and outputs in the model.
    # Format (time, value)
    # Note: in an actual application, the data would be received from a sensor or other source.
    # The structure below is used to emulate the sensor.
    example_data = [
        (0, {'x': 1.83}), 
        (0.1, {'x': 5.81}), 
        (0.2, {'x': 9.75}), 
        (0.3, {'x': 13.51}), 
        (0.4, {'x': 17.20}), 
        (0.5, {'x': 20.87}), 
        (0.6, {'x': 24.37}), 
        (0.7, {'x': 27.75}), 
        (0.8, {'x': 31.09}), 
        (0.9, {'x': 34.30}), 
        (1.0, {'x': 37.42}),
        (1.1, {'x': 40.43}),
        (1.2, {'x': 43.35}),
        (1.3, {'x': 46.17}),
        (1.4, {'x': 48.91}),
        (1.5, {'x': 51.53}),
        (1.6, {'x': 54.05}),
        (1.7, {'x': 56.50}),
        (1.8, {'x': 58.82}),
        (1.9, {'x': 61.05}),
        (2.0, {'x': 63.20}),
        (2.1, {'x': 65.23}),
        (2.2, {'x': 67.17}),
        (2.3, {'x': 69.02}),
        (2.4, {'x': 70.75}),
        (2.5, {'x': 72.40})
    ] 

    # Step 3: Send data to server, checking periodically for a prediction result.
    LAST_PREDICTION_TIME = None
    for i in range(len(example_data)):
        # Send data to server
        print(f'{example_data[i][0]}s: Sending data to server... ', end='')
        session.send_data(time = example_data[i][0], **example_data[i][1])

        # Check for a prediction result
        status = session.get_prediction_status()
        if LAST_PREDICTION_TIME != status["last prediction"]: 
            # New prediction result
            LAST_PREDICTION_TIME = status["last prediction"]
            print('Prediction Completed')
            
            # Get prediction
            # Prediction is returned as a type uncertain_data, so you can manipulate it like that datatype.
            # See https://nasa.github.io/prog_algs/uncertain_data.html
            t, prediction = session.get_predicted_toe()
            print(f'Predicted ToE (using state from {t}s): ')
            pprint(prediction.mean)

            # Get Predicted future states
            # You can also get the predicted future states of the model.
            # States are saved according to the prediction configuration parameter 'save_freq' or 'save_pts'
            # In this example we have it setup to save every 1 second.
            # Return type is UnweightedSamplesPrediction (since we're using the monte carlo predictor)
            # See https://nasa.github.io/prog_algs
            t, event_states = session.get_predicted_event_state()
            print(f'Predicted Event States (using state from {t}s): ')
            es_means = [(event_states.times[i], event_states.snapshot(i).mean) for i in range(len(event_states.times))]
            for time, es_mean in es_means:
                print(f"\t{time}s: {es_mean}")

            # Note: you can also get the predicted future states of the model (see get_predicted_states()) or performance parameters (see get_predicted_performance_metrics())

        else:
            print('No prediction yet')
            # No updated prediction, send more data and check again later.
        sleep(0.1)

    # Notice that the prediction wasn't updated every time step. It takes a bit of time to perform a prediction.

    # Note: You can also get the model from prog_server to work with directly.
    model = session.get_model()

# This allows the module to be executed directly
if __name__ == '__main__':
    run_example()
