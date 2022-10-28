# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This example performs a state estimation and prediction with uncertainty given a Prognostics Model with a specific prediction horizon. This prediction horizon marks the end of the "time of interest" for the prediction. Often this represents the end of a mission or sufficiently in the future where the user is unconcerned with the events
 
Method: An instance of the Thrown Object model in prog_models is created, and the prediction process is achieved in three steps:
    1) State estimation of the current state is performed using a chosen state_estimator, and samples are drawn from this estimate
    2) Prediction of future states (with uncertainty) and the times at which the event thresholds will be reached, within the prediction horizon. All events outside the horizon come back as None and are ignored in metrics

Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time event is predicted to occur (with uncertainty)
"""

from prog_models.models.thrown_object import ThrownObject
from prog_algs import *
from pprint import pprint

def run_example():
    # Step 1: Setup model & future loading
    def future_loading(t, x = None):
        return {}
    m = ThrownObject(process_noise = 0.25, measurement_noise = 0.2)
    initial_state = m.initialize()

    # Step 2: Demonstrating state estimator
    print("\nPerforming State Estimation Step...")

    # Step 2a: Setup
    NUM_SAMPLES = 1000
    filt = state_estimators.ParticleFilter(m, initial_state, num_particles = NUM_SAMPLES)
    # VVV Uncomment this to use UKF State Estimator VVV
    # filt = state_estimators.UnscentedKalmanFilter(batt, initial_state)

    # Step 2b: One step of state estimator
    u = m.InputContainer({})  # No input for ThrownObject
    filt.estimate(0.1, u, m.output(initial_state))

    # Note: in a prognostic application the above state estimation 
    # step would be repeated each time there is new data. 
    # Here we're doing one step to demonstrate how the state estimator is used

    # Step 3: Demonstrating Predictor
    print("\nPerforming Prediction Step...")

    # Step 3a: Setup Predictor
    mc = predictors.MonteCarlo(m)

    # Step 3b: Perform a prediction
    # THIS IS WHERE WE DIVERGE FROM THE THROWN_OBJECT_EXAMPLE
    # Here we set a prediction horizon
    # We're saying we are not interested in any events that occur after this time
    PREDICTION_HORIZON = 7.75
    samples = filt.x  # Since we're using a particle filter, which is also sample-based, we can directly use the samples, without changes
    STEP_SIZE = 0.01
    mc_results = mc.predict(samples, future_loading, dt=STEP_SIZE, horizon = PREDICTION_HORIZON)
    print("\nPredicted Time of Event:")
    metrics = mc_results.time_of_event.metrics()
    pprint(metrics)  # Note this takes some time
    mc_results.time_of_event.plot_hist(keys = 'impact')
    mc_results.time_of_event.plot_hist(keys = 'falling')

    print("\nSamples where impact occurs before horizon: {:.2f}%".format(metrics['impact']['number of samples']/NUM_SAMPLES*100))
    
    # Step 4: Show all plots
    import matplotlib.pyplot as plt  # For plotting
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
