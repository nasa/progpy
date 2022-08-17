# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This example performs a state estimation and prediction with uncertainty given a Prognostics Model. Unlike basic_example, this example uses a model with multiple events (ThrownObject). Prediction only ends when all events are met
 
Method: An instance of the Thrown Object model in prog_models is created, and the prediction process is achieved in three steps:
    1) State estimation of the current state is performed using a chosen state_estimator, and samples are drawn from this estimate
    2) Prediction of future states (with uncertainty) and the times at which the event thresholds will be reached
Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time event is predicted to occur (with uncertainty)
"""

from prog_models.models.thrown_object import ThrownObject
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

from prog_algs import *

from pprint import pprint

def run_example():
    # Step 1: Setup model & future loading
    m = ThrownObject(process_noise = 0.25, measurement_noise = 0.2)
    initial_state = m.initialize()
    def future_loading(t, x = None):
        return m.InputContainer({})

    # Step 2: Demonstrating state estimator
    print("\nPerforming State Estimation Step...")

    # Step 2a: Setup
    NUM_SAMPLES = 1000
    filt = state_estimators.ParticleFilter(m, initial_state, num_particles = NUM_SAMPLES)
    # VVV Uncomment this to use UKF State Estimator VVV
    # filt = state_estimators.UnscentedKalmanFilter(batt, initial_state)

    # Step 2b: Print & Plot Prior State
    u = m.InputContainer({})  # No input for ThrownObject
    z = m.output(initial_state)  # Measured output (here as an example we use the model output instead)
    # Note: In practice, replace this with actual measured data
    filt.estimate(0.1, u, z)

    # Note: in a prognostic application the above state estimation step would be repeated each time
    #   there is new data. Here we're doing one step to demonstrate how the state estimator is used

    # Step 3: Demonstrating Predictor
    print("\nPerforming Prediction Step...")

    # Step 3a: Setup Predictor
    mc = predictors.MonteCarlo(m)

    # Step 3b: Perform a prediction
    samples = filt.x  # Since we're using a particle filter, which is also sample-based, we can directly use the samples, without changes
    STEP_SIZE = 0.01
    mc_results = mc.predict(samples, future_loading, dt=STEP_SIZE, horizon=8)
    print("\nPredicted Time of Event:")
    pprint(mc_results.time_of_event.metrics())  # Note this takes some time
    mc_results.time_of_event.plot_hist(keys = 'impact')
    mc_results.time_of_event.plot_hist(keys = 'falling')
    
    # Step 4: Show all plots
    import matplotlib.pyplot as plt  # For plotting
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
