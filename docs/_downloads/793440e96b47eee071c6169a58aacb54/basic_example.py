# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This example performs a state estimation and prediction with uncertainty given a Prognostics Model.
 
Method: An instance of the ThrownObject model in prog_models is created, and the prediction process is achieved in three steps:
    1) State estimation of the current state is performed using a chosen state_estimator, and samples are drawn from this estimate
    2) Prediction of future states (with uncertainty) and the times at which the event threshold will be reached
    3) Metrics tools are used to further investigate the results of prediction
Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time event is predicted to occur (with uncertainty)
    iii) Various prediction metrics
    iv) Figures illustrating results
"""

from prog_models.models import ThrownObject
from prog_algs import *

def run_example():
    # Step 1: Setup model & future loading
    m = ThrownObject(process_noise = 1)
    def future_loading(t, x = None):
        # No load for a thrown object
        return m.InputContainer({})
    initial_state = m.initialize()

    # Step 2: Demonstrating state estimator
    # The state estimator is used to estimate the system state given sensor data. 
    print("\nPerforming State Estimation Step")

    # Step 2a: Setup
    filt = state_estimators.ParticleFilter(m, initial_state)
    # VVV Uncomment this to use UKF State Estimator VVV
    # filt = state_estimators.UnscentedKalmanFilter(m, initial_state)

    # Step 2b: Print & Plot Prior State
    print("Prior State:", filt.x.mean)
    print('\nevent state: ', m.event_state(filt.x.mean))
    fig = filt.x.plot_scatter(label='prior')

    # Step 2c: Perform state estimation step, given some measurement, above what's expected
    example_measurements = m.OutputContainer({'x': 7.5})
    t = 0.1
    u = future_loading(t)
    filt.estimate(t, u, example_measurements)  # Update state, given (example) sensor data

    # Step 2d: Print & Plot Resulting Posterior State
    # Note the posterior state is greater than the predicted state of 5.95
    # This is because of the high measurement 
    print("\nPosterior State:", filt.x.mean)
    # Event state for 'falling' is less, because velocity has decreased
    print('\nEvent State: ', m.event_state(filt.x.mean))
    filt.x.plot_scatter(fig=fig, label='posterior')  # Add posterior state to figure from prior state

    # Note: in a prognostic application the above state estimation step would be repeated each time
    #   there is new data. Here we're doing one step to demonstrate how the state estimator is used

    # Step 3: Demonstrating Prediction
    print("\n\nPerforming Prediction Step")

    # Step 3a: Setup Predictor
    mc = predictors.MonteCarlo(m)

    # Step 3b: Perform a prediction
    NUM_SAMPLES = 50
    STEP_SIZE = 0.01
    mc_results = mc.predict(filt.x, future_loading, n_samples = NUM_SAMPLES, dt=STEP_SIZE, save_freq=STEP_SIZE)
    print('Predicted time of event (ToE): ', mc_results.time_of_event.mean)
    # Here there are 2 events predicted, when the object starts falling, and when it impacts the ground.

    # Step 3c: Analyze the results

    # Note: The results of a sample-based prediction can be accessed by sample, e.g.,
    states_sample_1 = mc_results.states[1]
    # now states_sample_1[n] corresponds to times[n] for the first sample

    # You can also access a state distribution at a specific time using the .snapshot function
    states_time_1 = mc_results.states.snapshot(1)
    # now you have all the samples corresponding to times[1]

    # You can also access the final state (of type UncertainData), like so:
    # Note: to get a more accurate final state, you can decrease the step size.
    final_state = mc_results.time_of_event.final_state
    print('State when object starts falling: ', final_state['falling'].mean)
    
    # You can also use the metrics package to generate some useful metrics on the result of a prediction
    print("\nEOD Prediction Metrics")

    from prog_algs.metrics import prob_success
    print('\tPortion between 3.65 and 3.8: ', mc_results.time_of_event.percentage_in_bounds([3.65, 3.8], keys='falling'))
    print('\tAssuming ground truth 3.7: ', mc_results.time_of_event.metrics(ground_truth=3.7, keys='falling'))
    print('\tP(Success) if mission ends at 7.6: ', prob_success(mc_results.time_of_event, 7.6, keys='impact'))

    # Plot state transition 
    # Here we will plot the states at t0, 25% to ToE, 50% to ToE, 75% to ToE, and ToE
    # You should see the states move together (i.e., velocity is lowest and highest when closest to the ground (before impact, and at beginning, respectively))
    fig = mc_results.states.snapshot(0).plot_scatter(label = "t={} s".format(int(mc_results.times[0])))  # 0
    quarter_index = int(len(mc_results.times)/4)
    mc_results.states.snapshot(quarter_index).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[quarter_index])))  # 25%
    mc_results.states.snapshot(quarter_index*2).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[quarter_index*2])))  # 50%
    mc_results.states.snapshot(quarter_index*3).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[quarter_index*3])))  # 75%
    mc_results.states.snapshot(-1).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[-1])))  # 100%

    # Plot time of event for each event
    # If you dont see many bins here, this is because there is not much variety in the estimate. 
    # You can increase the number of bins, decrease step size, or increase the number of samples to see more of a distribution
    mc_results.time_of_event.plot_hist(keys='impact')
    mc_results.time_of_event.plot_hist(keys='falling')
    
    # Step 4: Show all plots
    import matplotlib.pyplot as plt  # For plotting
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
