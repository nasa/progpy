# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This example extends the "basic example" to perform a state estimation and prediction with uncertainty given a more complicated model. Models, state estimators, and predictors can be switched out. See documentation nasa.github.io/progpy for description of options
 
Method: An instance of the BatteryCircuit model in prog_models is created, and the prediction process is achieved in three steps:
    1) State estimation of the current state is performed using a chosen state_estimator, and samples are drawn from this estimate
    2) Prediction of future states (with uncertainty) and the times at which the event threshold will be reached
    3) Metrics tools are used to further investigate the results of prediction

Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time event is predicted to occur (with uncertainty)
    iii) Various prediction metrics
    iv) Figures illustrating results
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChemEOD as Battery

from prog_algs.state_estimators import ParticleFilter as StateEstimator
# VVV Uncomment this to use UKF State Estimator VVV
# from prog_algs.state_estimators import UnscentedKalmanFilter as StateEstimator

from prog_algs.predictors import MonteCarlo as Predictor
# VVV Uncomment this to use UnscentedTransform Predictor VVV
# from prog_algs.predictors import UnscentedTransformPredictor as Predictor

def run_example():
    # Step 1: Setup model & future loading
    # Measurement noise
    R_vars = {
        't': 2, 
        'v': 0.02
    }
    batt = Battery(process_noise = 0.25, measurement_noise = R_vars)
    # Creating the input containers outside of the function accelerates prediction
    loads = [
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

    initial_state = batt.initialize()

    # Step 2: Demonstrating state estimator
    print("\nPerforming State Estimation Step")

    # Step 2a: Setup
    filt = StateEstimator(batt, initial_state)
    
    # Step 2b: Print & Plot Prior State
    print("Prior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])
    fig = filt.x.plot_scatter(label='prior')

    # Step 2c: Perform state estimation step
    example_measurements = batt.OutputContainer({'t': 32.2, 'v': 3.915})
    t = 0.1
    u = future_loading(t)
    filt.estimate(t, u, example_measurements)

    # Step 2d: Print & Plot Resulting Posterior State
    print("\nPosterior State:", filt.x.mean)
    print('\tSOC: ', batt.event_state(filt.x.mean)['EOD'])
    filt.x.plot_scatter(fig=fig, label='posterior')  # Add posterior state to figure from prior state

    # Note: in a prognostic application the above state estimation step would be repeated each time
    #   there is new data. Here we're doing one step to demonstrate how the state estimator is used

    # Step 3: Demonstrating Predictor
    print("\n\nPerforming Prediction Step")

    # Step 3a: Setup Predictor
    mc = Predictor(batt)

    # Step 3b: Perform a prediction
    NUM_SAMPLES = 25
    STEP_SIZE = 0.1
    SAVE_FREQ = 100  # How often to save results
    mc_results = mc.predict(filt.x, future_loading, n_samples = NUM_SAMPLES, dt=STEP_SIZE, save_freq = SAVE_FREQ)
    print('ToE', mc_results.time_of_event.mean)

    # Step 3c: Analyze the results

    # Note: The results of a sample-based prediction can be accessed by sample, e.g.,
    from prog_algs.predictors import UnweightedSamplesPrediction
    if isinstance(mc_results, UnweightedSamplesPrediction):
        states_sample_1 = mc_results.states[1]
    # now states_sample_1[n] corresponds to times[n] for the first sample

    # You can also access a state distribution at a specific time using the .snapshot function
    states_time_1 = mc_results.states.snapshot(1)
    # now you have all the samples corresponding to times[1]

    # Print Results
    print('Results: ')
    for i, time in enumerate(mc_results.times):
        print('\nt = {}'.format(time))
        print('\tu = {}'.format(mc_results.inputs.snapshot(i).mean))
        print('\tx = {}'.format(mc_results.states.snapshot(i).mean))
        print('\tz = {}'.format(mc_results.outputs.snapshot(i).mean))
        print('\tevent state = {}'.format(mc_results.event_states.snapshot(i).mean))

    # You can also access the final state (of type UncertainData), like so:
    final_state = mc_results.time_of_event.final_state
    print('Final state @EOD: ', final_state['EOD'].mean)
    
    # You can also use the metrics package to generate some useful metrics on the result of a prediction
    print("\nEOD Prediction Metrics")

    from prog_algs.metrics import prob_success
    print('\tPortion between 3005.2 and 3005.6: ', mc_results.time_of_event.percentage_in_bounds([3005.2, 3005.6]))
    print('\tAssuming ground truth 3002.25: ', mc_results.time_of_event.metrics(ground_truth=3005.25))
    print('\tP(Success) if mission ends at 3002.25: ', prob_success(mc_results.time_of_event, 3005.25))

    # Plot state transition 
    # Here we will plot the states at t0, 25% to ToE, 50% to ToE, 75% to ToE, and ToE
    fig = mc_results.states.snapshot(0).plot_scatter(label = "t={} s".format(int(mc_results.times[0])))  # 0
    quarter_index = int(len(mc_results.times)/4)
    mc_results.states.snapshot(quarter_index).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[quarter_index])))  # 25%
    mc_results.states.snapshot(quarter_index*2).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[quarter_index*2])))  # 50%
    mc_results.states.snapshot(quarter_index*3).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[quarter_index*3])))  # 75%
    mc_results.states.snapshot(-1).plot_scatter(fig = fig, label = "t={} s".format(int(mc_results.times[-1])))  # 100%

    mc_results.time_of_event.plot_hist()
    
    # Step 4: Show all plots
    import matplotlib.pyplot as plt  # For plotting
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
