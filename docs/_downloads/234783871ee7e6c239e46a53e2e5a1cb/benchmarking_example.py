# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs benchmarking for a state estimation and prediction with uncertainty given a Prognostics Model. The process and benchmarking analysis are run for various sample sizes. 
 
Method: An instance of the BatteryCircuit model in prog_models is created, state estimation is set up with a chosen state_estimator, and prediction is set up with a chosen predictor.
        Prediction of future states (with uncertainty) is then performed for various sample sizes. 
        Metrics are calculated and displayed for each run. 

Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction for each distinct sample size
    ii) Time event is predicted to occur (with uncertainty)
    iii) Various prediction metrics, including alpha-lambda metric 
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

from prog_algs import *

def run_example():
    # Step 1: Setup Model and Future Loading
    batt = Battery()
    def future_loading(t, x={}):
        # Variable (piece-wise) future loading scheme 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        elif (t < 3000):
            i = 2
        else:
            i = 3
        return batt.InputContainer({'i': i})

    # Step 2: Setup Predictor 
    pred = predictors.MonteCarlo(batt, dt= 0.05)

    # Step 3: Estimate State
    x0 = batt.initialize()
    state_estimator = state_estimators.ParticleFilter(batt, x0)
    # Send in some data to estimate state
    z1 = batt.OutputContainer({'t': 32.2, 'v': 3.915})
    z2 = batt.OutputContainer({'t': 32.3, 'v': 3.91})
    state_estimator.estimate(0.1, future_loading(0.1), z1)
    state_estimator.estimate(0.2, future_loading(0.2), z2)

    # Step 4: Benchmark Predictions
    # Here we're comparing the results given different numbers of samples
    print('Benchmarking...')
    import time  # For timing prediction
    from prog_algs.metrics import samples as metrics 

    # Perform benchmarking for each number of samples
    sample_counts = [1, 2, 5, 10]
    for sample_count in sample_counts:
        print('\nRun 1 ({} samples)'.format(sample_count))
        start = time.perf_counter()
        pred_results = pred.predict(state_estimator.x, future_loading, n_samples = sample_count)
        toe = pred_results.time_of_event.key("EOD")  # Looking at EOD event
        end = time.perf_counter()
        print('\tMSE:     {:4.2f}s'.format(metrics.mean_square_error(toe, 3005.4)))
        print('\tRMSE:     {:4.2f}s'.format(metrics.root_mean_square_error(toe, 3005.4)))
        print('\tRuntime: {:4.2f}s'.format(end - start))

    # This same approach can be applied for benchmarking and comparing other changes 
    # For example: different sampling methods, prediction algorithms, step sizes, models

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
