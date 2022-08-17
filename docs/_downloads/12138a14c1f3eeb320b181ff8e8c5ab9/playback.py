# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example performs state estimation and prediction using playback data. 
 
Method: An instance of the BatteryCircuit model in prog_models is created, the state estimation is set up by defining a state_estimator, and the prediction method is set up by defining a predictor.
        Prediction is then performed using playback data. For each data point:
        1) The necessary data is extracted (time, current load, output values) and corresponding values defined (t, i, and z)
        2) The current state estimate is performed and samples are drawn from this distribution
        3) Prediction performed to get future states (with uncertainty) and the times at which the event threshold will be reached
    
Results: 
    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time event is predicted to occur (with uncertainty)
    iii) Various prediction metrics
    iv) Figures illustrating results
"""

from prog_models.models import BatteryCircuit as Battery
# VVV Uncomment this to use Electro Chemistry Model VVV
# from prog_models.models import BatteryElectroChem as Battery

from prog_algs.state_estimators import UnscentedKalmanFilter as StateEstimator
# VVV Uncomment this to use UnscentedKalmanFilter instead VVV
# from prog_algs.state_estimators import ParticleFilter as StateEstimator

from prog_algs.predictors import ToEPredictionProfile

from prog_algs.predictors import UnscentedTransformPredictor as Predictor
# VVV Uncomment this to use MonteCarloPredictor instead
# from prog_algs.predictors import MonteCarlo as Predictor

from prog_algs.uncertain_data.multivariate_normal_dist import MultivariateNormalDist

import csv
import matplotlib.pyplot as plt
import numpy as np

# Constants
NUM_SAMPLES = 20
NUM_PARTICLES = 1000 # For state estimator (if using ParticleFilter)
TIME_STEP = 1
PREDICTION_UPDATE_FREQ = 50 # Number of steps between prediction update
PLOT = True
PROCESS_NOISE = 1e-4 # Percentage process noise
MEASUREMENT_NOISE = 1e-4 # Percentage measurement noise
X0_COV = 1 # Covariance percentage with initial state
GROUND_TRUTH = {'EOD':2780}
ALPHA = 0.05
BETA = 0.90
LAMBDA_VALUE = 1500

def run_example():
    # Setup Model
    batt = Battery()

    # Initial state
    x0 = batt.initialize()
    batt.parameters['process_noise'] = {key: PROCESS_NOISE * value for key, value in x0.items()}
    z0 = batt.output(x0)
    batt.parameters['measurement_noise'] = {key: MEASUREMENT_NOISE * value for key, value in z0.items()}
    x0 = MultivariateNormalDist(x0.keys(), list(x0.values()), np.diag([max(1e-9, X0_COV * abs(x)) for x in x0.values()]))

    # Setup State Estimation
    filt = StateEstimator(batt, x0, num_particles = NUM_PARTICLES)

    # Setup Prediction
    def future_loading(t, x=None):
        return {'i': 2.35}
    Q = np.diag([batt.parameters['process_noise'][key] for key in batt.states])
    R = np.diag([batt.parameters['measurement_noise'][key] for key in batt.outputs])
    mc = Predictor(batt, Q = Q, R = R)

    # Run Playback
    step = 0
    profile = ToEPredictionProfile()
    
    with open('examples/data_const_load.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            step += 1
            print("{} s: {} W, {} C, {} V".format(*row))
            t = float(row[0])
            i = {'i': float(row[1])/float(row[3])}
            z = {'t': float(row[2]), 'v': float(row[3])}

            # State Estimation Step
            filt.estimate(t, i, z) 
            eod = batt.event_state(filt.x.mean)['EOD']
            print("  - Event State: ", eod)

            # Prediction Step (every PREDICTION_UPDATE_FREQ steps)
            if (step%PREDICTION_UPDATE_FREQ == 0):
                mc_results = mc.predict(filt.x, future_loading, t0 = t, n_samples=NUM_SAMPLES, dt=TIME_STEP)
                metrics = mc_results.time_of_event.metrics()
                print('  - ToE: {} (sigma: {})'.format(metrics['EOD']['mean'], metrics['EOD']['std']))
                profile.add_prediction(t, mc_results.time_of_event)

        # Calculating Prognostic Horizon once the loop completes
        from prog_algs.uncertain_data.uncertain_data import UncertainData
        from prog_algs.metrics import samples as metrics

        def criteria_eqn(tte : UncertainData, ground_truth_tte : dict) -> dict:
            """
            Sample criteria equation for playback. 
            # UPDATE THIS CRITERIA EQN AND WHAT IS CALCULATED

            Args:
                tte : UncertainData
                    Time to event in UncertainData format.
                ground_truth_tte : dict
                    Dictionary of ground truth of time to event.
            """
            
            # Set an alpha value
            bounds = {}
            for key, value in ground_truth_tte.items():
                # Set bounds for precentage_in_bounds by adding/subtracting to the ground_truth
                alpha_calc = value * ALPHA
                bounds[key] = [value - alpha_calc, value + alpha_calc] # Construct bounds for all events
            percentage_in_bounds = tte.percentage_in_bounds(bounds)
            
            # Verify if percentage in bounds for this ground truth meets beta distribution percentage limit
            return {key: percentage_in_bounds[key] > BETA for key in percentage_in_bounds.keys()}

        # Generate plots for playback example
        playback_plots = profile.plot(GROUND_TRUTH, ALPHA, True)

        # Calculate prognostic horizon with ground truth, and print
        ph = profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH)
        print(f"Prognostic Horizon for 'EOD': {ph['EOD']}")

        # Calculate alpha lambda with ground truth, lambda, alpha, and beta, and print
        al = profile.alpha_lambda(GROUND_TRUTH, LAMBDA_VALUE, ALPHA, BETA)
        print(f"Alpha Lambda for 'EOD': {al['EOD']}")

        # Calculate cumulative relative accuracy with ground truth, and print
        cra = profile.cumulative_relative_accuracy(GROUND_TRUTH)
        print(f"Cumulative Relative Accuracy for 'EOD': {cra['EOD']}")

    input('Press any key to exit')

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
