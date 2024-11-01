

from progpy.datasets import nasa_battery
(desc, data) = nasa_battery.load_data(1)

import pandas as pd
dataset = pd.concat(data[5:32], ignore_index=True)

from progpy.predictors import MonteCarlo
from progpy.models import BatteryElectroChemEOD

from progpy.loading import Piecewise

import numpy as np
from progpy.state_estimators import UnscentedKalmanFilter
from progpy.uncertain_data import MultivariateNormalDist


batt = BatteryElectroChemEOD()

initial_state = batt.initialize() # Initialize model

PROCESS_NOISE = 1e-4            # Percentage process noise
MEASUREMENT_NOISE = 1e-4        # Percentage measurement noise

batt.parameters['process_noise'] = {key: PROCESS_NOISE * value for key, value in initial_state.items()}
z0 = batt.output(initial_state)
batt.parameters['measurement_noise'] = {key: MEASUREMENT_NOISE * value for key, value in z0.items()}

future_loading = Piecewise(
        InputContainer=batt.InputContainer,
        times=[600, 900, 1800, 3000],
        values={'i': [2, 1, 4, 2, 3]})


mc = MonteCarlo(batt)

NUM_SAMPLES = 100
STEP_SIZE = 1
PREDICTION_HORIZON = 5000

mc_results = mc.predict(initial_state, future_loading_eqn = future_loading, n_samples=NUM_SAMPLES, dt=STEP_SIZE, horizon = PREDICTION_HORIZON)


debug = 1