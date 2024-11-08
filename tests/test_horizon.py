# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from progpy import predictors
from progpy.models import ThrownObject

class TestHorizon(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__
    
    def test_horizon_ex(self):
        # Setup model 
        m = ThrownObject(process_noise=0.25, measurement_noise=0.2) 
        # Change parameters (to make simulation faster) 
        m.parameters['thrower_height'] = 1.0
        m.parameters['throwing_speed'] = 10.0
        initial_state = m.initialize()

        # Define future loading (necessary for prediction call) 
        def future_loading(t, x=None):
            return {}

        # Setup Predictor (smaller sample size for efficiency)
        mc = predictors.MonteCarlo(m)
        mc.parameters['n_samples'] = 50

        # Perform a prediction
        # With this horizon, all samples will reach 'falling', but only some will reach 'impact'
        PREDICTION_HORIZON = 2.127 
        STEP_SIZE = 0.001 
        mc_results = mc.predict(initial_state, future_loading, dt=STEP_SIZE, horizon=PREDICTION_HORIZON)

        # 'falling' happens before the horizon is met
        falling_res = [mc_results.time_of_event[iter]['falling'] for iter in range(mc.parameters['n_samples']) if mc_results.time_of_event[iter]['falling'] is not None]
        self.assertEqual(len(falling_res), mc.parameters['n_samples'])

        # 'impact' happens around the horizon, so some samples have reached this event and others haven't
        impact_res = [mc_results.time_of_event[iter]['impact'] for iter in range(mc.parameters['n_samples']) if mc_results.time_of_event[iter]['impact'] is not None]
        self.assertLess(len(impact_res), mc.parameters['n_samples'])

        # Try again with very low prediction_horizon, where no events are reached
        # Note: here we count how many None values there are for each event (in the above and below examples, we count values that are NOT None)
        mc_results_no_event = mc.predict(initial_state, future_loading, dt=STEP_SIZE, horizon=0.3)
        falling_res_no_event = [mc_results_no_event.time_of_event[iter]['falling'] for iter in range(mc.parameters['n_samples']) if mc_results_no_event.time_of_event[iter]['falling'] is None]
        impact_res_no_event = [mc_results_no_event.time_of_event[iter]['impact'] for iter in range(mc.parameters['n_samples']) if mc_results_no_event.time_of_event[iter]['impact'] is None]
        self.assertEqual(len(falling_res_no_event), mc.parameters['n_samples'])
        self.assertEqual(len(impact_res_no_event), mc.parameters['n_samples'])

        # Finally, try without horizon, all events should be reached for all samples
        mc_results_all_event = mc.predict(initial_state, future_loading, dt=STEP_SIZE)
        falling_res_all_event = [mc_results_all_event.time_of_event[iter]['falling'] for iter in range(mc.parameters['n_samples']) if mc_results_all_event.time_of_event[iter]['falling'] is not None]
        impact_res_all_event = [mc_results_all_event.time_of_event[iter]['impact'] for iter in range(mc.parameters['n_samples']) if mc_results_all_event.time_of_event[iter]['impact'] is not None]
        self.assertEqual(len(falling_res_all_event), mc.parameters['n_samples'])
        self.assertEqual(len(impact_res_all_event), mc.parameters['n_samples'])

# This allows the module to be executed directly
def run_tests():
    unittest.main()
    
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Horizon functionality")
    result = runner.run(load_test.loadTestsFromTestCase(TestHorizon)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
