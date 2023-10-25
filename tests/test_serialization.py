# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from progpy.models import BatteryElectroChemEOD as Battery 
from progpy.data_models import DMDModel


class TestSerialization(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_serialization_basic(self):
        batt = Battery()

        def future_loading_1(t, x=None):
            # Variable (piece-wise) future loading scheme
            if (t < 500):
                i = 3
            elif (t < 1000):
                i = 2
            elif (t < 1500):
                i = 0.5
            else:
                i = 4.5
            return batt.InputContainer({'i': i})

        def future_loading_2(t, x=None):
            # Variable (piece-wise) future loading scheme
            if (t < 300):
                i = 2
            elif (t < 800):
                i = 3.5
            elif (t < 1300):
                i = 4
            elif (t < 1600):
                i = 1.5
            else:
                i = 5
            return batt.InputContainer({'i': i})
        
        load_functions = [future_loading_1, future_loading_2]

        options_surrogate = {
            'save_freq': 1,  # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1,  # For DMD, this value is the time step of the training data
            'trim_data_to': 0.7  # Value between 0 and 1 that determines the fraction of data resulting from simulate_to_threshold that is used to train DMD surrogate model
        }

        batt.parameters['process_noise'] = 0

        # Generate surrogate model  
        surrogate_orig = batt.generate_surrogate(load_functions, **options_surrogate)
        # Serialize parameters
        save_json_dict = surrogate_orig.to_json()

        # Generate new surrogate with serialized version
        new_model = DMDModel.from_json(save_json_dict)

        # Check serialization
        self.assertEqual(surrogate_orig.parameters, new_model.parameters)
        
        # Check deserialization
        options_sim = {
            'save_freq': 1  # Frequency at which results are saved, or equivalently time step in results
        }

        # Define loading profile
        def future_loading(t, x=None):
            if (t < 600):
                i = 3
            elif (t < 1000):
                i = 2
            elif (t < 1500):
                i = 1.5
            else:
                i = 4
            return batt.InputContainer({'i': i})

        # Simulate to threshold using DMD approximation
        surrogate_results = surrogate_orig.simulate_to_threshold(future_loading,**options_sim)
        new_results = new_model.simulate_to_threshold(future_loading, **options_sim) 

        for i in range(min(len(surrogate_results.times), len(new_results.times))):
            self.assertEqual(surrogate_results.times[i], new_results.times[i])
            for key in surrogate_results.states[0].keys():
                self.assertAlmostEqual(surrogate_results.states[i][key], new_results.states[i][key], delta=3e-01)
            for key in surrogate_results.inputs[0].keys():
                self.assertAlmostEqual(surrogate_results.inputs[i][key], new_results.inputs[i][key], delta=3e-01)
            for key in surrogate_results.outputs[0].keys():
                self.assertAlmostEqual(surrogate_results.outputs[i][key], new_results.outputs[i][key], delta=3e-01)
            for key in surrogate_results.event_states[0].keys():
                self.assertAlmostEqual(surrogate_results.event_states[i][key], new_results.event_states[i][key], delta=3e-01)

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Serialization of Surrogate Model")
    result = runner.run(load_test.loadTestsFromTestCase(TestSerialization)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
