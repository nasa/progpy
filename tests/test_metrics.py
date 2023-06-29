# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import pickle
import unittest

from progpy.metrics import calc_metrics, samples, alpha_lambda, prob_success
from progpy.uncertain_data import UncertainData, UnweightedSamples, MultivariateNormalDist, ScalarData
from progpy.predictors import ToEPredictionProfile


class TestMetrics(unittest.TestCase):
    def test_toe_metrics_prev_name(self):
        # This is kept for backwards compatability
        self.assertIs(samples.eol_metrics, calc_metrics)

    def test_toe_metrics_list_dict(self):
        # This is kept for backwards compatability

        # Common checks
        def check_metrics(metrics):
            # True for all keys
            for key in keys:
                self.assertAlmostEqual(metrics[key]['min'], 0)
                for percentile in ['0.01', '0.1', '1']:
                    # Not enough samples for these
                    self.assertIsNone(metrics[key]['percentiles'][percentile])

            # Key specific
            self.assertAlmostEqual(metrics['a']['percentiles']['10'], 1)
            self.assertAlmostEqual(metrics['a']['percentiles']['25'], 2)
            self.assertAlmostEqual(metrics['a']['mean'], 4.5)
            self.assertAlmostEqual(metrics['a']['percentiles']['50'], 5)
            self.assertAlmostEqual(metrics['a']['percentiles']['75'], 7)
            self.assertAlmostEqual(metrics['a']['mean absolute deviation'], 2.5)
            self.assertAlmostEqual(metrics['a']['max'], 9)
            self.assertAlmostEqual(metrics['a']['std'], 2.8722813232690143)
            self.assertAlmostEqual(metrics['b']['percentiles']['10'], 1.1)
            self.assertAlmostEqual(metrics['b']['percentiles']['25'], 2.2)
            self.assertAlmostEqual(metrics['b']['mean'], 4.95)
            self.assertAlmostEqual(metrics['b']['percentiles']['50'], 5.5)
            self.assertAlmostEqual(metrics['b']['percentiles']['75'], 7.7)
            self.assertAlmostEqual(metrics['b']['mean absolute deviation'], 2.75)
            self.assertAlmostEqual(metrics['b']['max'], 9.9)
            self.assertAlmostEqual(metrics['b']['std'], 3.159509455595916)
            self.assertAlmostEqual(metrics['c']['percentiles']['10'], 0.04)
            self.assertAlmostEqual(metrics['c']['percentiles']['25'], 0.16)
            self.assertAlmostEqual(metrics['c']['mean'], 1.14)
            self.assertAlmostEqual(metrics['c']['percentiles']['50'], 1.0)
            self.assertAlmostEqual(metrics['c']['percentiles']['75'], 1.96)
            self.assertAlmostEqual(metrics['c']['mean absolute deviation'], 0.928)
            self.assertAlmostEqual(metrics['c']['max'], 3.24)
            self.assertAlmostEqual(metrics['c']['std'], 1.074094967868298)

        u_samples = [{'a': i, 'b': i*1.1, 'c': (i/5)**2} for i in range(10)]
        keys = ['a', 'b', 'c']
        metrics = calc_metrics(u_samples)

        check_metrics(metrics)
        for key in keys:
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertNotIn(key2, metrics[key])

        metrics = calc_metrics(u_samples, 5.0)
        check_metrics(metrics)
        for key in keys:
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 3.86)

        metrics = calc_metrics(u_samples, ground_truth = {'a': 5.0, 'b': 4.5, 'c': 1.5})
        check_metrics(metrics)
        for key in keys:
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 1.012)

        # Empty Sample Set
        with self.assertRaises(ValueError):
            calc_metrics([]) 

    def test_toe_metrics_mvnd(self):
        # Common checks
        def check_metrics(metrics):
            mean_dict = {key: value for (key, value) in zip(keys, mean)}
            for key in keys:
                self.assertAlmostEqual(metrics[key]['mean'], mean_dict[key])
                self.assertAlmostEqual(metrics[key]['median'], mean_dict[key])
                self.assertAlmostEqual(metrics[key]['percentiles']['50'], mean_dict[key])
                self.assertAlmostEqual(metrics[key]['std'], 1, 1)
        mean = [10, 11, 12]
        keys = ['a', 'b', 'c']
        covar = [
            [1, 0.1, 0.1], 
            [0.1, 1, 0.1], 
            [0.1, 0.1, 1]]
        dist = MultivariateNormalDist(keys, mean, covar)
        metrics = calc_metrics(dist)
        dist.metrics()
        check_metrics(metrics)

        metrics = calc_metrics(dist, 11)
        dist.metrics(ground_truth=11)
        check_metrics(metrics)
        self.assertAlmostEqual(metrics['a']['ground truth percentile'], 84.3, -1)
        self.assertAlmostEqual(metrics['b']['ground truth percentile'], 50, -1)
        self.assertAlmostEqual(metrics['c']['ground truth percentile'], 15.4, -1)

        # P(success)
        p_success = prob_success(dist, 11)
        self.assertAlmostEqual(p_success['a'], 0.1575, 1)
        self.assertAlmostEqual(p_success['b'], 0.5, 1)
        self.assertAlmostEqual(p_success['c'], 0.8425, 1)

    def test_toe_metrics_scalar(self):
        # Common checks
        def check_metrics(metrics):
            for key in scalar.keys():
                self.assertAlmostEqual(metrics[key]['min'], data[key])
                for value in metrics[key]['percentiles'].values():
                    self.assertAlmostEqual(value, data[key])
                self.assertAlmostEqual(metrics[key]['mean'], data[key])
                self.assertAlmostEqual(metrics[key]['median'], data[key])
                self.assertAlmostEqual(metrics[key]['max'], data[key])
                self.assertAlmostEqual(metrics[key]['std'], 0)
                self.assertAlmostEqual(metrics[key]['mean absolute deviation'], 0)

        data = {
                'a': 10,
                'b': 11,
                'c': 12
            }
        scalar = ScalarData(data)
        metrics = calc_metrics(scalar)
        self.assertDictEqual(scalar.metrics(), metrics)
        check_metrics(metrics)

        # Check with ground truth
        metrics = calc_metrics(scalar, 11)
        self.assertDictEqual(scalar.metrics(ground_truth=11), metrics)
        check_metrics(metrics)
        for key in data.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 1)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 0)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 1)

        # Check with limited samples
        metrics = calc_metrics(scalar, n_samples = 1000)
        self.assertDictEqual(scalar.metrics(n_samples = 1000), metrics)
        for key in scalar.keys():
            self.assertIsNone(metrics[key]['percentiles']['0.01'])
            metrics[key]['percentiles']['0.01'] = data[key]  # Fill so we can check everything else below
        check_metrics(metrics)

        # Check broken samples
        with self.assertRaises(TypeError):
            calc_metrics(scalar, n_samples='abc')

        with self.assertRaises(TypeError):
            calc_metrics(scalar, n_samples=[])

        # P(success)
        p_success = prob_success(scalar, 11)
        self.assertAlmostEqual(p_success['a'], 0)  # After all samples
        self.assertAlmostEqual(p_success['b'], 0)  # Exactly equal
        self.assertAlmostEqual(p_success['c'], 1)  # Before all samples

    def test_toe_metrics_specific_keys(self):
        data = {
                'a': 10,
                'b': 11,
                'c': 12
            }
        scalar = ScalarData(data)
        metrics = scalar.metrics(keys=['a', 'b'])
        self.assertNotIn('c', metrics)
        self.assertIn('a', metrics)
        self.assertIn('b', metrics)

        # Check P(MS)
        p_success = prob_success(scalar, 11, keys=['a', 'c'])
        self.assertNotIn('b', p_success)
        self.assertIn('a', p_success)
        self.assertIn('c', p_success)

    def test_toe_metrics_u_samples(self):
        # Common checks
        def check_metrics(metrics):
            # True for all keys
            for key in u_samples.keys():
                self.assertAlmostEqual(metrics[key]['min'], 0)
                for percentile in ['0.01', '0.1', '1']:
                    # Not enough samples for these
                    self.assertIsNone(metrics[key]['percentiles'][percentile])

            # Key specific
            self.assertAlmostEqual(metrics['a']['percentiles']['10'], 1)
            self.assertAlmostEqual(metrics['a']['percentiles']['25'], 2)
            self.assertAlmostEqual(metrics['a']['mean'], 4.5)
            self.assertAlmostEqual(metrics['a']['percentiles']['50'], 5)
            self.assertAlmostEqual(metrics['a']['percentiles']['75'], 7)
            self.assertAlmostEqual(metrics['a']['mean absolute deviation'], 2.5)
            self.assertAlmostEqual(metrics['a']['max'], 9)
            self.assertAlmostEqual(metrics['a']['std'], 2.8722813232690143)
            self.assertAlmostEqual(metrics['b']['percentiles']['10'], 1.1)
            self.assertAlmostEqual(metrics['b']['percentiles']['25'], 2.2)
            self.assertAlmostEqual(metrics['b']['mean'], 4.95)
            self.assertAlmostEqual(metrics['b']['percentiles']['50'], 5.5)
            self.assertAlmostEqual(metrics['b']['percentiles']['75'], 7.7)
            self.assertAlmostEqual(metrics['b']['mean absolute deviation'], 2.75)
            self.assertAlmostEqual(metrics['b']['max'], 9.9)
            self.assertAlmostEqual(metrics['b']['std'], 3.159509455595916)
            self.assertAlmostEqual(metrics['c']['percentiles']['10'], 0.04)
            self.assertAlmostEqual(metrics['c']['percentiles']['25'], 0.16)
            self.assertAlmostEqual(metrics['c']['mean'], 1.14)
            self.assertAlmostEqual(metrics['c']['percentiles']['50'], 1.0)
            self.assertAlmostEqual(metrics['c']['percentiles']['75'], 1.96)
            self.assertAlmostEqual(metrics['c']['mean absolute deviation'], 0.928)
            self.assertAlmostEqual(metrics['c']['max'], 3.24)
            self.assertAlmostEqual(metrics['c']['std'], 1.074094967868298)

        data = [{'a': i, 'b': i*1.1, 'c': (i/5)**2} for i in range(10)]
        u_samples = UnweightedSamples(data)
        metrics = calc_metrics(u_samples)
        self.assertDictEqual(u_samples.metrics(), metrics)

        check_metrics(metrics)
        for key in u_samples.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertNotIn(key2, metrics[key])

        metrics = calc_metrics(u_samples, 5.0)
        self.assertDictEqual(u_samples.metrics(ground_truth=5.0), metrics)
        check_metrics(metrics)
        for key in u_samples.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 3.86)

        ground_truth = {'a': 5.0, 'b': 4.5, 'c': 1.5}
        metrics = calc_metrics(u_samples, ground_truth = ground_truth)
        self.assertDictEqual(u_samples.metrics(ground_truth = ground_truth), metrics)
        check_metrics(metrics)
        for key in u_samples.keys():
            for key2 in ['mean absolute percentage error', 'relative accuracy', 'ground truth percentile']:   
                self.assertIn(key2, metrics[key])
        self.assertAlmostEqual(metrics['a']['mean absolute error'], 2.5)
        self.assertAlmostEqual(metrics['b']['mean absolute error'], 2.75)
        self.assertAlmostEqual(metrics['c']['mean absolute error'], 1.012)

        # Empty Sample Set
        with self.assertRaises(ValueError):
            calc_metrics(UnweightedSamples([])) 

        # P(success)
        p_success = prob_success(u_samples, 5.0)
        self.assertAlmostEqual(p_success['a'], 0.4)
        self.assertAlmostEqual(p_success['b'], 0.5)
        self.assertAlmostEqual(p_success['c'], 0)

    def test_toe_metrics_ground_truth(self):
        # Wrong type 
        with self.assertRaises(TypeError):
            calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth='abc')

        # Below samples
        calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=0)

        # At sample
        calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=1)

        # Above samples
        calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=2)

        # NaN
        calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=float('nan'))

        # Inf
        calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=float('inf'))

        # -Inf
        calc_metrics(UnweightedSamples([{'a': 1.0}]), ground_truth=-float('inf'))

    def test_toe_profile_metrics(self):
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(10):
            # a will shift upward from 0-19 to 9-28
            # b is always a-1
            # c is always a * 2, and will therefore always have twice the spread
            data = [{'a': j, 'b': j -1 , 'c': (j-4.5) * 2 + 4.5} for j in range(i, i+20)]
            profile.add_prediction(
                10-i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )

        # Test 1: Ground truth at median
        ground_truth = {'a': 9.0, 'b': 8.0, 'c': 18.0}
        lambda_value = 8  # Almost at prediction 
        alpha = 0.5
        beta = 0.05  # 5% is really bad
        metrics = alpha_lambda(profile, ground_truth, lambda_value, alpha, beta)
        # Result at t=8
        # a
        #     toe: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        #     Bounds: [8.5 - 9.5](0.05%)
        # b
        # toe: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # Bounds: [8.0 - 8.0](0.0%)
        # c
        #     toe: [-0.5, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5, 29.5, 31.5, 33.5, 35.5, 37.5]
        #     Bounds: [13.0 - 23.0](0.25%)
        # {'a': True, 'b': False, 'c': True}
        self.assertTrue(metrics['a'])
        self.assertFalse(metrics['b'])
        self.assertTrue(metrics['c'])

        # Now lets do it at t=5
        lambda_value = 5
        metrics = alpha_lambda(profile, ground_truth, lambda_value, alpha, beta)
        # Here all should be true
        # a
        #     toe: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        #     Bounds: [7.0 - 11.0](0.15%)
        # b
        #     toe: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        #     Bounds: [6.5 - 9.5](0.15%)
        # c
        #     toe: [5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5, 29.5, 31.5, 33.5, 35.5, 37.5, 39.5, 41.5, 43.5]
        #     Bounds: [11.5 - 24.5](0.3%)
        # {'a': True, 'b': True, 'c': True}
        self.assertTrue(metrics['a'])
        self.assertTrue(metrics['b'])
        self.assertTrue(metrics['c'])
        self.assertDictEqual(metrics, profile.alpha_lambda(ground_truth, lambda_value, alpha, beta))

        # Now lets try specifying only keys a and b
        metrics = alpha_lambda(profile, ground_truth, lambda_value, alpha, beta, keys=['a', 'b'])
        self.assertIn('a', metrics)
        self.assertIn('b', metrics)
        self.assertNotIn('c', metrics)

    def test_toe_profile_metrics(self):
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(10):
            # a will shift upward from 0-19 to 9-28
            # b is always a-1
            # c is always a * 2, and will therefore always have twice the spread
            data = [{'a': j, 'b': j -1 , 'c': (j-4.5) * 2 + 4.5} for j in range(i, i+20)]
            profile.add_prediction(
                10-i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )

        # Test 1: Ground truth at median
        ground_truth = {'a': 9.0, 'b': 8.0, 'c': 18.0}
        lambda_value = 8  # Almost at prediction 
        alpha = 0.5
        beta = 0.05  # 5% is really bad
        metrics = alpha_lambda(profile, ground_truth, lambda_value, alpha, beta)
        
        # Test pickling ToEPredictionProfile
        pickle.dump(profile, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(profile, pickle_converted_result)
        # Test pickling alpha_lambda result
        pickle.dump(metrics, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(metrics, pickle_converted_result)

    def test_toe_profile_prognostic_horizon(self):
        profile = ToEPredictionProfile()  # Empty profile
        # Define test sample ground truth
        GROUND_TRUTH = {'a': 9.0, 'b': 8.0, 'c': 18.0}
        # Create rudimentary criteria equation
        def criteria_eqn(tte : UncertainData, ground_truth_tte : dict) -> dict:
            """
            Sample criteria equation for unittesting. 

            Args:
                tte : UncertainData
                    Time to event in UncertainData format.
                ground_truth_tte : dict
                    Dictionary of ground truth of time to event.
            """
            result = {}
            for key, value in ground_truth_tte.items():
                result[key] = abs(tte.mean[key] - value) < 0.6
            return result
        # Prognostic horizon metric testing
        # Test 0: Empty profile (should return None for all)
        self.assertDictEqual(profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH), {'a': None, 'b': None, 'c': None})

        # Loading profile
        for i in range(10):
            # a will shift upward from 0-19 to 9-28
            # b is always a-1
            # c is always a * 2, and will therefore always have twice the spread
            data = [{'a': j, 'b': j -1 , 'c': (j-4.5) * 2 + 4.5} for j in range(i, i+20)]
            profile.add_prediction(
                10-i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )
        # Test 1: simple 1 criteria met
        self.assertDictEqual(profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH), {'a': None, 'b': None, 'c': 10.0})
        # Test 2: all criteria are met at different times
        GROUND_TRUTH = {'a': 10.0, 'b': 10.0, 'c': 18.0} # Adjust ground truth to match 3 criteria
        self.assertDictEqual(profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH), {'a': 1.0, 'b': 2.0, 'c': 10.0})
        # Test 3: 2 criteria are met at once (at 10.0)
        GROUND_TRUTH = {'a': 9.0, 'b': 14.0, 'c': 18.0}
        self.assertDictEqual(profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH), {'a': None, 'b': 10.0, 'c': 10.0})
        # Test 4: at least one criteria is met at beginning of the prediction
        GROUND_TRUTH = {'a': 10, 'b': 8.0, 'c': 10.0}
        self.assertDictEqual(profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH), {'a': 1.0, 'b': None, 'c': None})
        # Test 5: criteria not met for any
        GROUND_TRUTH = {'a': 9.0, 'b': 8.0, 'c': 8.0}
        self.assertDictEqual(profile.prognostic_horizon(criteria_eqn, GROUND_TRUTH), {'a': None, 'b': None, 'c': None})

    def test_toe_profile_cumulative_relative_accuracy(self):
        profile = ToEPredictionProfile()  # Empty profile
        # Loading profile
        for i in range(10):
            data = [{'a': j, 'b': j -1 , 'c': (j-4.5) * 2 + 4.5} for j in range(i, i+20)]
            profile.add_prediction(
                10-i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )
        # Test positive floats ground truth
        GROUND_TRUTH = {'a': 9.0, 'b': 8.0, 'c': 18.0}
        self.assertEqual(profile.cumulative_relative_accuracy(GROUND_TRUTH), {'a': 0.4444444444444445, 'b': 0.375, 'c': 0.6388888888888888})
        # Test negative floats ground truth
        GROUND_TRUTH = {'a': -9.0, 'b': -8.0, 'c': -18.0}
        self.assertEqual(profile.cumulative_relative_accuracy(GROUND_TRUTH), {'a': 3.555555555555556, 'b': 3.625, 'c': 3.305555555555556})
        # Test ground truth values of 0; already caught by relative_accuracy
        with self.assertRaises(ZeroDivisionError):
            GROUND_TRUTH = {'a': 0, 'b': 0, 'c': 0}
            raise_error = profile.cumulative_relative_accuracy(GROUND_TRUTH)
        # Test ground truth in invalid input types; already caught by relative_accuracy
        with self.assertRaises(TypeError):
            GROUND_TRUTH = []
            raise_error = profile.cumulative_relative_accuracy(GROUND_TRUTH)

    def test_toe_profile_monotonicity(self):
        # Test monotonically increasing and decreasing
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(10):
            data = [{'a': 1+i/10, 'b': 2-i/5} for i in range(10)]
            profile.add_prediction(
                i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )
        self.assertDictEqual(profile.monotonicity(), {'a': 1.0, 'b': 1.0})

        # Test no monotonicity
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(0,11):
            data = [{'a': i + i*(i%2-0.5), 'b': i + i*(i%2-0.5)}]*10
            profile.add_prediction(
                i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )
        self.assertDictEqual(profile.monotonicity(), {'a': 0.0, 'b': 0.0})

        # Test monotonicity between range [0,1]
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(11):
            data = [{'a': i+i*(i%3-0.5), 'b': i+i*(i%3-0.5)}] * 10
            profile.add_prediction(
                i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )
        self.assertDictEqual(profile.monotonicity(), {'a': 0.4, 'b': 0.4})

        # Test mixed
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(11):
            data = [{'a': i+i*(i%3-0.5), 'b': i + i*(i%2-0.5), 'c': 1+i/10}] * 10
            profile.add_prediction(
                i,  # Time (reverse so data is decreasing)
                UnweightedSamples(data)  # ToE Prediction
            )
        self.assertDictEqual(profile.monotonicity(), {'a': 0.4, 'b': 0.0, 'c': 1.0})

        # Test MultivariateNormalDist
        profile = ToEPredictionProfile()  # Empty profile
        covar = [[0.1, 0.01], [0.01, 0.1]]
        for i in range(11):
            data = [{'a': i, 'b': i*(i%3+5)}] * 11
            profile.add_prediction(
                i,  # Time (reverse so data is decreasing)
                MultivariateNormalDist(data[i].keys(), data[i].values(), covar)  # ToE Prediction
            )
        self.assertDictEqual(profile.monotonicity(), {'a': 0.0, 'b': 0.5})

        # Test ScalarData
        profile = ToEPredictionProfile()  # Empty profile
        for i in range(11):
            data = {'a': i+i*(i%3-0.5), 'b': i+i*(i%3-0.5)}
            profile.add_prediction(
                i,  # Time (reverse so data is decreasing)
                ScalarData(data)  # ToE Prediction
            )
        self.assertDictEqual(profile.monotonicity(), {'a': 0.4, 'b': 0.4})


# This allows the module to be executed directly    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Metrics")
    result = runner.run(l.loadTestsFromTestCase(TestMetrics)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
