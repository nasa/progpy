# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
import numpy as np
import pickle
import sys
import unittest

from progpy import PrognosticsModel
from progpy.predictors import UnscentedTransformPredictor, MonteCarlo, ToEPredictionProfile
from progpy.uncertain_data import MultivariateNormalDist, UnweightedSamples, ScalarData
from progpy.models import BatteryCircuit, ThrownObject
from progpy.state_estimators import UnscentedKalmanFilter
from progpy.predictors.prediction import UnweightedSamplesPrediction, Prediction
from progpy.metrics import samples

# This ensures that the directory containing predictor_template is in the Python search directory
from os.path import dirname, join
sys.path.append(join(dirname(__file__), ".."))


class MockProgModel(PrognosticsModel):
    states = ['a', 'b', 'c', 't']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    events = ['e1', 'e2']
    default_parameters = {
        'p1': 1.2,
    }

    def initialize(self, u={}, z={}):
        return {'a': 1, 'b': 5, 'c': -3.2, 't': 0}

    def next_state(self, x, u, dt):
        x['a']+= u['i1']*dt
        x['c']-= u['i2']
        x['t']+= dt
        return x

    def output(self, x):
        return {'o1': x['a'] + x['b'] + x['c']}
    
    def event_state(self, x):
        t = x['t']
        return {
            'e1': max(1-t/5.0,0),
            'e2': max(1-t/15.0,0)
            }

    def threshold_met(self, x):
        return {key : value < 1e-6 for (key, value) in self.event_state(x).items()}


class TestPredictors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._m = ThrownObject(process_noise=0, measurement_noise=0)
        def future_loading(t, x=None):
            return cls._m.InputContainer({})
        cls._s = cls._m.generate_surrogate([future_loading], state_keys=['v'], dt=0.1, save_freq=0.1, events='impact')

    def test_pred_template(self):
        from predictor_template import TemplatePredictor
        m = MockProgModel()
        pred = TemplatePredictor(m)

    def test_UTP_Broken(self):
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        
        with self.assertRaises(ValueError):
            # Invalid event strategy - first (not supported)
            pred.predict(samples, dt=0.2, num_samples=3, save_freq=1, event_strategy='first')

    def test_UTP_ThrownObject(self):
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])

        # No future loading (i.e., no load)
        results = pred.predict(samples, dt=0.01, save_freq=1)
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.21, 0)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 4.15, 0)
        # self.assertAlmostEqual(mc_results.times[-1], 9, 1)  # Saving every second, last time should be around the 1s after impact event (because one of the sigma points fails afterwards)

        # Test setting dt at class level (otherwise default of 1 will be used and this wont work)
        pred['dt'] = 0.01
        results = pred.predict(samples, save_freq=1)
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.21, 0)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 4.15, 0)

        # Setting event manually
        results = pred.predict(samples, dt=0.01, save_freq=1, events=['falling'])
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)
        self.assertNotIn('impact', results.time_of_event.mean)

        # Setting event in construction
        pred = UnscentedTransformPredictor(m, events=['falling'])
        results = pred.predict(samples, dt=0.01, save_freq=1)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)
        self.assertNotIn('impact', results.time_of_event.mean)

        # Override event set in construction
        results = pred.predict(samples, dt=0.01, save_freq=1, events=['falling', 'impact'])
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.21, 0)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 4.15, 0)

        # String event
        results = pred.predict(samples, dt=0.01, save_freq=1, events='impact')
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 7.785, 5)
        self.assertNotIn('falling', results.time_of_event.mean)

        # Invalid event
        with self.assertRaises(ValueError):
            results = pred.predict(samples, dt=0.01, save_freq=1, events='invalid')
        with self.assertRaises(ValueError):
            # Mix valid, invalid
            results = pred.predict(samples, dt=0.01, save_freq=1, events=['falling', 'invalid'])
        with self.assertRaises(ValueError):
            # Empty
            results = pred.predict(samples, dt=0.01, save_freq=1, events=[])
        with self.assertRaises(TypeError):
            results = pred.predict(samples, dt=0.01, save_freq=1, events=45)

    def test_UTP_ThrownObject_One_Event(self):
        # Test thrown object, similar to test_UKP_ThrownObject, but with only the 'falling' event
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        def future_loading(t, x={}):
            return {}

        results = pred.predict(samples, future_loading, dt=0.01, events=['falling'], save_freq=1)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 0)
        self.assertTrue('impact' not in results.time_of_event.mean)
        self.assertAlmostEqual(results.times[-1], 3, 1)  # Saving every second, last time should be around the nearest 1s before falling event

    def test_UKP_Battery(self):
        def future_loading(t, x=None):
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
            return {'i': i}

        batt = BatteryCircuit()

        ## State Estimation - perform a single ukf state estimate step
        filt = UnscentedKalmanFilter(batt, batt.parameters['x0'])

        example_measurements = {'t': 32.2, 'v': 3.915}
        t = 0.1
        filt.estimate(t, future_loading(t), example_measurements)

        ## Prediction - Predict EOD given current state
        # Setup prediction
        ut = UnscentedTransformPredictor(batt)

        # Predict with a step size of 0.1
        results = ut.predict(filt.x, future_loading, dt=0.1)
        self.assertAlmostEqual(results.time_of_event.mean['EOD'], 3004, -2)

        # Test Metrics
        s = results.time_of_event.sample(100).key('EOD')
        samples.eol_metrics(s)  # Kept for backwards compatibility

    def test_MC_Broken(self):
        m = ThrownObject()
        mc = MonteCarlo(m)

        with self.assertRaises(ValueError):
            # Invalid event strategy
            mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, event_strategy='fdksl')

    def test_MC_ThrownObject(self):
        m = ThrownObject()
        mc = MonteCarlo(m)
        
        # Test with empty future loading (i.e., no load)
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1)
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.0, 5)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)

        # event_strategy='all' should act the same
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, event_strategy='all')
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.0, 5)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)

        # Setting event manually
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events=['falling'])
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)
        self.assertNotIn('impact', results.time_of_event.mean)

        # Setting event in construction
        mc = MonteCarlo(m, events=['falling'])
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1)
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)
        self.assertNotIn('impact', results.time_of_event.mean)

        # Override event set in construction
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events=['falling', 'impact'])
        self.assertAlmostEqual(results.time_of_event.mean['falling'], 3.8, 5)
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.0, 5)

        # String event
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events='impact')
        self.assertAlmostEqual(results.time_of_event.mean['impact'], 8.0, 5)
        self.assertNotIn('falling', results.time_of_event.mean)

        # Invalid event
        with self.assertRaises(ValueError):
            results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events='invalid')
        with self.assertRaises(ValueError):
            # Mix valid, invalid
            results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events=['falling', 'invalid'])
        with self.assertRaises(ValueError):
            # Empty
            results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events=[])
        with self.assertRaises(TypeError):
            results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, events=45)
        
        # Empty with horizon
        results = mc.predict(m.initialize(), dt=0.2, num_samples=3, save_freq=1, horizon=3, events=[])
        
        # TODO(CT): Events in other predictor
    
    def test_MC_ThrownObject_First(self):
        # Test thrown object, similar to test_UKP_ThrownObject, but with only the first event (i.e., 'falling')

        m = ThrownObject()
        mc = MonteCarlo(m)
        mc_results = mc.predict(m.initialize(), dt=0.2, event_strategy='first', num_samples=3, save_freq=1)

        self.assertAlmostEqual(mc_results.time_of_event.mean['falling'], 3.8, 10)
        self.assertTrue('impact' not in mc_results.time_of_event.mean)
        self.assertAlmostEqual(mc_results.times[-1], 3, 1)  # Saving every second, last time should be around the nearest 1s before falling event

    def test_prediction_mvnormaldist(self):
        times = list(range(10))
        covar = [[0.1, 0.01], [0.01, 0.1]]
        means = [{'a': 1+i/10, 'b': 2-i/5} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = Prediction(times, states)

        self.assertEqual(p.mean, means)
        self.assertEqual(p.snapshot(0), states[0])
        self.assertEqual(p.snapshot(-1), states[-1])
        self.assertEqual(p.times[0], times[0])
        self.assertEqual(p.times[-1], times[-1])

        # Out of range
        with self.assertRaises(Exception):
            tmp = p.times[10]

    def test_prediction_uwsamples(self):
        times = list(range(10))
        states = [UnweightedSamples([{'a': i} for i in range(10)]), 
            UnweightedSamples([{'a': i} for i in range(1, 11)]), 
            UnweightedSamples([{'a': i} for i in range(-1, 9)])]
        p = UnweightedSamplesPrediction(times, states)

        self.assertEqual(p[0], states[0])
        self.assertEqual(p[-1], states[-1])
        self.assertEqual(p.snapshot(0), UnweightedSamples([{'a': 0}, {'a': 1}, {'a': -1}]))
        self.assertEqual(p.snapshot(-1), UnweightedSamples([{'a': 9}, {'a': 10}, {'a': 8}]))
        self.assertEqual(p.times[0], times[0])
        self.assertEqual(p.times[-1], times[-1])

        # Out of range
        with self.assertRaises(Exception):
            tmp = p[10]

        with self.assertRaises(Exception):
            tmp = p.sample(10)

        with self.assertRaises(Exception):
            tmp = p.time(10)

        # Bad type
        with self.assertRaises(Exception):
            tmp = p.sample('abc')
    
    def test_prediction_profile(self):
        profile = ToEPredictionProfile()
        self.assertEqual(len(profile), 0)

        profile.add_prediction(0, ScalarData({'a': 1, 'b': 2, 'c': -3.2}))
        profile.add_prediction(1, ScalarData({'a': 1.1, 'b': 2.2, 'c': -3.1}))
        profile.add_prediction(0.5, ScalarData({'a': 1.05, 'b': 2.1, 'c': -3.15}))
        self.assertEqual(len(profile), 3)
        for (t_p, t_p_real) in zip(profile.keys(), [0, 0.5, 1]):
            self.assertAlmostEqual(t_p, t_p_real)

        profile[0.75] = ScalarData({'a': 1.075, 'b': 2.15, 'c': -3.125})
        self.assertEqual(len(profile), 4)
        for (t_p, t_p_real) in zip(profile.keys(), [0, 0.5, 0.75, 1]):
            self.assertAlmostEqual(t_p, t_p_real)
        self.assertEqual(profile[0.75], ScalarData({'a': 1.075, 'b': 2.15, 'c': -3.125}))

        del profile[0.5]
        self.assertEqual(len(profile), 3)
        for (t_p, t_p_real) in zip(profile.keys(), [0, 0.75, 1]):
            self.assertAlmostEqual(t_p, t_p_real)
        for ((t_p, toe), t_p_real) in zip(profile.items(), [0, 0.75, 1]):
            self.assertAlmostEqual(t_p, t_p_real)
        with self.assertRaises(Exception):
            tmp = profile[0.5]
            # 0.5 doesn't exist anymore

    def test_pickle_UTP_ThrownObject_pickle_result(self): # PREDICTION TEST
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        def future_loading(t, x={}):
            return {}

        mc_results = pred.predict(samples, future_loading, dt=0.01, save_freq=1)
        pickle.dump(mc_results, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(mc_results, pickle_converted_result)

    def test_UTP_ThrownObject_One_Event_pickle_result(self): # PREDICTION TEST
        # Test thrown object, similar to test_UKP_ThrownObject, but with only the 'falling' event
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        def future_loading(t, x={}):
            return {}

        mc_results = pred.predict(samples, future_loading, dt=0.01, events=['falling'], save_freq=1)
        pickle.dump(mc_results, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(mc_results, pickle_converted_result)

    def test_UKP_Battery_pickle_result(self):
        def future_loading(t, x = None):
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
            return {'i': i}

        batt = BatteryCircuit()

        ## State Estimation - perform a single ukf state estimate step
        filt = UnscentedKalmanFilter(batt, batt.parameters['x0'])

        example_measurements = {'t': 32.2, 'v': 3.915}
        t = 0.1
        filt.estimate(t, future_loading(t), example_measurements)

        ## Prediction - Predict EOD given current state
        # Setup prediction
        ut = UnscentedTransformPredictor(batt)

        # Predict with a step size of 0.1
        mc_results = ut.predict(filt.x, future_loading, dt=0.1)
        pickle.dump(mc_results, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(mc_results, pickle_converted_result)

    def test_pickle_prediction_mvnormaldist(self):
        times = list(range(10))
        covar = [[0.1, 0.01], [0.01, 0.1]]
        means = [{'a': 1+i/10, 'b': 2-i/5} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = Prediction(times, states)

        p2 = pickle.loads(pickle.dumps(p))
        self.assertEqual(p2, p)

    def test_pickle_prediction_uwsamples(self):
        times = list(range(10))
        states = [UnweightedSamples(list(range(10))), 
            UnweightedSamples(list(range(1, 11))), 
            UnweightedSamples(list(range(-1, 9)))]
        p = UnweightedSamplesPrediction(times, states)

        p2 = pickle.loads(pickle.dumps(p))
        self.assertEqual(p2, p)
    
    def test_pickle_prediction_profile(self):
        profile = ToEPredictionProfile()
        self.assertEqual(len(profile), 0)

        profile.add_prediction(0, ScalarData({'a': 1, 'b': 2, 'c': -3.2}))
        profile.add_prediction(1, ScalarData({'a': 1.1, 'b': 2.2, 'c': -3.1}))
        profile.add_prediction(0.5, ScalarData({'a': 1.05, 'b': 2.1, 'c': -3.15}))
        self.assertEqual(len(profile), 3)
        
        p2 = pickle.loads(pickle.dumps(profile))
        self.assertEqual(p2, profile)

    # Testing LazyUTPrediction
    def test_pickle_UTP_ThrownObject(self):
        m = ThrownObject()
        pred = UnscentedTransformPredictor(m)
        samples = MultivariateNormalDist(['x', 'v'], [1.83, 40], [[0.1, 0.01], [0.01, 0.1]])
        def future_loading(t, x={}):
            return {}
        mc_results = pred.predict(samples, future_loading, dt=0.01, save_freq=1)
        # LazyUTPrediction objects from pre
        pred_op = mc_results.outputs
        pred_es = mc_results.event_states

        pickle.dump(pred_op, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(pred_op, pickle_converted_result)
        
        pickle.dump(pred_es, open('predictor_test.pkl', 'wb'))
        pickle_converted_result = pickle.load(open('predictor_test.pkl', 'rb'))
        self.assertEqual(pred_es, pickle_converted_result)

    def test_profile_plot(self):
        profile = ToEPredictionProfile()
        profile.add_prediction(0, ScalarData({'a': 1, 'b': 2, 'c': -3.2}))
        profile.add_prediction(1, ScalarData({'a': 1.1, 'b': 2.2, 'c': -3.1}))
        profile.add_prediction(0.5, ScalarData({'a': 1.05, 'b': 2.1, 'c': -3.15}))

        # No ground truth or alpha provided
        no_gt_alpha_plots = profile.plot(show=True)

        # Ground truth provided, no alpha provided
        sample_gt = {'a': 1.075, 'b': 2.15, 'c': -3.125}
        gt_no_alpha_plots = profile.plot(ground_truth=sample_gt, show=True)

        # Ground truth and alpha provided
        sample_alpha = 0.50
        gt_and_alpha_plots = profile.plot(ground_truth=sample_gt, alpha=sample_alpha, show=True)

    def test_prediction_monotonicity(self):
        times = list(range(10))
        covar = [[0.1, 0.01], [0.01, 0.1]]

        # Test monotonically increasing and decreasing
        means = [{'a': 1+i/10, 'b': 2-i/5} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = Prediction(times, states)
        self.assertDictEqual(p.monotonicity(), {'a': 1.0, 'b': 1.0})

        # Test no monotonicity
        means = [{'a': i*(i%2-1), 'b': i*(i%2-1)} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = Prediction(times, states)
        self.assertDictEqual(p.monotonicity(), {'a': 0.0, 'b': 0.0})

        # Test monotonicity between range [0,1]
        means = [{'a': i*(i%3-1), 'b': i*(i%3-1)} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = Prediction(times, states)
        self.assertDictEqual(p.monotonicity(), {'a': 0.2222222222222222, 'b': 0.2222222222222222})

        # Test mixed
        means = [{'a': i, 'b': i*(i%3+5)} for i in range(10)]
        states = [MultivariateNormalDist(means[i].keys(), means[i].values(), covar) for i in range(10)]
        p = Prediction(times, states)
        self.assertDictEqual(p.monotonicity(), {'a': 1, 'b': 0.5555555555555556})

        # Test Scalar
        samples = [{'a': 1+i/10, 'b': 2-i/5, 'c': i*(i%2-1), 'd': i*(i%3-1)} for i in range(10)]
        states = [ScalarData(samples[i]) for i in range(10)]
        p = Prediction(times, states)
        self.assertDictEqual(p.monotonicity(), {'a': 1, 'b': 1, 'c': 0, 'd': 0.2222222222222222})

        # Test UnweightedSamples
        samples = [{'a': 1+i/10, 'b': 2-i/5, 'c': i*(i%2-1), 'd': i*(i%3-1)} for i in range(10)]
        states = [UnweightedSamples([samples[i]]) for i in range(10)]
        p = Prediction(times, states)
        self.assertDictEqual(p.monotonicity(), {'a': 1, 'b': 1, 'c': 0, 'd': 0.2222222222222222})

    def _test_surrogate_pred(self, Predictor, **kwargs):
        s = self._s
        p = Predictor(s, **kwargs)
        def future_loading(t, x= None):
            return s.InputContainer({})
        x0 = s.initialize()
        x0 = MultivariateNormalDist(x0.keys(), x0.values(), np.diag([1e-8 * xi for xi in x0.values()]))
        result = p.predict(x0, future_loading, horizon = 50, dt = 0.1)

    def test_utp_surrogate(self):
        self._test_surrogate_pred(UnscentedTransformPredictor, Q = np.diag([1e-3, 1e-3, 1e-7, 1e-7]))

    def test_mc_surrogate(self):
        self._test_surrogate_pred(MonteCarlo)
    
    def test_mc_num_samples(self):
        """
        This test confirms that monte carlos sampling logic works as expected
        """
        m = ThrownObject()
        def future_load(t, x=None):
            return m.InputContainer({})

        pred = MonteCarlo(m)

        # First test- scalar input
        x_scalar = ScalarData({'x': 10, 'v': 0})
        # Should default to 100 samples
        result = pred.predict(x_scalar, future_load)
        self.assertEqual(len(result.time_of_event), 100)
        # Repeat with less samples
        result = pred.predict(x_scalar, future_load, n_samples=10)
        self.assertEqual(len(result.time_of_event), 10)
        
        # Second test- Same, but with multivariate normal input
        # Behavior should be the same
        x_mvnormal = MultivariateNormalDist(['x', 'v'], [10, 0], [[0.1, 0], [0, 0.1]])
        # Should default to 100 samples
        result = pred.predict(x_mvnormal, future_load)
        self.assertEqual(len(result.time_of_event), 100)
        # Repeat with less samples
        result = pred.predict(x_mvnormal, future_load, n_samples=10)
        self.assertEqual(len(result.time_of_event), 10)

        # Third test- UnweightedSamples input
        x_uwsamples = UnweightedSamples([{'x': 10, 'v': 0}, {'x': 9.9, 'v': 0.1}, {'x': 10.1, 'v': -0.1}])
        # Should default to same as x_uwsamples - HERE IS THE DIFFERENCE FROM OTHER TYPES
        result = pred.predict(x_uwsamples, future_load)
        self.assertEqual(len(result.time_of_event), 3)
        # Should be exact same data, in the same order
        for i in range(3):
            self.assertEqual(result.states[i][0]['x'], x_uwsamples[i]['x'])
            self.assertEqual(result.states[i][0]['v'], x_uwsamples[i]['v'])
        # Repeat with more samples
        result = pred.predict(x_uwsamples, future_load, n_samples=10)
        self.assertEqual(len(result.time_of_event), 10)

# This allows the module to be executed directly    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Predictor")
    from unittest.mock import patch
    with patch('matplotlib.pyplot.show') as p:
        result = runner.run(l.loadTestsFromTestCase(TestPredictors)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
