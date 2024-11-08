# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from os.path import dirname, join
import numpy as np
import random
import sys
import unittest
sys.path.append(join(dirname(__file__), ".."))

from progpy import PrognosticsModel, LinearModel
from progpy.models import ThrownObject, BatteryElectroChem, PneumaticValveBase, BatteryElectroChemEOD
from progpy.state_estimators import ParticleFilter, KalmanFilter, UnscentedKalmanFilter
from progpy.uncertain_data import ScalarData, MultivariateNormalDist, UnweightedSamples

def equal_cov(pair1, pair2):
    """
    Compare 2 covariance matricies, considering the order

    Args:
        pair1 (tuple[list, array[array[float]]]):
            keys (list[str]): Labels for keys
            covar (array[array[float]]): Covariance matrix
        pair2 (tuple[list, array[array[float]]]):
            keys (list[str]): Labels for keys
            covar (array[array[float]]): Covariance matrix
    """
    (keys1, cov1) = pair1
    (keys2, cov2) = pair2
    mapping = {i: keys2.index(key) for i, key in enumerate(keys1)}
    return all([cov1[i][j] == cov2[mapping[i]][mapping[j]] for i in range(len(keys1)) for j in range(len(keys1))])


class MockProgModel(PrognosticsModel):
    states = ['a', 'b', 'c', 't']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    events = ['e1', 'e2']
    default_parameters = {
        'p1': 1.2,
    }

    def initialize(self, u = {}, z = {}):
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


class MockProgModel2(MockProgModel):
    outputs = ['o1', 'o2']
    def output(self, x):
        return self.OutputContainer({
            'o1': x['a'] + x['b'] + x['c'], 
            'o2': 7
            })


class TestStateEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._m_mock = MockProgModel()
        cls._m = ThrownObject()
        def future_loading(t, x= None):
            return cls._m.InputContainer({})
        cls._future_loading = future_loading
        cls._s = cls._m.generate_surrogate([future_loading], state_keys=['v'], dt=0.1, save_freq=0.1)

    def test_state_est_template(self):
        from state_estimator_template import TemplateStateEstimator
        se = TemplateStateEstimator(self._m_mock, {'a': 0.0, 'b': 0.0, 'c': 0.0, 't':0.0})

    def __test_state_est(self, filt, m):
        x = m.initialize()

        self.assertTrue(all(key in filt.x.mean for key in m.states))

        # run for a while
        dt = 0.2
        u = m.InputContainer({})
        last_time = 0
        for i in range(500):
            # Get simulated output (would be measured in a real application)
            x = m.next_state(x, u, dt)
            z = m.output(x)

            # Estimate New State every few steps
            if i % 8 == 0:
                # This is to test dt
                # Without dt, this would fail
                last_time = (i+1)*dt
                filt.estimate((i+1)*dt, u, z, dt=dt)

        if last_time != (i+1)*dt:
            # Final estimate
            filt.estimate((i+1)*dt, u, z, dt=dt)

        # Check results - make sure it converged
        x_est = filt.x.mean
        for key in m.states:
            # should be close to right
            self.assertAlmostEqual(x_est[key], x[key], delta=0.4)

    def __test_state_est_no_dt(self, filt, m):
        x = m.initialize()
        filt['dt'] = 0.2

        self.assertTrue(all(key in filt.x.mean for key in m.states))

        # run for a while
        dt = 0.2
        u = m.InputContainer({})
        last_time = 0
        for i in range(500):
            # Get simulated output (would be measured in a real application)
            x = m.next_state(x, u, dt)
            z = m.output(x)

            # Estimate New State every few steps
            if i % 8 == 0:
                # This is to test dt setting at the estimator lvl
                # Without dt, this would fail
                last_time = (i+1)*dt
                filt.estimate((i+1)*dt, u, z)

        if last_time != (i+1)*dt:
            # Final estimate
            filt.estimate((i+1)*dt, u, z)

        # Check results - make sure it converged
        x_est = filt.x.mean
        for key in m.states:
            # should be close to right
            self.assertAlmostEqual(x_est[key], x[key], delta=0.4)

    def test_UKF(self):
        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)
        x_guess = {'x': 1.75, 'v': 35} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = UnscentedKalmanFilter(m, x_guess)
        self.__test_state_est(filt, m)

        filt = UnscentedKalmanFilter(m, x_guess)
        self.__test_state_est_no_dt(filt, m)

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)

        # Test UnscentedKalmanFilter ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 35})
        filt_scalar = UnscentedKalmanFilter(m, x_scalar)
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_scalar.keys()), x_scalar.cov), 
                (list(filt_scalar.x.keys()), filt_scalar.x.cov)))

        # Test UnscentedKalmanFilter MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], np.array([2, 10]), np.array([[1, 0], [0, 1]]))
        filt_mvnd = UnscentedKalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)))

        # Now with a different order
        x_mvnd = MultivariateNormalDist(['v', 'x'], np.array([10, 2]), np.array([[1, 0], [0, 2]]))
        filt_mvnd = UnscentedKalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)), "Covs are not equal for multivariate in different order")

        # Test UnscentedKalmanFilter UnweightedSamples
        x_us = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
        filt_us = UnscentedKalmanFilter(m, x_us)
        mean1 = filt_us.x.mean
        mean2 = x_us.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_us.keys()), x_us.cov), 
                (list(filt_us.x.keys()), filt_us.x.cov)))

        with self.assertRaises(Exception):
            # Not linear model
            UnscentedKalmanFilter(BatteryElectroChem, {})

        with self.assertRaises(Exception):
            # Missing states
            UnscentedKalmanFilter(ThrownObject, {})
        
    def __incorrect_input_tests(self, filter):
        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'c': 2}
        # Missing Key 'b'
        with self.assertRaises(KeyError):
            filter(m, x0)

        class IncompleteModel:
            states = ['a', 'b']
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(NotImplementedError):
            filter(m, x0)

        class IncompleteModel:
            outputs = []
            def next_state(self):
                pass
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(NotImplementedError):
            filter(m, x0)

        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def output(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(NotImplementedError):
            filter(m, x0)
        class IncompleteModel:
            outputs = []
            states = ['a', 'b']
            def next_state(self):
                pass
        m = IncompleteModel()
        x0 = {'a': 0, 'b': 2}
        with self.assertRaises(NotImplementedError):
            filter(m, x0)

    def test_UKF_incorrect_input(self):
        self.__incorrect_input_tests(UnscentedKalmanFilter)

    def test_PF_limit_check(self):
        class OneInputOneOutputOneEventLM(LinearModel):
            inputs = ['u1']
            states = ['x1']
            outputs = ['x1+1']
            events = ['x1 == 10']

            A = np.array([[0]])
            B = np.array([[1]])
            C = np.array([[1]])
            D = np.array([[1]])
            F = np.array([[-0.1]])
            G = np.array([[1]])

            default_parameters = {
                'process_noise': 0.1,
                'measurement_noise': 0.1,
                'x0': {
                    'x1': 0
                }
            }

        m = OneInputOneOutputOneEventLM()
        pf = ParticleFilter(m, {'x1': 10}, num_particles = 5)

        # Without state limits
        pf.estimate(1, {'u1': 1}, {'x1+1': 12})
        self.assertAlmostEqual(pf.x.mean['x1'], 11, delta=0.2)

        # With state limits
        OneInputOneOutputOneEventLM.state_limits = {
            'x1': (0, 10)
        }
        pf.estimate(2, {'u1': 1}, {'x1+1': 13})
        self.assertLessEqual(pf.x.mean['x1'], 10)  # Limited to 10 now
    
    def test_PF_step(self):
        m = PneumaticValveBase()

        # Generate data
        cycle_time = 20
        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2:
                return m.InputContainer({
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Open Valve
                    'uTop': False,
                    'uBot': True
                })
            return m.InputContainer({
                'pL': 3.5e5,
                'pR': 2.0e5,
                # Close Valve
                'uTop': True,
                'uBot': False
            })

        config = {
                'dt': 0.01,
                'save_freq': 1,
            }
        simulated_results = m.simulate_to(10, future_loading, **config)

        # Setup PF
        x0 = m.initialize(future_loading(0))
        filt = ParticleFilter(m, x0, num_particles = 100)
        t=0
        for u, z in zip(simulated_results.inputs, simulated_results.outputs):
            filt.estimate(t, u, z, dt = 3)
            t += config['save_freq']

    def test_PF(self):
        m = ThrownObject(process_noise={'x': 0.75, 'v': 0.75}, measurement_noise=1)
        x_guess = {'x': 1.75, 'v': 38.5} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = ParticleFilter(m, x_guess, num_particles = 1000, measurement_noise = {'x': 1})
        self.__test_state_est(filt, m)

        filt = ParticleFilter(m, x_guess, num_particles = 1000, measurement_noise = {'x': 1})
        self.__test_state_est_no_dt(filt, m)

        # Test ParticleFilter ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 38.5})
        filt_scalar = ParticleFilter(m, x_scalar, num_particles = 20) # Sample count does not affect ScalarData testing
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue((filt_scalar.x.cov == x_scalar.cov).all())
        
        # Test ParticleFilter MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], np.array([2, 10]), np.array([[1, 0], [0, 1]]))
        filt_mvnd = ParticleFilter(m, x_mvnd, num_particles = 100000)
        for k, v in filt_mvnd.x.mean.items():
            self.assertAlmostEqual(v, x_mvnd.mean[k], delta = 0.01)
        for i in range(len(filt_mvnd.x.cov)):
            for j in range(len(filt_mvnd.x.cov[i])):
                self.assertAlmostEqual(filt_mvnd.x.cov[i][j], x_mvnd.cov[i][j], delta=0.1)

        # Test ParticleFilter UnweightedSamples
        uw_input = []
        x_bounds, v_bounds, x0_samples = 5, 5, 10000
        for i in range(x0_samples):
            uw_input.append({'x': random.randrange(-x_bounds, x_bounds), 'v': random.randrange(-v_bounds, v_bounds)})
        x_us = UnweightedSamples(uw_input)
        filt_us = ParticleFilter(m, x_us, num_particles = 100000)
        for k, v in filt_us.x.mean.items():
            self.assertAlmostEqual(v, x_us.mean[k], delta=0.025)
        for i in range(len(filt_us.x.cov)):
            for j in range(len(filt_us.x.cov[i])):
                self.assertAlmostEqual(filt_us.x.cov[i][j], x_us.cov[i][j], delta=0.1)

        # Test x0 if-else Control
        # Case 0: isinstance(x0, UncertainData) 
        x_scalar = ScalarData({'x': 1.75, 'v': 38.5}) # Testing with ScalarData
        filt_scalar = ParticleFilter(m, x_scalar, num_particles = 20) # Sample count does not affect ScalarData testing
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue((filt_scalar.x.cov == x_scalar.cov).all())
        
    def test_PF_incorrect_input(self):
        self.__incorrect_input_tests(ParticleFilter)

    def _test_state_est_surrogate(self, StateEst):
        m = self._m
        s = self._s

        # Setup ParticleFilter
        x0 = s.initialize()
        filt = StateEst(s, x0)

        # Test
        x0 = m.initialize()
        x = m.next_state(x0, {}, 0.1)
        filt.estimate(0.1, s.InputContainer({}), m.output(x))
        self.assertIsInstance(filt.x.mean, s.StateContainer)
        # mean = filt.x.mean
        # self.assertAlmostEqual(mean['x'], x['x'], delta=10)
        # self.assertAlmostEqual(mean['v'], x['v'], delta=1)
        # es = m.event_state(x)
        # for key in es.keys():
        #     self.assertAlmostEqual(mean[key], es[key], delta=0.2)

    def test_PF_Surrogate(self):
        self._test_state_est_surrogate(ParticleFilter)

    def test_UKF_Surrogate(self):
        self._test_state_est_surrogate(UnscentedKalmanFilter)

    def test_KF_Surrogate(self):
        self._test_state_est_surrogate(KalmanFilter)
    
    def test_KF(self):
        class ThrownObject(LinearModel):
            inputs = []  # no inputs, no way to control
            states = ['x', 'v']
            outputs = ['x']
            events = ['falling', 'impact']

            A = np.array([[0, 1], [0, 0]])
            E = np.array([[0], [-9.81]])
            C = np.array([[1, 0]])
            F = None # Will override method

            default_parameters = {
                'thrower_height': 1.83, 
                'throwing_speed': 40, 
                'g': -9.81 
            }

            def initialize(self, u=None, z=None):
                return self.StateContainer({
                    'x': self.parameters['thrower_height'], 
                    'v': self.parameters['throwing_speed'] 
                    })
            
            def threshold_met(self, x):
                return {
                    'falling': x['v'] < 0,
                    'impact': x['x'] <= 0
                }

            def event_state(self, x): 
                x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2) # Use speed and position to estimate maximum height
                return {
                    'falling': np.maximum(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
                    'impact': np.maximum(x['x']/x_max,0) if x['v'] < 0 else 1  # 1 until falling begins, then it's fraction of height
                }

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)
        x_guess = {'x': 1.75, 'v': 35} # Guess of initial state, actual is {'x': 1.83, 'v': 40}

        filt = KalmanFilter(m, x_guess)
        self.__test_state_est(filt, m)

        filt = KalmanFilter(m, x_guess)
        self.__test_state_est_no_dt(filt, m)

        m = ThrownObject(process_noise=5e-2, measurement_noise=5e-2)

        # Test KalmanFilter ScalarData
        x_scalar = ScalarData({'x': 1.75, 'v': 35})
        filt_scalar = KalmanFilter(m, x_scalar)
        mean1 = filt_scalar.x.mean
        mean2 = x_scalar.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_scalar.keys()), x_scalar.cov), 
                (list(filt_scalar.x.keys()), filt_scalar.x.cov)))

        # Test KalmanFilter MultivariateNormalDist
        x_mvnd = MultivariateNormalDist(['x', 'v'], np.array([2, 10]), np.array([[1, 0], [0, 1]]))
        filt_mvnd = KalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)))

        # Now with a different order
        x_mvnd = MultivariateNormalDist(['v', 'x'], np.array([10, 2]), np.array([[1, 0], [0, 2]]))
        filt_mvnd = KalmanFilter(m, x_mvnd)
        mean1 = filt_mvnd.x.mean
        mean2 = x_mvnd.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_mvnd.keys()), x_mvnd.cov), 
                (list(filt_mvnd.x.keys()), filt_mvnd.x.cov)), "Covs are not equal for multivariate in different order")

        # Test KalmanFilter UnweightedSamples
        x_us = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
        filt_us = KalmanFilter(m, x_us)
        mean1 = filt_us.x.mean
        mean2 = x_us.mean
        self.assertSetEqual(set(mean1.keys()), set(mean2.keys()))
        for k in mean1.keys():
            self.assertEqual(mean1[k], mean2[k])
        self.assertTrue(
            equal_cov(
                (list(x_us.keys()), x_us.cov), 
                (list(filt_us.x.keys()), filt_us.x.cov)))

        with self.assertRaises(Exception):
            # Not linear model
            KalmanFilter(BatteryElectroChem, {})

        with self.assertRaises(Exception):
            # Missing states
            KalmanFilter(ThrownObject, {})

    def test_KF_descending(self):
        # Example introduced by @CuiiGen in https://github.com/nasa/progpy/issues/220
        class Descending(LinearModel):
            inputs = ['u']
            states = ['x']
            outputs = ['x']
            events = ['zero']
            A = np.array([[0]])
            B = np.array([[0]])
            E = np.array([[-1]])
            C = np.array([[1]])
            D = np.array([[0]])
            F = None
            default_parameters = {
                'x0': {
                    'x': 10
                },
                'process_noise': 0,
                'measurement_noise': 5
            }

            def initialize(self, u=None, z=None):
                return self.StateContainer(self.default_parameters['x0'])

            def threshold_met(self, x):
                return {
                    'zero': x['x'] <= 0
                }

            def event_state(self, x):
                return {
                    'zero': x['x'] > 0
                }

        m = Descending()

        def future_loading(t, x=None):
            return m.InputContainer({'u': 1})

        config = {
            'dt': 0.01,
            'save_freq': 0.01,
            'events': 'zero'
        }
        simulation_result = m.simulate_to_threshold(future_loading, **config)

        states = simulation_result.states
        outputs = simulation_result.outputs
        inputs = simulation_result.inputs
        states.plot()
        outputs.plot()

        x0 = MultivariateNormalDist(['x'], [0], [[0.01]])
        kf = KalmanFilter(m, x0)
        times = simulation_result.times
        for t, u, z in zip(times, inputs.data, outputs.data):
            kf.estimate(t, u, z)

    def test_PF_particle_ordering(self):
        """
        This is testing for a bug found by @mstraut where particle filter was mixing up the keys if users:
          1. Do not call m.initialize(), and instead
          2. provide a state as a dictionary instead of a state container, and
          3. order the states in a different order than m.states
        """
        m = BatteryElectroChemEOD()
        x0 = m.parameters['x0']  # state as a dictionary with the wrong order
        filt = ParticleFilter(m, x0, num_particles=2)
        for key in m.states:
            self.assertEqual(filt.particles[key][0], x0[key])
            self.assertEqual(filt.particles[key][1], x0[key])

# This allows the module to be executed directly    
def main():
     # This ensures that the directory containing StateEstimatorTemplate is in the python search directory
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting State Estimators")
    result = runner.run(l.loadTestsFromTestCase(TestStateEstimators)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
