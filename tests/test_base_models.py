# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy
import io
import numpy as np
from os.path import dirname, join
import pickle
import sys
import unittest

# This ensures that the directory containing ProgModelTemplate is in the python search directory
sys.path.append(join(dirname(__file__), ".."))

from progpy import PrognosticsModel, CompositeModel
from progpy.models import ThrownObject, BatteryElectroChemEOD
from progpy.models.test_models.linear_models import (OneInputNoOutputNoEventLM, OneInputOneOutputNoEventLM, OneInputTwoStatesNoOutputNoEventLM, OneInputNoOutputOneEventLM, OneInputOneOutputNoEventLMPM)
from progpy.models.test_models.linear_thrown_object import (LinearThrownObject, LinearThrownDiffThrowingSpeed, LinearThrownObjectUpdatedInitializedMethod, LinearThrownObjectDiffDefaultParams)


class MockModel():
    states = ['a', 'b', 'c', 't']
    inputs = ['i1', 'i2']
    outputs = ['o1']
    default_parameters = {
        'p1': 1.2,
        'x0': {'a': 1, 'b': 5, 'c': -3.2, 't': 0}
    }

    def initialize(self, u={}, z={}):
        return self.StateContainer(self.parameters['x0'])

    def next_state(self, x, u, dt):
        return self.StateContainer({
                    'a': x['a'] + u['i1']*dt,
                    'b': x['b'],
                    'c': x['c'] - u['i2'],
                    't': x['t'] + dt
                })

    def output(self, x):
        return self.OutputContainer({'o1': x['a'] + x['b'] + x['c']})


class MockProgModel(MockModel, PrognosticsModel):
    events = ['e1', 'e2']

    def event_state(self, x):
        t = x['t']
        return {
            'e1': max(1-t/5.0,0),
            'e2': max(1-t/15.0,0)
            }

    def threshold_met(self, x):
        return {
            key: value < 1e-6 for (key, value) in self.event_state(x).items()}

def derived_callback(config):
    return {
        'p2': config['p1']  # New config
    }

def derived_callback2(config):
    return {  # Testing chained update
        'p3': config['p2'], 
    }

def derived_callback3(config):
    return {  # Testing 2nd chained update
        'p4': -2 * config['p2'],
    }

class MockModelWithDerived(MockProgModel):
    param_callbacks = {
            'p1': [derived_callback],
            'p2': [derived_callback2, derived_callback3]
        }


class TestModels(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = io.StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_non_container(self):
        class MockProgModelStateDict(MockProgModel):
            def next_state(self, x, u, dt):
                return {
                    'a': x['a'] + u['i1']*dt,
                    'b': x['b'],
                    'c': x['c'] - u['i2'],
                    't': x['t'] + dt
                }
        
        m = MockProgModelStateDict(
            process_noise_dist='none',
            measurement_noise_dist='none')
        
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Any event, default
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0)
        self.assertAlmostEqual(times[-1], 5.0, 5)
        self.assertAlmostEqual(outputs[-1]['o1'], -13.2)
        self.assertIsInstance(outputs[-1], m.OutputContainer)

        class MockProgModelStateNdarray(MockProgModel):
            def next_state(self, x, u, dt):
                return np.array([
                    [x['a'] + u['i1']*dt],
                    [x['b']],
                    [x['c'] - u['i2']],
                    [x['t'] + dt]]
                )

        m = MockProgModelStateNdarray(
            process_noise_dist='none', 
            measurement_noise_dist='none')

        # Any event, default
        (times, _, _, outputs, _) = m.simulate_to_threshold(
            load,
            {'o1': 0.8},
            dt=0.5,
            save_freq=1.0)
        self.assertAlmostEqual(times[-1], 5.0, 5)

    def test_integration_type(self):
        m_default_integration = LinearThrownObject(process_noise=0, measurement_noise=0)  
        m_rk4_integration = LinearThrownObject(integration_method='rk4', process_noise=0, measurement_noise=0)

        # compare two models with different integration types
        x_default = m_default_integration.initialize()
        u_default = m_default_integration.InputContainer({})
        x_default = m_default_integration.next_state(x_default, u_default, 0.1)
        
        x_rk4 = m_rk4_integration.initialize()
        u_rk4 = m_rk4_integration.InputContainer({})
        x_rk4 = m_rk4_integration.next_state(x_rk4, u_rk4, 0.1)

        # The two models should have close but different values for x
        self.assertAlmostEqual(x_default['x'], x_rk4['x'], delta=0.1)
        self.assertNotEqual(x_default['x'], x_rk4['x'])

        # Velocity should be exactly the same (because it's linear)
        self.assertEqual(x_default['v'], x_rk4['v'])

        # Now, if we change the integrator for the default model, it should be the same as the rk4 model
        m_default_integration.parameters['integration_method'] = 'rk4'
        x_default = m_default_integration.initialize()
        u_default = m_default_integration.InputContainer({})
        x_default = m_default_integration.next_state(x_default, u_default, 0.1)

        # Both velocity (v) and position (x) should be equal
        # This is because we changed the integration method to rk4 for the default model
        self.assertEqual(x_default['v'], x_rk4['v'])
        self.assertEqual(x_default['x'], x_rk4['x'])

    def test_parameters_statelikematrixwrapper(self):
        """
        This is testing a very specific case where a state container from one model is used to define the noise from another.
        """
        m0 = OneInputNoOutputNoEventLM()
        m1 = OneInputTwoStatesNoOutputNoEventLM(process_noise=m0.parameters['process_noise'])
        self.assertSetEqual(set(m1.parameters['process_noise'].keys()), set(m1.states))

    def test_integration_type_scipy(self):
        # SciPy Integrator test.
        # Here we will set the integrator to various scipy integration methods and make sure that it works
        from scipy.integrate import RK45, RK23, DOP853, Radau, BDF, LSODA

        m_default_integration = LinearThrownObject(process_noise=0, measurement_noise=0)
        x_default = m_default_integration.initialize()
        u_default = m_default_integration.InputContainer({})
        x_default = m_default_integration.next_state(x_default, u_default, 0.1)

        # RK45
        m = LinearThrownObject(integration_method=RK45, process_noise=0, measurement_noise=0)
        x = m.initialize()
        u = m.InputContainer({})
        x = m.next_state(x, u, 0.1)
        self.assertAlmostEqual(x['v'], x_default['v'], delta=0.1)
        self.assertAlmostEqual(x['x'], x_default['x'], delta=0.1)
        self.assertNotEqual(x['v'], x_default['v'])
        self.assertNotEqual(x['x'], x_default['x'])

        # RK23
        m = LinearThrownObject(integration_method=RK23, process_noise=0, measurement_noise=0)
        x = m.initialize()
        u = m.InputContainer({})
        x = m.next_state(x, u, 0.1)
        # V is equal because it's linear
        self.assertEqual(x['v'], x_default['v'])
        self.assertAlmostEqual(x['x'], x_default['x'], delta = 0.1)
        self.assertNotEqual(x['x'], x_default['x'])

        # DOP853
        m = LinearThrownObject(integration_method=DOP853, process_noise=0, measurement_noise=0)
        x = m.initialize()
        u = m.InputContainer({})
        x = m.next_state(x, u, 0.1)
        # V is equal because it's linear
        self.assertAlmostEqual(x['v'], x_default['v'])
        self.assertAlmostEqual(x['x'], x_default['x'], delta=0.1)
        self.assertNotEqual(x['x'], x_default['x'])

        # Radau
        m = LinearThrownObject(integration_method=Radau, process_noise=0, measurement_noise=0)
        x = m.initialize()
        u = m.InputContainer({})
        x = m.next_state(x, u, 0.1)
        # V is equal because it's linear
        self.assertEqual(x['v'], x_default['v'])
        self.assertAlmostEqual(x['x'], x_default['x'], delta=0.1)
        self.assertNotEqual(x['x'], x_default['x'])

        # BDF
        m = LinearThrownObject(integration_method=BDF, process_noise=0, measurement_noise=0)
        x = m.initialize()
        u = m.InputContainer({})
        x = m.next_state(x, u, 0.1)
        # V is equal because it's linear
        self.assertEqual(x['v'], x_default['v'])
        self.assertAlmostEqual(x['x'], x_default['x'], delta=0.1)
        self.assertNotEqual(x['x'], x_default['x'])

        # LSODA
        m = LinearThrownObject(integration_method=LSODA, process_noise=0, measurement_noise=0)
        x = m.initialize()
        u = m.InputContainer({})
        x = m.next_state(x, u, 0.1)
        # V is equal because it's linear
        self.assertEqual(x['v'], x_default['v'])
        self.assertAlmostEqual(x['x'], x_default['x'], delta=0.1)
        self.assertNotEqual(x['x'], x_default['x'])

    def test_integration_type_error(self):
        with self.assertRaises(ValueError):
            # unsupported integration type
            m = LinearThrownObject(integration_method='invalid')

        with self.assertRaises(TypeError):
            # change integration type for a discrete model
            m = ThrownObject(integration_method='rk4')

        # Repeat with setting in params
        m = LinearThrownObject()
        with self.assertRaises(ValueError):
            # unsupported integration type
            m.parameters['integration_method'] = 'invalid'

        m = ThrownObject()
        with self.assertRaises(TypeError):
            # change integration type for a discrete model
            m.parameters['integration_method'] = 'rk4'

    def test_size(self):
        m = MockProgModel()
        size = sys.getsizeof(m)
        self.assertLess(size, 20000)

        # Adding a parameter
        m.parameters['test'] = 8675309
        size2 = sys.getsizeof(m)

        # Check that size increases
        self.assertLess(size, size2)

        # Size difference should be slightly more than the size of key & value
        # The difference is overhead for the dict
        # Note- have to use an uncommon number for this to work
        # This is because of python's memory allocation
        diff = size2 - size
        sum_of_parts = sys.getsizeof('test') + sys.getsizeof(8675309)
        self.assertLess(sum_of_parts, diff)
        self.assertLess(diff-sum_of_parts, 100)

        # Adding other attributes
        m._test_value = 123456789
        size3 = sys.getsizeof(m)

        # Check that size increases
        self.assertLess(size2, size3)

        # Size difference should be slightly more than the size of key & value
        # The difference is overhead for the dict
        diff = size3 - size2
        sum_of_parts = sys.getsizeof('_test_value') + sys.getsizeof(123456789)
        self.assertLess(sum_of_parts, diff)
        self.assertLess(diff-sum_of_parts, 100)

        # Add list into parameters
        m.parameters['test_list'] = [79534, 84392, 93333, -243934, 23233]
        size4 = sys.getsizeof(m)

        # Check that size increases
        self.assertLess(size3, size4)

        # Size difference should be slightly more than the size of key & value
        # The difference is overhead for the dict
        diff = size4 - size3
        sum_of_parts = sys.getsizeof('test_list') + sys.getsizeof([79534, 84392, 93333, -243934, 23233])
        self.assertLess(sum_of_parts, diff)

        # Note that adding as a parameter is expected to be similar
        # This is because it uses the same logic in the callback

        # Adding something already seen
        m.parameters['recursive'] = m
        size5 = sys.getsizeof(m)

        # Check that size increases (for key)
        self.assertLess(size4, size5)

        # Size difference should be slightly more than the size of key
        # size of value is skipped because it is already seen
        # The difference is overhead for the dict
        diff = size5 - size4
        sum_of_parts = sys.getsizeof('recursive')
        self.assertLess(sum_of_parts, diff)
        self.assertLess(diff-sum_of_parts, 100)

    def test_templates(self):
        import prog_model_template
        m = prog_model_template.ProgModelTemplate()

    def test_derived(self):
        m = MockModelWithDerived()
        self.assertAlmostEqual(m.parameters['p1'], 1.2, 5)
        self.assertAlmostEqual(m.parameters['p2'], 1.2, 5)
        self.assertAlmostEqual(m.parameters['p3'], 1.2, 5)
        self.assertAlmostEqual(m.parameters['p4'], -2.4, 5)

        m.parameters['p1'] = 2.4
        self.assertAlmostEqual(m.parameters['p1'], 2.4, 5)
        self.assertAlmostEqual(m.parameters['p2'], 2.4, 5)
        self.assertAlmostEqual(m.parameters['p3'], 2.4, 5)
        self.assertAlmostEqual(m.parameters['p4'], -4.8, 5)

        m.parameters['p2'] = 5
        self.assertAlmostEqual(m.parameters['p1'], 2.4, 5)  # No change
        self.assertAlmostEqual(m.parameters['p2'], 5, 5)
        self.assertAlmostEqual(m.parameters['p3'], 5, 5)
        self.assertAlmostEqual(m.parameters['p4'], -10, 5)

    def test_broken_models(self):


        class missing_states(PrognosticsModel):
            inputs = ['i1', 'i2']
            outputs = ['o1']
            parameters = {'process_noise': 0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        

        class empty_states(PrognosticsModel):
            states = []
            inputs = ['i1', 'i2']
            outputs = ['o1']
            parameters = {'process_noise': 0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        

        class missing_inputs(PrognosticsModel):
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise': 0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        

        class missing_outputs(PrognosticsModel):
            states = ['x1', 'x2']
            inputs = ['i1']
            parameters = {'process_noise': 0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        

        class missing_initialize(PrognosticsModel):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise': 0.1}
            def next_state(self, x, u, dt):
                pass
            def output(self, x):
                pass
        

        class missing_output(PrognosticsModel):
            inputs = ['i1']
            states = ['x1', 'x2']
            outputs = ['o1']
            parameters = {'process_noise': 0.1}
            def initialize(self, u, z):
                pass
            def next_state(self, x, u, dt):
                pass

        with self.assertRaises(TypeError):
            m = missing_states()

        m = empty_states()
        self.assertEqual(len(m.states), 0)

        m = missing_inputs()
        self.assertEqual(len(m.inputs), 0)

        m = missing_outputs()
        self.assertEqual(len(m.outputs), 0)

        m = missing_initialize()
        # Should work- initialize is now optional

        m = missing_output()
        # Should work- output is now optional

    def __noise_test(self, noise_key, dist_key, keys):
        m = MockProgModel(**{noise_key: 0.0})
        for key in keys:
            self.assertIn(key, m.parameters[noise_key])
            self.assertAlmostEqual(m.parameters[noise_key][key], 0.0)

        i = 0
        noise = {}
        for key in keys:
            noise[key] = i
            i += 1
        m = MockProgModel(**{noise_key: noise})
        for key in keys:
            self.assertIn(key, m.parameters[noise_key])
            self.assertAlmostEqual(m.parameters[noise_key][key], noise[key])

        def add_one(self, x):
            return {key: value + 1 for (key, value) in x.items()}
        m = MockProgModel(**{noise_key: add_one})
        x = getattr(m, "apply_{}".format(noise_key))({key: 1 for key in keys})
        self.assertEqual(x[keys[0]], 2)

        with self.assertRaises(Exception):
            noise = []
            m = MockProgModel(**{noise_key: noise})

        # Test that it ignores process_noise_dist in case where process_noise is a function
        m = MockProgModel(**{noise_key: add_one, dist_key: 'invalid one'})
        x = getattr(m, "apply_{}".format(noise_key))({key: 1 for key in keys})
        self.assertEqual(x[keys[0]], 2)

        # Invalid dist
        with self.assertRaises(ValueError):
            noise = {key: 0.0 for key in keys}
            m = MockProgModel(**{noise_key: noise, dist_key: 'invalid one'})

        # Invalid dist
        with self.assertRaises(ValueError):
            m = MockProgModel(**{noise_key: 0, dist_key: 'invalid one'})

        # Valid distributions
        m = MockProgModel(**{noise_key: 0, dist_key: 'uniform'})
        m = MockProgModel(**{noise_key: 0, dist_key: 'gaussian'})
        m = MockProgModel(**{noise_key: 0, dist_key: 'normal'})
        m = MockProgModel(**{noise_key: 0, dist_key: 'triangular'})

    def test_isdiscrete_iscontinuous(self):
        m = ThrownObject()
        self.assertTrue(m.is_discrete)
        self.assertFalse(m.is_continuous)

        m = BatteryElectroChemEOD()
        self.assertTrue(m.is_continuous)
        self.assertFalse(m.is_discrete)
        
    def test_process_noise(self):
        self.__noise_test('process_noise', 'process_noise_dist', MockProgModel.states)

        m = MockProgModel()

        # All states except for the last one
        noise = {key: 1 for key in list(m.states)[:-1]}
        m.parameters['process_noise'] = noise
        for key in list(m.states)[:-1]:
            self.assertEqual(m.parameters['process_noise'][key], 1)
        # That key should be 0 (default)
        self.assertEqual(m.parameters['process_noise'][list(m.states)[-1]], 0)

        # Const noise
        m.parameters['process_noise_dist'] = 'constant'
        x = m.StateContainer({key: 0 for key in m.states})
        x = m.apply_process_noise(x)
        for key in list(m.states)[:-1]:
            self.assertAlmostEqual(x[key], 1.0)
        self.assertAlmostEqual(x[list(m.states)[-1]], 0.0)

    def test_measurement_noise(self):
        self.__noise_test('measurement_noise', 'measurement_noise_dist', MockProgModel.outputs)

        m = MockProgModel()

        # All outputs except for the last one
        noise = {key: 1 for key in list(m.outputs)[:-1]}
        m.parameters['measurement_noise'] = noise
        for key in list(m.outputs)[:-1]:
            self.assertEqual(m.parameters['measurement_noise'][key], 1)
        # That key should be 0 (default)
        self.assertEqual(m.parameters['measurement_noise'][list(m.outputs)[-1]], 0)

    def test_prog_model(self):
        m = MockProgModel() # Should work- sets default
        m = MockProgModel(process_noise = 0.0)
        x0 = m.initialize()
        self.assertSetEqual(set(x0.keys()), set(m.parameters['x0'].keys()))
        for key, value in m.parameters['x0'].items():
            self.assertEqual(value, x0[key])
        x = m.next_state(x0, {'i1': 1, 'i2': 2.1}, 0.1)
        self.assertAlmostEqual(x['a'], 1.1, 6)
        self.assertAlmostEqual(x['c'], -5.3, 6)
        self.assertEqual(x['b'], 5)
        z = m.output(x)
        self.assertAlmostEqual(z['o1'], 0.8, 5)
        e = m.event_state({'t': 0})
        t = m.threshold_met({'t': 0})
        self.assertAlmostEqual(e['e1'], 1.0, 5)
        self.assertFalse(t['e1'])
        e = m.event_state({'t': 5})
        self.assertAlmostEqual(e['e1'], 0.0, 5)
        t = m.threshold_met({'t': 5})
        self.assertTrue(t['e1'])
        t = m.threshold_met({'t': 10})
        self.assertTrue(t['e1'])

    def test_default_es_and_tm(self):
        # Test 1: TM only
        class NoES(MockModel, PrognosticsModel):
            events = ['e1', 'e2']

            def threshold_met(self, _):
                return {'e1': False, 'e2': True}

        m = NoES()

        self.assertDictEqual(m.threshold_met({}), {'e1': False, 'e2': True})
        self.assertDictEqual(m.event_state({}), {'e1': 1.0, 'e2': 0.0})

        # Test 2: ES only
        class NoTM(MockModel, PrognosticsModel):
            events = ['e1', 'e2']

            def event_state(self, _):
                return {'e1': 0.0, 'e2': 1.0}
        
        m = NoTM()

        self.assertDictEqual(m.threshold_met({}), {'e1': True, 'e2': False})
        self.assertDictEqual(m.event_state({}), {'e1': 0.0, 'e2': 1.0})

        # Test 3: Neither ES or TM 
        class NoESTM(MockModel, PrognosticsModel):
            events = []

        m = NoESTM()
        self.assertDictEqual(m.threshold_met({}), {})
        self.assertDictEqual(m.event_state({}), {})

    def test_pickle(self):
        m = MockProgModel(p1=1.3)
        pickle.dump(m, open('model_test.pkl', 'wb'))
        m2 = pickle.load(open('model_test.pkl', 'rb'))
        isinstance(m2, MockProgModel)
        self.assertEqual(m.parameters['p1'], m2.parameters['p1'])
        self.assertEqual(m, m2)

    def test_sim_to_thresh(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Any event, default
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0)
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # Any event, initial state 
        x0 = {'a': 1, 'b': 5, 'c': -3.2, 't': -1}
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, x=x0)
        self.assertAlmostEqual(times[-1], 6.0, 5)
        self.assertAlmostEqual(states[0]['t'], -1.0, 5)

        # Any event, manual
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, events=['e1', 'e2'])
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # Any event, manual - specified strategy
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, events=['e1', 'e2'], event_strategy='first')
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # both e1, e2
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, events=['e1', 'e2'], event_strategy='all')
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # Any event, manual - unexpected strategy
        with self.assertRaises(ValueError):
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, events=['e1', 'e2'], event_strategy='fljsdk')

        # Only event 2
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, events=['e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # Only event 2 - threshold keys
        with self.assertWarns(Warning):
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, threshold_keys=['e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # Threshold before event
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, horizon=5.0, events='e2')
        self.assertAlmostEqual(times[-1], 5.0, 5)

        # Threshold after event
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, horizon=20.0, events=['e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # No thresholds
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, horizon=20.0, events=[])
        self.assertAlmostEqual(times[-1], 20.0, 5)

        # No thresholds and no horizon
        with self.assertRaises(ValueError):
            (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, events=[])

        # No events and no horizon
        m_noevents = OneInputNoOutputNoEventLM()
        with self.assertRaises(ValueError):
            (times, inputs, states, outputs, event_states) = m_noevents.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0)

        # Custom thresholds met eqn- both keys
        def thresh_met(thresholds):
            return all(thresholds.values())
        config = {'dt': 0.5, 'save_freq': 1.0, 'horizon': 20.0, 'thresholds_met_eqn': thresh_met}
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(load, {'o1': 0.8}, **config, events=['e1', 'e2'])
        self.assertAlmostEqual(times[-1], 15.0, 5)

        # With no events and no horizon, but a threshold met eqn
        # Should still run
        def thresh_met(thresholds):
            return True
        linear_load = lambda t, x=None: m_noevents.InputContainer({'u1': 1})
        (times, inputs, states, outputs, event_states) = m_noevents.simulate_to_threshold(linear_load, {'o1': 0.8}, dt=0.5, save_freq=1.0, thresholds_met_eqn=thresh_met)
        self.assertListEqual(times, [0, 0.5])  # Only one step

        with self.assertRaises(ValueError):
            result = m.simulate_to_threshold(load, {'o1': 0.8}, events=['e1', 'e2', 'e3'], dt=0.5, save_freq=1.0)

    def test_sim_past_thresh(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        result = m.simulate_to(6, load, {'o1': 0.8}, dt=0.5, save_freq=1.0)
        self.assertAlmostEqual(result.times[-1], 6.0, 5)

    def test_sim_namedtuple_access(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        z = {'o1': 0.8}
        (times, inputs, states, outputs, event_states) = m.simulate_to(6, load, z, dt=0.5, save_freq=1.0)
        named_results = m.simulate_to(6, load, z, dt=0.5, save_freq=1.0)
        self.assertEqual(times, named_results.times)
        self.assertEqual(inputs, named_results.inputs)
        self.assertEqual(states, named_results.states)
        self.assertEqual(outputs, named_results.outputs)
        self.assertEqual(event_states, named_results.event_states)
        
    def test_next_time_fcn(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Any event, default
        result = m.simulate_to_threshold(load, {'o1': 0.8}, dt=1, save_freq=1e-99)
        self.assertEqual(len(result.times), 6)

        def next_time(t, x):
            return 0.5

        # With next_time
        result = m.simulate_to_threshold(load, {'o1': 0.8}, save_freq=1e-99, dt=next_time)
        self.assertEqual(len(result.times), 11)

    def test_sim_measurement_noise(self):
        m = MockProgModel(process_noise=0.0, measurement_noise=1)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Simulate
        (times, inputs, states, outputs, event_states) = m.simulate_to(3.5, load, {'o1': 0.8}, dt=0.5, save_freq=1.0)

        # Check times
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 5)
        self.assertAlmostEqual(times[-1], 3.5, 5)  # Save last step (even though it's not on a savepoint)
        
        # Check inputs
        self.assertEqual(len(inputs), 5)
        for i in inputs:
            i0 = {'i1': 1, 'i2': 2.1}
            for key in i.keys():
                self.assertEqual(i[key], i0[key], "Future loading error")

        # Check states
        self.assertEqual(len(states), 5)
        a = [1, 2, 3, 4, 4.5]
        c = [-3.2, -7.4, -11.6, -15.8, -17.9]
        for (ai, ci, x) in zip(a, c, states):
            self.assertAlmostEqual(x['a'], ai, 5)
            self.assertEqual(x['b'], 5)
            self.assertAlmostEqual(x['c'], ci, 5)

        # Check outputs
        self.assertEqual(len(outputs), 5)
        o = [2.8, -0.4, -3.6, -6.8, -8.4]
        for (oi, z) in zip(o, outputs): 
            # Noise will make output not equal the expected
            self.assertNotEqual(round(z['o1'], 6), round(oi, 6))

        # Now with no measurement Noise
        m = MockProgModel(process_noise = 0.0, measurement_noise = 0.0)
        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Simulate
        (times, inputs, states, outputs, event_states) = m.simulate_to(3.5, load, {'o1': 0.8}, dt=0.5, save_freq=1.0)

        # Check times
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 5)
        self.assertAlmostEqual(times[-1], 3.5, 5)  # Save last step (even though it's not on a savepoint)
        
        # Check inputs
        self.assertEqual(len(inputs), 5)
        for i in inputs:
            i0 = {'i1': 1, 'i2': 2.1}
            for key in i.keys():
                self.assertEqual(i[key], i0[key], "Future loading error")

        # Check states
        self.assertEqual(len(states), 5)
        a = [1, 2, 3, 4, 4.5]
        c = [-3.2, -7.4, -11.6, -15.8, -17.9]
        for (ai, ci, x) in zip(a, c, states):
            self.assertAlmostEqual(x['a'], ai, 5)
            self.assertEqual(x['b'], 5)
            self.assertAlmostEqual(x['c'], ci, 5)

        # Check outputs
        self.assertEqual(len(outputs), 5)
        o = [2.8, -0.4, -3.6, -6.8, -8.4]
        for (oi, z) in zip(o, outputs): 
            # Lack of noise will make output as expected
            self.assertEqual(round(z['o1'], 6), round(oi, 6))

            
    def test_sim_prog(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        
        # Check inputs
        (times, inputs, states, outputs, event_states) = m.simulate_to(0, load, {'o1': 0.8})
        self.assertEqual(len(times), 1)

        with self.assertRaises(ValueError):
            m.simulate_to(-30, load, {'o1': 0.8})

        with self.assertRaises(ValueError):
            m.simulate_to([12], load, {'o1': 0.8})

        with self.assertRaises(ValueError):
            m.simulate_to(12, load, {'o2': 0.9})

        with self.assertRaises(ValueError):
            m.simulate_to(12, 132, {'o1': 0.8})

        # Simulate
        (times, inputs, states, outputs, event_states) = m.simulate_to(3.5, load, {'o1': 0.8}, dt=0.5, save_freq=1.0)

        # Check times
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 5)
        self.assertAlmostEqual(times[-1], 3.5, 5) # Save last step (even though it's not on a savepoint)
        
        # Check inputs
        self.assertEqual(len(inputs), 5)
        for i in inputs:
            i0 = {'i1': 1, 'i2': 2.1}
            for key in i.keys():
                self.assertEqual(i[key], i0[key], "Future loading error")

        # Check states
        self.assertEqual(len(states), 5)
        a = [1, 2, 3, 4, 4.5]
        c = [-3.2, -7.4, -11.6, -15.8, -17.9]
        for (ai, ci, x) in zip(a, c, states):
            self.assertAlmostEqual(x['a'], ai, 5)
            self.assertEqual(x['b'], 5)
            self.assertAlmostEqual(x['c'], ci, 5)

        # Check outputs
        self.assertEqual(len(outputs), 5)
        o = [2.8, -0.4, -3.6, -6.8, -8.4]
        for (oi, z) in zip(o, outputs):
            self.assertAlmostEqual(z['o1'], oi, 5)

        # Check event_states
        self.assertEqual(len(event_states), 5)
        e = [1.0, 0.8, 0.6, 0.4, 0.3]
        for (ei, es) in zip(e, event_states):
            self.assertAlmostEqual(es['e1'], ei, 5)

        # Check last state saving
        (times, inputs, states, outputs, event_states) = m.simulate_to(3, load, {'o1': 0.8}, dt=0.5, save_freq=1.0)
        for t in range(0, 4):
            self.assertAlmostEqual(times[t], t, 5)
        self.assertEqual(len(times), 4, "Should be 4 elements in times") # Didn't save last state (because same as savepoint)

        # Check dt > save_freq
        (times, inputs, states, outputs, event_states) = m.simulate_to(3, load, {'o1': 0.8}, dt=0.5, save_freq=0.1)
        for t in range(0, 7):
            self.assertAlmostEqual(times[t], t/2, 5)
        self.assertEqual(len(times), 7, "Should be 7 elements in times") # Didn't save last state (because same as savepoint)

        # Custom Savepoint test - with last state saving
        (times, inputs, states, outputs, event_states) = m.simulate_to(3, load, {'o1': 0.8}, dt=0.5, save_freq=99.0, save_pts=[1.45, 2.45])
        # Check times
        self.assertAlmostEqual(times[0], 0, 5)
        self.assertAlmostEqual(times[1], 1.5, 5)
        self.assertAlmostEqual(times[2], 2.5, 5)
        self.assertEqual(len(times), 4)
        self.assertAlmostEqual(times[-1], 3.0, 5) # Save last step (even though it's not on a savepoint)
        
        # Custom Savepoint test
        (times, inputs, states, outputs, event_states) = m.simulate_to(2.5, load, {'o1': 0.8}, dt=0.5, save_freq=99.0, save_pts=[1.45, 2.45])
        # Check times
        self.assertAlmostEqual(times[0], 0, 5)
        self.assertAlmostEqual(times[1], 1.5, 5)
        self.assertAlmostEqual(times[2], 2.5, 5)
        self.assertEqual(len(times), 3)
        # Last step is a savepoint        

    def test_vectorization(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        a = np.array([1, 2, 3, 4, 4.5])
        b = np.array([5]*5)
        c = np.array([-3.2, -7.4, -11.6, -15.8, -17.9])
        t = np.array([0, 0.5, 1, 1.5, 2])
        dt = 0.5
        x0 = {'a': deepcopy(a), 'b': deepcopy(b), 'c': deepcopy(c), 't': deepcopy(t)}
        x = m.next_state(x0, load(0), dt)
        for xa, xa0 in zip(x['a'], a):
            self.assertAlmostEqual(xa, xa0+dt)

    def test_sim_prog_inproper_config(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}
        
        # Check inputs
        with self.assertRaises(TypeError):
            result = m.simulate_to(0, load, {'o1': 0.8}, dt=[1, 2])

        with self.assertRaises(ValueError):
            result = m.simulate_to(0, load, {'o1': 0.8}, dt=-1)

        with self.assertRaises(TypeError):
            result = m.simulate_to(0, load, {'o1': 0.8}, save_freq=[1, 2])

        with self.assertRaises(ValueError):
            result = m.simulate_to(0, load, {'o1': 0.8}, save_freq=-1)

        with self.assertRaises(TypeError):
            result = m.simulate_to_threshold(load, {'o1': 0.8}, horizon=[1, 2])

        with self.assertRaises(ValueError):
            result = m.simulate_to_threshold(load, {'o1': 0.8}, horizon=-1)
        
        with self.assertRaises(TypeError):
            result = m.simulate_to_threshold(load, {'o1': 0.8}, thresholds_met_eqn=-1)

        # incorrect number of arguments
        t_met = lambda a, b: print(a, b)
        with self.assertRaises(ValueError):
            result = m.simulate_to_threshold(load, {'o1': 0.8}, thresholds_met_eqn=t_met)

    def test_sim_modes(self):
        m = ThrownObject(process_noise=0, measurement_noise=0)

        def load(t, x=None):
            return m.InputContainer({})

        # Default mode should be auto
        result = m.simulate_to_threshold(load, save_freq=0.75, save_pts=[1.5, 2.5])
        self.assertListEqual(result.times, [0, 0.75, 1.5, 2.25, 2.5, 3, 3.75])  

        # Auto step size
        result = m.simulate_to_threshold(load, dt='auto', save_freq=0.75, save_pts=[1.5, 2.5])
        self.assertListEqual(result.times, [0, 0.75, 1.5, 2.25, 2.5, 3, 3.75])  

        # Auto step size with a max of 2
        result = m.simulate_to_threshold(load, dt=('auto', 2), save_freq=0.75, save_pts=[1.5, 2.5])
        self.assertListEqual(result.times, [0, 0.75, 1.5, 2.25, 2.5, 3, 3.75])  

        # Constant step size of 2
        result = m.simulate_to_threshold(load, dt=('constant', 2), save_freq=0.75, save_pts=[1.5, 2.5])
        self.assertListEqual(result.times, [0, 2, 4])  

        # Constant step size of 2
        result = m.simulate_to_threshold(load, dt=2, save_freq=0.75, save_pts=[1.5, 2.5])
        self.assertListEqual(result.times, [0, 2, 4])  

        # With save_freq==0
        result = m.simulate_to_threshold(load, dt=2, save_freq=0)
        self.assertListEqual(result.times, [0, 2, 4])

        # With save_freq==0 and auto step size
        result = m.simulate_to_threshold(load, dt=('auto', 2), save_freq=0)
        self.assertListEqual(result.times, [0, 2, 4])

        result = m.simulate_to_threshold(load, dt=2, save_pts=[2.5])
        self.assertListEqual(result.times, [0, 4])  

    def test_sim_rk4(self):
        # With non-linear model
        m = ThrownObject()

        def load(t, x=None):
            return m.InputContainer({})
        
        with self.assertRaises(TypeError):
            m.simulate_to_threshold(load, integration_method='rk4')

        # With linear model
        m = LinearThrownObject(process_noise=0, measurement_noise=0)

        result = m.simulate_to_threshold(load, dt = 0.1, integration_method='rk4')
        self.assertAlmostEqual(result.times[-1], 8.3)

    # when range specified when state doesn't exist or entered incorrectly
    def test_state_limits(self):
        m = MockProgModel()
        m.state_limits = {
            't': (-100, 100)
        }
        x0 = m.initialize()

        def load(t, x=None):
            return m.InputContainer({'i1': 1, 'i2': 2.1})

        # inside bounds using simulate_to
        x0['t'] = 0
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        # now using the apply_limits function
        x0['t'] = 0
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], 0, 9)

        # outside low boundary
        x0['t'] = -200
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)
        self.assertAlmostEqual(states[1]['t'], -100)

        x0['t'] = -200
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], -100, 9)

        # outside high boundary
        x0['t'] = 200
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)
        self.assertAlmostEqual(states[1]['t'], 100)

        x0['t'] = 200
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], 100, 9)

        # at low boundary
        x0['t'] = -100
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        x0['t'] = -100
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], -100, 9)

        # at high boundary
        x0['t'] = 100
        (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)
        self.assertGreaterEqual(states[1]['t'], -100)
        self.assertLessEqual(states[1]['t'], 100)

        x0['t'] = 100
        x = m.apply_limits(x0)
        self.assertAlmostEqual(x['t'], 100, 9)

        # Vectorized inputs - high and low
        x0 = m.StateContainer(np.array([[1]*3, [5]*3, [-3.2]*3,[50, -150, 125]]))
        x = m.apply_limits(x0)
        self.assertListEqual(list(x['t']), [50, -100, 100])

        # when state doesn't exist
        with self.assertRaises(Exception):
            x0['n'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)

        # when state entered incorrectly
        with self.assertRaises(Exception):
            x0['t'] = 'f'
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)

        # when boundary entered incorrectly
        with self.assertRaises(Exception):
            m.state_limits = { 't': ('f', 100) }
            x0['t'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)

        with self.assertRaises(Exception):
            m.state_limits = { 't': (-100, 'f') }
            x0['t'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)

        with self.assertRaises(Exception):
            m.state_limits = { 't': (100) }
            x0['t'] = 0
            (times, inputs, states, outputs, event_states) = m.simulate_to(0.001, load, {'o1': 0.8}, x=x0)

    def test_progress_bar(self):
        m = MockProgModel(process_noise=0.0)

        def load(t, x=None):
            return {'i1': 1, 'i2': 2.1}

        # Define output redirection
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        # Test progress bar matching
        simulate_results = m.simulate_to_threshold(load, {'o1': 0.8}, dt=0.5, save_freq=1.0, print=False, progress=True)
        capture_split =  [l+"%" for l in capturedOutput.getvalue().split("%") if l][:11]
        percentage_vals = [0, 9, 19, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(capture_split)):
            actual = '%s |%s| %s%% %s' % ("Progress", "█" * percentage_vals[i] + '-' * (100 - percentage_vals[i]), str(percentage_vals[i])+".0","")
            self.assertEqual(capture_split[i].strip(), actual.strip())
        
    def test_containers(self):
        m = ThrownObject()
        c1 = m.StateContainer({'x': 1.7, 'v': 40})
        c2 = m.StateContainer(np.array([[1.7], [40]]))
        self.assertEqual(c1, c2)
        self.assertListEqual(list(c1.keys()), m.states)

        input_c1 = m.InputContainer({})
        input_c2 = m.InputContainer(np.array([]))
        self.assertEqual(input_c1, input_c2)
        self.assertListEqual(list(input_c1.keys()), m.inputs)

        output_c1 = m.OutputContainer({'x': 1.7})
        output_c2 = m.OutputContainer(np.array([[1.7]]))
        self.assertEqual(output_c1, output_c2)
        self.assertListEqual(list(output_c1.keys()), m.outputs)

    def test_thrown_object_drag(self):
        def future_load(t, x=None):
            return {}
        event = 'impact'
        m_nd = ThrownObject(process_noise_dist='none')
        
        # Create no drag model ('cd' = 0)
        m_nd.parameters['cd'] = 0
        simulated_results_nd = m_nd.simulate_to_threshold(future_load, events=[event], dt=0.005, save_freq=1)
        # Create default drag model ('cd' = 0.007)
        m_df = ThrownObject(process_noise_dist='none')
        simulated_results_df = m_df.simulate_to_threshold(future_load, events=[event], dt=0.005, save_freq=1)
        # Create high drag model ('cd' = 1.0)
        m_hi = ThrownObject(process_noise_dist='none')
        m_hi.parameters['cd'] = 1
        simulated_results_hi = m_hi.simulate_to_threshold(future_load, events=[event], dt=0.005, save_freq=1)

        # Test no drag simulated results different from default
        self.assertNotEqual(simulated_results_nd.times, simulated_results_df.times)
        self.assertNotEqual(simulated_results_nd.states, simulated_results_df.states)
        self.assertGreater(simulated_results_nd.times[-1], simulated_results_df.times[-1])

        # Test high drag simulated results different from default
        self.assertNotEqual(simulated_results_hi.times, simulated_results_df.times)
        self.assertNotEqual(simulated_results_hi.states, simulated_results_df.states)
        self.assertLess(simulated_results_hi.times[-1], simulated_results_df.times[-1])

        # Test high drag simulated results different from no drag
        self.assertNotEqual(simulated_results_hi.times, simulated_results_nd.times)
        self.assertNotEqual(simulated_results_hi.states, simulated_results_nd.states)

    def test_composite_broken(self):
        m1 = OneInputOneOutputNoEventLM()
        # Insufficient number of models
        with self.assertRaises(ValueError):
            CompositeModel([])
        with self.assertRaises(ValueError):
            CompositeModel([m1])
        
        # Wrong type
        with self.assertRaises(ValueError):
            CompositeModel([m1, m1, 'abc'])

        # Incorrect named format
        with self.assertRaises(ValueError):
            # Too many elements
            CompositeModel([('a', m1, 'Something else'), ('b', m1)])
        with self.assertRaises(ValueError):
            # Not a string
            CompositeModel([(m1, m1)])
        with self.assertRaises(ValueError):
            # Not a model
            CompositeModel([('a', 'b')])
        with self.assertRaises(ValueError):
            # Too few elements
            CompositeModel([(m1, )])

        # Incorrect connections
        with self.assertRaises(ValueError):
            # without model name
            CompositeModel([m1, m1], connections=[('z1', 'u1')])
        with self.assertRaises(ValueError):
            # broken in
            CompositeModel([m1, m1], connections=[('z1', 'OneInputOneOutputNoEventLM.u1')])
        with self.assertRaises(ValueError):
            # broken out
            CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.z1', 'u1')])
        with self.assertRaises(ValueError):
            # Switched
            CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.u1', 'OneInputOneOutputNoEventLM_2.z1')])
        with self.assertRaises(ValueError):
            # Improper format - too long
            CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.z1', 'OneInputOneOutputNoEventLM.u1', 'Something else')])
        with self.assertRaises(ValueError):
            # Improper format - not a string
            CompositeModel([m1, m1], connections=[(m1, m1)])
        with self.assertRaises(ValueError):
            # Improper format - too short
            CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.z1', )])
        with self.assertRaises(ValueError):
            # Improper format - not a tuple
            CompositeModel([m1, m1], connections=['m1'])

        # Incorrect outputs
        with self.assertRaises(ValueError):
            # without model name
            CompositeModel([m1, m1], outputs=['z1'])
        with self.assertRaises(ValueError):
            # extra
            CompositeModel([m1, m1], outputs=['OneInputOneOutputNoEventLM.z1', 'OneInputOneOutputNoEventLM_2.z1', 'z1'])

    def test_composite(self):        
        m1 = OneInputOneOutputNoEventLM()
        m2 = OneInputNoOutputOneEventLM()
        m1_withpm = OneInputOneOutputNoEventLMPM()

        # Test with no connections
        m_composite = CompositeModel([m1, m1])
        self.assertSetEqual(m_composite.states, {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        self.assertSetEqual(m_composite.inputs, {'OneInputOneOutputNoEventLM.u1', 'OneInputOneOutputNoEventLM_2.u1'})
        self.assertSetEqual(m_composite.outputs, {'OneInputOneOutputNoEventLM.z1', 'OneInputOneOutputNoEventLM_2.z1'})
        self.assertSetEqual(m_composite.events, set())
        self.assertSetEqual(m_composite.performance_metric_keys, set(), "Shouldn't have any performance metrics")

        x0 = m_composite.initialize()
        self.assertSetEqual(set(x0.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        self.assertEqual(x0['OneInputOneOutputNoEventLM_2.x1'], 0)
        self.assertEqual(x0['OneInputOneOutputNoEventLM.x1'], 0)
        # Only provide non-zero input for the first model
        u = m_composite.InputContainer({'OneInputOneOutputNoEventLM.u1': 1, 'OneInputOneOutputNoEventLM_2.u1': 0})
        x = m_composite.next_state(x0, u, 1)
        self.assertSetEqual(set(x.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 0)
        self.assertEqual(x['OneInputOneOutputNoEventLM.x1'], 1)
        z = m_composite.output(x)
        self.assertSetEqual(set(z.keys()), {'OneInputOneOutputNoEventLM_2.z1', 'OneInputOneOutputNoEventLM.z1'})
        self.assertEqual(z['OneInputOneOutputNoEventLM_2.z1'], 0)
        self.assertEqual(z['OneInputOneOutputNoEventLM.z1'], 1)
        pm = m_composite.performance_metrics(x)
        self.assertSetEqual(set(pm.keys()), set())

        # With Performance Metrics
        # Everything else should behave the same, so we're only testing the performance metrics
        m_composite = CompositeModel([m1_withpm, m1_withpm])
        self.assertSetEqual(m_composite.performance_metric_keys, {'OneInputOneOutputNoEventLMPM_2.x1+1', 'OneInputOneOutputNoEventLMPM.x1+1'})

        x0 = m_composite.initialize()
        u = m_composite.InputContainer({'OneInputOneOutputNoEventLMPM.u1': 1, 'OneInputOneOutputNoEventLMPM_2.u1': 0})
        x = m_composite.next_state(x0, u, 1)
        pm = m_composite.performance_metrics(x)
        self.assertSetEqual(set(pm.keys()), {'OneInputOneOutputNoEventLMPM_2.x1+1', 'OneInputOneOutputNoEventLMPM.x1+1'})
        self.assertEqual(pm['OneInputOneOutputNoEventLMPM_2.x1+1'], 1)
        self.assertEqual(pm['OneInputOneOutputNoEventLMPM.x1+1'], 2)

        # Test with connections - output, no event
        m_composite = CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.z1', 'OneInputOneOutputNoEventLM_2.u1')])
        # Additional state to store output
        self.assertSetEqual(m_composite.states, {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1', 'OneInputOneOutputNoEventLM.z1'})
        # One less input - since it's internally connected
        self.assertSetEqual(m_composite.inputs, {'OneInputOneOutputNoEventLM.u1',})
        self.assertSetEqual(m_composite.outputs, {'OneInputOneOutputNoEventLM.z1', 'OneInputOneOutputNoEventLM_2.z1'})
        self.assertSetEqual(m_composite.events, set())

        x0 = m_composite.initialize()
        self.assertSetEqual(set(x0.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1', 'OneInputOneOutputNoEventLM.z1'})
        self.assertEqual(x0['OneInputOneOutputNoEventLM_2.x1'], 0)
        self.assertEqual(x0['OneInputOneOutputNoEventLM.x1'], 0)
        self.assertEqual(x0['OneInputOneOutputNoEventLM.z1'], 0)
        # Only provide non-zero input for first model
        u = m_composite.InputContainer({'OneInputOneOutputNoEventLM.u1': 1})
        x = m_composite.next_state(x0, u, 1)
        self.assertSetEqual(set(x.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1', 'OneInputOneOutputNoEventLM.z1'})
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 1) # Propagates through, because of the order. If the connection were the other way it wouldn't
        self.assertEqual(x['OneInputOneOutputNoEventLM.x1'], 1)
        z = m_composite.output(x)
        self.assertSetEqual(set(z.keys()), {'OneInputOneOutputNoEventLM_2.z1', 'OneInputOneOutputNoEventLM.z1'})
        self.assertEqual(z['OneInputOneOutputNoEventLM_2.z1'], 1)
        self.assertEqual(z['OneInputOneOutputNoEventLM.z1'], 1)

        # Propagate again
        x = m_composite.next_state(x, u, 1)
        self.assertSetEqual(set(x.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1', 'OneInputOneOutputNoEventLM.z1'})
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 3) # 1 + 2
        self.assertEqual(x['OneInputOneOutputNoEventLM.x1'], 2)

        # Test with connections - state, no event
        m_composite = CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.x1', 'OneInputOneOutputNoEventLM_2.u1')])
        # No additional state to store output, since state is used for the connection
        self.assertSetEqual(m_composite.states, {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        # One less input - since it's internally connected
        self.assertSetEqual(m_composite.inputs, {'OneInputOneOutputNoEventLM.u1',})
        self.assertSetEqual(m_composite.outputs, {'OneInputOneOutputNoEventLM.z1', 'OneInputOneOutputNoEventLM_2.z1'})
        self.assertSetEqual(m_composite.events, set())
        
        x0 = m_composite.initialize()
        self.assertSetEqual(set(x0.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        self.assertEqual(x0['OneInputOneOutputNoEventLM_2.x1'], 0)
        self.assertEqual(x0['OneInputOneOutputNoEventLM.x1'], 0)
        # Only provide non-zero input for model 1
        u = m_composite.InputContainer({'OneInputOneOutputNoEventLM.u1': 1})
        x = m_composite.next_state(x0, u, 1)
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 1) # Propagates through, because of the order. If the connection were the other way it wouldn't
        self.assertEqual(x['OneInputOneOutputNoEventLM.x1'], 1)
        z = m_composite.output(x)
        self.assertEqual(z['OneInputOneOutputNoEventLM_2.z1'], 1)
        self.assertEqual(z['OneInputOneOutputNoEventLM.z1'], 1)

        # Propagate again
        x = m_composite.next_state(x, u, 1)
        self.assertSetEqual(set(x.keys()), {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 3)  # 1 + 2
        self.assertEqual(x['OneInputOneOutputNoEventLM.x1'], 2)

        # Test with connections - two events
        m_composite = CompositeModel([m2, m2], connections=[('OneInputNoOutputOneEventLM.x1', 'OneInputNoOutputOneEventLM_2.u1')])
        self.assertSetEqual(m_composite.states, {'OneInputNoOutputOneEventLM_2.x1', 'OneInputNoOutputOneEventLM.x1'})
        # One less input - since it's internally connected
        self.assertSetEqual(m_composite.inputs, {'OneInputNoOutputOneEventLM.u1',})
        self.assertSetEqual(m_composite.outputs, set())
        self.assertSetEqual(m_composite.events, {'OneInputNoOutputOneEventLM.x1 == 10', 'OneInputNoOutputOneEventLM_2.x1 == 10'})

        x0 = m_composite.initialize()
        u = m_composite.InputContainer({'OneInputNoOutputOneEventLM.u1': 1})
        x = m_composite.next_state(x0, u, 1)  # 1, 1
        x = m_composite.next_state(x, u, 1)  # 2, 3
        x = m_composite.next_state(x, u, 1)  # 3, 6
        tm = m_composite.threshold_met(x)
        self.assertSetEqual(set(tm.keys()), {'OneInputNoOutputOneEventLM.x1 == 10', 'OneInputNoOutputOneEventLM_2.x1 == 10'})
        self.assertFalse(tm['OneInputNoOutputOneEventLM.x1 == 10'])
        self.assertFalse(tm['OneInputNoOutputOneEventLM_2.x1 == 10'])

        x = m_composite.next_state(x, u, 1)  # 4, 10
        es = m_composite.event_state(x)
        self.assertSetEqual(set(es.keys()), {'OneInputNoOutputOneEventLM.x1 == 10', 'OneInputNoOutputOneEventLM_2.x1 == 10'})
        self.assertEqual(es['OneInputNoOutputOneEventLM.x1 == 10'], 0.6)
        self.assertEqual(es['OneInputNoOutputOneEventLM_2.x1 == 10'], 0.0)
        tm = m_composite.threshold_met(x)
        self.assertSetEqual(set(tm.keys()), {'OneInputNoOutputOneEventLM.x1 == 10', 'OneInputNoOutputOneEventLM_2.x1 == 10'})
        self.assertFalse(tm['OneInputNoOutputOneEventLM.x1 == 10'])
        self.assertTrue(tm['OneInputNoOutputOneEventLM_2.x1 == 10'])

        # Test with outputs specified
        m_composite = CompositeModel([m1, m1], connections=[('OneInputOneOutputNoEventLM.x1', 'OneInputOneOutputNoEventLM_2.u1')], outputs=['OneInputOneOutputNoEventLM_2.z1'])
        self.assertSetEqual(m_composite.states, {'OneInputOneOutputNoEventLM_2.x1', 'OneInputOneOutputNoEventLM.x1'})
        self.assertSetEqual(m_composite.inputs, {'OneInputOneOutputNoEventLM.u1',})
        # One less output
        self.assertSetEqual(set(m_composite.outputs), {'OneInputOneOutputNoEventLM_2.z1', })
        self.assertSetEqual(m_composite.events, set())
        x0 = m_composite.initialize()
        z = m_composite.output(x0)
        self.assertSetEqual(set(z.keys()), {'OneInputOneOutputNoEventLM_2.z1', })

        # With Names
        m_composite = CompositeModel([('m1', m1), ('m2', m2)], connections=[('m1.x1', 'm2.u1')])
        self.assertSetEqual(m_composite.states, {'m2.x1', 'm1.x1'})
        self.assertSetEqual(m_composite.inputs, {'m1.u1',})
        self.assertSetEqual(m_composite.outputs, {'m1.z1', })
        self.assertSetEqual(m_composite.events, {'m2.x1 == 10', })
    
    # Fill parameters with different types of objects instead
    def test_parameter_equality(self):
        m1 = LinearThrownObject()
        m2 = LinearThrownObject()

        self.assertTrue(m1.parameters == m2.parameters) #Checking to see if the parameters are equal
        self.assertTrue(m2.parameters == m1.parameters) #Parameters should be equal

        m3 = LinearThrownDiffThrowingSpeed() # A model with a different throwing speed
        self.assertFalse(m1.parameters == m3.parameters)
        self.assertFalse(m3.parameters == m1.parameters) # Checking both directions 

        m4 = LinearThrownObjectDiffDefaultParams() # Model with an extra default parameter.

        self.assertFalse(m1.parameters == m4.parameters)
        self.assertFalse(m4.parameters == m1.parameters) # checking both directions

        m5 = LinearThrownObjectUpdatedInitializedMethod() # Model with incorrectly initialized throwing height, but same parameters

        self.assertFalse(m1.parameters == m5.parameters) 
        self.assertFalse(m5.parameters == m1.parameters) 

        self.assertTrue(m1.parameters == m2.parameters) # Checking to see previous equal statements stay the same
        self.assertTrue(m2.parameters == m1.parameters) 

    def test_direct_params(self):
        m1 = LinearThrownObject()
        print('test')

        # Accessing parameters directly
        self.assertEqual(m1.parameters['g'], m1['g'])

        # Setting parameters
        m1['g'] *= 2  # Doubling 
        self.assertEqual(m1.parameters['g'], m1['g'])

        # Updating from parameters
        m1.parameters['g'] *= 2
        self.assertEqual(m1.parameters['g'], m1['g'])

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Base Models")
    result = runner.run(load_test.loadTestsFromTestCase(TestModels)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
