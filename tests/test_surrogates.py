# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import chaospy as cp
from io import StringIO
import sys
import unittest
import warnings

from progpy.data_models import PCE
from progpy.models import ThrownObject, BatteryElectroChemEOD, DCMotorSP
from progpy.models.test_models.linear_models import OneInputNoOutputOneEventLM, OneInputNoOutputTwoEventLM, TwoInputNoOutputOneEventLM, TwoInputNoOutputTwoEventLM


class TestSurrogate(unittest.TestCase):
    def setUp(self):
        # set stdout (so it wont print)
        sys.stdout = StringIO()
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def tearDown(self):
        sys.stdout = sys.__stdout__
        warnings.filterwarnings("default", category=DeprecationWarning)

    def test_surrogate_improper_input(self):
        m = ThrownObject()

        def load_eqn(t=None, x=None):
            return m.InputContainer({})
        
        with self.assertRaises(TypeError):
            m.generate_surrogate(None)
        with self.assertRaises(ValueError):
            m.generate_surrogate([])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], method='invalid')
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], save_pts=[10])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], trim_data_to=-1)
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], trim_data_to='invalid')
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], trim_data_to=1.5)
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], state_keys=['invalid'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], state_keys=['x', 'invalid', 'v'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], input_keys=['invalid'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], output_keys=['invalid'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], output_keys=['invalid', 'x'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], event_keys=['invalid'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], event_keys=['falling', 'impact', 'invalid'])
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], stability_tol=-1)
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], stability_tol='invalid')
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], training_noise=-1)
        with self.assertRaises(ValueError):
            m.generate_surrogate([load_eqn], training_noise=['invalid'])
    
    def test_surrogate_basic_thrown_object(self):
        m = ThrownObject(process_noise=0, measurement_noise=0)

        def load_eqn(t=None, x=None):
            return m.InputContainer({})
        
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', training_noise=0)
        self.assertEqual(surrogate.dt, 0.25)

        self.assertListEqual(surrogate.states, [stateTest for stateTest in m.states if (stateTest not in m.outputs and stateTest not in m.events)] + m.outputs + m.events)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'events': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate.parameters['measurement_noise'] = 0
        surrogate.parameters['process_noise'] = 0
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)

        MSE = m.calc_error(surrogate_results.times, surrogate_results.inputs, surrogate_results.outputs)
        self.assertLess(MSE, 10)  # Pretty good approx

        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

    def test_surrogate_basic_battery(self):
        m = BatteryElectroChemEOD(process_noise=0)

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
            return m.InputContainer({'i': i})
        
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
            return m.InputContainer({'i': i})
        load_functions = [future_loading_1, future_loading_2]

        options_surrogate = {
            'save_freq': 1,  # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1,  # For DMD, this value is the time step of the training data
            'state_keys': ['Vsn', 'Vsp', 'tb'],  # Define internal states to be included in surrogate model
            'trim_data_to': 0.7,  # Trim data to this fraction of the time series
            'output_keys': ['v'],  # Define outputs to be included in surrogate model 
            'training_noise': 0
        }

        surrogate = m.generate_surrogate(load_functions, **options_surrogate)
        self.assertSetEqual(set(surrogate.states), set(['tb', 'Vsn', 'Vsp'] + options_surrogate['output_keys'] + m.events))
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, options_surrogate['output_keys'])
        self.assertListEqual(surrogate.events, m.events)

        options_sim = {
            'save_freq': 1,  # Frequency at which results are saved, or equivalently time step in results
            'dt': 0.1,
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
            return m.InputContainer({'i': i})

        result = m.simulate_to_threshold(future_loading, **options_sim)
        surrogate_results = surrogate.simulate_to_threshold(future_loading, **options_sim)

        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=250)
        MSE = m.calc_error(surrogate_results.times, surrogate_results.inputs, surrogate_results.outputs)
        self.assertLess(MSE, 0.02) # Pretty good approx
    
    def test_surrogate_subsets(self):
        m = ThrownObject(process_noise=0, measurement_noise=0)
        def load_eqn(t=None, x=None):
            return m.InputContainer({})

        # Perfect subset
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', state_keys=['x', 'v'], training_noise=0)
        surrogate.parameters['process_noise'] = 0
        surrogate.parameters['measurement_noise'] = 0
        self.assertEqual(surrogate.dt, 0.25)

        self.assertListEqual(surrogate.states, [stateTest for stateTest in m.states if (stateTest not in m.inputs and stateTest not in m.outputs and stateTest not in m.events)] + m.outputs + m.events)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'events': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])
        
        # State subset
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', state_keys=['v'], training_noise=0)
        self.assertListEqual(surrogate.states, ['v'] + m.outputs + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', state_keys='v', training_noise=0)
        self.assertListEqual(surrogate.states, ['v'] + m.outputs + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'events': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

        # Events subset
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', event_keys=['impact'], training_noise=0)
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', event_keys='impact', training_noise=0)
        self.assertListEqual(surrogate.states, [stateTest for stateTest in m.states if (stateTest not in m.inputs and stateTest not in m.outputs and stateTest not in m.events)] + m.outputs + ['impact'] + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, m.outputs)
        self.assertListEqual(surrogate.events, ['impact'])

        options = {
            'events': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertEqual(surrogate_results.states[i]['x'], surrogate_results.outputs[i]['x'])
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1.25)
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

        # Outputs - Empty
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', output_keys=[], training_noise=0)
        self.assertListEqual(surrogate.states, m.states + m.events + m.inputs)
        self.assertListEqual(surrogate.inputs, m.inputs)
        self.assertListEqual(surrogate.outputs, [])
        self.assertListEqual(surrogate.events, m.events)

        options = {
            'events': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }
        result = m.simulate_to_threshold(load_eqn, **options)
        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        self.assertAlmostEqual(surrogate_results.times[-1], result.times[-1], delta=0.26)
        for i in range(min(len(result.times), len(surrogate_results.times))):
            self.assertListEqual(list(surrogate_results.inputs[i].keys()), list(surrogate_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_results.states[i]['x'], result.states[i]['x'], delta=8)
            self.assertAlmostEqual(surrogate_results.states[i]['v'], result.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_results.states[i]['falling'], result.event_states[i]['falling'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['falling'], surrogate_results.event_states[i]['falling'])
            self.assertAlmostEqual(surrogate_results.states[i]['impact'], result.event_states[i]['impact'], delta=0.1)
            self.assertEqual(surrogate_results.states[i]['impact'], surrogate_results.event_states[i]['impact'])

    def test_surrogate_thrown_object_with_noise(self):
        m = ThrownObject()
        def load_eqn(t=None, x=None):
            return m.InputContainer({})
        
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', training_noise=0)
        surrogate_noise = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', training_noise=0.01)
        self.assertEqual(surrogate.dt, 0.25)

        self.assertListEqual(surrogate.states, surrogate_noise.states)
        self.assertListEqual(surrogate.inputs, surrogate_noise.inputs)
        self.assertListEqual(surrogate.outputs, surrogate_noise.outputs)
        self.assertListEqual(surrogate.events, surrogate_noise.events)

        options = {
            'events': 'impact',
            'save_freq': 0.25,
            'dt': 0.25
        }

        surrogate_results = surrogate.simulate_to_threshold(load_eqn, **options)
        surrogate_noise_results = surrogate_noise.simulate_to_threshold(load_eqn, **options)

        position = [surrogate_results.states[iter]['x'] for iter in range(len(surrogate_results.times))]
        position_noise = [surrogate_noise_results.states[iter]['x'] for iter in range(len(surrogate_noise_results.times))]

        MSE = sum([(position[iter] - position_noise[iter])**2 for iter in range(min(len(position),len(position_noise)))])/min(len(position),len(position_noise))
        self.assertLess(MSE, 10) 
        self.assertNotEqual(MSE, 0)

        self.assertAlmostEqual(surrogate_results.times[-1], surrogate_noise_results.times[-1], delta=0.26)
        for i in range(min(len(surrogate_results.times), len(surrogate_noise_results.times))):
            self.assertListEqual(list(surrogate_noise_results.inputs[i].keys()), list(surrogate_noise_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_noise_results.states[i]['x'], surrogate_results.states[i]['x'], delta=6)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['v'], surrogate_results.states[i]['v'], delta=1)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['falling'], surrogate_results.states[i]['falling'], delta=0.1)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['impact'], surrogate_results.states[i]['impact'], delta=0.5)
            self.assertEqual(surrogate_noise_results.states[i]['x'], surrogate_noise_results.outputs[i]['x'])
            self.assertEqual(surrogate_noise_results.states[i]['falling'], surrogate_noise_results.event_states[i]['falling'])
            self.assertEqual(surrogate_noise_results.states[i]['impact'], surrogate_noise_results.event_states[i]['impact'])
            self.assertAlmostEqual(surrogate_noise_results.states[i]['falling'], surrogate_results.event_states[i]['falling'], delta=0.1)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['impact'], surrogate_results.event_states[i]['impact'], delta=0.5)
            if i > 0:
                self.assertNotEqual(surrogate_noise_results.states[i]['x'] - surrogate_results.states[i]['x'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['v'] - surrogate_results.states[i]['v'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['falling'] - surrogate_results.states[i]['falling'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['impact'] - surrogate_results.states[i]['impact'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['falling']-surrogate_results.event_states[i]['falling'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['impact'], surrogate_results.event_states[i]['impact'], 0)
                
    def test_surrogate_battery_with_noise(self):
        m = BatteryElectroChemEOD(process_noise=0)
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
            return m.InputContainer({'i': i})
        
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
            return m.InputContainer({'i': i})
        load_functions = [future_loading_1, future_loading_2]

        options_surrogate = {
            'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1, # For DMD, this value is the time step of the training data
            'trim_data_to': 0.7, # Trim data to this fraction of the time series
            'output_keys': ['v'], # Define outputs to be included in surrogate model 
            'training_noise': 0
        }
        options_surrogate_noise = {
            'save_freq': 1, # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1, # For DMD, this value is the time step of the training data
            'trim_data_to': 0.7, # Trim data to this fraction of the time series
            'output_keys': ['v'], # Define outputs to be included in surrogate model 
            'training_noise': 0.02
        }

        surrogate = m.generate_surrogate(load_functions, **options_surrogate)
        surrogate_noise = m.generate_surrogate(load_functions, **options_surrogate_noise)
        
        self.assertListEqual(surrogate.states, surrogate_noise.states)
        self.assertListEqual(surrogate.inputs, surrogate_noise.inputs)
        self.assertListEqual(surrogate.outputs, surrogate_noise.outputs)
        self.assertListEqual(surrogate.events, surrogate_noise.events)

        options_sim = {
            'save_freq': 1, # Frequency at which results are saved, or equivalently time step in results
            'dt': 0.1,
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
            return m.InputContainer({'i': i})

        surrogate_results = surrogate.simulate_to_threshold(future_loading, **options_sim)
        surrogate_noise_results = surrogate_noise.simulate_to_threshold(future_loading, **options_sim)

        voltage = [surrogate_results.states[iter]['v'] for iter in range(len(surrogate_results.times))]
        voltage_noise = [surrogate_noise_results.states[iter]['v'] for iter in range(len(surrogate_noise_results.times))]

        MSE = sum([(voltage[iter] - voltage_noise[iter])**2 for iter in range(min(len(voltage),len(voltage_noise)))])/min(len(voltage),len(voltage_noise))
        self.assertLess(MSE, 0.02) # Pretty good approx            
        self.assertNotEqual(MSE, 0)

        self.assertAlmostEqual(surrogate_results.times[-1], surrogate_noise_results.times[-1], delta=20)
        for i in range(min(len(surrogate_results.times), len(surrogate_noise_results.times))):
            self.assertListEqual(list(surrogate_noise_results.inputs[i].keys()), list(surrogate_noise_results.inputs[i].keys()))
            self.assertAlmostEqual(surrogate_noise_results.states[i]['tb'], surrogate_results.states[i]['tb'], delta=10)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['Vo'], surrogate_results.states[i]['Vo'], delta=0.1)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['Vsn'], surrogate_results.states[i]['Vsn'], delta=0.5)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['Vsp'], surrogate_results.states[i]['Vsp'], delta=0.1)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['qnB'], surrogate_results.states[i]['qnB'], delta=55)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['qnS'], surrogate_results.states[i]['qnS'], delta=8)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['qpB'], surrogate_results.states[i]['qpB'], delta=55)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['qpS'], surrogate_results.states[i]['qpS'], delta=8)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['v'], surrogate_results.states[i]['v'], delta=0.3)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['EOD'], surrogate_results.states[i]['EOD'],delta=0.1)
            self.assertEqual(surrogate_noise_results.states[i]['v'], surrogate_noise_results.outputs[i]['v'])
            self.assertEqual(surrogate_noise_results.states[i]['EOD'], surrogate_noise_results.event_states[i]['EOD'])
            self.assertAlmostEqual(surrogate_noise_results.states[i]['v'], surrogate_results.outputs[i]['v'], delta=0.3)
            self.assertAlmostEqual(surrogate_noise_results.states[i]['EOD'], surrogate_results.event_states[i]['EOD'], delta=0.1)
            if i > 0:
                self.assertNotEqual(surrogate_noise_results.states[i]['tb'] - surrogate_results.states[i]['tb'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['Vo'] - surrogate_results.states[i]['Vo'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['Vsn'] - surrogate_results.states[i]['Vsn'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['Vsp'] - surrogate_results.states[i]['Vsp'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['qnB'] - surrogate_results.states[i]['qnB'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['qnS'] - surrogate_results.states[i]['qnS'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['qpB'] - surrogate_results.states[i]['qpB'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['qpS'] - surrogate_results.states[i]['qpS'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['v'] - surrogate_results.states[i]['v'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['EOD'] - surrogate_results.states[i]['EOD'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['v'] - surrogate_results.outputs[i]['v'], 0)
                self.assertNotEqual(surrogate_noise_results.states[i]['EOD'] - surrogate_results.event_states[i]['EOD'], 0)
    
    def test_surrogate_output_interp(self):
        m = BatteryElectroChemEOD(process_noise=0)
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
            return m.InputContainer({'i': i})
        
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
            return m.InputContainer({'i': i})
        load_functions = [future_loading_1, future_loading_2]

        options_surrogate = {
            'save_freq': 5, # For DMD, this value is the time step for which the surrogate model is generated
            'dt': 0.1, # For DMD, this value is the time step of the training data
            'trim_data_to': 0.7, # Trim data to this fraction of the time series
            'training_noise': 0
        }

        surrogate = m.generate_surrogate(load_functions, **options_surrogate)

        options_sim = {
            'save_freq': 3, # Frequency at which results are saved, or equivalently time step in results
            'save_pts': [7] # Add save points to check functionality 
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
            return m.InputContainer({'i': i})

        surrogate_results = surrogate.simulate_to_threshold(future_loading, **options_sim)
        self.assertEqual((surrogate_results.times[1]-surrogate_results.times[0]),options_sim['save_freq'])
        self.assertEqual(options_sim['save_pts'][0] in surrogate_results.times, True)

    def test_surrogate_options(self):
        m = ThrownObject()
        def load_eqn(t=None, x=None):
            return m.InputContainer({})
        
        # treat warnings as exceptions
        with self.assertWarns(UserWarning):
            surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, events = 'impact', state_keys = ['v'], training_noise = 0)

        warnings.filterwarnings("error")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # This is needed to check that warning doesn't occur
        try:
            # Set sufficiently large failure tolerance
            surrogate = m.generate_surrogate([load_eqn], dt = 0.1, save_freq = 0.25, events = 'impact', state_keys = ['v'], stability_tol=1e99)
        except Warning as w:
            if w is not DeprecationWarning:  # Ignore deprecation warnings
                self.fail('Warning raised')
        
        # Reset Warnings
        warnings.filterwarnings("default")
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def test_surrogate_use_error_cases(self):
        m = ThrownObject()
        def load_eqn(t=None, x=None):
            return m.InputContainer({})
        
        surrogate = m.generate_surrogate([load_eqn], dt=0.1, save_freq=0.25, events='impact', training_noise=0)

        # Reset warnings seen so warning will occur
        from progpy import exceptions
        exceptions.warnings_seen = set() 

        with self.assertWarns(Warning):
            surrogate.simulate_to_threshold(load_eqn, dt=0.05)

    def test_pce_no_input(self):
        m = ThrownObject()
        with self.assertRaises(ValueError):
            pce = PCE.from_model(m, m.initialize(), {}, [])

    def test_pce_no_state(self):
        m = DCMotorSP()
        with self.assertRaises(ValueError):
            pce = PCE.from_model(m, m.initialize(), {}, [])

    def _pce_tests(self, m):
        x0 = m.initialize()
        input_dists = {'u1': cp.Uniform(0.5, 2), 'u2': cp.Uniform(0.5, 2)}
        # This is to handle cases where there are <2 inputs
        input_dists = {key: input_dists[key] for key in m.inputs}
        pce = PCE.from_model(m, x0, input_dists, times=[i*10 for i in range(5)], N=250)
        pce_result = pce.time_of_event(x0, lambda t, x=None: pce.InputContainer({'u1': 1, 'u2': 0.75}))
        gt_result = m.time_of_event(x0, lambda t, x=None: m.InputContainer({'u1': 1, 'u2': 0.75}))
        for event in m.events:
            self.assertAlmostEqual(pce_result[event], gt_result[event], delta=1)

        input_dists = {'u1': cp.Normal(1, 0.5), 'u2': cp.Normal(0.75, 0.5)}
        # This is to handle cases where there are <2 inputs
        input_dists = {key: input_dists[key] for key in m.inputs}
        pce = PCE.from_model(m, x0, input_dists, times=[i*10 for i in range(6)], N=250)
        pce_result = pce.time_of_event(x0, lambda t, x=None: pce.InputContainer({'u1': 1, 'u2': 0.75}))
        gt_result = m.time_of_event(x0, lambda t, x=None: m.InputContainer({'u1': 1, 'u2': 0.75}))
        for event in m.events:
            self.assertAlmostEqual(pce_result[event], gt_result[event], delta=1)

        pce_result = pce.time_of_event(x0, lambda t, x=None: pce.InputContainer({'u1': 1.5, 'u2': 1}))
        gt_result = m.time_of_event(x0, lambda t, x=None: m.InputContainer({'u1': 1.5, 'u2': 1}))
        for event in m.events:
            self.assertAlmostEqual(pce_result[event], gt_result[event], delta=1.25)
    
    def test_pce_oneinput_oneevent(self):
        m = OneInputNoOutputOneEventLM()
        self._pce_tests(m)

    def test_pce_oneinput_twoevent(self):
        m = OneInputNoOutputTwoEventLM()
        self._pce_tests(m)

    def test_pce_twoinput_oneevent(self):
        m = TwoInputNoOutputOneEventLM()
        self._pce_tests(m) 

    def test_pce_twoinput_twoevent(self):
        m = TwoInputNoOutputTwoEventLM()
        self._pce_tests(m) 
            
# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Surrogate")
    result = runner.run(load_test.loadTestsFromTestCase(TestSurrogate)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
