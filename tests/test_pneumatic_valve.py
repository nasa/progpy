# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
from numpy import array
import sys
import unittest

from progpy.models.pneumatic_valve import PneumaticValve, PneumaticValveWithWear, PneumaticValveBase


class TestPneumaticValve(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    @unittest.skip
    def test_pneumatic_valve_vectorized(self):
        m = PneumaticValveWithWear(process_noise=0)

        cycle_time = 20

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2:
                return {
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Open Valve
                    'uTop': False,
                    'uBot': True
                }
            return {
                'pL': 3.5e5,
                'pR': 2.0e5,
                # Close Valve
                'uTop': True,
                'uBot': False
            }

        x0 = {
            'x': array([0]*4),
            'v': array([0]*4),
            'Ai': array([0]*4),
            'r': array([6000]*4),
            'k': array([48000]*4),
            'Aeb': array([1e-5]*4),
            'Aet': array([1e-5]*4),
            'condition': array([1]*4),
            'mTop': array([0.069, 0.048, 0.027, 0.06]),
            'mBot': array([9.45e-4, 0.008, 0.016, 0.024]),
            'pDiff': array([1.5e5]*4),
            'wb': array([0]*4),
            'wi': array([0]*4),
            'wk': array([0]*4),
            'wr': array([0]*4),
            'wt': array([0]*4)
        }

        m.next_state(x0, future_loading(0), 0.1)

    def test_pneumatic_valve_with_wear(self):
        # Test using PneumaticValveWithWear
        m = PneumaticValveWithWear(process_noise=0)

        cycle_time = 20

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2:
                return {
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Open Valve
                    'uTop': False,
                    'uBot': True
                }
            return {
                'pL': 3.5e5,
                'pR': 2.0e5,
                # Close Valve
                'uTop': True,
                'uBot': False
            }

        u0 = future_loading(0)
        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'condition': 1,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 0,
            'wt': 0
        }

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'condition': 1,
            'k': 48000,
            'mBot': 0.008534524658,
            'mTop': 0.048044198904,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 0,
            'wt': 0
        }
        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = m.output(x)
        z_test = {
            "Q": 0,
            "iB": True, 
            "iT": False,
            "pB": 0.91551734,
            "pT": 3.7319387914459899,
            "x": 0
        }
        for key in m.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 7)

        m.parameters['x0']['wr'] = 1

        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 1,
            'wt': 0
        }

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'k': 48000,
            'mBot': 8.5345246578678786736e-3,
            'mTop': 0.048044198903933490206,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 1,
            'wt': 0
        }
        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        config = {'dt': 0.01, 'horizon': 800, 'save_freq': 60}
        (times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_loading, m.output(m.initialize(future_loading(0))), **config)# , 'save_freq': 60
        self.assertAlmostEqual(times[-1], 782.53, 0)

        # Test warning when setting overwritten parameters
        with self.assertWarns(UserWarning):
            m.parameters['wb'] = 1

        with self.assertWarns(UserWarning):
            m.parameters['wi'] = 1

        with self.assertWarns(UserWarning):
            m.parameters['wk'] = 1

        with self.assertWarns(UserWarning):
            m.parameters['wr'] = 1

        with self.assertWarns(UserWarning):
            m.parameters['wt'] = 1

    def test_pneumatic_valve_base(self):
        # Test using PneumaticValveBase
        m = PneumaticValveBase(process_noise=0)
        param = m.parameters

        cycle_time = 20

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2:
                return {
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Open Valve
                    'uTop': False,
                    'uBot': True
                }
            return {
                'pL': 3.5e5,
                'pR': 2.0e5,
                # Close Valve
                'uTop': True,
                'uBot': False
            }

        u0 = future_loading(0)
        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'condition': 1,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR']
        }
        param['wb'] = 0
        param['wi'] = 0
        param['wk'] = 0
        param['wr'] = 0
        param['wt'] = 0

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'condition': 1,
            'k': 48000,
            'mBot': 0.008534524658,
            'mTop': 0.048044198904,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR']
        }
        param['wb'] = 0
        param['wi'] = 0
        param['wk'] = 0
        param['wr'] = 0
        param['wt'] = 0

        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = m.output(x)
        z_test = {
            "Q": 0,
            "iB": True,
            "iT": False,
            "pB": 0.91551734,
            "pT": 3.7319387914459899,
            "x": 0
        }
        for key in m.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 7)

        param['x0']['wr'] = 1

        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR']
        }
        param['wb'] = 0
        param['wi'] = 0
        param['wk'] = 0
        param['wr'] = 1
        param['wt'] = 0

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'k': 48000,
            'mBot': 8.5345246578678786736e-3,
            'mTop': 0.048044198903933490206,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR']
        }
        param['wb'] = 0
        param['wi'] = 0
        param['wk'] = 0
        param['wr'] = 1
        param['wt'] = 0

        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        config = {'dt': 0.01, 'horizon': 800, 'save_freq': 60}
        results = m.simulate_to_threshold(future_loading, m.output(m.initialize(future_loading(0))), **config)
        self.assertAlmostEqual(results.times[-1], 782.53, 0)
        
    def test_pneumatic_valve(self):
        self.assertEqual(PneumaticValve, PneumaticValveWithWear)

    def test_pneumatic_valve_namedtuple_access(self):
        m = PneumaticValveWithWear(process_noise=0)
        cycle_time = 20

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time/2:
                return {
                    'pL': 3.5e5,
                    'pR': 2.0e5,
                    # Open Valve
                    'uTop': False,
                    'uBot': True
                }
            return {
                'pL': 3.5e5,
                'pR': 2.0e5,
                # Close Valve
                'uTop': True,
                'uBot': False
            }

        u0 = future_loading(0)
        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'condition': 1,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 0,
            'wt': 0
        }

        for key in m.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        
        x = m.next_state(x0, future_loading(0), 0.1)
        x_test = {
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'Ai': 0,
            'condition': 1,
            'k': 48000,
            'mBot': 0.008534524658,
            'mTop': 0.048044198904,
            'r': 6000,
            'v': 0,
            'x': 0,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 0,
            'wt': 0
        }
        for key in m.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = m.output(x)
        z_test = {
            "Q": 0,
            "iB": True,
            "iT": False,
            "pB": 0.91551734,
            "pT": 3.7319387914459899,
            "x": 0
        }
        for key in m.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 7)

        m.parameters['x0']['wr'] = 1

        x0 = m.initialize(u0)
        x0_test = {
            'x': 0,
            'v': 0,
            'Ai': 0,
            'r': 6000,
            'k': 48000,
            'Aeb': 1e-5,
            'Aet': 1e-5,
            'mTop': 0.067876043046,
            'mBot': 9.445596253581e-4,
            'pDiff': u0['pL'] - u0['pR'],
            'wb': 0,
            'wi': 0,
            'wk': 0,
            'wr': 1,
            'wt': 0
        }

        config = {'dt': 0.01, 'horizon': 800, 'save_freq': 60}
        named_results = m.simulate_to_threshold(future_loading, m.output(m.initialize(future_loading(0))), **config)
        times = named_results.times
        inputs = named_results.inputs
        states = named_results.states
        outputs = named_results.outputs
        event_states = named_results.event_states

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Pneumatic Valve model")
    result = runner.run(load_test.loadTestsFromTestCase(TestPneumaticValve)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
