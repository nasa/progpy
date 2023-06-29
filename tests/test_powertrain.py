# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from progpy.models import ESC, DCMotor, Powertrain, PropellerLoad


class TestPowertrain(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__
        
    def test_propeller_load(self):
        m = PropellerLoad()

        # initial state
        x0 = m.initialize()
        self.assertSetEqual(set(x0.keys()), {'t_l'})
        self.assertEqual(x0['t_l'], 0)

        # Check state transition
        x = m.next_state(x0, m.InputContainer({'v_rot': 2}), None)
        self.assertSetEqual(set(x.keys()), {'t_l'})
        self.assertEqual(x['t_l'], m.parameters['C_q'] * 4)

        x = m.next_state(x0, m.InputContainer({'v_rot': -2}), None)
        self.assertEqual(x['t_l'], m.parameters['C_q'] * 4)

        # Check output
        z = m.output(x)
        self.assertSetEqual(set(z.keys()), {'t_l'})
        self.assertEqual(z['t_l'], x['t_l'])

        # Events
        self.assertSetEqual(set(m.events), set())

        # State limits
        x_limited = m.apply_limits(m.StateContainer({'t_l': -1}))
        self.assertEqual(x_limited['t_l'], 0)

        # Updating derived parameters 
        c_q = m.parameters['C_q']
        m.parameters['c_q'] *= 2
        self.assertEqual(m.parameters['C_q'], c_q * 2)

        m.parameters['D'] = 1
        self.assertEqual(m.parameters['C_q'], m.parameters['c_q'] * m.parameters['rho'])

    def test_powertrain(self):
        esc = ESC()
        motor = DCMotor()
        powertrain = Powertrain(esc, motor)
        def future_loading(t, x=None):
            return powertrain.InputContainer({
                'duty': 1,
                'v': 23
            })
        
        results = powertrain.simulate_to(2, future_loading, dt=2e-5, save_freq=0.1)
        # Add additional tests

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Powertrain model")
    result = runner.run(load_test.loadTestsFromTestCase(TestPowertrain)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
