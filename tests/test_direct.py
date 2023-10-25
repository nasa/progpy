# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import numpy as np
import sys
import time
import unittest

from progpy import PrognosticsModel
from progpy.models import ThrownObject


class TestDirect(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_inherited_direct(self):
        m = ThrownObject()
        self.assertFalse(m.is_direct)

        x = m.initialize()
        tic = time.perf_counter()
        m.time_of_event(x, dt = 0.05)
        toc = time.perf_counter()
        t_nondirect = toc - tic

        class DirectThrownObject(ThrownObject):
            def time_of_event(self, x, **kwargs):
                # calculate time when object hits ground given x['x'] and x['v']
                # 0 = x0 + v0*t - 0.5*g*t^2
                g = self.parameters['g']
                t_impact = -(x['v'] + np.sqrt(x['v']*x['v'] - 2*g*x['x']))/g
                # 0 = v0 - g*t
                t_falling = -x['v']/g
                return {'impact': t_impact, 'falling': t_falling}

        m_direct = DirectThrownObject()
        x = m_direct.initialize()
        self.assertTrue(m_direct.is_direct)
        tic = time.perf_counter()
        m_direct.time_of_event(x, dt=0.05)
        toc = time.perf_counter()
        t_direct = toc-tic
        # Direct should be at least 10x faster
        self.assertLess(t_direct, t_nondirect/10)

        # Direct should have same events, states, inputs, outputs as non-direct 
        self.assertListEqual(m_direct.events, m.events)
        self.assertListEqual(m_direct.states, m.states)
        self.assertListEqual(m_direct.inputs, m.inputs)
        self.assertListEqual(m_direct.outputs, m.outputs)

        # State Transition, Output, etc. should still work.
        u = m_direct.InputContainer({})
        x_direct = m_direct.next_state(x, u, dt=0.05)
        x = m.next_state(x, u, dt=0.05)
        self.assertEqual(x, x_direct)
        z_direct = m_direct.output(x)
        self.assertSetEqual(set(list(z_direct.keys())), set(m_direct.outputs))
        z = m.output(x)
        self.assertEqual(z, z_direct)
        es = m_direct.event_state(x)
        self.assertSetEqual(set(list(es.keys())), set(m_direct.events))

    def test_non_inherited_direct(self):
        # Test case where direct model is not inherited from state transition model and therefore ONLY has direct functionality
        class DirectInheritedThrownObject(ThrownObject):
            def time_of_event(self, x, **kwargs):
                # calculate time when object hits ground given x['x'] and x['v']
                # 0 = x0 + v0*t - 0.5*g*t^2
                g = self.parameters['g']
                t_impact = -(x['v'] + np.sqrt(x['v']*x['v'] - 2*g*x['x']))/g
                # 0 = v0 - g*t
                t_falling = -x['v']/g
                return {'impact': t_impact, 'falling': t_falling}

        class DirectNonInheritedThrownObject(PrognosticsModel):
            states = ['x', 'v']
            events = ['impact', 'falling']
            default_parameters = {
                'g': -9.81,
            }
            def time_of_event(self, x, **kwargs):
                # calculate time when object hits ground given x['x'] and x['v']
                # 0 = x0 + v0*t - 0.5*g*t^2
                g = self.parameters['g']
                t_impact = -(x['v'] + np.sqrt(x['v']*x['v'] - 2*g*x['x']))/g
                # 0 = v0 - g*t
                t_falling = -x['v']/g
                return {'impact': t_impact, 'falling': t_falling}

        m_inherited = DirectInheritedThrownObject()
        m_non_inherited = DirectNonInheritedThrownObject()
        self.assertTrue(m_non_inherited.is_direct)

        x = m_inherited.initialize()
        # Time of event should work the same
        self.assertEqual(m_inherited.time_of_event(x), m_non_inherited.time_of_event(x))

        # Using output should warn
        with self.assertWarns(UserWarning):
            # Should warn that outputs are not supported
            m_non_inherited.output(x)

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Direct models")
    result = runner.run(load_test.loadTestsFromTestCase(TestDirect)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
