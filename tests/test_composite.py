# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest

from progpy import CompositeModel
from progpy.models.test_models.linear_models import (
    OneInputOneOutputNoEventLM,
    OneInputNoOutputOneEventLM,
    OneInputOneOutputNoEventLMPM,
    OneInputOneOutputOneEventLM)

class TestCompositeModel(unittest.TestCase):
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
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 3)  # 1 + 2
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
        self.assertEqual(x['OneInputOneOutputNoEventLM_2.x1'], 3) # 1 + 2
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
        x = m_composite.next_state(x0, u, 1) # 1, 1
        x = m_composite.next_state(x, u, 1) # 2, 3
        x = m_composite.next_state(x, u, 1) # 3, 6
        tm = m_composite.threshold_met(x)
        self.assertSetEqual(set(tm.keys()), {'OneInputNoOutputOneEventLM.x1 == 10', 'OneInputNoOutputOneEventLM_2.x1 == 10'})
        self.assertFalse(tm['OneInputNoOutputOneEventLM.x1 == 10'])
        self.assertFalse(tm['OneInputNoOutputOneEventLM_2.x1 == 10'])

        x = m_composite.next_state(x, u, 1) # 4, 10
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
        self.assertSetEqual(set(z.keys()), {'OneInputOneOutputNoEventLM_2.z1'})

        # With Names
        m_composite = CompositeModel([('m1', m1), ('m2', m2)], connections=[('m1.x1', 'm2.u1')])
        self.assertSetEqual(m_composite.states, {'m2.x1', 'm1.x1'})
        self.assertSetEqual(m_composite.inputs, {'m1.u1',})
        self.assertSetEqual(m_composite.outputs, {'m1.z1', })
        self.assertSetEqual(m_composite.events, {'m2.x1 == 10', })

    def test_composite_pm(self):
        m = OneInputOneOutputOneEventLM()
        m_composite = CompositeModel([m, m], connections=[('OneInputOneOutputOneEventLM_2.pm1', 'OneInputOneOutputOneEventLM.u1')])
        self.assertSetEqual(m_composite.states, {'OneInputOneOutputOneEventLM_2.pm1', 'OneInputOneOutputOneEventLM.x1', 'OneInputOneOutputOneEventLM_2.x1'})
        self.assertSetEqual(m_composite.inputs, {'OneInputOneOutputOneEventLM_2.u1',})
        x0 = m_composite.initialize()
        u = m_composite.InputContainer({'OneInputOneOutputOneEventLM_2.u1': 1})
        x = m_composite.next_state(x0, u, 1)
        x = m_composite.next_state(x0, u, 1)
        self.assertAlmostEqual(x['OneInputOneOutputOneEventLM.x1'], 3)  # extra 1 from pm
        self.assertAlmostEqual(x['OneInputOneOutputOneEventLM_2.x1'], 2)

def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Composite Models")
    result = runner.run(load_test.loadTestsFromTestCase(TestCompositeModel)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
