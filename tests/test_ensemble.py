# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import numpy as np
import sys
import unittest

from progpy import EnsembleModel
from progpy.models.test_models.linear_models import OneInputOneOutputOneEventLM, OneInputOneOutputOneEventAltLM


class TestEnsemble(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_no_model(self):
        with self.assertRaises(ValueError):
            EnsembleModel([])

    def test_single_model(self):
        # An ensemble model with one model should raise an exception

        m = OneInputOneOutputOneEventLM()
        

        EnsembleModel([m])

    def test_wrong_type(self):
        # An ensemble model with a non-model should raise an exception

        m = OneInputOneOutputOneEventLM()
        with self.assertRaises(TypeError):
            EnsembleModel(m)
        with self.assertRaises(TypeError):
            EnsembleModel([m, 1])
        with self.assertRaises(TypeError):
            EnsembleModel(77)
        with self.assertRaises(TypeError):
            EnsembleModel([m, m, m, 77])

    def test_single_model(self):
        """
        This tests that the ensemble model works with a single model, ensuring that inputs, states, outputs, and events are correctly handled.
        """
        m = OneInputOneOutputOneEventLM()
        em = EnsembleModel([m])

        # inputs, states, outputs, and events should be the same as the single model
        self.assertSetEqual(set(em.inputs), {'u1'})
        self.assertSetEqual(set(em.states), {'x1'})
        self.assertSetEqual(set(em.outputs), {'z1'})
        self.assertSetEqual(set(em.events), {'x1 == 10'})

        # Initialize
        x_t0 = em.initialize()
        self.assertEqual(x_t0['x1'], 0)

        # State transition
        u = em.InputContainer({'u1': 1})
        x_t1 = em.next_state(x_t0, u, 1)
        self.assertEqual(x_t1['x1'], 1)

        # Output
        z = em.output(x_t1)
        self.assertEqual(z['z1'], 1)

        # Event state
        es = em.event_state(x_t1)
        self.assertEqual(es['x1 == 10'], 0.9)

        # Threshold met
        self.assertFalse(em.threshold_met(x_t1)['x1 == 10'])

        # Transition again
        x_t2 = em.next_state(x_t1, u, 2)

        # Threshold met
        # x1 == 3
        self.assertFalse(em.threshold_met(x_t2)['x1 == 10'])

        

    def test_two_models_identical(self):
        """
        This tests that the ensemble model works with two identical model-types with identical states, inputs, etc., one with slightly altered parameters.

        The result is that each state, output, etc. is combined using the specified aggregation method.
        """
        m = OneInputOneOutputOneEventLM()
        m2 = OneInputOneOutputOneEventLM(x0={'x1': 2})

        # Make sure they're not the same - matrices are 3x their original values
        # The result is a model where state changes 3x as fast.
        # Event state degrades 9x as fast, since B and F compound
        m2.B = np.array([[3]])
        m2.C = np.array([[3]])
        m2.F = np.array([[-0.3]])

        # An ensemble model with two models should work
        em = EnsembleModel([m, m2])

        # Since they're identical, the inputs, state, etc. should be the same
        self.assertEqual(em.inputs, m.inputs)
        self.assertEqual(em.states, m.states)
        self.assertEqual(em.outputs, m.outputs)
        self.assertEqual(em.events, m.events)

        # The resulting initial state should be exactly between the two:
        x_t0 = em.initialize()
        self.assertEqual(x_t0['x1'], 1)

        # Same with state transition
        u = em.InputContainer({'u1': 1})
        x_t1 = em.next_state(x_t0, u, 1)
        # m would give 2, m2 would give 4
        self.assertEqual(x_t1['x1'], 3)

        # Same with output
        z = em.output(x_t1)
        # m would give 3, m2 would give 9
        self.assertEqual(z['z1'], 6)

        # Same with event state
        es = em.event_state(x_t1)
        # m would give 0.7, m2 would give 0.1
        self.assertEqual(es['x1 == 10'], 0.4)

        # performance metrics
        pm = em.performance_metrics(x_t1)
        self.assertEqual(pm['pm1'], 4)

        # Time of event 
        toe = em.time_of_event(x_t0, lambda t, x=None: u, dt=1e-3)
        self.assertAlmostEqual(toe['x1 == 10'], 4.8895)

        # threshold met should be false
        self.assertFalse(em.threshold_met(x_t1)['x1 == 10'])

        # Transition again
        x_t2 = em.next_state(x_t1, u, 1)
        # threshold met should be true (because one of 2 models says it is)
        self.assertTrue(em.threshold_met(x_t2)['x1 == 10'])

    def test_two_models_different(self):
        """
        This tests that the ensemble model works with two different model-types with different states, inputs, etc. Tests how the ensemble model handles the different values.

        The result is that the different states, inputs, etc. are combined into a single set without being aggregated (unlike test_two_models_identical).
        """
        m = OneInputOneOutputOneEventLM()
        m2 = OneInputOneOutputOneEventAltLM()
        em = EnsembleModel([m, m2])

        # inputs, states, outputs, events should be a combination of the two models
        self.assertSetEqual(set(em.inputs), {'u1', 'u2'})
        self.assertSetEqual(set(em.states), {'x1', 'x2'})
        self.assertSetEqual(set(em.outputs), {'z1', 'z2'})
        self.assertSetEqual(set(em.events), {'x1 == 10', 'x2 == 5'})

        # Initialize - should be combination of the two
        x_t0 = em.initialize()
        self.assertEqual(x_t0['x1'], 0)
        self.assertEqual(x_t0['x2'], 0)

        # State transition - should be combination of the two
        u = em.InputContainer({'u1': 1, 'u2': 2})
        x_t1 = em.next_state(x_t0, u, 1)
        self.assertEqual(x_t1['x1'], 1)
        self.assertEqual(x_t1['x2'], 2)

        # Output - should be combination of the two
        z = em.output(x_t1)
        self.assertEqual(z['z1'], 1)
        self.assertEqual(z['z2'], 2)

        # Event state - should be combination of the two
        es = em.event_state(x_t1)
        self.assertEqual(es['x1 == 10'], 0.9)
        self.assertEqual(es['x2 == 5'], 0.6)

        # Threshold met - should be combination of the two
        self.assertFalse(em.threshold_met(x_t1)['x1 == 10'])
        self.assertFalse(em.threshold_met(x_t1)['x2 == 5'])

        # Transition again
        x_t2 = em.next_state(x_t1, u, 2)

        # Threshold met - should be combination of the two
        # x1 == 3, x2 == 6
        self.assertFalse(em.threshold_met(x_t2)['x1 == 10'])
        self.assertTrue(em.threshold_met(x_t2)['x2 == 5'])

    def test_two_models_alt_aggrigation(self):
        """
        This test repeats test_two_models_identical with different aggrigation method.
        """
        m = OneInputOneOutputOneEventLM()
        m2 = OneInputOneOutputOneEventLM(x0={'x1': 2})

        # Make sure they're not the same - 3x the impact
        m2.B = np.array([[3]])
        m2.C = np.array([[3]])
        m2.F = np.array([[-0.3]])

        # An ensemble model with two models should work
        em = EnsembleModel([m, m2], aggregation_method=np.max)

        # The resulting initial state should be max of the two:
        x_t0 = em.initialize()
        self.assertEqual(x_t0['x1'], 2)

        # Same with state transition
        u = em.InputContainer({'u1': 1})
        x_t1 = em.next_state(x_t0, u, 1)
        # m would give 3, m2 would give 5
        self.assertEqual(x_t1['x1'], 5)

        # Same with output
        z = em.output(x_t1)
        # m would give 5, m2 would give 15
        self.assertEqual(z['z1'], 15)

        # Same with event state
        es = em.event_state(x_t1)
        # m would give 0.5, m2 would give -0.5
        self.assertEqual(es['x1 == 10'], 0.5)

        # threshold met should be false
        self.assertFalse(em.threshold_met(x_t1)['x1 == 10'])

        # Next state
        x2 = em.next_state(x_t1, u, 2)
        # threshold met should be true (because both of the models agree)
        self.assertTrue(em.threshold_met(x2)['x1 == 10'])

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Ensemble models")
    result = runner.run(load_test.loadTestsFromTestCase(TestEnsemble)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
