# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from progpy import MixtureOfExpertsModel
from progpy.models.test_models.other_models import OneInputTwoOutputsOneEvent, OneInputTwoOutputsOneEvent_alt


class TestMoE(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def testSameModel(self):
        m1 = OneInputTwoOutputsOneEvent(a=2.3, b=0.75, c=0.75)
        m2 = OneInputTwoOutputsOneEvent(a=1.19) # best option
        m3 = OneInputTwoOutputsOneEvent(a=0.95, b=0.85, c=0.85)

        m_moe = MixtureOfExpertsModel((m1, m2, m3))
        self.assertSetEqual(set(m_moe.inputs), set(OneInputTwoOutputsOneEvent.inputs + OneInputTwoOutputsOneEvent.outputs))
        self.assertSetEqual(set(m_moe.outputs), set(OneInputTwoOutputsOneEvent.outputs))
        self.assertSetEqual(set(m_moe.events), set(OneInputTwoOutputsOneEvent.events))
        self.assertSetEqual(set(m_moe.states), {'OneInputTwoOutputsOneEvent.x0', 'OneInputTwoOutputsOneEvent_2.x0', 'OneInputTwoOutputsOneEvent_3.x0', 'OneInputTwoOutputsOneEvent._score', 'OneInputTwoOutputsOneEvent_2._score', 'OneInputTwoOutputsOneEvent_3._score'})

        x0 = m_moe.initialize()

        # Scores defaulted to 0.5
        self.assertEqual(x0['OneInputTwoOutputsOneEvent._score'], 0.5)
        self.assertEqual(x0['OneInputTwoOutputsOneEvent_2._score'], 0.5)
        self.assertEqual(x0['OneInputTwoOutputsOneEvent_3._score'], 0.5)

        # Next_state uses first model (since scores are equal)
        x = m_moe.next_state(x0, m_moe.InputContainer({'u0': 2}), 1)
        # Scores dont change, since outputs are not provided
        self.assertEqual(x['OneInputTwoOutputsOneEvent._score'], 0.5)
        self.assertEqual(x['OneInputTwoOutputsOneEvent_2._score'], 0.5)
        self.assertEqual(x['OneInputTwoOutputsOneEvent_3._score'], 0.5)
        z = m_moe.output(x)  # This is where it "chooses one"
        # Since scores are equal it should choose the first one
        # Which meanes x0 is 4.6 and b is 0.75
        self.assertEqual(z['x0+b'], 5.35)
        self.assertEqual(z['x0+c'], 5.35)

        # Update state with outputs
        x = m_moe.next_state(
            x,
            m_moe.InputContainer({ # Input corresponding to _2
                'u0': 2,
                'x0+b': 5.76,
                'x0+c': 5.
            }),
            1)
        # This time the scores changed so that _2 is the best one
        self.assertEqual(x['OneInputTwoOutputsOneEvent._score'], 0.49)
        self.assertEqual(x['OneInputTwoOutputsOneEvent_2._score'], 0.51)
        # _3 is somewhere in between the worst and best
        self.assertGreater(x['OneInputTwoOutputsOneEvent_3._score'], 0.49)
        self.assertLess(x['OneInputTwoOutputsOneEvent_3._score'], 0.51)

        z = m_moe.output(x)  # This is where it "chooses one"
        # Since scores are not equal it should choose the best one (_2)
        # Which meanes x0 is 4.76 and b is 1
        self.assertEqual(z['x0+b'], 5.76)
        self.assertEqual(z['x0+c'], 5.76)

        # Scaling (used to prevent saturation)
        x['OneInputTwoOutputsOneEvent_2._score'] = 0.999
        x = m_moe.next_state(
            x,
            m_moe.InputContainer({ # Input corresponding to _2
                'u0': 2,
                'x0+b': 8.14,
                'x0+c': 8.14
            }),
            1)
        self.assertEqual(x['OneInputTwoOutputsOneEvent._score'], (0.49-0.01)*0.8)
        self.assertEqual(x['OneInputTwoOutputsOneEvent_2._score'], (0.999+0.01)*0.8)
        self.assertGreater(x['OneInputTwoOutputsOneEvent_3._score'], 0.48*0.8)
        self.assertLess(x['OneInputTwoOutputsOneEvent_3._score'], 0.52*0.8)

    def test_heterogeneous_models(self):
        m1 = OneInputTwoOutputsOneEvent(a=2.3, b=0.75, c=0.75)
        m2 = OneInputTwoOutputsOneEvent(a=1.19) # best option
        m3 = OneInputTwoOutputsOneEvent_alt(a=1.17, d=0.85, c=0.85)  # different class

        m_moe = MixtureOfExpertsModel((m1, m2, m3))
        self.assertSetEqual(set(m_moe.inputs), set(OneInputTwoOutputsOneEvent.inputs + OneInputTwoOutputsOneEvent.outputs + OneInputTwoOutputsOneEvent_alt.outputs))
        self.assertSetEqual(set(m_moe.outputs), set(OneInputTwoOutputsOneEvent.outputs + OneInputTwoOutputsOneEvent_alt.outputs))
        self.assertSetEqual(set(m_moe.events), set(OneInputTwoOutputsOneEvent.events + OneInputTwoOutputsOneEvent_alt.events))
        self.assertSetEqual(set(m_moe.states), {'OneInputTwoOutputsOneEvent.x0', 'OneInputTwoOutputsOneEvent_2.x0', 'OneInputTwoOutputsOneEvent_alt.x0', 'OneInputTwoOutputsOneEvent._score', 'OneInputTwoOutputsOneEvent_2._score', 'OneInputTwoOutputsOneEvent_alt._score'})

        x0 = m_moe.initialize()

        # Next_state uses first model (since scores are equal)
        x = m_moe.next_state(x0, m_moe.InputContainer({'u0': 2}), 1)
        z = m_moe.output(x)  # This is where it "chooses one"
        # Since scores are equal it should choose the first one
        # Which meanes x0 is 4.6 and b, c are 0.75, and d is 0.85 (and x0 for that one is 2.34)
        self.assertEqual(z['x0+b'], 5.35)
        self.assertEqual(z['x0+c'], 5.35)
        self.assertEqual(z['x0+d'], 3.19)

        es = m_moe.event_state(x)
        self.assertEqual(es['x0==10'], 0.54) # 1 - 0.46/10 (uses x0 for model 1)
        self.assertEqual(es['x0==7'], 1-2.34/7) # (uses x0 for model 3)

        tm = m_moe.threshold_met(x)
        self.assertFalse(tm['x0==10'])
        self.assertFalse(tm['x0==7'])

        x['OneInputTwoOutputsOneEvent_alt.x0'] = 20 # Will only effect the state for model 3
        tm = m_moe.threshold_met(x)
        self.assertFalse(tm['x0==10'])
        self.assertTrue(tm['x0==7'])

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting MoE models")
    result = runner.run(load_test.loadTestsFromTestCase(TestMoE)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()