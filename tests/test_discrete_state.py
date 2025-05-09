# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest

from progpy import create_discrete_state
from progpy.models.test_models.tank_model import Tank, ValveState

class TestDiscreteStates(unittest.TestCase):
    def test_random_transition(self):
        # Setup
        StateType = create_discrete_state(10)
        x = StateType(1)
        self.assertEqual(x, StateType(1))

        # Less than transition
        x += 0.4
        self.assertEqual(x, StateType(1))

        # greater than transition
        # Note - since this is random, it may not transition still. So here we check that it transitions more than more than 5 of 10 times (will on average transtion 90% of the time)
        transition_count = 0
        for _ in range(10):
            transition_count += x+0.51 != StateType(1)
        self.assertGreater(transition_count, 5)

        # greater than transition - negative
        # Should still transition for negative numbers so long as absolute value is >= 0.5
        # Note - since this is random, it may not transition still. So here we check that it transitions more than more than 5 of 10 times (will on average transtion 90% of the time)
        transition_count = 0
        for _ in range(10):
            transition_count += x-0.51 != StateType(1)
        self.assertGreater(transition_count, 5)

    def test_sequential_transition(self):
        # Setup
        StateType = create_discrete_state(10, transition='sequential')
        x = StateType(1)
        self.assertEqual(x, StateType(1))

        # Less than transition
        x += 0.4
        self.assertEqual(x, StateType(1))

        # greater than transition
        self.assertEqual(x + 0.51, StateType(2))
        self.assertEqual(x - 0.51, StateType(0))
        self.assertEqual(x + 1.51, StateType(3))

        # lower bound
        self.assertEqual(x - 1.51, StateType(0))
        
        # upper bound
        self.assertEqual(x + 10.51, StateType(9))
    
    def test_custom_transition(self):
        # Define transition function
        import random
        def transition(current_state, amount_added):
            # this is an example function- in reality it could be anything
            # Transition in this case is from 1-> any state and
            #  if not in state 1 can only transition back to 1
            if current_state == type(current_state)(1) and amount_added > 0.5:
                return random.randint(0, len(type(current_state))-1)
            elif amount_added > 0.5:
                return 1
            # No transition
            return current_state

        StateType = create_discrete_state(10, transition=transition)
        x = StateType(1)
        self.assertEqual(x, StateType(1))

        # Less than transition
        x += 0.4
        self.assertEqual(x, StateType(1))

        # From 1- 
        # greater than transition
        # Note - since this is random, it may not transition still. So here we check that it transitions more than more than 5 of 10 times (will on average transtion 90% of the time)
        transition_count = 0
        for _ in range(10):
            transition_count += x+0.51 != StateType(1)
        self.assertGreater(transition_count, 5)

        # transition from other number should always yeald 1
        for i in [0, *list(range(2, 10))]:
            self.assertEqual(StateType(i)+1, StateType(1))
    
    def test_tank(self):
        m = Tank()
        def stupid_load(t, x=None):
            # Doesn't open valve
            if x is None:
                return m.InputContainer({'q_in': 0.1, 'valve_command': ValveState.closed})
            return m.InputContainer({'q_in': 0.1, 'valve_command': x['valve']})
        
        result = m.simulate_to_threshold(stupid_load, events='full', save_freq=1, horizon=25)
        for x in result.states:
            self.assertEqual(x['valve'], ValveState.closed)

        def smart_load(t, x=None):
            if x is None:
                # First step
                return m.InputContainer({'q_in': 0.1, 'valve_command': ValveState.closed})

            if (x['valve'] == ValveState.closed) and (x['h'] >= m['height']*0.8):
                # If closed, open at 80% full
                return m.InputContainer({'q_in': 0.1, 'valve_command': ValveState.open})
            elif (x['valve'] == ValveState.open) and (x['h'] <= m['height']*0.6):
                # If open, close at 60% full
                return m.InputContainer({'q_in': 0.1, 'valve_command': ValveState.closed})
            
            # Default- dont control valve
            return m.InputContainer({'q_in': 0.1, 'valve_command': x['valve']})
        
        result = m.simulate_to_threshold(smart_load, events='full', save_freq=1, horizon=25)
        for x, next_x in zip(result.states[:-1], result.states[1:]):
            if x['h'] >= 0.8:
                self.assertEqual(next_x['valve'], ValveState.open)
            elif x['h'] <= 0.6:
                self.assertEqual(next_x['valve'], ValveState.closed)

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Discrete States")
    result = runner.run(load_test.loadTestsFromTestCase(TestDiscreteStates)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()