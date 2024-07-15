# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

from copy import deepcopy
from io import StringIO
import matplotlib.pyplot as plt
import pickle
import sys
import unittest
from unittest.mock import patch
import warnings

from progpy.data_models import LSTMStateTransitionModel, DataModel, DMDModel
from progpy.models import ThrownObject


class TestDataModel(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__
      
    def _test_simple_case(
        self,
        DataModelType,
        m=ThrownObject(),
        max_error=2,
        TIMESTEP=0.01,
        WITH_STATES=True,
        WITH_DT=True,
        **kwargs):

        # Step 1: Generate data
        def future_loading(t, x=None):
            return m.InputContainer({})  # No input for thrown object

        data = m.simulate_to_threshold(future_loading, events='impact', save_freq=TIMESTEP, dt=TIMESTEP)

        if WITH_STATES:
            kwargs['states'] = [data.states, data.states]

        if WITH_DT:
            kwargs['dt'] = TIMESTEP

        # Step 2: Generate model
        m2 = DataModelType.from_data(
            times=[data.times, data.times],
            inputs=[data.inputs, data.inputs],
            outputs=[data.outputs, data.outputs],
            event_states=[data.event_states, data.event_states],
            output_keys=list(m.outputs),
            save_freq=TIMESTEP,
            validation_split=0.5,
            **kwargs)
        
        self.assertIsInstance(m2, DataModelType)
        self.assertIsInstance(m2, DataModel)
        self.assertListEqual(m2.outputs, list(m.outputs))
        
        # Step 3: Use model to simulate_to time of threshold
        t_counter = 0
        x_counter = m.initialize()
        if isinstance(m2, DMDModel):
            future_loading2 = future_loading
        else:
            def future_loading2(t, x=None):
                # Future Loading is a bit complicated here
                # Loading for the resulting model includes the data inputs,
                # and the output from the last timestep
                nonlocal t_counter, x_counter
                z = m.output(x_counter)
                z = m2.InputContainer(z.matrix)
                x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
                t_counter = t
                return z

        results2 = m2.simulate_to(data.times[-1], future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

        # Have to do it this way because the other way (i.e., using the LSTM model- the states are not a subset)
        # Compare RMSE of the results to the original data
        error = m.calc_error(results2.times, results2.inputs, results2.outputs)
        self.assertLess(error, max_error)

        _stdout = sys.stdout
        sys.stdout = StringIO()
        actual_out = StringIO()
        m2.summary(file=actual_out)
        self.assertEqual(sys.stdout.getvalue(), '')
        self.assertNotEqual(actual_out.getvalue(), '')
        sys.stdout = _stdout

        return m2

    def test_early_stopping(self):
        m = ThrownObject()
        
        def future_loading(t, x=None):
            return m.InputContainer({})  # No input for thrown object 

        cfg = {
            'dt': 0.1,
            'save_freq': 0.1,
            'window': 4,
            'epochs': 75
        }

        # No early stopping
        _stdout = sys.stdout
        sys.stdout = StringIO()
        m2 = LSTMStateTransitionModel.from_model(m, [future_loading], early_stop=False, **cfg)
        end = sys.stdout.getvalue().rsplit("Epoch ", 1)[1]
        value = int(end.split(f"/{cfg['epochs']}", 1)[0])
        self.assertEqual(value, cfg['epochs'])

        # With early stopping (default)
        sys.stdout = StringIO()
        # Default = True
        m2 = LSTMStateTransitionModel.from_model(m, [future_loading], **cfg)
        end = sys.stdout.getvalue().rsplit("Epoch ", 1)[1]
        value = int(end.split(f"/{cfg['epochs']}", 1)[0])
        self.assertNotEqual(value, cfg['epochs'])
        sys.stdout = _stdout

    def test_lstm_simple(self):
        m = self._test_simple_case(LSTMStateTransitionModel, window=5, epochs=20, max_error=3)
        self.assertListEqual(m.inputs, ['x_t-1'])
        # Use set below so there's no issue with ordering
        keys = ['x_t-1', 'x_t-2', 'x_t-3', 'x_t-4', 'x_t-5']
        keys.extend([f'_model_output{i}' for i in range(16)])
        self.assertSetEqual(set(m.states), set(keys))

        # Create from model
        LSTMStateTransitionModel(m.parameters['output_model'], m.parameters['state_model'], output_keys=['x'])
        try:
            # Test pickling model m
            with self.assertWarns(RuntimeWarning):
                # Will raise warning suggesting using save and load from keras.
                pickled_m = pickle.dumps(m)
                m2 = pickle.loads(pickled_m)
                self.assertIsInstance(m2, LSTMStateTransitionModel)
                self.assertIsInstance(m2, DataModel)
                self.assertListEqual(m2.outputs, ['x'])

            # Deepcopy test
            m3 = deepcopy(m2)
        except Exception:
            warnings.warn("Pickling not supported for LSTMStateTransitionModel on this system")
            pass

        # More tests in examples.lstm_model

    def test_dmd_simple(self):
        self._test_simple_case(DMDModel, max_error=25)

        # Inferring dt
        self._test_simple_case(DMDModel, max_error=8, WITH_DT=False)

        # Without velocity, DMD doesn't perform well
        m = self._test_simple_case(DMDModel, WITH_STATES=False, max_error=100)

        # Test pickling model m
        pickled_m = pickle.dumps(m)
        m2 = pickle.loads(pickled_m)
        self.assertIsInstance(m2, DMDModel)
        self.assertIsInstance(m2, DataModel)
        self.assertListEqual(m2.outputs, ['x'])

        # Deepcopy test
        m3 = deepcopy(m2)

    def test_lstm_from_model_thrown_object(self):
        TIMESTEP = 0.01

        # Step 1: Generate dataed
        m = ThrownObject()

        def future_loading(t, x=None):
            return m.InputContainer({})  # No input for thrown object
        
        m3 = LSTMStateTransitionModel.from_model(
            m,
            [future_loading for _ in range(5)],
            dt=[TIMESTEP, TIMESTEP/2, TIMESTEP/4, TIMESTEP*2, TIMESTEP*4],
            window=2,
            units=(16, ),  # Units as tuple
            epochs=20)

        # Should get keys from original model
        self.assertSetEqual(set(m3.inputs), set(['dt', 'x_t-1']))
        self.assertSetEqual(set(m3.outputs), set(m.outputs))

        # Step 3: Use model to simulate_to time of threshold
        t_counter = 0
        x_counter = m.initialize()

        def future_loading2(t, x=None):
            # Future Loading is a bit complicated here
            # Loading for the resulting model includes the data inputs, 
            # and the output from the last timestep
            nonlocal t_counter, x_counter
            z = m.output(x_counter)
            z = m3.InputContainer(
                {
                    'x_t-1': z['x'],
                    'dt': TIMESTEP
                })
            x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
            t_counter = t
            return z
        
        results2 = m3.simulate_to(7.5, future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

        # Have to do it this way because the other way (i.e., using the LSTM model- the states are not a subset)
        # Compare RMSE of the results to the original data
        error = m.calc_error(results2.times, results2.inputs, results2.outputs)
        self.assertLess(error, 2)

        # Now with no 'dt'
        #----------------------------------------------------
        m3 = LSTMStateTransitionModel.from_model(
            m,
            [future_loading for _ in range(5)],
            dt = [TIMESTEP, TIMESTEP/2, TIMESTEP/4, TIMESTEP*2, TIMESTEP*4],
            window=2,
            epochs=30,
            add_dt = False)

        # Should get keys from original model
        self.assertSetEqual(set(m3.inputs), set(['x_t-1']))
        self.assertSetEqual(set(m3.outputs), set(m.outputs))

        # Step 3: Use model to simulate_to time of threshold
        t_counter = 0
        x_counter = m.initialize()
        def future_loading2(t, x=None):
            # Future Loading is a bit complicated here 
            # Loading for the resulting model includes the data inputs, 
            # and the output from the last timestep
            nonlocal t_counter, x_counter
            z = m.output(x_counter)
            z = m3.InputContainer(
                {
                    'x_t-1': z['x']
                })
            x_counter = m.next_state(x_counter, future_loading(t), t - t_counter)
            t_counter = t
            return z
        
        results2 = m3.simulate_to(7.5, future_loading2, dt=TIMESTEP, save_freq=TIMESTEP)

        # Have to do it this way because the other way (i.e., using the LSTM model- the states are not a subset)
        # Compare RMSE of the results to the original data
        error = m.calc_error(results2.times, results2.inputs, results2.outputs)
        self.assertLess(error, 2)

    def test_improper_input(self):
        # Input, output format
        with self.assertRaises(Exception, msg='No outputs'):
            LSTMStateTransitionModel.from_data(inputs=[1])
        with self.assertRaises(Exception, msg='No inputs'):
            LSTMStateTransitionModel.from_data(outputs=[1])
        with self.assertRaises(ValueError, msg='Empty inputs/outputs'):
            LSTMStateTransitionModel.from_data(inputs=[], outputs=[])
        with self.assertRaises(ValueError, msg="Uneven Inputs/outputs"):
            LSTMStateTransitionModel.from_data([1], [1, 2])
        with self.assertRaises(ValueError, msg="Uneven Inputs/outputs-2"):
            LSTMStateTransitionModel.from_data([1, 2, 3], [1, 2])
        with self.assertRaises(TypeError, msg="Element Format"):
            LSTMStateTransitionModel.from_data(1, [2])
        with self.assertRaises(TypeError, msg="Element Format-2"):
            LSTMStateTransitionModel.from_data([1], 2)
        with self.assertRaises(TypeError, msg="Not iterable"):
            LSTMStateTransitionModel.from_data(12, 12)

        # Other Configurables
        with self.assertRaises(ValueError, msg="Too small sequence length"):
            LSTMStateTransitionModel.from_data([1], [2], window=0)
        with self.assertRaises(TypeError, msg="Non-scalar sequence length"):
            LSTMStateTransitionModel.from_data([1], [2], window=[])

        with self.assertRaises(ValueError, msg="Too few layers"):
            LSTMStateTransitionModel.from_data([1], [2], layers=0)
        with self.assertRaises(TypeError, msg="Non-scalar layers"):
            LSTMStateTransitionModel.from_data([1], [2], layers=[])

        with self.assertRaises(ValueError, msg="Too few epochs"):
            LSTMStateTransitionModel.from_data([1], [2], epochs=0)
        with self.assertRaises(TypeError, msg="Non-scalar epochs"):
            LSTMStateTransitionModel.from_data([1], [2], epochs=[])

        with self.assertRaises(TypeError, msg="Normalize type"):
            LSTMStateTransitionModel.from_data([1], [2], normalize=[1, 2])

        with self.assertRaises(TypeError, msg="Dropout type"):
            LSTMStateTransitionModel.from_data([1], [2], dropout = 'abc')   
        with self.assertRaises(ValueError, msg="Negative dropout"):
            LSTMStateTransitionModel.from_data([1], [2], dropout = -1)   

        with self.assertRaises(ValueError, msg="Zero units"):
            LSTMStateTransitionModel.from_data([1], [2], units = 0)  

        with self.assertRaises(ValueError, msg="Units, layers mismatch"):
            LSTMStateTransitionModel.from_data([1], [2], units = [1], layers=2)  
        with self.assertRaises(ValueError, msg="Units, layers mismatch-2"):
            LSTMStateTransitionModel.from_data([1], [2], units = [1, 2, 3], layers=2)     

        with self.assertRaises(TypeError, msg="Validation Split type"):
            LSTMStateTransitionModel.from_data([1], [2], validation_split = 'abc')   
        with self.assertRaises(ValueError, msg="Negative Validation Split"):
            LSTMStateTransitionModel.from_data([1], [2], validation_split = -1e-5)    
        with self.assertRaises(ValueError, msg="Validation Split of 1"):
            LSTMStateTransitionModel.from_data([1], [2], validation_split = 1.0)    

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Data Models")

    with patch('matplotlib.pyplot.show'):
        result = runner.run(load_test.loadTestsFromTestCase(TestDataModel)).wasSuccessful()
    plt.close('all')
    
    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
