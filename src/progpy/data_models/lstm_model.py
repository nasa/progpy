# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import abc
import importlib.util
from itertools import chain
import matplotlib.pyplot as plt
from numbers import Number
import numpy as np
import sys
from warnings import warn

from progpy.exceptions import warn_once
from progpy.data_models import DataModel
from progpy.sim_result import SimResult


class LSTMStateTransitionModel(DataModel):
    """
    .. versionadded:: 1.4.0

    A State Transition Model with no :term:`event` using an Keras LSTM Model.
    State transition models map from the :term:`input` at time t and :term:`output` at time t-1 plus historical data from a set window to the :term:`output` at time t.

    Most users will use the :py:func:`LSTMStateTransitionModel.from_data` method to create a model, but the model can be created by passing in a model directly into the constructor. The LSTM model in this method maps from [u_t-n+1, z_t-n, ..., u_t, z_t-1] to z_t. Past :term:`input` are stored in the :term:`model` internal :term:`state`. Actual calculation of :term:`output` is performed when :py:func:`LSTMStateTransitionModel.output` is called. When using in simulation that may not be until the simulation results are accessed.

    .. note::
        ProgPy must be installed with [datadriven] option to use LSTM model. either

        pip3 install 'progpy[datadriven]'

        or (if using local version)

        pip3 install '.[datadriven]'

    Args:
        output_model (keras.Model): If a state model is present, maps from the state_model outputs to model :term:`output`. Otherwise, maps from model inputs to model :term:`output`
        state_model (keras.Model, optional): Keras model to use for state transition
        event_state_model (keras.Model, optional): If a state model is present, maps from the state_model outputs to :term:`event state`. Otherwise, maps from model inputs to :term:`event state`
        t_met_model (keras.Model, optional): If a state model is present, maps from the state_model outputs to if the threshold has been met. Otherwise, maps from model inputs to if the threshold has not been met

    Keyword Args:
        input_keys (list[str]): List of input keys
        output_keys (list[str]): List of output keys
        event_keys (list[str]): List of event keys

    Attributes:
        model (keras.Model): Keras model to use for state transition

    See Also:
        LSTMStateTransitionModel.from_data
        LSTMStateTransitionModel.from_model
        examples.lstm_model
    """

    default_params = {
        'process_noise': 0,  # Default 0 noise
        'measurement_noise': 0,  # Default 0 noise
    }

    def __init__(self, output_model, state_model=None, event_state_model=None, t_met_model=None, **kwargs):
        n_outputs = output_model.output.shape[1]
        n_internal = 0 if state_model is None else state_model.output.shape[1]
        input_shape = output_model.input.shape if state_model is None else state_model.input.shape
        n_inputs = input_shape[-1]-n_outputs
        if event_state_model is not None:
            n_events = event_state_model.output.shape[-1]
            if t_met_model is not None and t_met_model.output.shape[-1]/2 != n_events:
                raise ValueError("t_met output length must be twice that of event state")
        elif t_met_model is not None:
            n_events = t_met_model.output.shape[-1]/2
        else:
            n_events = 0

        # Setup inputs, outputs, states
        self.outputs = kwargs.get('output_keys', [f'z{i}' for i in range(n_outputs)])
        self.events = kwargs.get(
            'event_keys', 
            getattr(self, 'events', [f'event{i}' for i in range(n_events)]))  # If overridden- use that
        input_keys = kwargs.get('input_keys', [f'u{i}' for i in range(n_inputs)])
        self.inputs = input_keys.copy()
        # Outputs from the last step are part of input
        self.inputs.extend([f'{z_key}_t-1' for z_key in self.outputs])

        # States are in format [u_t-n+1, z_t-n, ..., u_t, z_t-1]
        self.states = []
        for j in range(input_shape[1]-1, -1, -1):
            self.states.extend([f'{input_i}_t-{j}' for input_i in input_keys])
            self.states.extend([f'{output_i}_t-{j+1}' for output_i in self.outputs])
        self.states.extend([f'_model_output{i}' for i in range(n_internal)])

        kwargs['window'] = input_shape[1]
        self.history = kwargs.get('history', None)
        if 'history' in kwargs:
            # Delete to prevent copying below
            del kwargs['history']

        super().__init__(**kwargs)

        # Set parameters without copying
        # Putting it in the parameters dictionary simplifies pickling
        self.parameters.__setitem__('state_model', state_model, _copy = False)
        self.parameters.__setitem__('output_model', output_model, _copy = False)
        self.parameters.__setitem__('event_state_model', event_state_model, _copy = False)
        self.parameters.__setitem__('t_met_model', t_met_model, _copy = False)
        self.parameters.__setitem__('history', self.history, _copy = False)        

    def __getstate__(self):
        warn_once("LSTMStateTransitionModel uses a Keras model, which does not always support pickling. We recommend that you use the keras save and load model functions instead with m.model", RuntimeWarning)
        return ((), self.parameters.data)

    def __eq__(self, other):
        # Needed because we add .model, which is not present in the parent class
        if not isinstance(other, LSTMStateTransitionModel):
            return False
        return super().__eq__(other) and self.model == other.model

    def initialize(self, u=None, z=None):
        """
        Initialize the model with the given inputs and outputs. For LSTMStateTransitionModel, the initial state is set to None. As data is added (i.e., next_state is called), the initial state is updated.

        This is one of the rare models that cannot be initialized right away.

        Args:
            u (InputContainer, optional): first input
            z (OutputContainer, optional): first output

        Returns:
            StateContainer: First State
        """
        return self.StateContainer(np.array([[None] for _ in self.states]))

    def next_state(self, x, u, _):
        # Rotate new input into state
        input_data = u.matrix

        if self.parameters['state_model'] is None:
            states = x.matrix[len(input_data):]
            return self.StateContainer(np.vstack((states, input_data)))
            
        states = x.matrix[len(input_data):-self.parameters['state_model'].output_shape[1]]
        states = np.vstack((states, input_data))

        if states[0,0] is None:
            return self.StateContainer(np.vstack((states, x.matrix[-self.parameters['state_model'].output_shape[1]:])))
        else:
            # Enough data has been received to calculate output
            # Format input into np array with shape (1, window, num_inputs)
            m_input = states.reshape(1, self.parameters['window'], len(self.inputs))
            m_input = np.array(m_input, dtype=np.float64)
            internal_states = self.parameters['state_model'](m_input).numpy().T
        result = np.array(np.vstack((states, internal_states)), dtype=np.float64)
        return self.StateContainer(result)

    def output(self, x):
        if x.matrix[0, 0] is None:
            warn(f"Output estimation is not available until at least {1+self.parameters['window']} timesteps have passed.")
            return self.OutputContainer(np.array([[None] for _ in self.outputs]))

        # Enough data has been received to calculate output
        # Pass internal states into model to calculate output
        if self.parameters['state_model'] is None:
            m_input = x.matrix.reshape(1, self.parameters['window'], len(self.inputs))
            internal_states = np.array(m_input, dtype=np.float64)
        else:
            internal_states = x.matrix[-self.parameters['state_model'].output_shape[1]:].T
        m_output = self.parameters['output_model'](internal_states)

        if 'normalization' in self.parameters:
            m_output *= self.parameters['normalization'][1]
            m_output += self.parameters['normalization'][0]

        return self.OutputContainer(m_output.numpy().T)

    def event_state(self, x):
        if self.parameters['event_state_model'] is None:
            warn('No event state model exists- returning empty event state')
            return {key: None for key in self.events}
        
        if x.matrix[0,0] is None:
            warn(f"Event state estimation is not available until at least {1+self.parameters['window']} timesteps have passed.")
            return {key: None for key in self.events}

        # Enough data has been received to calculate output
        # Pass internal states into model to calculate output
        if self.parameters['state_model'] is None:
            m_input = x.matrix.reshape(1, self.parameters['window'], len(self.inputs))
            internal_states = np.array(m_input, dtype=np.float64)
        else:
            internal_states = x.matrix[-self.parameters['state_model'].output_shape[1]:].T
        m_event_state = self.parameters['event_state_model'](internal_states)

        return {key: value for key, value in zip(self.events, m_event_state[0])}

    def threshold_met(self, x):
        if self.parameters['t_met_model'] is None:
            warn('No threshold met model exists- returning empty t_met')
            return {key: None for key in self.events}
        
        if x.matrix[0,0] is None:
            warn(f"Threshold met estimation is not available until at least {1+self.parameters['window']} timesteps have passed.")
            return {key: None for key in self.events}

        # Enough data has been received to calculate output
        # Pass internal states into model to calculate output
        if self.parameters['state_model'] is None:
            m_input = x.matrix.reshape(1, self.parameters['window'], len(self.inputs))
            internal_states = np.array(m_input, dtype=np.float64)
        else:
            internal_states = x.matrix[-self.parameters['state_model'].output_shape[1]:].T
        m_t_met = self.parameters['t_met_model'](internal_states)
        m_t_met = [np.argmax(m_t_met[0][i*2:(i+1)*2]) == 0 for i in range(len(self.events))]

        return {key: value for key, value in zip(self.events, m_t_met)}

    def summary(self, file= sys.stdout, expand_nested=False, show_trainable=False):
        print('LSTM State Transition Model: ', file=file)
        print("Inputs: ", self.inputs, file=file)
        print("Outputs: ", self.outputs, file=file)
        print("Window_size: ", self.parameters['window'], file=file)
        if self.parameters['state_model'] is not None:
            print('\nState Model: ', file=file)
            self.parameters['state_model'].summary(print_fn=file.write, expand_nested=expand_nested, show_trainable=show_trainable)
        
        print('\nOutput Model: ', file=file)
        self.parameters['output_model'].summary(print_fn=file.write, expand_nested=expand_nested, show_trainable=show_trainable)

        if self.parameters['event_state_model'] is not None:
            print('\nEvent State Model: ', file=file)
            self.parameters['event_state_model'].summary(print_fn=file.write, expand_nested=expand_nested, show_trainable=show_trainable)
        
    @staticmethod
    def pre_process_data(inputs, outputs, event_states=None, t_met=None, window=10, **kwargs):
        """
        Pre-process data for the LSTMStateTransitionModel. This is run inside from_data to convert the data into the desired format 

        Args:
            inputs (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
            outputs (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
            event_states (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
            t_met (List[ndarray or SimulationResult]): Data to be processed. Each element is of format, ndarray or SimulationResult
            window (int): Length of a single sequence

        Returns:
            Tuple[ndarray, ndarray]: pre-processed data (input, output). Where input is of size (num_sequences, window, num_inputs) and output is of size (num_sequences, num_outputs)
        """

        u_all = []
        z_all = []
        es_all = []
        t_all = []

        DataModel.check_data_format(inputs, outputs)

        if event_states is not None and len(inputs) != len(event_states):
            raise ValueError("Inputs must be same length as event_states")

        for i in range(len(inputs)):
            # Each item (u, z) is a 1-d array, a 2-d array, or a SimResult

            # Process Input
            u = inputs[i]
            if isinstance(u, SimResult):
                if len(u[0].keys()) == 0:
                    # No inputs
                    u = []
                else:
                    u = np.array([u_i.matrix[:, 0] for u_i in u])

            if len(u) > 0 and len(u) <= window:
                raise TypeError(f"Not enough data for window size {window}. Only {len(u)} elements present.")

            if isinstance(u, (list, np.ndarray)):
                if len(u) == 0:
                    # No inputs
                    u_i = []
                elif np.isscalar(u[0]):
                    # Input is 1-d array (i.e., 1 input)
                    # Note: 1 is added to account for current time (current input used to predict output at time i)
                    u_i = [[[u[i+j]] for j in range(1, window+1)] for i in range(len(u)-window)]
                elif isinstance(u[0], (list, np.ndarray)):
                    # Input is d-d array
                    # Note: 1 is added to account for current time (current input used to predict output at time i)
                    n_inputs = len(u[0])
                    u_i = [[[u[i+j][k] for k in range(n_inputs)] for j in range(1, window+1)] for i in range(len(u)-window)]
                else:
                    raise TypeError(f"Unsupported input type: {type(u)} for internal element (data[0][i]")  
            else:
                raise TypeError(f"Unsupported data type: {type(u)}. input u must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")


            # Process Output
            z = outputs[i]
            if isinstance(z, SimResult):
                if len(z[0].keys()) == 0:
                    # No outputs
                    z = []
                else:
                    z = np.array([z_i.matrix[:, 0] for z_i in z])

            if isinstance(z, (list, np.ndarray)):
                if len(z) != len(u) and len(u) != 0 and len(z) != 0:
                    # Checked here to avoid SimResults from accidentially
                    # triggering this check
                    raise IndexError(f"Number of outputs ({len(z)}) does not match number of inputs ({len(u)})")

                if len(z) == 0:
                    # No outputs
                    z_i = []
                elif np.isscalar(z[0]):
                    # Output is 1-d array (i.e., 1 output)
                    z_i = [[z[i]] for i in range(window, len(z))]
                elif isinstance(z[0], (list, np.ndarray)):
                    # Input is d-d array
                    n_outputs = len(z[0])
                    z_i = [[z[i][k] for k in range(n_outputs)] for i in range(window, len(z))]
                else:
                    raise TypeError(f"Unsupported input type: {type(z)} for internal element (output[i])")

                # Also add to input (past outputs are part of input)
                z_ii = [[z[i+j] for j in range(window)] for i in range(len(z_i))]
                # ISSUE- TODO z_ii has too many dimensions
                if len(u_i) == 0:
                    u_i = z_ii
                else:
                    for k in range(len(z_ii)):
                        for j in range(window):
                            u_i[k][j].extend(z_ii[k][j])
            else:
                raise TypeError(f"Unsupported data type: {type(z)}. output z must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            
            if event_states is not None:
                es = event_states[i]
                if isinstance(es, SimResult):
                    if len(es[0].keys()) == 0:
                        # No event_states
                        es = []
                    else:
                        es = np.array([[es_i[key] for key in es_i.keys()] for es_i in es])

                if isinstance(es, (list, np.ndarray)):
                    if len(es) != len(u) and len(u) != 0 and len(es) != 0:
                        # Checked here to avoid SimResults from accidentially
                        # triggering this check
                        raise IndexError(f"Number of event_states ({len(es)}) does not match number of inputs ({len(u)})")

                    if len(es) == 0:
                        # No outputs
                        es_i = []
                    elif np.isscalar(es[0]):
                        # Output is 1-d array (i.e., 1 output)
                        es_i = [[es[i]] for i in range(window, len(es))]
                    elif isinstance(es[0], (list, np.ndarray)):
                        # Input is d-d array
                        n_events = len(es[0])
                        es_i = [[es[i][k] for k in range(n_events)] for i in range(window, len(es))]
                    else:
                        raise TypeError(f"Unsupported input type: {type(es)} for internal element (es[i])")

                else:
                    raise TypeError(f"Unsupported data type: {type(es)}. event state must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            else:
                es_i = []

            if t_met is not None:
                t = t_met[i]
                if isinstance(t, SimResult):
                    if len(t[0].keys()) == 0:
                        # No event_states
                        t = []
                    else:
                        t = np.array([[t_i[key] for key in t_i.keys()] for t_i in t])

                if isinstance(t, (list, np.ndarray)):
                    if len(t) != len(u) and len(u) != 0 and len(t) != 0:
                        # Checked here to avoid SimResults from accidentially
                        # triggering this check
                        raise IndexError(f"Number of t_met ({len(t)}) does not match number of inputs ({len(u)})")

                    if len(t) == 0:
                        # No outputs
                        t_i = []
                    elif np.isscalar(t[0]):
                        # Output is 1-d array (i.e., 1 output)
                        t_i = [[1, 0] if t[i] else [0, 1] for i in range(window, len(t))]
                    elif isinstance(t[0], (list, np.ndarray)):
                        # Input is d-d array
                        n_events = len(t[0])
                        # True = 1, 0; False = 0, 1
                        t_i = [list(chain.from_iterable((1, 0) if t[i][k] else (0, 1) for k in range(n_events))) for i in range(window, len(t))]
                    else:
                        raise TypeError(f"Unsupported input type: {type(t[0])} for internal element (t[i])")  

                else:
                    raise TypeError(f"Unsupported data type: {type(t)}. t_met must be in format List[Tuple[np.array, np.array]] or List[Tuple[SimResult, SimResult]]")
            else:
                t_i = []

            u_all.extend(u_i)
            z_all.extend(z_i)
            es_all.extend(es_i)
            t_all.extend(t_i)
        
        u_all = np.array(u_all, dtype=np.float64)
        z_all = np.array(z_all, dtype=np.float64)
        es_all = np.array(es_all, dtype=np.float64)
        t_all = np.array(t_all, dtype=np.float64)
        return (u_all, z_all, es_all, t_all)

    @classmethod
    def from_data(cls, inputs, outputs, event_states=None, t_met=None, **kwargs):
        """
        Generate a LSTMStateTransitionModel from data

        Args:
            inputs (list[np.array]): 
                list of :term:`input` data for use in data. Each element is the inputs for a single run of size (n_times, n_inputs)
            outputs (list[np.array]): 
                list of :term:`output` data for use in data. Each element is the outputs for a single run of size (n_times, n_outputs)
            event_states (list[np.array], optional): 
                list of :term:`event state` data for use in data. Each element is the event state for a single run of size (n_times, n_events)
            t_met (list[np.array], optional): 
                list of :term:`threshold` met data for use in data. Each element is if the threshold has been met for a single run of size (n_times, n_events) 

        Keyword Args:
            window (int): 
                Number of historical points used in the model. I.e, if window is 3, the model will map from [t-3, t-2, t-1] to t
            input_keys (list[str]): 
                List of keys to use to identify :term:`input`. If not supplied u[#] will be used to idenfiy inputs
            output_keys (list[str]): 
                List of keys to use to identify :term:`output`. If not supplied z[#] will be used to idenfiy outputs
            event_keys (list[str]):
                List of keys to use to identify events for :term:`event state` and :term:`threshold` met. If not supplied event[#] will be used to identify events
            validation_percentage (float): 
                Percentage of data to use for validation, between 0-1
            epochs (int): 
                Number of epochs (i.e., iterations) to train the model. More epochs means better results (to a point), but more time to train. Note: large numbers of epochs may result in overfitting.
            layers (int): 
                Number of LSTM layers to use. More layers can represent more complex systems, but are less efficient. Note: 2 layers is typically enough for most complex systems. Default: 1
            units (int or list[int]): 
                number of units (i.e., dimensionality of output state) used in each lstm layer. Using a scalar value will use the same number of units for each layer.
            activation (str or list[str]): 
                Activation function to use for each layer
            dropout (float): 
                Dropout rate to be applied. Dropout helps avoid overfitting
            normalize (bool): 
                If the data should be normalized. This is recommended for most cases.
            early_stopping (bool):
                If early stopping is desired. Default is True
            early_stop.cfg (dict):
                Configuration to pass into early stopping callback (if enabled). See keras documentation (https://keras.io/api/callbacks/early_stopping) for options. E.g., {'patience': 5}

        Returns:
            LSTMStateTransitionModel: Generated Model

        See Also:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        """
        params = { # default_params
            'window': 128,
            'validation_split': 0.25,
            'epochs': 100,
            'prediction_steps': 1,
            'layers': 1,
            'units': 16,
            'activation': 'tanh',
            'dropout': 0.1,
            'normalize': True,
            'early_stop': True,
            'early_stop.cfg': {'patience': 3, 'monitor': 'loss'},
        }.copy()  # Copy is needed to avoid updating default

        params.update(LSTMStateTransitionModel.default_params)
        params.update(kwargs)

        # Input Validation
        if not np.isscalar(params['window']):
            raise TypeError(f"window must be an integer greater than 0, not {type(params['window'])}")
        if params['window'] <= 0:
            raise ValueError(f"window must be greater than 0, got {params['window']}")
        if not np.isscalar(params['layers']):
            raise TypeError(f"layers must be an integer greater than 0, not {type(params['layers'])}")
        if params['layers'] <= 0:
            raise ValueError(f"layers must be greater than 0, got {params['layers']}")
        if np.isscalar(params['units']):
            params['units'] = [params['units'] for _ in range(params['layers'])]
        if not isinstance(params['units'], (abc.Sequence, np.ndarray)):
            raise TypeError(f"units must be a Sequence (e.g., list or tuple) of integers, not {type(params['units'])}")
        if len(params['units']) != params['layers']:
            raise ValueError(f"units must be a list of integers of length {params['layers']}, got {params['units']}")
        for i in range(params['layers']):
            if params['units'][i] <= 0:
                raise ValueError(f"units[{i}] must be greater than 0, got {params['units'][i]}")
        if not isinstance(params['dropout'], Number):
            raise TypeError(f"dropout must be an float greater than or equal to 0, not {type(params['dropout'])}")
        if params['dropout'] < 0:
            raise ValueError(f"dropout must be greater than or equal to 0, got {params['dropout']}")
        if not isinstance(params['activation'], (list, tuple, np.ndarray)):
            params['activation'] = [params['activation'] for _ in range(params['layers'])]
        if not np.isscalar(params['validation_split']):
            raise TypeError(f"validation_split must be an float between 0 and 1, not {type(params['validation_split'])}")
        if params['validation_split'] < 0 or params['validation_split'] >= 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {params['validation_split']}")
        if not np.isscalar(params['epochs']):
            raise TypeError(f"epochs must be an integer greater than 0, not {type(params['epochs'])}")
        if params['epochs'] < 1:
            raise ValueError(f"epochs must be greater than 0, got {params['epochs']}")
        if np.isscalar(inputs):  # Is scalar (e.g., SimResult)
            inputs = [inputs]
        if np.isscalar(outputs):
            outputs = [outputs]
        if not isinstance(params['normalize'], bool):
            raise TypeError(f"normalize must be a boolean, not {type(params['normalize'])}")

        # Prepare datasets
        (u_all, z_all, es_all, t_all) = LSTMStateTransitionModel.pre_process_data(inputs, outputs, event_states=event_states, t_met=t_met, **params)

        # Normalize
        if params['normalize']:
            n_inputs = len(inputs[0][0])
            u_mean = np.mean(u_all[:, 0, :n_inputs], axis=0)
            u_std = np.std(u_all[:, 0, :n_inputs], axis=0)
            # If there's no variation- don't normalize 
            u_std[u_std == 0] = 1
            z_mean = np.mean(z_all, axis=0)
            z_std = np.std(z_all, axis=0)
            # If there's no variation- don't normalize 
            z_std[z_std == 0] = 1

            # Add output (since z_t-1 is last input)
            u_mean = np.hstack((u_mean, z_mean))
            u_std = np.hstack((u_std, z_std))

            z_all = (z_all - z_mean)/z_std

            # u_mean and u_std act on the column vector form (from inputcontainer)
            # so we need to transpose them to a column vector
            params['normalization'] = (z_mean, z_std)

        # Tensorflow is imported here to avoid importing it if not needed
        try:
            from tensorflow import keras
        except ImportError as e:
            raise ImportError("Missing required dependencies for data-driven model. ProgPy was imported directly. Instead import with datadriven dependencies using pip3 install progpy[datadriven] or pip3 install -e '.[datadriven]' (if installing from local copy)")

        # Build model
        callbacks = [ ]

        if params['early_stop']:
            callbacks.append(keras.callbacks.EarlyStopping(**params['early_stop.cfg']))

        inputs = keras.Input(shape=u_all.shape[1:])
        x = inputs
        if params['normalize']:
            x = keras.layers.Normalization(mean = u_mean, variance = u_std**2)(inputs)
        for i in range(params['layers']):
            if i == params['layers'] - 1:
                # Last layer
                x = keras.layers.LSTM(params['units'][i], activation=params['activation'][i])(x)
            else:
                # Intermediate layer
                x = keras.layers.LSTM(params['units'][i], activation=params['activation'][i], return_sequences=True)(x)
        
        if params['dropout'] > 0:
            # Dropout prevents overfitting
            x = keras.layers.Dropout(params['dropout'])(x)

        outputs = [keras.layers.Dense(z_all.shape[1] if z_all.ndim == 2 else 1, name='output')(x)]
        output_data = [z_all]
        
        if event_states is not None:
            outputs.append(keras.layers.Dense(es_all.shape[1] if es_all.ndim == 2 else 1, name='event_state')(x))
            output_data.append(es_all)
        
        if t_met is not None and t_all.shape[1] > 0:
            n_events = round(t_all.shape[1]/2)
            # Layer for each event
            t_met_layers = [keras.layers.Dense(2, activation="softmax") for _ in range(n_events)]
            t_met_layers_output = [layer(x) for layer in t_met_layers]
            if len(t_met_layers) == 1:
                outputs.append(t_met_layers_output[-1])
            else:
                # Concatenate layers
                outputs.append(keras.layers.Concatenate(name='t_met')(t_met_layers_output))
            output_data.append(t_all)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="mse", metrics=[["mae"]]*len(outputs))
        
        # Train model
        history = model.fit(
            u_all,
            output_data,
            epochs=params['epochs'],
            callbacks=callbacks,
            validation_split=params['validation_split'])

        # Split model into separate models
        n_state_layers = params['layers'] + 1 + (params['dropout'] > 0) + (params['normalize'])
        output_layer_input = keras.layers.Input(model.layers[n_state_layers-1].output.shape[1:])
        output_layer = model.get_layer('output')(output_layer_input)
        state_model = keras.Model(model.input, model.layers[n_state_layers-1].output)
        output_model = keras.Model(output_layer_input, output_layer)
        if event_states is None:
            event_state_model = None
        else:
            event_state_layer = model.get_layer('event_state')(output_layer_input)
            event_state_model = keras.Model(output_layer_input, event_state_layer)            

        if t_met is None:
            t_met_model = None
        else:
            t_met_layers = [layer(output_layer_input) for layer in t_met_layers]
            if len(t_met_layers) == 1:
                t_met_layer = t_met_layers[0]
            else:  # Concat layer exists
                t_met_layer = model.get_layer('t_met')(t_met_layers)
            t_met_model = keras.Model(output_layer_input, t_met_layer)  

        return cls(output_model, state_model, event_state_model, t_met_model, history = history, **params)
        
    def simulate_to_threshold(self, future_loading_eqn, first_output=None, events=None, **kwargs):
        t = kwargs.get('t0', 0)
        dt = kwargs.get('dt', 0.1)
        x = kwargs.get('x', self.initialize(future_loading_eqn(t), first_output))

        # configuring next_time function to define prediction time step, default is constant dt
        if callable(dt):
            dt_mode = 'function'
        elif isinstance(dt, tuple):
            dt_mode = dt[0]
            dt = dt[1]               
        elif isinstance(dt, str):
            dt_mode = dt
            if dt_mode == 'constant':
                dt = 1.0  # Default
            else:
                dt = np.inf
        else:
            dt_mode = 'constant'

        if dt_mode in ('constant', 'auto'):
            def next_time(t, x):
                return dt
        elif dt_mode != 'function':
            raise Exception(f"'dt' mode {dt_mode} not supported. Must be 'constant', 'auto', or a function")

        # Simulate until passing minimum number of steps
        # TODO Suggestion (Matteo): 
        # normalize data before prediction loop starts; de-normalize them after loop.
        # This way, normalization could be handled using functions that normalize nd arrays all at once, 
        # avoiding to normalize data at each step of the simulation. 
        # I don't know if this interferes with how the PrognosticsModel class works. 
        while x.matrix[0,0] is None:
            if 'horizon' in kwargs and t > kwargs['horizon']:
                raise Exception(f'Not enough timesteps to reach minimum number of steps for model simulation')
            dt = next_time(t, x)
            t = t + dt/2
            # Use state at midpoint of step to best represent the load during the duration of the step
            u = future_loading_eqn(t, x)
            t = t + dt/2
            x = self.next_state(x, u, dt)

        # Now do actual simulate_to_threshold
        kwargs['t0'] = t
        x.matrix = np.array(x.matrix, dtype=np.float64)
        kwargs['x'] = x
        if 'horizon' in kwargs:
            if kwargs['horizon'] < t:
                raise ValueError('Prediction horizon does not allow enough steps to fully initialize model')
            kwargs['horizon'] = kwargs['horizon'] - t
        return super().simulate_to_threshold(future_loading_eqn, first_output, events, **kwargs)
    
    def plot_history(self, metrics=None):
        """
        Plot the trianing history for the keras model. 

        Args:
            metrics (list[str], optional): Metrics to plot (e.g., [loss]). Defaults to all metrics in history.

        Raises:
            Exception: No history is available (e.g., because you supplied a model to the constructor but didn't provide the history as a kwarg)

        Returns:
            list[plt.figure]: List of Figures, one for each metric
        """
        if self.history is None:
            raise Exception("Cannot plot history- no history is available")

        if metrics is None: 
            metrics = self.history.history.keys()
        
        plts = []
        for key in metrics:
            if key[:4] == 'val_':
                plt.figure(plts[list(metrics).index(key[4:])].number)
            else:
                plts.append(plt.figure())
            plt.plot(self.history.history[key], label=key)
            plt.xlabel('epochs')
            plt.ylabel(key)
            plt.legend()
        
        return plts
