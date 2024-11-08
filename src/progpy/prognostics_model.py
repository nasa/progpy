# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC
from collections import abc, namedtuple
from copy import deepcopy
import itertools
import json
from numbers import Number
import numpy as np
from typing import List, Mapping  # Still needed until v3.9
from warnings import warn

from progpy.exceptions import ProgModelStateLimitWarning, warn_once
from progpy.loading import Piecewise
from progpy.sim_result import SimResult, LazySimResult
from progpy.utils import ProgressBar, calc_error, input_validation
from progpy.utils.containers import DictLikeMatrixWrapper, InputContainer, OutputContainer
from progpy.utils.next_state import next_state_functions
from progpy.utils.parameters import PrognosticsModelParameters
from progpy.utils.serialization import CustomEncoder, custom_decoder
from progpy.utils.size import getsizeof


class PrognosticsModel(ABC):
    """
    A general time-variant state space :term:`model` of system degradation behavior.

    The PrognosticsModel class is a wrapper around a mathematical model of a system as represented by a state, output, input, event_state and threshold equation.

    A Model also has a parameters structure, which contains fields for various model parameters.

    Keyword Args
    ------------
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state).
          Can be number (e.g., .2) applied to every state, a dictionary of values for each
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)
        integration_method: Optional, str or OdeSolver
          Integration method used by next_state in continuous models, e.g. 'rk4' or 'euler' (default: 'euler'). Could also be a SciPy integrator (e.g., scipy.integrate.RK45). If the model is discrete, this parameter will raise an exception.

    Additional parameters specific to the model

    Raises
    ------
        TypeError

    Example
    -------
        >>> m = PrognosticsModel(process_noise=3.2)
        Traceback (most recent call last):
            ...
        TypeError: Must have `states` attribute
        >>> m2 = PrognosticsModel(integration_method='rk4')
        Traceback (most recent call last):
            ...
        TypeError: Must have `states` attribute
        >>> m3 = PrognosticsModel(integration_method=sp.integrate.RK45)
        Traceback (most recent call last):
            ...
        NameError: name 'sp' is not defined

    Attributes
    ----------
        is_vectorized : bool, optional
            True if the model is vectorized, False otherwise. Default is False
        default_parameters : dict[str, float], optional
            Default parameters for the model class
        parameters : dict[str, float]
            Parameters for the specific model object. This is created automatically from the default_parameters and kwargs
        state_limits: dict[str, tuple[float, float]], optional
            Limits on the state variables format {'state_name': (lower_limit, upper_limit)}
        param_callbacks : dict[str, list[function]], optional
            Callbacks for derived parameters
        inputs: list[str], optional
            Identifiers for each :term:`input`
        states: list[str]
            Identifiers for each :term:`state`
        outputs: list[str], optional
            Identifiers for each :term:`output`
        performance_metric_keys: list[str], optional
            Identifiers for each performance metric
        events: list[str], optional
            Identifiers for each :term:`event` predicted
        StateContainer : DictLikeMatrixWrapper
            Class for state container - used for representing :term:`state`
        OutputContainer : DictLikeMatrixWrapper
            Class for output container - used for representing :term:`output`
        InputContainer : DictLikeMatrixWrapper
            Class for input container - used for representing :term:`input`
    """
    is_vectorized = False

    # Configuration Parameters for model
    default_parameters = {
        'process_noise': 0.0,
        'measurement_noise': 0.0
    }

    # Configurable state range limit
    state_limits = {
        # 'state': (lower_limit, upper_limit)
    }

    # inputs = []     # Identifiers for each input
    # states = []     # Identifiers for each state
    # outputs = []    # Identifiers for each output
    # performance_metric_keys = []  # Identifies for each performance metric
    # events = []       # Identifiers for each event
    param_callbacks = {}  # Callbacks for derived parameters

    SimulationResults = namedtuple(
        'SimulationResults',
        ['times', 'inputs', 'states', 'outputs', 'event_states'])

    def __init__(self, **kwargs):
        # Default params for any model
        params = PrognosticsModel.default_parameters.copy()

        # Add params specific to the model
        params.update(self.__class__.default_parameters)

        # Add params specific passed via command line arguments
        params.update(kwargs)

        PrognosticsModel.__setstate__(self, params)

    def __eq__(self, other: "PrognosticsModel") -> bool:
        """
        Check if two models are equal
        """
        return self.__class__ == other.__class__ and self.parameters == other.parameters

    def __str__(self) -> str:
        return "{} Prognostics Model (Events: {})".format(type(self).__name__, self.events)

    def __getstate__(self) -> dict:
        return self.parameters.data

    def __setstate__(self, params: dict) -> None:
        # This method is called when de-pickling and in construction.
        # It builds the model from the parameters
        
        if not hasattr(self, 'inputs'):
            self.inputs = []
        self.n_inputs = len(self.inputs)

        if not hasattr(self, 'states'):
            raise TypeError('Must have `states` attribute')
        if not isinstance(self.states, abc.Iterable):
            raise TypeError(f'model.states must be a list or set, was {type(self.states)}')
        self.n_states = len(self.states)

        if not hasattr(self, 'events'):
            self.events = []
        if not isinstance(self.events, abc.Iterable):
            raise TypeError(f'model.events must be a list or set, was {type(self.events)}')
        self.n_events = len(self.events)

        if not hasattr(self, 'outputs'):
            self.outputs = []
        if not isinstance(self.outputs, abc.Iterable):
            raise TypeError(f'model.outputs must be a list or set, was {type(self.outputs)}')
        self.n_outputs = len(self.outputs)

        if not hasattr(self, 'performance_metric_keys'):
            self.performance_metric_keys = []
        self.n_performance = len(self.performance_metric_keys)

        # Setup Containers
        # These containers should be used instead of dictionaries for models 
        # that use the internal matrix state

        states = self.states

        class StateContainer(DictLikeMatrixWrapper):
            def __init__(self, data):
                super().__init__(states, data)
        self.StateContainer = StateContainer

        inputs = self.inputs

        class InputContainer(DictLikeMatrixWrapper):
            def __init__(self, data):
                super().__init__(inputs, data)
        self.InputContainer = InputContainer

        outputs = self.outputs

        class OutputContainer(DictLikeMatrixWrapper):
            def __init__(self, data):
                super().__init__(outputs, data)
        self.OutputContainer = OutputContainer

        self.parameters = PrognosticsModelParameters(self, params, self.param_callbacks)

    def __getitem__(self, arg):
        return self.parameters[arg]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def initialize(self, u=None, z=None):
        """
        Calculate initial state given inputs and outputs. If not defined for a model, it will return parameters['x0']

        Parameters
        ----------
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        z : OutputContainer
            Outputs, with keys defined by model.outputs \n
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3}) given outputs = ['t', 'v']

        Returns
        -------
        x : StateContainer
            First state, with keys defined by model.states \n
            e.g., x = StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Example
        -------
            :
            >>> from progpy.models import BatteryCircuit
            >>> m = BatteryCircuit()    # Replace above with specific model being simulated ^
            >>> u = m.InputContainer({'u1': 3.2})
            >>> z = m.OutputContainer({'z1': 2.2})
            >>> x = m.initialize(u, z)  # Initialize first state
        """
        return self.StateContainer(self.parameters['x0'])

    def apply_measurement_noise(self, z):
        """
        Apply measurement noise to the measurement

        Parameters
        ----------
        z : OutputContainer
            output, with keys defined by model.outputs \n
            e.g., z = m.OutputContainer({'abc': 332.1, 'def': 221.003}) given outputs = ['abc', 'def']

        Returns
        -------
        z : OutputContainer
            output, with applied noise, with keys defined by model.outputs \n
            e.g., z = m.OutputContainer({'abc': 332.2, 'def': 221.043}) given outputs = ['abc', 'def']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit()
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> z = m.apply_measurement_noise(z)

        Note
        ----
        Configured using parameters `measurement_noise` and `measurement_noise_dist`
        """
        z.matrix += np.random.normal(0, self.parameters['measurement_noise'].matrix, size=z.matrix.shape)
        return z

    def apply_process_noise(self, x, dt: float = 1):
        """
        Apply process noise to the state

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        dt : float, optional
            Time step (e.g., dt = 0.1)

        Returns
        -------
        x : StateContainer
            state, with applied noise, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.2, 'def': 221.043}) given states = ['abc', 'def']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> u = m.InputContainer({'u1': 3.2})
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> x = m.apply_process_noise(x)    # {'tb': 292.04728504765023, 'qb': 7856.398056966799, 'qcp': 0.1736435913459429, 'qcs': 0.053705829558158086}

        Note
        ----
        Configured using parameters `process_noise` and `process_noise_dist`
        """
        x.matrix += dt * np.random.normal(0, self.parameters['process_noise'].matrix, size=x.matrix.shape)
        return x

    def dx(self, x, u):
        """
        Calculate the first derivative of state `x` at a specific time `t`, given state and input

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']

        Returns
        -------
        dx : StateContainer
            First derivative of state, with keys defined by model.states \n
            e.g., dx = m.StateContainer({'abc': 3.1, 'def': -2.003}) given states = ['abc', 'def']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit()  # Replace with specific model being simulated
        >>> u = m.InputContainer({'i': 2.0})
        >>> z = m.OutputContainer({'v': 3.2, 't': 295})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> dx = m.dx(x, u) # Returns first derivative of state given input u

        See Also
        --------
        next_state

        Note
        ----
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models,
        and `next_state` for discrete, where the behavior cannot be described by the first derivative
        """
        raise NotImplementedError('dx not defined - please use next_state()')

    def next_state(self, x, u, dt: float):
        """
        State transition equation: Calculate next state

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        u : InputContainer
            Inputs, with keys defined by model.inputs \n
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        dt : float
            Timestep size in seconds (≥ 0) \n
            e.g., dt = 0.1

        Returns
        -------
        x : StateContainer
            Next state, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> u = m.InputContainer({'u1': 3.2})
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> x = m.next_state(x, u, 0.1) # Returns state at 3.1 seconds given input u

        ???{'tb': 292.1, 'qb': nan, 'qcp': nan, 'qcs': nan}

        See Also
        --------
        dx

        Note
        ----
        A model should overwrite either `next_state` or `dx`. Override `dx` for continuous models, and `next_state` for discrete, where the behavior cannot be described by the first derivative
        """
        dx = self.dx(x, u)
        return self.StateContainer({key: x[key] + dx[key]*dt for key in dx.keys()})

    @property
    def is_continuous(self) -> bool:
        """
        Returns
        -------
        is_continuous : bool
            True if model is continuous, False if discrete
        """
        return type(self).dx != PrognosticsModel.dx

    @property
    def is_discrete(self) -> bool:
        """
        Returns
        -------
        is_discrete : bool
            True if model is discrete, False if continuous
        """
        return type(self).dx == PrognosticsModel.dx

    def apply_limits(self, x):
        """
        Apply state bound limits. Any state outside of limits will be set to the closest limit.

        Parameters
        ----------
        x : StateContainer or dict
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        x : StateContainer or dict
            Bounded state, with keys defined by model.states
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> u = m.InputContainer({'u1': 3.2})
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> x = m.apply_limits(x) # Returns bounded state
        """
        for (key, limit) in self.state_limits.items():
            if np.any(np.array(x[key]) < limit[0]):
                warn("State {} limited to {} (was {})".format(key, limit[0], x[key]), ProgModelStateLimitWarning)
                x[key] = np.maximum(x[key], limit[0])
            if np.any(np.array(x[key]) > limit[1]):
                warn("State {} limited to {} (was {})".format(key, limit[1], x[key]), ProgModelStateLimitWarning)
                x[key] = np.minimum(x[key], limit[1])
        return x

    def performance_metrics(self, x) -> dict:
        """
        Calculate performance metrics where

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        
        Returns
        -------
        pm : dict
            Performance Metrics, with keys defined by model.performance_metric_keys. \n
            e.g., pm = {'tMax':33, 'iMax':19} given performance_metric_keys = ['tMax', 'iMax']

        Example
        -------
        >>> from progpy.models import BatteryElectroChemEOD
        >>> m = BatteryElectroChemEOD() # Replace with specific model being simulated
        >>> u = m.InputContainer({'u1': 3.2})
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> pm = m.performance_metrics(x)   # {'max_i': array([8.83810109])}
        """
        return {}

    observables = performance_metrics  # For backwards compatibility

    def output(self, x):
        """
        Calculate :term:`output` given state

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        z : OutputContainer
            Outputs, with keys defined by model.outputs. \n
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3}) given outputs = ['t', 'v']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> u = m.InputContainer({'u1': 3.2})
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> z = m.output(x)
        {'t': 292.1, 'v': 4.182999999010731}

        """
        if self.is_direct:
            warn_once('This Direct Model does not support output estimation. Did you mean to call time_of_event?')
        else:
            warn_once('This model does not support output estimation.')
        return self.OutputContainer({})

    def __output(self, x):
        """
        Calls output, which calculates next state forward one timestep, and then adds noise

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states \n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        z : OutputContainer
            Outputs, with keys defined by model.outputs. \n
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3} )given outputs = ['t', 'v']

        """

        # Calculate next state, forward one timestep
        z = self.output(x)

        # Add measurement noise
        return self.apply_measurement_noise(z)

    def event_state(self, x) -> dict:
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occurred))

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states\n
            e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

        Returns
        -------
        event_state : dict
            Event States, with keys defined by prognostics_model.events.\n
            e.g., event_state = {'EOL':0.32} given events = ['EOL']

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> u = m.InputContainer({'u1': 3.2})
        >>> z = m.OutputContainer({'z1': 2.2})
        >>> x = m.initialize(u, z) # Initialize first state
        >>> event_state = m.event_state(x)  # {'EOD': 1.0}

        Note
        ----
        If not overridden, will return 0.0 if threshold_met returns True, otherwise 1.0. If neither threshold_met or event_state is overridden, will return an empty dictionary (i.e., no events)

        See Also
        --------
        threshold_met
        """
        if type(self).threshold_met == PrognosticsModel.threshold_met:
            # Neither Threshold Met nor Event States are overridden
            return {}

        return {key: 1.0-float(t_met)
            for (key, t_met) in self.threshold_met(x).items()} 
    
    def threshold_met(self, x) -> dict:
        """
        For each event threshold, calculate if it has been met

        Args:
            x (StateContainer):
                state, with keys defined by model.states\n
                e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        
        Returns:
            thresholds_met (dict):
                If each threshold has been met (bool), with keys defined by prognostics_model.events\n
                e.g., thresholds_met = {'EOL': False} given events = ['EOL']

        Example:
            >>> from progpy.models import BatteryCircuit
            >>> m = BatteryCircuit() # Replace with specific model being simulated
            >>> u = m.InputContainer({'u1': 3.2})
            >>> z = m.OutputContainer({'z1': 2.2})
            >>> x = m.initialize(u, z) # Initialize first state
            >>> threshold_met = m.threshold_met(x)  # {'EOD': False}

        Note:
            If not overridden, will return True if event_state is <= 0, otherwise False. If neither threshold_met or event_state is overridden, will return an empty dictionary (i.e., no events)
        
        See Also:
            event_state
        """
        if type(self).event_state == PrognosticsModel.event_state:
            # Neither Threshold Met nor Event States are overridden
            return {}

        return {key: event_state <= 0
            for (key, event_state) in self.event_state(x).items()} 

    @property
    def is_state_transition_model(self) -> bool:
        """
        .. versionadded:: 1.5.0

        If the model is a "state transition model" - i.e., a model that uses state transition differential equations to propagate state forward.

        Returns:
            bool: if the model is a state transition model
        """
        has_default_next_state = type(self).next_state == PrognosticsModel.next_state
        has_integrator_next_state = type(self).next_state in next_state_functions.values()
        has_overridden_transition = not (has_default_next_state or has_integrator_next_state) or type(self).dx != PrognosticsModel.dx
        return has_overridden_transition and len(self.states) > 0

    @property
    def is_direct(self) -> bool:
        """
        .. versionadded:: 1.5.0
        
        If the model is a "direct model" - i.e., a model that directly estimates time of event from system state, rather than using state transition. This is useful for data-driven models that map from sensor data to time of event, and for physics-based models where state transition differential equations can be solved.

        Returns:
            bool: if the model is a direct model
        """
        return type(self).time_of_event != PrognosticsModel.time_of_event

    def state_at_event(self, x, future_loading_eqn = lambda t,x=None: {}, **kwargs):
        """
        .. versionadded:: 1.5.0

        Calculate the :term:`state` at the time that each :term:`event` occurs (i.e., the event :term:`threshold` is met). state_at_event can be implemented by a direct model. For a state transition model, this returns the state at which threshold_met returns true for each event.

        Args:
            x (StateContainer):
                state, with keys defined by model.states \n
                e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
            future_loading_eqn (abc.Callable, optional):
                Function of (t) -> z used to predict future loading (output) at a given time (t). Defaults to no outputs

        Returns:
            state_at_event (dict[str, StateContainer]):
                state at each events occurrence, with keys defined by model.events \n
                e.g., state_at_event = {'impact': {'x1': 10, 'x2': 11}, 'falling': {'x1': 15, 'x2': 20}} given events = ['impact', 'falling'] and states = ['x1', 'x2']

        Note:
            Also supports arguments from :py:meth:`simulate_to_threshold`

        See Also:
            threshold_met
        """
        params = {
            'future_loading_eqn': future_loading_eqn,
        }
        params.update(kwargs)

        events_remaining = self.events.copy()
        t = 0
        state_at_event = {}
        while len(events_remaining) > 0:
            result = self.simulate_to_threshold(x = x, t0 = t, **params)
            for key, value in result.event_states[-1].items():
                if value <= 0 and key not in state_at_event:
                    events_remaining.remove(key)
                    state_at_event[key] = result.states[-1]
            x = result.states[-1]
            t = result.times[-1]
        return state_at_event

    def time_of_event(self, x, future_loading_eqn = lambda t,x=None: {}, **kwargs) -> dict:
        """
        .. versionadded:: 1.5.0

        Calculate the time at which each :term:`event` occurs (i.e., the event :term:`threshold` is met). time_of_event must be implemented by any direct model. For a state transition model, this returns the time at which threshold_met returns true for each event. A model that implements this is called a "direct model".

        Args:
            x (StateContainer):
                state, with keys defined by model.states \n
                e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
            future_loading_eqn (abc.Callable, optional)
                Function of (t) -> z used to predict future loading (output) at a given time (t). Defaults to no outputs

        Returns:
            time_of_event (dict)
                time of each event, with keys defined by model.events \n
                e.g., time_of_event = {'impact': 8.2, 'falling': 4.077} given events = ['impact', 'falling']

        Note:
            Also supports arguments from :py:meth:`simulate_to_threshold`

        See Also:
            threshold_met
        """
        params = {
            'future_loading_eqn': future_loading_eqn,
        }
        params.update(kwargs)

        events_remaining = self.events.copy()
        t = 0
        time_of_event = {}
        while len(events_remaining) > 0:
            result = self.simulate_to_threshold(x = x, t0 = t, **params)
            for key, value in result.event_states[-1].items():
                if value <= 0 and key not in time_of_event:
                    events_remaining.remove(key)
                    time_of_event[key] = result.times[-1]
            x = result.states[-1]
            t = result.times[-1]
        return time_of_event

    def simulate_to(self, time : float, future_loading_eqn: abc.Callable = lambda t,x=None: {}, first_output=None, **kwargs) -> namedtuple:
        """
        Simulate prognostics model for a given number of seconds

        Parameters
        ----------
        time : float
            Time to which the model will be simulated in seconds (≥ 0.0) \n
            e.g., time = 200
        future_loading_eqn : abc.Callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)
        first_output : OutputContainer, optional
            First measured output, needed to initialize state for some classes. Can be omitted for classes that don't use this
        
        Returns
        -------
        times: list[float]
            Times for each simulated point
        inputs: SimResult
            Future input (from future_loading_eqn) for each time in times
        states: SimResult
            Estimated states for each time in times
        outputs: SimResult
            Estimated outputs for each time in times
        event_states: SimResult
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurrence, for each time in times
        
        Raises
        ------
        ProgModelInputException

        Note:
            See simulate_to_threshold for supported keyword arguments

        See Also
        --------
        simulate_to_threshold

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> def future_load_eqn(t, x = None):
        ...     if t < 5.0: # Load is 2.0 for first 5 seconds
        ...         return m.InputContainer({'i': 2.0})
        ...     else:
        ...         return m.InputContainer({'i': 2.2})
        >>> first_output = m.OutputContainer({'v': 3.2, 't': 295})
        >>> (results) = m.simulate_to(200, future_load_eqn, first_output)

        """
        # Input Validation
        if not isinstance(time, Number) or time < 0:
            raise ValueError("'time' must be positive, was {} (type: {})".format(time, type(time)))

        # Override threshold_met_eqn and horizon
        kwargs['thresholds_met_eqn'] = lambda x: False
        kwargs['horizon'] = time

        return self.simulate_to_threshold(future_loading_eqn, first_output, **kwargs)
 
    def simulate_to_threshold(self, future_loading_eqn: abc.Callable = None, first_output = None, events: list = None, **kwargs) -> namedtuple:
        """
        Simulate prognostics model until any or specified threshold(s) have been met

        Parameters
        ----------
        future_loading_eqn : abc.Callable
            Function of (t) -> z used to predict future loading (output) at a given time (t)

        Keyword Arguments
        -----------------
        events: abc.Sequence[str] or str, optional
            Keys for events that will trigger the end of simulation.
            If blank, simulation will occur if any event will be met ()
        event_strategy: str, optional
            Strategy for stopping evaluation. Default is 'first'. One of:\n
            * *first*: Will stop when first event in `events` list is reached.\n
            * *all*: Will stop when all events in `events` list have been reached
        t0 : float, optional
            Starting time for simulation in seconds (default: 0.0) \n
        dt : float, tuple, str, or function, optional
            float: constant time step (s), e.g. dt = 0.1\n
            function (t, x) -> dt\n
            tuple: (mode, dt), where modes could be constant or auto. If auto, dt is maximum step size\n
            str: mode - 'auto' or 'constant'\n
        integration_method: str, optional
            Integration method, e.g. 'rk4' or 'euler' (default: 'euler')
        save_freq : float, optional
            Frequency at which output is saved (s), e.g., save_freq = 10. A save_freq of 0 will save every step. \n
        save_pts : list[float], optional
            Additional ordered list of custom times where output is saved (s), e.g., save_pts= [50, 75] \n
        eval_pts : list[float], optional
            Additional ordered list of custom times where simulation is guarenteed to be evaluated (though results are not saved, as with save_pts) when dt is auto (s), e.g., eval_pts= [50, 75] \n
        horizon : float, optional
            maximum time that the model will be simulated forward (s), e.g., horizon = 1000 \n
        first_output : OutputContainer, optional
            First measured output, needed to initialize state for some classes. Can be omitted for classes that don't use this
        x : StateContainer, optional
            initial state, e.g., x= m.StateContainer({'x1': 10, 'x2': -5.3})\n
        thresholds_met_eqn : abc.Callable, optional
            custom equation to indicate logic for when to stop sim f(thresholds_met) -> bool\n
        print : bool, optional
            toggle intermediate printing, e.g., print = True\n
            e.g., m.simulate_to_threshold(eqn, z, dt=0.1, save_pts=[1, 2])
        progress : bool, optional
            toggle progress bar printing, e.g., progress = True\n
    
        Returns
        -------
        times: list[float]
            Times for each simulated point
        inputs: SimResult
            Future input (from future_loading_eqn) for each time in times
        states: SimResult
            Estimated states for each time in times
        outputs: SimResult
            Estimated outputs for each time in times
        event_states: SimResult
            Estimated event state (e.g., SOH), between 1-0 where 0 is event occurrence, for each time in times
        
        Raises
        ------
        ValueError

        See Also
        --------
        simulate_to

        Example
        -------
        >>> from progpy.models import BatteryCircuit
        >>> m = BatteryCircuit() # Replace with specific model being simulated
        >>> def future_load_eqn(t, x = None):
        ...    if t< 5.0: # Load is 3.0 for first 5 seconds
        ...        return m.InputContainer({'i': 2.0})
        ...    else:
        ...        return m.InputContainer({'i': 2.2})
        >>> first_output = m.OutputContainer({'v': 3.2, 't': 295})
        >>> (results) = m.simulate_to_threshold(future_load_eqn, first_output)

        Note
        ----
        configuration of the model is set through model.parameters.\n
        """
        # Input Validation
        if first_output and not all(key in first_output for key in self.outputs):
            raise ValueError("Missing key in 'first_output', must have every key in model.outputs")

        if future_loading_eqn is None:
            future_loading_eqn = lambda t,x=None: self.InputContainer({})
        elif not (callable(future_loading_eqn)):
            raise ValueError("'future_loading_eqn' must be callable f(t)")

        if 'threshold_keys' in kwargs:
            warn('Please use the keyword argument `events` instead of `threshold_keys`')
            if events is None:
                events = kwargs['threshold_keys']
            else:
                warn('Both `events` and `threshold_keys` were set. `events` will be used.')

        if events is None:
            events = self.events.copy()
        if not isinstance(events, abc.Iterable):
            # must be string or list-like
            raise TypeError(f'`events` must be a single event string or list of events. Was unsupported type {type(events)}.')
        if isinstance(events, str):
            # A single event
            events = [events]
        if (events is not None) and not all([key in self.events for key in events]):
            raise ValueError("`events` must be event names")
        if not isinstance(events, list):
            # Change to list because of the limits of jsonify
            events = list(events)

        # Configure
        config = {  # Defaults
            't0': 0.0,
            'dt': ('auto', 1.0),
            'eval_pts': [],
            'event_strategy': 'first',
            'save_pts': [],
            'save_freq': 10.0,
            'horizon': 1e100,  # Default horizon (in s), essentially inf
            'print': False,
            'x': None,
            'progress': False
        }
        config.update(kwargs)
        
        # Configuration validation
        if not isinstance(config['dt'], (Number, tuple, str)) and not callable(config['dt']):
            raise TypeError("'dt' must be a number or function, was a {}".format(type(config['dt'])))
        if isinstance(config['dt'], Number) and config['dt'] < 0:
            raise ValueError("'dt' must be positive, was {}".format(config['dt']))
        if not isinstance(config['save_freq'], Number) and not isinstance(config['save_freq'], tuple):
            raise TypeError("'save_freq' must be a number, was a {}".format(type(config['save_freq'])))
        if (isinstance(config['save_freq'], Number) and config['save_freq'] < 0) or \
            (isinstance(config['save_freq'], tuple) and config['save_freq'][1] < 0):
            raise ValueError("'save_freq' must be positive, was {}".format(config['save_freq']))
        if not isinstance(config['save_pts'], abc.Iterable):
            raise TypeError("'save_pts' must be list or array, was a {}".format(type(config['save_pts'])))
        if not isinstance(config['horizon'], Number):
            raise TypeError("'horizon' must be a number, was a {}".format(type(config['horizon'])))
        if config['horizon'] < 0:
            raise ValueError("'horizon' must be positive, was {}".format(config['horizon']))
        if config['x'] is not None and not all([state in config['x'] for state in self.states]):
            raise ValueError("'x' must contain every state in model.states")
        if 'thresholds_met_eqn' in config and not callable(config['thresholds_met_eqn']):
            raise TypeError("'thresholds_met_eqn' must be callable (e.g., function or lambda)")
        if 'thresholds_met_eqn' in config and config['thresholds_met_eqn'].__code__.co_argcount != 1:
            raise ValueError("'thresholds_met_eqn' must accept one argument (thresholds)-> bool")
        if not isinstance(config['print'], bool):
            raise TypeError("'print' must be a bool, was a {}".format(type(config['print'])))

        # Setup
        t = config['t0']
        u = future_loading_eqn(t)
        if config['x'] is not None:
            x = deepcopy(config['x'])
        else:
            x = self.initialize(u, first_output)

        if not isinstance(x, self.StateContainer):
            x = self.StateContainer(x)
        
        # Optimization
        output = self.__output
        threshold_met_eqn = self.threshold_met
        event_state = self.event_state
        load_eqn = future_loading_eqn
        next_state = self.next_state
        apply_noise = self.apply_process_noise
        apply_limits = self.apply_limits
        progress = config['progress']

        # Threshold Met Equations
        if config['event_strategy'] in ('first', 'any'):
            def check_thresholds(thresholds_met):
                t_met = [thresholds_met[key] for key in events]
                if len(t_met) > 0 and not np.isscalar(list(t_met)[0]):
                    return np.any(t_met)
                return any(t_met)
        elif config['event_strategy'] == 'all':
            def check_thresholds(thresholds_met):
                t_met = [thresholds_met[key] for key in events]
                if len(t_met) > 0 and not np.isscalar(list(t_met)[0]):
                    return np.all(t_met)
                return all(t_met)
        else:
            raise ValueError(f"Invalid value for `event_strategy`: {config['event_strategy']}. Should be either 'all' or 'first'")

        if 'thresholds_met_eqn' in config:
            check_thresholds = config['thresholds_met_eqn']
            events = []
        elif events is None: 
            # Note: Setting events to be all events if it is None
            events = self.events
        elif len(events) == 0:
            check_thresholds = lambda _: False

        if len(events) == 0 and config.get('thresholds_met_eqn', None) is None and 'horizon' not in kwargs:
            raise ValueError("Running simulate to threshold for a model with no events requires a horizon to be set. Otherwise simulation would never end.")

        # Initialization of save arrays
        saved_times = []
        saved_inputs = []
        saved_states = []  
        saved_outputs = []
        saved_event_states = []
        horizon = t+config['horizon']
        if isinstance(config['save_freq'], tuple):
            # Tuple used to specify start and frequency
            t_step = config['save_freq'][1]
            # Use starting time or the next multiple
            t_start = config['save_freq'][0]
            start = max(t_start, t - (t-t_start)%t_step)
            iterator = itertools.count(start, t_step)
        else:
            # Otherwise - start is t0
            t_step = config['save_freq']
            iterator = itertools.count(t, t_step)
        next(iterator) # Skip current time
        next_save = next(iterator)
        save_pt_index = 0
        save_pts = config['save_pts']
        eval_pt_index = 0
        eval_pts = config['eval_pts'].copy()  # Copy because we may change it

        # configure optional intermediate printing
        if config['print']:
            def update_all():
                saved_times.append(t)
                saved_inputs.append(u)
                saved_states.append(deepcopy(x))  # Avoid optimization where x is not copied
                saved_outputs.append(output(x))
                saved_event_states.append(event_state(x))
                print("Time: {}\n\tInput: {}\n\tState: {}\n\tOutput: {}\n\tEvent State: {}\n"
                    .format(
                        saved_times[-1],
                        saved_inputs[-1],
                        saved_states[-1],
                        saved_outputs[-1],
                        saved_event_states[-1]))  
        else:
            def update_all():
                saved_times.append(t)
                saved_inputs.append(u)
                saved_states.append(deepcopy(x))  # Avoid optimization where x is not copied

        # configuring next_time function to define prediction time step, default is constant dt
        if callable(config['dt']):
            next_time = config['dt']
            dt_mode = 'function'
        elif isinstance(config['dt'], tuple):
            dt_mode = config['dt'][0]
            dt = config['dt'][1]                
        elif isinstance(config['dt'], str):
            dt_mode = config['dt']
            if dt_mode == 'constant':
                dt = 1.0  # Default
            else:
                dt = np.inf
        else:
            dt_mode = 'constant'
            dt = config['dt']  # saving to optimize access in while loop

        if dt_mode == 'constant':
            def next_time(t, x):
                return dt
        elif dt_mode == 'auto':
            if isinstance(future_loading_eqn, Piecewise):
                eval_pts.extend(future_loading_eqn.times)
                eval_pts = sorted(eval_pts)
            def next_time(t, x=None):
                next_save_pt = save_pts[save_pt_index] if save_pt_index < len(save_pts) else float('inf')
                next_eval_pt = eval_pts[eval_pt_index] if eval_pt_index < len(eval_pts) else float('inf')
                opts = (item for item in (dt, next_save-t, next_save_pt-t, next_eval_pt-t) if item > 0)
                return min(*opts)
        elif dt_mode != 'function':
            raise ValueError(f"'dt' mode {dt_mode} not supported. Must be 'constant', 'auto', or a function")
        
        # Auto Container wrapping
        dt0 = next_time(t, x) - t
        if not isinstance(u, DictLikeMatrixWrapper):
            # Wrapper around the future loading equation
            def load_eqn(t, x):
                u = future_loading_eqn(t, x)
                return self.InputContainer(u)

        if not isinstance(next_state(x.copy(), u, dt0), DictLikeMatrixWrapper):
            # Wrapper around the next state equation
            def next_state(x, u, dt):
                x = self.next_state(x, u, dt)
                return self.StateContainer(x)

        if not isinstance(self.output(x), DictLikeMatrixWrapper):
            # Wrapper around the output equation
            def output(x):
                # Calculate output, convert to output container
                z = self.output(x)
                z = self.OutputContainer(z)

                # Add measurement noise
                return self.apply_measurement_noise(z)

        # Simulate
        update_all()
        if progress:
            simulate_progress = ProgressBar(100, "Progress")
            last_percentage = 0

        if 'integration_method' in config:
            # Update integration method for the duration of the simulation
            old_integration_method = self.parameters.get('integration_method', 'euler')
            self.parameters['integration_method'] = config['integration_method']
       
        while t < horizon:
            dt_i = next_time(t, x)
            t_load = t + dt_i/2  # Saving as separate variable reduces likelihood of floating point error
            # Use state at midpoint of step to best represent the load during the duration of the step
            # This is sometimes referred to as 'leapfrog integration'
            u = load_eqn(t_load, x)
            t += dt_i
            x = next_state(x, u, dt_i)
            x = apply_noise(x, dt_i)
            x = apply_limits(x)

            # Save if at appropriate time
            if (t >= next_save):
                next_save = next(iterator)
                update_all()
                if (save_pt_index < len(save_pts)) and (t >= save_pts[save_pt_index]):
                    # Prevent double saving when save_pt and save_freq align
                    save_pt_index += 1
            elif (save_pt_index < len(save_pts)) and (t >= save_pts[save_pt_index]):
                # (save_pt_index < len(save_pts)) covers when t is past the last savepoint
                # Otherwise save_pt_index would be out of range
                save_pt_index += 1
                update_all()
            elif (eval_pt_index < len(eval_pts)) and (t >= eval_pts[eval_pt_index]):
                # (eval_pt_index < len(eval_pts)) covers when t is past the last evaluation point
                # Otherwise eval_pt_index would be out of range
                eval_pt_index += 1

            # Update progress bar
            if config['progress']:
                percentages = [1-val for val in event_state(x).values()]
                percentages.append((t/horizon))
                converted_iteration = int(max(min(100, max(percentages)*100), 0))
                if converted_iteration - last_percentage > 1:
                    simulate_progress(converted_iteration)
                    last_percentage = converted_iteration

            # Check thresholds
            if check_thresholds(threshold_met_eqn(x)):
                break
        
        # Save final state
        if saved_times[-1] != t:
            # This check prevents double recording when the last state was a savepoint
            update_all()
        
        if not saved_outputs:
            # saved_outputs is empty, so it wasn't calculated in simulation - used cached result
            saved_outputs = LazySimResult(self.__output, saved_times, saved_states) 
            saved_event_states = LazySimResult(self.event_state, saved_times, saved_states)
        else:
            saved_outputs = SimResult(saved_times, saved_outputs, _copy=False)
            saved_event_states = SimResult(saved_times, saved_event_states, _copy=False)

        if 'integration_method' in config:
            # Reset integration method
            self.parameters['integration_method'] = old_integration_method
        
        return self.SimulationResults(
            saved_times, 
            SimResult(saved_times, saved_inputs, _copy=False), 
            SimResult(saved_times, saved_states, _copy=False), 
            saved_outputs, 
            saved_event_states
        )

    def __sizeof__(self):
        return getsizeof(self)

    def calc_error(self, times: List[float], inputs: List[InputContainer], outputs: List[OutputContainer], _loc = None, **kwargs) -> float:
        """Calculate Error between simulated and observed data using selected Error Calculation Method
        
        Args:
            times (list[float]): array of times for each sample
            inputs (list[InputContainer]): array of input dictionaries where input[x] corresponds to time[x]
            outputs (list[OutputContainer]): array of output dictionaries where output[x] corresponds to time[x]
        
        Keyword Args:
            method (str, optional): Error method to use when calculating error. Supported methods include:

              * MSE (Mean Squared Error) - DEFAULT
              * RMSE (Root Mean Squared Error)
              * MAX_E (Maximum Error)
              * MAE (Mean Absolute Error)
              * MAPE (Mean Absolute Percentage Error)
              * DTW (Dynamic Time Warping)
            x0 (StateContainer, optional): Initial state
            dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to time between samples.
            stability_tol (double, optional): Configurable cutoff
                Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
                In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
                stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
                before the model goes unstable in order to produce a valid estimate of mean squared error. 

                If the model goes unstable before stability_tol is met, a ValueError is raised.
                Else, model goes unstable after stability_tol is met, the mean squared error calculated from data up to the instability is returned.
            aggr_method (func, optional): When multiple runs are provided, users can state how to aggregate the results of the errors. Defaults to taking the mean.
            short_sim_penalty (double, optional): Only for MSE method: penalty added for simulation becoming unstable before stability_tol, added for each % below tol. If set to None, operation will return an error if simulation becomes unstable before stability_tol. Default is 100

        Returns:
            float: error

        See Also:
            :func:`calc_error.MSE`
            :func:`calc_error.RMSE`
            :func:`calc_error.MAX_E`
            :func:`calc_error.MAPE`
            :func:`calc_error.MAE`
            :func:'calc_error.DTW'
        """
        method = kwargs.get('method', 'MSE')

        method_map = {
            'mse': calc_error.MSE,
            'max_e': calc_error.MAX_E,
            'rmse': calc_error.RMSE,
            'mae': calc_error.MAE,
            'mape': calc_error.MAPE,
            'dtw': calc_error.DTW,
        }

        try:
            method = method_map[method.lower()]
        except KeyError:
            # If we get here, method is not supported
            raise KeyError(f"Error method '{method}' not supported")
        
        acceptable_types = {abc.Sequence, np.ndarray, SimResult, LazySimResult}

        if not all(isinstance(obj, tuple(acceptable_types)) for obj in [times, inputs, outputs]):
            type_error = f"Types passed in must be from the following: Sequence, np.ndarray, SimResult, or LazySimResult. Current types" \
                         f"{(' at data location (' + str(_loc) + ')' if _loc is not None else '')}" \
                         f": times = {type(times).__name__}, inputs = {type(inputs).__name__}, and outputs = {type(outputs).__name__}."
            raise TypeError(type_error)
        if len(times) != len(inputs) or len(inputs) != len(outputs):
            len_error = f"Times, inputs, and outputs must all be the same length. Current lengths" \
                        f"{(' at data location (' + str(_loc) + ')' if _loc is not None else '')}" \
                        f": times = {len(times)}, inputs = {len(inputs)}, outputs = {len(outputs)}."
            raise ValueError(len_error)
        if len(times) < 2:
            less_2_error = f"Must provide at least 2 data points for times, inputs, and outputs" \
                           f"{(' at data location (' + str(_loc) + ')' if _loc is not None else '')}."
            raise ValueError(less_2_error)

        # Determines if all values of arguments are iterables
        input_validation.all_none_iterable(times, 'times', loc=_loc) if _loc is not None else input_validation.all_none_iterable(times, 'times')
        input_validation.all_none_iterable(inputs, 'inputs', loc=_loc) if _loc is not None else input_validation.all_none_iterable(inputs, 'inputs')
        input_validation.all_none_iterable(outputs, 'outputs', loc=_loc) if _loc is not None else input_validation.all_none_iterable(outputs, 'outputs')

        dt = kwargs.get('dt', 1e99)
        aggr_method = kwargs.get('aggr_method', np.mean)
        kwargs['stability_tol'] = kwargs.get('stability_tol', 0.95)

        if isinstance(times[0], str):
            raise TypeError("Times values cannot be strings")
        if isinstance(times[0], abc.Iterable):
            # Calculate error for each
            error = []
            for r, (t, i, z) in enumerate(zip(times, inputs, outputs)):
                run_updated = str(r) if _loc is None else _loc + f', {str(r)}'
                error.append(self.calc_error(t, i, z, _loc=run_updated, **kwargs))
            return aggr_method(error)
                
        # Checks stability_tol is within bounds
        if not isinstance(kwargs['stability_tol'], Number):
            raise TypeError(f"Keyword argument 'stability_tol' must be either a int, float, or double.")
        if kwargs['stability_tol'] > 1 or kwargs['stability_tol'] <= 0:
            raise ValueError(f"Configurable cutoff must be some float value in the domain (0, 1]. Received {kwargs['stability_tol']}.")

        # Type and Value checking dt to make sure it has correctly passed in values.
        if not isinstance(dt, Number):
            raise TypeError(f"Keyword argument 'dt' must be either a int, float, or double.")
        if dt <= 0:
            raise ValueError(f"Keyword argument 'dt' must a initialized to a value greater than 0. Currently passed in {dt}.")
        
        if 'x0' in kwargs.keys() and not isinstance(kwargs['x0'], (self.StateContainer, dict)):
            raise TypeError(f"Keyword argument 'x0' must be initialized to a Dict or StateContainer, not a {type(kwargs['x0']).__name__}.")
        
        return method(self, times, inputs, outputs, **kwargs)


    def estimate_params(self, runs: List[tuple] = None, keys: List[str] = None, times: List[float] = None, inputs: List[InputContainer] = None,
                        outputs: List[OutputContainer] = None, method: str = 'nelder-mead', **kwargs) -> None:
        """Estimate the model parameters given data. Overrides model parameters

        Keyword Args:
            keys (list[str] or list[tuple[str]]):
                Parameter keys to optimize. Use tuple for nested parameters. For example, key ('x0', 'a') corresponds to m.parameters['x0']['a'].
            times (list[float]):
                Array of times for each sample
            inputs (list[InputContainer]):
                Array of input containers where input[x] corresponds to time[x]
            outputs (list[OutputContainer]):
                Array of output containers where output[x] corresponds to time[x]
            method (str, optional):
                Optimization method- see scipy.optimize.minimize for options
            tol (int, optional):
                Tolerance for termination. Depending on the provided minimization method, specifying tolerance sets solver-specific options to tol
            error_method (str, optional):
                Method to use in calculating error. See calc_error for options
            bounds (tuple or dict, optional):
                Bounds for optimization in format ((lower1, upper1), (lower2, upper2), ...) or {key1: (lower1, upper1), key2: (lower2, upper2), ...}
            options (dict, optional):
                Options passed to optimizer. see scipy.optimize.minimize for options
            runs (list[tuple], depreciated):
                data from all runs, where runs[0] is the data from run 0. Each run consists of a tuple of arrays of times, input dicts, and output dicts. Use inputs, outputs, states, times, etc. instead

        Returns:
            OptimizeResult: Scipy minimize Optimization Result from estimating parameters. See scipy's scipy.optimize.OptimizeResult documentation for details. 

        See: examples.param_est
        """
        from scipy.optimize import minimize

        if keys is None:
            # if no keys provided, use all keys that are Numbers
            keys = [key for key in self.parameters.keys() if isinstance(self.parameters[key], Number)]
        
        if isinstance(keys, set):
            raise ValueError(f"Can not pass in keys as a Set. Sets are unordered by construction, so bounds may be out of order.")
        
        for key in keys:
            if isinstance(key, (tuple, list)):
                tmp = self.parameters
                keys_so_far = ''
                for key_element in key:
                    if not isinstance(tmp, Mapping) or key_element not in tmp:
                        raise ValueError(f"Key '{keys_so_far}[{key_element}]' not in model parameters")
                    keys_so_far += f'[{key_element}]'
                    tmp = tmp[key_element]
            else:
                if key not in self.parameters:
                    raise ValueError(f"Key '{key}' not in model parameters")

        config = {
            'error_method': 'MSE',
            'bounds': tuple((-np.inf, np.inf) for _ in keys),
            'options': None,
            'tol': None
        }
        config.update(kwargs)

        if isinstance(times, set) or isinstance(inputs, set) or isinstance(outputs, set):
            raise TypeError(f"Times, inputs, and outputs cannot be a set. Sets are unordered by definition, so passing in arguments as sets may result in incorrect behavior.")

        # if parameters not in parent wrapper sequence, then place them into one.
        if isinstance(times, np.ndarray):
            times = times.tolist()
        if isinstance(inputs, np.ndarray):
            inputs = inputs.tolist()
        if isinstance(outputs, np.ndarray):
            outputs = outputs.tolist()
        if not runs and times and inputs and outputs:
            if not isinstance(times[0], (abc.Sequence, np.ndarray)):
                times = [times]
            if not isinstance(inputs[0], (abc.Sequence, np.ndarray)):
                inputs = [inputs]
            if not isinstance(outputs[0], (abc.Sequence, np.ndarray)):
                outputs = [outputs]

        # If depreciated feature runs is not provided (will be removed in future version)
        if runs is None:
            # Check if required times, inputs, and outputs are present
            missing_args = []
            for arg in ('times', 'inputs', 'outputs'):
                if locals().get(arg) is None:
                    missing_args.append(arg)
            if len(missing_args) > 0:
                # Concat into string
                missing_args_str = ', '.join(missing_args)
                raise ValueError(f"Missing keyword arguments {missing_args_str}")
            # Check lengths of args
            if len(times) != len(inputs) or len(inputs) != len(outputs):
                raise ValueError(f"Times, inputs, and outputs must be same length. Length of times: {len(times)}, Length of inputs: {len(inputs)}, Length of outputs: {len(outputs)}")
            if len(times) == 0:
                # Since inputs, times, and outputs are already confirmed to be the same length, only check that one is not empty
                raise ValueError(f"Times, inputs, and outputs must have at least one element")
            # For now- convert to runs
            runs = [(t, u, z) for t, u, z in zip(times, inputs, outputs)]

        # Convert bounds
        if isinstance(config['bounds'], dict):
            # Allows for partial bounds definition, and definition by key name
            for key in config['bounds'].keys():
                if key not in self.parameters:
                    warn(f"{key} is not a valid parameter (i.e., it is not a parameter present in this model) and should not be passed in to the bounds") 
            config['bounds'] = [config['bounds'].get(key, (-np.inf, np.inf)) for key in keys]
        else:
            if not isinstance(config['bounds'], abc.Iterable):
                raise ValueError("Bounds must be a tuple of tuples or a dict, was {}".format(type(config['bounds'])))
            if len(config['bounds']) != len(keys):
                raise ValueError(f"Bounds must be same length as keys. There were {len(config['bounds'])} Bounds given whereas there are {len(keys)} Keys. To define partial bounds, use a dict (e.g., {{'param1': {(0, 5)}, 'param3': {(-5.5, 10)}}})")
        for bound in config['bounds']:
            if (isinstance(bound, set)):
                raise TypeError(f"The Bound {bound} cannot be a Set. Sets are unordered by construction, so bounds may be out of order.")
            if (not isinstance(bound, abc.Iterable)) or (len(bound) != 2):
                raise ValueError("Each bound must be a tuple of format (lower, upper), was {}".format(type(config['bounds'])))

        if 'x0' in kwargs and not isinstance(kwargs['x0'], self.StateContainer):
            # Convert here so it isn't done every call of calc_error
            kwargs['x0'] = [self.StateContainer(x_i) for x_i in kwargs['x0']]

        # Set noise to 0
        m_noise, self.parameters['measurement_noise'] = self.parameters['measurement_noise'], 0
        p_noise, self.parameters['process_noise'] = self.parameters['process_noise'], 0

        for i, (times, inputs, outputs) in enumerate(runs):
            has_changed = False
            if len(times) != len(inputs) or len(inputs) != len(outputs):
                raise ValueError(f"Times, inputs, and outputs must be same length for the run at index {i}. Length of times: {len(times)}, Length of inputs: {len(inputs)}, Length of outputs: {len(outputs)}")
            if len(times) == 0:
                raise ValueError(f"Times, inputs, and outputs for Run {i} must have at least one element")
            if not isinstance(inputs[0], self.InputContainer):
                inputs = [self.InputContainer(u_i) for u_i in inputs]
                has_changed = True
            if isinstance(outputs, np.ndarray):
                outputs = [self.OutputContainer(u_i) for u_i in outputs]
                has_changed = True
            if has_changed:
                runs[i] = (times, inputs, outputs)

        def optimization_fcn(params):
            for key, param in zip(keys, params):
                if isinstance(key, (tuple, list)):
                    tmp = self.parameters
                    for key_element in key[:-1]:
                        tmp = tmp[key_element]
                    tmp[key[-1]] = param
                else:
                    self.parameters[key] = param
            err = 0
            for run in runs:
                try:
                    err += self.calc_error(run[0], run[1], run[2], method = config['error_method'], **kwargs)
                except Exception:
                    return 1e99 
                    # If it doesn't work (i.e., throws an error), don't use it
            return err
        
        params = []
        for key in keys:
            if isinstance(key, (tuple, list)):
                tmp = self.parameters
                for key_element in key[:-1]:
                    tmp = tmp[key_element]
                params.append(tmp[key[-1]])
            else:
                params.append(self.parameters[key])

        res = minimize(optimization_fcn, params, method=method, bounds=config['bounds'], options=config['options'], tol=config['tol'])

        if not res.success:
            warn(f"Parameter Estimation did not converge: {res.message}")

        for x, key in zip(res.x, keys):
            if isinstance(key, (tuple, list)):
                tmp = self.parameters
                for key_element in key[:-1]:
                    tmp = tmp[key_element]
                tmp[key[-1]] = x
            else:
                self.parameters[key] = x
        
        # Reset noise
        self.parameters['measurement_noise'] = m_noise
        self.parameters['process_noise'] = p_noise

        return res


    def generate_surrogate(self, load_functions: List[abc.Callable], method: str = 'dmd', **kwargs):
        """
        Generate a surrogate model to approximate the higher-fidelity model 

        Parameters
        ----------
        load_functions : List[abc.Callable]
            Each index is a callable loading function of (t, x = None) -> z used to predict future loading (output) at a given time (t) and state (x)
        method : str, optional
            list[ indicating surrogate modeling method to be used 

        Keyword Arguments
        -----------------
        dt : float or abc.Callable, optional
            Same as in simulate_to_threshold; for DMD, this value is the time step of the training data\n
        save_freq : float, optional
            Same as in simulate_to_threshold; for DMD, this value is the time step with which the surrogate model is generated  \n
        state_keys: List[str], optional
            List of state keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        input_keys: List[str], optional
            List of input keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        output_keys: List[str], optional
            List of output keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n
        event_keys: List[str], optional
            List of event_state keys to be included in the surrogate model generation. keys must be a subset of those defined in the PrognosticsModel  \n   
        ...: optional
            Keyword arguments from simulate_to_threshold (except save_pts)

        Returns
        -------
        SurrogateModel(): class
            Instance of SurrogateModel class

        Example
        -------
        See examples/generate_surrogate
        """
        from progpy.data_models import SURROGATE_METHOD_LOOKUP

        if method not in SURROGATE_METHOD_LOOKUP.keys():
            raise ValueError("Method {} not supported. Supported methods: {}".format(method, SURROGATE_METHOD_LOOKUP.keys()))

        # Configure
        config = {  # Defaults
            'save_freq': 1.0,
            'state_keys': self.states.copy(),
            'input_keys': self.inputs.copy(),
            'output_keys': self.outputs.copy(),
            'event_keys': self.events.copy(),
        }
        config.update(kwargs)

        # Validate user inputs
        if not isinstance(load_functions, abc.Iterable):
            raise TypeError(f"load_functions must be a list or list-like object, was {type(load_functions)}")

        if len(load_functions) <= 0:
            raise ValueError("load_functions must contain at least one element")
        if 'save_pts' in config.keys():
            raise ValueError("'save_pts' is not a valid input for DMD Surrogate Model.")

        if isinstance(config['input_keys'], str):
            config['input_keys'] = [config['input_keys']]
        if not all([x in self.inputs for x in config['input_keys']]):
            raise ValueError(f"Invalid 'input_keys' value ({config['input_keys']}), must be a subset of the model's inputs ({self.inputs}).")
        
        if isinstance(config['state_keys'], str):
            config['state_keys'] = [config['state_keys']]
        if not all([x in self.states for x in config['state_keys']]):
            raise ValueError(f"Invalid 'state_keys' input value ({config['state_keys']}), must be a subset of the model's states ({self.states}).")

        if isinstance(config['output_keys'], str):
            config['output_keys'] = [config['output_keys']]
        if not all([x in self.outputs for x in config['output_keys']]):
            raise ValueError(f"Invalid 'output_keys' input value ({config['output_keys']}), must be a subset of the model's outputs ({self.outputs}).")

        if isinstance(config['event_keys'], str):
            config['event_keys'] = [config['event_keys']]
        if not all([x in self.events for x in config['event_keys']]):
            raise ValueError(f"Invalid 'event_keys' input value ({config['event_keys']}), must be a subset of the model's events ({self.events}).")

        return SURROGATE_METHOD_LOOKUP[method](self, load_functions, **config)
    
    def to_json(self) -> str:
        """
        Serialize parameters as JSON objects 

        Returns:
            str: Serialized PrognosticsModel parameters as string

        See Also
        --------
        from_json

        Note
        ----
        This method only serializes the values of the prognostics model parameters (model.parameters)
        """
        return json.dumps(self.parameters.data, cls=CustomEncoder)
    
    @classmethod
    def from_json(cls, data: str):
        """
        Create a new prognostics model from a previously generated model that was serialized as a JSON object

        Args:
            data (str): 
                JSON serialized parameters necessary to build a model 
                See to_json method 

        Returns:
            PrognosticsModel: Model generated from serialized parameters 

        See Also
        ---------
        to_json

        Note
        ----
        This serialization only works for models that include all parameters necessary to generate the model in model.parameters. 
        """
        extract_parameters = json.loads(data, object_hook = custom_decoder)
 
        return cls(**extract_parameters)
