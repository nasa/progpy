# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

# INSTRUCTIONS:
# 1. Copy this file- renaming to the name of your model
# 2. Rename the class as desired
# 3. Replace the events, inputs, states, outputs keys with those specific to the model
# 4. Uncomment either dx or next_state function. dx for continuous models, and next_state for discrete
# 5. Implement logic of model in each method

# Note: To preserve vectorization use numpy math function (e.g., maximum, minimum, sign, sqrt, etc.) instead of non-vectorized functions (max, min, etc.)

from numpy import inf

from prog_models import PrognosticsModel

# REPLACE THIS WITH DERIVED PARAMETER CALLBACKS (IF ANY)
# See examples.derived_params
#
# Each function defines one or more derived parameters as a function of the other parameters.
def example_callback(params):
    # Return format: dict of key: new value pair for at least one derived parameter
    return {
        "Example Parameter 1": params["Example Parameter 2"]-3
    }


class ProgModelTemplate(PrognosticsModel):
    """
    Template for Prognostics Model
    """

    # V Uncomment Below if the class is vectorized (i.e., if it can accept input to all functions as arrays) V
    # is_vectorized = True

    # REPLACE THE FOLLOWING LIST WITH EVENTS BEING PREDICTED
    events = [
        'Example Event'
    ]
    
    # REPLACE THE FOLLOWING LIST WITH INPUTS (LOADING)
    inputs = [
        'Example Input 1',
        'Example Input 2'
    ]

    # REPLACE THE FOLLOWING LIST WITH STATES
    states = [
        'Examples State 1',
        'Examples State 2',
        'Examples State 3',
        'Examples State 4'
    ]

    # REPLACE THE FOLLOWING LIST WITH OUTPUTS (MEASURED VALUES)
    outputs = [
        'Example Output 1',
        'Example Output 2'
    ]

    # REPLACE THE FOLLOWING LIST WITH PERFORMANCE METRICS
    # i.e., NON-MEASURED VALUES THAT ARE A FUNCTION OF STATE
    # e.g., maximum torque of a motor
    performance_metric_keys = [
        'metric 1',
    ]

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    # Note- everything required to configure the model
    # should be in parameters- this is to enable the serialization features
    default_parameters = {  # Set default parameters
        'Example Parameter 1': 0,
        'Example Parameter 2': 3,
        'process_noise': 0.1,  # Process noise
        'x0': {  # Initial state
            'Examples State 1': 1.5,
            'Examples State 2': -935,
            'Examples State 3': 42.1,
            'Examples State 4': 0
        }
    }

    # REPLACE THE FOLLOWING WITH STATE BOUNDS IF NEEDED
    state_limits = {
        # 'state': (lower_limit, upper_limit)
        # only specify for states with limits
        'Examples State 1': (0, inf),
        'Examples State 4': (-2, 3)
    }

    # Identify callbacks used by this model
    # See examples.derived_params
    # Format: "trigger": [callbacks]
    # Where trigger is the parameter that the derived parameters are derived from.
    # And callbacks are one or more callback functions that define parameters that are
    # derived from that parameter
    # REPLACE THIS WITH ACTUAL DERIVED PARAMETER CALLBACKS
    param_callbacks = {
        "Example Parameter 2": [example_callback]
    }

    # UNCOMMENT THIS FUNCTION IF YOU NEED CONSTRUCTION LOGIC (E.G., INPUT VALIDATION)
    # def __init__(self, **kwargs):
    #     """
    #     Constructor for model

    #     Note
    #     ----
    #     To use the JSON serialization capabilities in to_json and from_json, model.parameters must include everything necessary for initialize, including any keyword arguments.
    #     """
    #     # ADD OPTIONS CHECKS HERE

    #     # e.g., Checking for required parameters
    #     # if not 'required_param' in kwargs:
    #     #   throw Exception;

    #     super().__init__(**kwargs) # Run Parent constructor

    # Model state initialization - there are two ways to provide the logic to initialize model state.
    # 1. Provide the initial state in parameters['x0'], or
    # 2. Provide an Initialization function
    #
    # If following method 2, uncomment the initialize function, below.
    # Sometimes initial input (u) and initial output (z) are needed to initialize the model
    # In that case remove the '= None' for the appropriate argument
    # Note: If they are needed, that requirement propagated through to the simulate_to* functions
    # UNCOMMENT THIS FUNCTION FOR COMPLEX INITIALIZATION
    # def initialize(self, u=None, z=None):
    #     """
    #     Calculate initial state given inputs and outputs
    #
    #     Parameters
    #     ----------
    #     u : InputContainer
    #         Inputs, with keys defined by model.inputs.
    #         e.g., u = {'i':3.2} given inputs = ['i']
    #     z : OutputContainer
    #         Outputs, with keys defined by model.outputs.
    #         e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
    #
    #     Returns
    #     -------
    #     x : StateContainer
    #         First state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     """
    #
    #     # REPLACE BELOW WITH LOGIC TO CALCULATE INITIAL STATE
    #     # NOTE: KEYS FOR x0 MATCH 'states' LIST ABOVE
    #
    #     # YOU CAN ACCESS ANY PARAMETERS USING self.parameters[key]
    #     x0 = {
    #         'Examples State 1': 99.2,
    #         'Examples State 2': False,
    #         'Examples State 3': 44,
    #         'Examples State 4': 7.5
    #     }
    #     return self.StateContainer(x0)

    # UNCOMMENT THIS FUNCTION FOR CONTINUOUS MODELS
    # def dx(self, x, u):
    #     """
    #     Returns the first derivative of state `x` given state and input
    #
    #     Parameters
    #     ----------
    #     x : StateContainer
    #         state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     u : InputContainer
    #         Inputs, with keys defined by model.inputs.
    #         e.g., u = {'i':3.2} given inputs = ['i']
    #
    #     Returns
    #     -------
    #     dx : StateContainer
    #         First derivative of state, with keys defined by model.states
    #         e.g., dx = {'abc': 3.1, 'def': -2.003} given states = ['abc', 'def']
    # 
    #     Example
    #     -------
    #     | m = DerivProgModel() # Replace with specific model being simulated
    #     | u = {'u1': 3.2}
    #     | z = {'z1': 2.2}
    #     | x = m.initialize(u, z) # Initialize first state
    #     | dx = m.dx(x, u) # Returns first derivative of state given input u
    #     """
    #
    #     # REPLACE THE FOLLOWING WITH SOMETHING SPECIFIC TO YOUR MODEL
    #     dxdt = {
    #         'Examples State 1': 0.1,
    #         'Examples State 2': -2.3,
    #         'Examples State 3': 4.7,
    #         'Examples State 4': 220
    #     }
    #     return self.StateContainer(dxdt)

    # UNCOMMENT THIS FUNCTION FOR DISCRETE MODELS
    # def next_state(self, x, u, dt):
    #     """
    #     State transition equation: Calculate next state
    #
    #     Parameters
    #     ----------
    #     x : StateContainer
    #         state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     u : InputContainer
    #         Inputs, with keys defined by model.inputs.
    #         e.g., u = {'i':3.2} given inputs = ['i']
    #     dt : number
    #         Timestep size in seconds (≥ 0)
    #         e.g., dt = 0.1
    #
    #     Returns
    #     -------
    #     x : StateContainer
    #         Next state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     """
    #
    #     next_x = x
    #     # ADD LOGIC TO CALCULATE next_x from x
    #
    #     return self.StateContainer(next_x)

    def output(self, x):
        """
        Calculate output, z (i.e., measurable values) given the state x

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : OutputContainer
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE OUTPUTS
        # NOTE: KEYS FOR z MATCH 'outputs' LIST ABOVE
        z = self.OutputContainer({
            'Example Output 1': 0.0,
            'Example Output 2': 0.0
        })

        return z

    def event_state(self, x):
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occurred))

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        event_state : dict
            Event States, with keys defined by prognostics_model.events.
            e.g., event_state = {'EOL':0.32} given events = ['EOL']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE EVENT STATES
        # NOTE: KEYS FOR event_x MATCH 'events' LIST ABOVE
        event_x = {
            'Example Event': 0.95
        }

        return event_x
        
    # Note: Thresholds met equation below is not strictly necessary.
    # By default, threshold_met will check if event_state is ≤ 0 for each event
    def threshold_met(self, x):
        """
        For each event threshold, calculate if it has been met

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        thresholds_met : dict
            If each threshold has been met (bool), with keys defined by prognostics_model.events
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE IF THRESHOLDS ARE MET
        # NOTE: KEYS FOR t_met MATCH 'events' LIST ABOVE
        t_met = {
            'Example Event': False
        }

        return t_met

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
        | m = PrognosticsModel() # Replace with specific model being simulated
        | u = m.InputContainer({'u1': 3.2})
        | z = m.OutputContainer({'z1': 2.2})
        | x = m.initialize(u, z) # Initialize first state
        | pm = m.performance_metrics(x) # Returns {'tMax':33, 'iMax':19}
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE PERFORMANCE METRICS
        # NOTE: KEYS FOR p_metrics MATCH 'performance_metric_keys' LIST ABOVE
        p_metrics = {
            'metric1': 23
        }
        return p_metrics

    # V UNCOMMENT THE BELOW FUNCTION FOR DIRECT FUNCTIONS V
    # V i.e., a function that directly estimate ToE from  V
    # V x and future loading function                     V
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # def time_of_event(self, x, future_loading_eqn = lambda t,x=None: {}, **kwargs) -> dict:
    #     """
    #     Calculate the time at which each :term:`event` occurs (i.e., the event :term:`threshold` is met). time_of_event must be implemented by any direct model. For a state transition model, this returns the time at which threshold_met returns true for each event. A model that implements this is called a "direct model".

    #     Args:
    #         x (StateContainer):
    #             state, with keys defined by model.states \n
    #             e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    #         future_loading_eqn (abc.Callable, optional)
    #             Function of (t) -> z used to predict future loading (output) at a given time (t). Defaults to no outputs

    #     Returns:
    #         time_of_event (dict)
    #             time of each event, with keys defined by model.events \n
    #             e.g., time_of_event = {'impact': 8.2, 'falling': 4.077} given events = ['impact', 'falling']

    #     Note:
    #         Also supports arguments from :py:meth:`simulate_to_threshold`

    #     See Also:
    #         threshold_met
    #     """
    #     # REPLACE BELOW WITH LOGIC TO CALCULATE IF TIME THAT EVENT OCCURS
    #     # NOTE: KEYS FOR t_event MATCH 'events' LIST ABOVE
    #     t_event = {
    #         'Example Event': 2330
    #     }

    #     return t_event
