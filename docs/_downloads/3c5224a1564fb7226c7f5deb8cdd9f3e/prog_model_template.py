# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

# INSTRUCTIONS:
# 1. Copy this file- renaming to the name of your model
# 2. Rename the class as desired
# 3. Replace the events, inputs, states, outputs keys with those specific to the model
# 4. Uncomment either dx or next_state function. dx for continuous models, and next_state for discrete
# 5. Implement logic of model in each method

# Note: To preserve vectorization use numpy math function (e.g., maximum, minimum, sign, sqrt, etc.) instead of non-vectorized functions (max, min, etc.)

from prog_models import PrognosticsModel
from numpy import inf

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

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    default_parameters = { # Set default parameters
        'Example Parameter 1': 0,
        'Example Parameter 2': 3,
        'process_noise': 0.1, # Process noise
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


    def __init__(self, **kwargs):
        """
        Constructor for model
        """
        # ADD OPTIONS CHECKS HERE

        # e.g., Checking for required parameters
        # if not 'required_param' in kwargs: 
        #   throw Exception;

        # e.g. 2, Modify parameters
        # kwargs['some_param'] = some_function(kwargs['some_param'])

        super().__init__(**kwargs) # Run Parent constructor

    # Sometimes initial input (u) and initial output (z) are needed to initialize the model
    # In that case remove the '= None' for the appropriate argument
    # Note: If they are needed, that requirement propogated through to the simulate_to* functions
    def initialize(self, u=None, z=None):
        """
        Calculate initial state given inputs and outputs

        Parameters
        ----------
        u : dict
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']

        Returns
        -------
        x : dict
            First state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE INITIAL STATE
        # NOTE: KEYS FOR x0 MATCH 'states' LIST ABOVE

        # YOU CAN ACCESS ANY PARAMETERS USING self.parameters[key]
        x0 = {
            'Examples State 1': 99.2,
            'Examples State 2': False,
            'Examples State 3': 44,
            'Examples State 4': [1, 2, 3]
        }
        return self.StateContainer(x0)

    # UNCOMMENT THIS FUNCTION FOR CONTINUOUS MODELS
    # def dx(self, t, x, u):
    #     """
    #     Returns the first derivative of state `x` at a specific time `t`, given state and input

    #     Parameters
    #     ----------
    #     t : number
    #         Current timestamp in seconds (≥ 0)
    #         e.g., t = 3.4
    #     x : dict
    #         state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     u : dict
    #         Inputs, with keys defined by model.inputs.
    #         e.g., u = {'i':3.2} given inputs = ['i']

    #     Returns
    #     -------
    #     dx : dict
    #         First derivitive of state, with keys defined by model.states
    #         e.g., dx = {'abc': 3.1, 'def': -2.003} given states = ['abc', 'def']
        
    #     Example
    #     -------
    #     | m = DerivProgModel() # Replace with specific model being simulated
    #     | u = {'u1': 3.2}
    #     | z = {'z1': 2.2}
    #     | x = m.initialize(u, z) # Initialize first state
    #     | dx = m.dx(3.0, x, u) # Returns first derivative of state at 3 seconds given input u
    #     """

    #     # REPLACE THE FOLLOWING WITH SOMETHING SPECIFC TO YOUR MODEL
    #     dxdt = {
    #         'Examples State 1': 0.1,
    #         'Examples State 2': -2.3,
    #         'Examples State 3': 4.7,
    #         'Examples State 4': 220
    #     }
    #     return self.StateContainer(dxdt)

    # UNCOMMENT THIS FUNCTION FOR DISCRETE MODELS
    # def next_state(self, t, x, u, dt):
    #     """
    #     State transition equation: Calculate next state

    #     Parameters
    #     ----------
    #     t : number
    #         Current timestamp in seconds (≥ 0)
    #         e.g., t = 3.4
    #     x : dict
    #         state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     u : dict
    #         Inputs, with keys defined by model.inputs.
    #         e.g., u = {'i':3.2} given inputs = ['i']
    #     dt : number
    #         Timestep size in seconds (≥ 0)
    #         e.g., dt = 0.1
        

    #     Returns
    #     -------
    #     x : dict
    #         Next state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
    #     """

    #     next_x = x
    #     # ADD LOGIC TO CALCULATE next_x from x

    #     return self.StateContainer(next_x)

    def output(self, t, x):
        """
        Calculate next statem, forward one timestep

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : dict
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

    def event_state(self, t, x):
        """
        Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occured))

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
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
        
    # Note: Thresholds met equation below is not strictly necessary. By default threshold_met will check if event_state is ≤ 0 for each event
    def threshold_met(self, t, x):
        """
        For each event threshold, calculate if it has been met

        Parameters
        ----------
        t : number
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        x : dict
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        thresholds_met : dict
            If each threshold has been met (bool), with deys defined by prognostics_model.events
            e.g., thresholds_met = {'EOL': False} given events = ['EOL']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE IF THRESHOLDS ARE MET
        # NOTE: KEYS FOR t_met MATCH 'events' LIST ABOVE
        t_met = {
            'Example Event': False
        }

        return t_met
