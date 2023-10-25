# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from numpy import inf
from progpy import PrognosticsModel


class ParisLawCrackGrowth(PrognosticsModel):
    """
    .. versionadded:: 1.4.0
    A simple Paris Law Crack Growth :term:`model`

    Events: (1)
        CGF :   Crack Growth Fracture 

    Inputs/Loading: (2)
        |k_max :  Maximum crack growth 
        |k_min :  Minimum crack growth 

    States: (1)
        |c_l :   crack length
  
    Outputs: (1)
        |c_l :   crack length

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
        crack_limit : float
            crack length limit after which the crack growth fracture event is triggered
        c : float
            Material Constant
        m : float
            Material Constant
        dndt : float
            cycles per second
        x0 : dict[str, float]
            Initial :term:`state`
    """ 
    # Event: Crack Growth Fracture
    events = ['CGF']
    # Inputs are ['k_min', 'k_max']
    inputs = ['k_max', 'k_min']
    # State: Crack Length
    states = ['c_l']
    # Output: Crack Length
    outputs = ['c_l']

    # The default parameters
    default_parameters = {
        'crack_limit': 1e-4,
        'c': 3.24,
        'm': 0.1527,
        'dndt': 10, 
        'x0' : {
            'c_l': 0.00001,
        }  
    }
    
    state_limits = {
        'c_l': (0, inf),
    }

    # The model equations
    def dx(self, x, u):
        parameters = self.parameters
        r = (parameters['c']*(u['k_max'] - u['k_min'])**parameters['m'])*parameters['dndt']  # Paris Law Equation with respect to time
        dxdt = {'c_l': r}
        return self.StateContainer(dxdt)

    def output(self, x):
        return self.OutputContainer(x)

    def event_state(self, x) -> dict:
        return {
            'CGF': 1 - x['c_l'] / self.parameters['crack_limit']
        }

    def threshold_met(self, x):
        return {
           'CGF': x['c_l'] > self.parameters['crack_limit']
        }
