# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import math
from progpy import PrognosticsModel

def update_x0(params):
    return {'x0': 
        {
            'SOC': params['x0']['SOC'], 
            'v': params['v_L']}}


class SimplifiedBattery(PrognosticsModel):
    """
    .. versionadded:: 1.8.0

    Simplified battery model from [Sierra2019]_. Introduced in 2024 PHM Society Tutorial.

    :term:`Events<event>`: (2)
        EOD: End of Discharge (Complete)
        Low V: When voltage hits a specified threshold (VEOD)

    :term:`Inputs/Loading<input>`: (1)
        P: Power draw on the battery

    :term:`States<state>`: (2)
        | SOC: State of Charge
        | v: Voltage supplied by battery

    :term:`Outputs<output>`: (1)
        v: Voltage supplied by battery

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

    Note
    ---------
    Default parameters are for a Tattu battery.

    References
    -----------
    .. [Sierra2019] G. Sierra and M. Orchard and K. Goebel and C. Kulkarni, "Battery health management for small-size rotary-wing electric unmanned aerial vehicles: An efficient approach for constrained computing platforms," Reliability Engineering & System Safety, Volume 182,2019. https://www.sciencedirect.com/science/article/pii/S0951832018301406
    """

    inputs = ['P']
    states = ['SOC', 'v']
    outputs = ['v']
    events = ['EOD', 'Low V']

    state_limits = {
        'SOC': (0.0, 1.0),
        'v': (0, float('inf'))
    }

    default_parameters = {
        'E_crit': 202426.858,
        'v_L': 11.148,
        'lambda': 0.046,
        'gamma': 3.355,
        'mu': 2.759,
        'beta': 8.482,
        'R_int': 0.027,
        'VEOD': 9,

        'x0': {
            'SOC': 1,
            'v': 11.148
        }
    }

    param_callbacks = {
        'v_L': [update_x0]
    }

    def next_state(self, x, u, dt):
        x['SOC'] = x['SOC'] - u['P'] * dt / self['E_crit']

        v_oc = self['v_L'] - self['lambda']**(self['gamma']*x['SOC']) - self['mu'] * math.exp(-self['beta']* math.sqrt(x['SOC']))
        i = (v_oc - math.sqrt(v_oc**2 - 4 * self['R_int'] * u['P']))/(2 * self['R_int'])
        v = v_oc - i * self['R_int']

        x['v'] = v

        return x

    def output(self, x):
        return self.OutputContainer({
            'v': x['v']})
    
    def event_state(self, x):
        return {
            'EOD': x['SOC'],
            'Low V': (x['v'] - self['VEOD'])/(self['v_L'] - self['VEOD'])
        }
