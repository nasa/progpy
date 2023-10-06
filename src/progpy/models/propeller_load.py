# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

from progpy import PrognosticsModel


def update_Cq(params):
    return {
        'C_q': params['c_q'] * params['rho'] * pow(params['D'], 5)
    }


class PropellerLoad(PrognosticsModel):
    """
    .. versionadded:: 1.5.0

    This is a simple model of a propeller load. This model estimates load torque as a function of the rotational velocity. When the propeller is spinning faster, drag increases, and the propeller load on the torque increases.

    This model is typically used with the esc and dcmotor models to simulate a motor and propeller system.

    :term:`Events<event>`: (0)

    :term:`Inputs/Loading<input>`: (1):
        v_rot: Rotational Velocity (rad/sec)

    :term:`States<state>`: (1)
        t_l: Load Torque

    :term:`Outputs<output>`: (1):
        t_l: Load Torque
    """
    inputs = ['v_rot']
    states = ['t_l']
    outputs = ['t_l']

    param_callbacks = {
        'c_q': [update_Cq],
        'rho': [update_Cq],
        'D': [update_Cq],
    }

    default_parameters = {
        # Load parameters
        'c_q': 5.42e-7,  # coefficient of torque (APC data, derived) [dimensionless]
        'rho': 1.225,  # (Kg/m^3)
        'D': 0.381,  # (m)

        'x0': {
            't_l': 0,
        }
    }

    state_limits = {
        't_l': (0, np.inf),
    }

    def next_state(self, x, u, dt: float):
        return self.StateContainer({
            't_l': self.parameters['C_q']*u['v_rot']**2})
    
    def output(self, x):
        return x
