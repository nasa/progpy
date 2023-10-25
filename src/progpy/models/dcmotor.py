# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

from progpy import PrognosticsModel

RAD_TO_DEG = 180/np.pi
PI2 = 2 * np.pi


def backemf(theta):
    """
    Backemf for the current opsition of rotor (theta)
    
    Parameters
    ----------
    theta : float
        Current position of rotor
        
    Returns
    -------
    tuple[float, float, float]: f_a, f_b, f_c
        f_a, f_b, f_c are the three components of the backemf
    """
    theta *= RAD_TO_DEG  # convert rad to deg
    if 0. <= theta <= 60:
        return (1, -1, -(1/30)*theta+1)
    if 60 < theta <= 120:
        return (1, (1/30)*(theta-60)-1, -1)
    if 120 < theta <= 180:
        return (-(1/30)*(theta-120)+1, 1, -1)
    if 180 < theta <= 240:
        return (-1, 1, (1/30)*(theta-180)-1)
    if 240 < theta <= 300:
        return (-1, -(1/30)*(theta-240)+1, 1)
    return ((1/30)*(theta-300)-1, -1, 1)

# Derived Paramaters
def update_L1(params):
    return {
        'L1': params['L'] - params['M']
    }

def update_R_L1(params):
    return {
        'negR_L1': -params['R']/params['L1']
    }

def update_BC(params):
    L1 = params['L1']
    J = params['J']
    return {
        'Bc': np.array([[(1/L1), 0, 0, 0],
                [0, (1/L1), 0, 0],
                [0, 0, (1/L1), 0],
                [0, 0, 0, -(1/J)],
                [0, 0, 0, 0]
                ], dtype=np.float64)
    }

def update_AC(params):
    L1 = params['L1']
    Flx = params['K']
    J = params['J']
    B = params['B']
    Po = params['Po']
    negR_L1 = params['negR_L1']
    return {'Ac': np.array([[negR_L1, 0, 0, -(Flx/L1), 0],
            [0, negR_L1, 0, -(Flx/L1), 0],
            [0, 0, negR_L1, -(Flx/L1), 0],
            [((Flx)/J), ((Flx)/J), ((Flx)/J), -(B/J), 0],
            [0, 0, 0, (Po/2), 0]
            ])
    }


class DCMotor(PrognosticsModel):
    """
    .. versionadded:: 1.3.0

    :term:`Model<model>` of triple-phase brushlessDC Motor.

    References: [0]_, [1]_.

    This model was developed by NASA's System Wide Safety (SWS) Project. https://www.nasa.gov/aeroresearch/programs/aosp/sws/

    :term:`Events<event>`: (0)
        | None

    :term:`Inputs/Loading<input>`: (4)
        | v_a, v_b, v_c: Voltages at a, b, c
        | t_l: Torque from load

    :term:`States<state>`: (5)
        | i_a, i_b, i_c: Currents provided to motor 
        | v_rot: Rotational velocity (rad/sec)
        | theta: Angle of motor (rad)

    :term:`Outputs<output>`: (2)
        | v_rot: Rotational velocity (rad/sec)
        | theta: Angle of motor (rad)

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
        L : float
            Inductance (H)
        M: float
            Mutual inductance (H)
        R: float
            Resistance (Ohm)
        K: float
            back emf constant / Torque constant (V/rad/sec)  
        B: float
            Friction in motor / Damping (Not a function of thrust) (Nm/(rad/s))
        Po: float
            no of poles in rotor 
        J: float
            Load moment of inertia (neglecting motor shaft inertia) (Kg*m^2)
        x0 : dict[str, float]
            Initial :term:`state`

    References
    ----------
    .. [0] Matteo Corbetta, Chetan S. Kulkarni. An approach for uncertainty quantification and management of unmanned aerial vehicle health. 
    Annual Conference of the PHM Society, Scottsdale, AZ, 2019. http://papers.phmsociety.org/index.php/phmconf/article/view/847
    .. [1] George E. Gorospe Jr, Chetan S. Kulkarni, Edward Hogge, Andrew Hsu, and Natalie Ownby. A Study of the Degradation of Electronic Speed Controllers forBrushless DC Motors.
    Asia Pacific Conference of the Prognostics and Health Management Society, 2017. https://ntrs.nasa.gov/citations/20200000579
    """
    states = ['i_a', 'i_b', 'i_c', 'v_rot', 'theta']
    inputs = ['v_a', 'v_b', 'v_c', 't_l']
    outputs = ['v_rot', 'theta']
    param_callbacks = {
        'L': [update_L1],
        'L1': [update_R_L1, update_BC, update_AC],
        'M': [update_L1],
        'R': [update_R_L1, update_AC],
        'J': [update_BC, update_AC],
        'K': [update_AC],
        'B': [update_AC],
        'Po': [update_AC],
    }

    default_parameters = {
        # motor parameters
        'L': 83e-6,  # (H) inductance
        'M': 0,  # (H) Mutual inductance
        'R': 0.081,  # (Ohm) Resistance
        'K': 0.0265,  # (V/rad/sec)  back emf constant / Torque constant (Nm/A) 
        'B': 0,  # Nm/(rad/s) Friction in motor / Damping (Not a function of thrust)
        'Po': 28,  # no of poles in rotor

        # Load parameters
        'J': 26.967e-6,  # (Kg*m^2) Load moment of inertia (neglecting motor shaft inertia)

        # Matricies
        'Cc': np.array([[1, 1, 1, 1, 1]], dtype=np.float64),
        'Dc': np.array([0, 0, 0, 0], dtype=np.float64),

        # Initial State
        'x0': {
            'i_a': 0,
            'i_b': 0,
            'i_c': 0,
            'v_rot': 0,
            'theta': 0
        }
    }

    def next_state(self, x, u, dt: float):
        (F_a, F_b, F_c) = backemf(x['theta'])

        Ac = self.parameters['Ac'].copy()
        Ac[0][3] *= F_a
        Ac[1][3] *= F_b
        Ac[2][3] *= F_c
        Ac[3][0] *= F_a
        Ac[3][1] *= F_b
        Ac[3][2] *= F_c
        # TODO(CT): Move F_* to U_vector

        dxdt = np.dot(Ac, x.matrix) + np.dot(self.parameters['Bc'], u.matrix)
        x.matrix += dxdt * dt  # Update inplace
        x.matrix[4] %= PI2  # Wrap angle
        return x

    def output(self, x):
        return self.OutputContainer(x.matrix[3:])
