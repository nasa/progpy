# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import math
import numpy as np

from progpy import PrognosticsModel

# Derived Parameters
def update_L1(params):
    return {
        'L1': params['L'] - params['M']
    }

def update_Kv(params):
    # (rad/s)/V, inverse of Kt
    return {
        'Kv': 1.0 / params['Kt']
    }

def update_Km(params):
    # Nm/sqrt(W), motor constant, will be calculated on next line
    return {
        'Km': params['Kt'] / math.sqrt(params['R'])
    }

def update_B(params):
    # Estimate B according to empirical model 0.0051 * Km^1.9485 from:
    # Jeong et al., Improvement of Electric Propulsion System Model for Performance Analysis of Large-Size Multicopter UAVs. MDPI Applied Sciences.
    # This paper was for "large multicopter UAVs." 
    return {
        'B': 0.0051 * params['Km']**1.9485
    }

def update_J(params):
    # J is the sum of Jl (load) and Js (shaft)
    return {
        'J': params['Jl'] + params['Js']
    }


class DCMotorSP(PrognosticsModel):
    """
    .. versionadded:: 1.4.0

    :term:`Model<model>` of single-phase brushless DC Motor, as defined by the following equations:

    .. math:: \dfrac{di}{dt} = \dfrac{1}{L1}*(V-E-R*i)

    .. math:: \dfrac{d{\omega}}{dt} = \dfrac{1}{J_t} * (T_e - T_l - B * \omega)

    where:

    * i       current, A

    * :math:`\omega`   rotor speed, :math:`\dfrac{rad}{s}`

    * V       input voltage, V

    * E       back-emf voltage, V

    * R       armature resistance, Ohm

    * :math:`J_t`      total inertia (rotor + propeller or load), :math:`kg*m^2`

    * :math:`T_e`     driving torque (electrical), :math:`N*m`

    * :math:`T_l`     load torque (mechanical), :math:`N*m`

    * B       friction coefficient, :math:`\dfrac{N*m}{rad/s}`

    * t       time, :math:`s`

    :term:`Events<event>`: (0)
        | None

    :term:`Inputs/Loading<input>`: (2)
        | v: Voltage
        | t_l: Torque from load

    :term:`States<state>`: (2)
        | i: current (A)
        | v_rot: Rotational velocity (rad/sec)

    :term:`Outputs<output>`: (1)
        | v_rot: Rotational velocity (rad/sec)

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
            Self-inductance (H)
        M: float
            Mutual inductance (H)
        R: float
            Armature Resistance (Ohm)
        Kt: float
            back emf constant / Torque constant (V/rad/sec)  
        B: float
            Friction in motor / Damping (Not a function of thrust) (Nm/(rad/s))
        J: float
            Total load moment of inertia (motor shaft + load) (Kg*m^2) - alternately, you can set these separately as Js and Jl
        Js: float
            Moment of inertia of motor shaft (kg*m^2) - one component of J
        Jl: float
            Moment of inertia from load (kg*m^2) - one component of J. Note load is whatever the motor is attached to (e.g., propeller, valve, axil, etc.)
        x0 : dict[str, float]
            Initial :term:`state`
    """
    states = ['i', 'v_rot']
    inputs = ['v', 't_l']
    outputs = ['v_rot']

    param_callbacks = {
        'L': [update_L1],
        'M': [update_L1],
        'R': [update_Km, update_B],
        'Kt': [update_Kv, update_Km, update_B],
        'Km': [update_B],
        'Js': [update_J],
        'Jl': [update_J]
        }

    default_parameters = dict(L=83.0e-6,
                              M=0.0,
                              R=0.081,
                              Kt=0.0265258,
                              Js=2.69e-5,
                              Jl=1e-4,
                              x0={'i': 0.0, 'v_rot': 0.0}
                              )

    def dx(self, x, u):
        # Get parameters
        parameters     = self.parameters
        friction_coeff = parameters['B']
        inertia        = parameters['J']
        inductance     = parameters['L']

        # Get input
        load_torque    = u['t_l']
        voltage        = u['v']

        # Forcing on electrical and mechanical equations
        back_emf_v     = parameters['Kt'] * x['v_rot']
        el_torque      = parameters['Kt'] * x['i']

        # Rate of change (state-space) form
        didt    = 1.0 / inductance * (voltage - back_emf_v - parameters['R'] * x['i'])
        dvrotdt = 1.0 / inertia * (el_torque - load_torque - friction_coeff * x['v_rot'])

        return self.StateContainer(np.array([
            np.atleast_1d(didt),            # current
            np.atleast_1d(dvrotdt)          # rotor speed
        ]))

    def output(self, x):
        rotor_speed = x['v_rot']
        return self.OutputContainer(np.array([
            np.atleast_1d(rotor_speed),
            ]))
