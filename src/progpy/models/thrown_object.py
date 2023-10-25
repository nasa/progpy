# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from progpy import PrognosticsModel, LinearModel


def calc_lumped_param(params):
    return {
        'lumped_param': 0.5 * params['rho'] * params['cd'] * params['A'] / params['m']
    }


class ThrownObject(PrognosticsModel):
    """
    Simple Non-Linear :term:`model` that similates an object thrown into the air with air resistance

    :term:`Events<event>`: (2)
        | falling: The object is falling
        | impact: The object has hit the ground

    :term:`Inputs/Loading<input>`: (0)

    :term:`States<state>`: (2)
        | x: Position in space (m)
        | v: Velocity in space (m/s)

    :term:`Outputs<output>`: (1)
        | x: Position in space (m)

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
        g : Optional, float
            Acceleration due to gravity (m/s^2). Default is 9.81 m/s^2 (standard gravity)
        rho : Optional, float
            Air density (kg/m^3). Default is 1.225 (air density at sea level). Used in drag calculation
        A : Optional, float
            Cross sectional area of object (m^2)
        m : Optional, float
            Mass of object (kg)
        cd : Optional, float
            Coefficient of drag
        thrower_height : Optional, float
            Height of the thrower (m). Default is 1.83 m
        throwing_speed : Optional, float
            Speed at which the ball is thrown (m/s). Default is 40 m/s
    """

    inputs = []
    states = ['x', 'v']
    outputs = ['x']
    events = ['falling', 'impact']

    is_vectorized = True

    default_parameters = {
        'thrower_height': 1.83,
        'throwing_speed': 40,
        'g': -9.81,
        'rho': 1.225,
        'A': 0.05,
        'm': 0.145,
        'cd': 0.007,
        'process_noise': 0.0
    }

    param_callbacks = {
        'rho': [calc_lumped_param],
        'A': [calc_lumped_param],
        'm': [calc_lumped_param],
        'cd': [calc_lumped_param]
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self.parameters['thrower_height'],  # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed']   # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            })
    
    def next_state(self, x, u, dt: float):
        next_x = x['x'] + x['v']*dt
        drag_acc = self.parameters['lumped_param'] * x['v'] * x['v']
        next_v = x['v'] + (self.parameters['g'] - drag_acc*np.sign(x['v']))*dt
        return self.StateContainer(np.array([
                np.atleast_1d(next_x),
                np.atleast_1d(next_v)  # Acceleration of gravity
            ]))

    def output(self, x):
        return self.OutputContainer(np.array([[x['x']]]))

    def threshold_met(self, x) -> dict:
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    def event_state(self, x) -> dict:
        # Use speed and position to estimate maximum height
        x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2)
        # 1 until falling begins
        x_max = np.where(x['v'] > 0, x['x'], x_max)
        return {
            'falling': np.maximum(x['v']/self.parameters['throwing_speed'], 0),  # Throwing speed is max speed
            'impact': np.maximum(x['x']/x_max, 0)  # then it's fraction of height
        }


class LinearThrownObject(LinearModel):
    inputs = []
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    D = np.array([[1]])
    E = np.array([[0], [-9.81]])
    C = np.array([[1, 0]])
    F = None  # Will override method
    

    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81  # Acceleration due to gravity in m/s^2
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self.parameters['thrower_height'],  # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed']  # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            })
    
    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    def event_state(self, x):
        x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2)  # Use speed and position to estimate maximum height
        return {
            'falling': np.maximum(x['v']/self.parameters['throwing_speed'], 0),  # Throwing speed is max speed
            'impact': np.maximum(x['x']/x_max, 0) if x['v'] < 0 else 1  # 1 until falling begins, then it's fraction of height
        }
