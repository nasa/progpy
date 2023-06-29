# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import numpy as np

from progpy import LinearModel


class LinearThrownObject(LinearModel):
    inputs = [] 
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    C = np.array([[1, 0]])
    E = np.array([[0], [-9.81]])
    F = None # Will override method

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
        x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2) # Use speed and position to estimate maximum height
        return {
            'falling': np.maximum(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
            'impact': np.maximum(x['x']/x_max,0) if x['v'] < 0 else 1  # 1 until falling begins, then it's fraction of height
        }

class LinearThrownObjectNoE(LinearThrownObject):
    E = np.array([[0], [-9.81]])

class LinearThrownDiffThrowingSpeed(LinearThrownObject):
    inputs = [] 
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    C = np.array([[1, 0]])
    D = np.array([[1]])
    E = np.array([[0], [-9.81]])
    F = None # Will override method

    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 20,  # m/s
        'g': -9.81  # Acceleration due to gravity in m/s^2
    }

class LinearThrownObjectWrongB(LinearThrownObject):
    inputs = [] 
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[1, 0]])
    D = np.array([[1]])
    E = np.array([[0], [-9.81]])
    F = None # Will override method


# Wrong x stateContainer parameter. Has Throwing_speed when it should be thrower_height
class LinearThrownObjectUpdatedInitializedMethod(LinearThrownObject):
    inputs = [] 
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    C = np.array([[1, 0]])
    D = np.array([[1]])
    E = np.array([[0], [-9.81]])
    F = None # Will override method

    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81  # Acceleration due to gravity in m/s^2
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self.parameters['throwing_speed'],  # Thrown, so initial altitude is height of thrower
            })

class LinearThrownObjectDiffDefaultParams(LinearThrownObject):
    inputs = [] 
    states = ['x', 'v']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1], [0, 0]])
    C = np.array([[1, 0]])
    D = np.array([[1]])
    E = np.array([[0], [-9.81]])
    F = None # Will override method

    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81,  # Acceleration due to gravity in m/s^2
        'x': 1111
    }

class LinearThrownObjectFourStates(LinearThrownObject):
    inputs = []
    states = ['x', 'v', 'y' ,'z']
    outputs = ['x']
    events = ['impact']

    A = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    C = np.array([[0, 1, 2, 3]])
    E = np.array([[0], [1], [2], [3]])
