# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import numpy as np

from progpy import LinearModel

class FNoneNoEventStateLM(LinearModel):
    """
    Simple model that increases state by u1 every step. 
    """
    inputs = ['u1']
    states = ['x1']

    A = np.array([[0]])
    B = np.array([[1]])
    C = np.empty((0,1))
    F = None

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }

class OneInputNoOutputNoEventLM(LinearModel):
    """
    Simple model that increases state by u1 every step. 
    """
    inputs = ['u1']
    states = ['x1']

    A = np.array([[0]])
    B = np.array([[1]])
    C = np.empty((0,1))
    F = np.empty((0,1))

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }

class OneInputTwoStatesNoOutputNoEventLM(LinearModel):
    """
    Simple model that increases state by u1 every step. 
    """
    inputs = ['u1']
    states = ['x1', 'x2']

    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1], [0]])
    C = np.empty((0,2))
    F = np.empty((0,2))

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0,
            'x2': 0
        }
    }


class OneInputOneOutputNoEventLM(LinearModel):
    """
    Simple model that increases state by u1 every step. 
    """
    inputs = ['u1']
    states = ['x1']
    outputs = ['z1']

    A = np.array([[0]])
    B = np.array([[1]])
    C = np.array([[1]])
    F = np.empty((0, 1))

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }


class OneInputOneOutputOneEventLM(OneInputOneOutputNoEventLM):
    events = ['x1 == 10']
    performance_metric_keys = ['pm1']

    F = np.array([[-0.1]])
    G = np.array([[1]])

    def performance_metrics(self, x) -> dict:
        return {'pm1': x['x1'] + 1}

class OneInputOneOutputOneEventAltLM(LinearModel):
    """
    Simple model that increases state by u1 every step. Event occurs when state == 10
    """
    inputs = ['u2']
    states = ['x2']
    outputs = ['z2']
    performance_metric_keys = ['pm2']
    events = ['x2 == 5']

    A = np.array([[0]])
    B = np.array([[1]])
    C = np.array([[1]])
    F = np.array([[-0.2]])
    G = np.array([[1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x2': 0
        }
    }

    def performance_metrics(self, x) -> dict:
        return {'pm2': x['x2'] + 1}


class OneInputOneOutputNoEventLMPM(OneInputOneOutputNoEventLM):
    """
    Same as OneInputOneOutputNoEventLM, but with performance metrics defined as a function. Has a single performance metric that is always the state, plus 1
    """
    performance_metric_keys = ['x1+1']

    def performance_metrics(self, x) -> dict:
        return {'x1+1': x['x1']+1}


class OneInputNoOutputOneEventLM(OneInputNoOutputNoEventLM):
    """
    Simple model that increases state by u1 every step. Event occurs when state == 10
    """
    events = ['x1 == 10']

    F = np.array([[-0.1]])
    G = np.array([[1]])


class OneInputNoOutputTwoEventLM(LinearModel):
    """
    Simple model that increases state by u1 every step. Event occurs when state == 10, 5
    """
    inputs = ['u1', 'u2']
    states = ['x1']
    events = ['x1 == 10', 'x1 == 5']

    A = np.array([[0]])
    B = np.array([[1, 0.5]])
    C = np.empty((0,1))
    D = np.empty((0,1))
    F = np.array([[-0.1], [-0.2]])
    G = np.array([[1], [1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }


class TwoInputNoOutputOneEventLM(LinearModel):
    """
    Simple model that increases state by u1+0.5*u2 every step. Event occurs when state == 10
    """
    inputs = ['u1', 'u2']
    states = ['x1']
    events = ['x1 == 10']

    A = np.array([[0]])
    B = np.array([[1, 0.5]])
    C = np.empty((0,1))
    D = np.empty((0,1))
    F = np.array([[-0.1]])
    G = np.array([[1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }


class TwoInputNoOutputTwoEventLM(LinearModel):
    """
    Simple model that increases state by u1+0.5*u2 every step. Event occurs when state == 10
    """
    inputs = ['u1', 'u2']
    states = ['x1']
    events = ['x1 == 10', 'x1 == 5']

    A = np.array([[0]])
    B = np.array([[1, 0.5]])
    C = np.empty((0,1))
    D = np.empty((0,1))
    F = np.array([[-0.1], [-0.2]])
    G = np.array([[1], [1]])

    default_parameters = {
        'process_noise': 0,
        'x0': {
            'x1': 0
        }
    }
