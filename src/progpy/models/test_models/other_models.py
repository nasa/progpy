# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

from progpy import PrognosticsModel


class OneInputTwoOutputsOneEvent(PrognosticsModel):
    """
    Simple example model where x0 increases by a * u0
    """
    inputs = ['u0']
    states = ['x0']
    outputs = ['x0+b', 'x0+c']
    events = ['x0==10']

    default_parameters = {
        'x0': {  # Initial State
            'x0': 0
        },
        'a': 1,
        'b': 1,
        'c': 1
    }

    def dx(self, x, u):
        return self.StateContainer({
            'x0': self.parameters['a'] * u['u0']
        })

    def output(self, x):
        return self.OutputContainer({
            'x0+b': x['x0'] + self.parameters['b'],
            'x0+c': x['x0'] + self.parameters['c']
        })

    def event_state(self, x):
        return {
            'x0==10': 1-x['x0']/10
        }
    
    def threshold_met(self, x):
        return {'x0==10': x['x0']>=10}

class OneInputTwoOutputsOneEvent_alt(PrognosticsModel):
    """
    Simple example model where x0 increases by a * u0. Designed to be slightly different than OneInputTwoOutputsOneEvent
    """
    inputs = ['u0']
    states = ['x0']
    outputs = ['x0+d', 'x0+c']
    events = ['x0==10', 'x0==7']

    default_parameters = {
        'x0': {  # Initial State
            'x0': 0
        },
        'a': 1,
        'd': 1,
        'c': 1
    }

    def dx(self, x, u):
        return self.StateContainer({
            'x0': self.parameters['a'] * u['u0']
        })

    def output(self, x):
        return self.OutputContainer({
            'x0+d': x['x0'] + self.parameters['d'],
            'x0+c': x['x0'] + self.parameters['c']
        })

    def event_state(self, x):
        return {
            'x0==10': 1-x['x0']/10,
            'x0==7': 1-x['x0']/7
        }
    
    def threshold_met(self, x):
        return {
            'x0==10': x['x0']>=10,
            'x0==7': x['x0']>=7
        }

