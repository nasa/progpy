# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
An example illustrating the creation of a new state estimator.

In this example a basic state estimator is constructed by subclassing the StateEstimator class. This StateEstimator is then demonstrated with a ThrownObject model
"""

from prog_algs.state_estimators import StateEstimator
from prog_algs.uncertain_data import ScalarData
import random

class BlindlyStumbleEstimator(StateEstimator):
    """
    A new state estimator. This is not a very effective state estimator, but one that technically works. It blindly stumbles towards the correct state by randomly generating a new state each timestep and selecting the state that's most consistant with the measurements.

    I do not in any universe recommend using this state estimator for anything other then demonstrating a bad state estimator. It's intended as an example of creating a new state estimation algorithm.

    This state estimator was created by copying the state estimator template and filling out each function with the logic for this algorithm
    """
    def __init__(self, model, x0, measurement = None):
        """
        Initialize the state estimator

        Args:
            model (PrognosticsModel): Model to be used in state estimation
            x0 (dict): Initial State
        """
        self.m = model
        self.state = x0

    def estimate(self, t, u, z):
        """
        Update the state estimate

        Args:
            t (Number): Time
            u (dict): Inputs (load) for time t
            z (dict): Measured output at time t
        """
        # Generate new candidate state
        x2 = {key : float(value) + 10*(random.random()-0.5) for (key,value) in self.state.items()}

        # Calculate outputs
        z_est = self.m.output(t, self.state)
        z_est2 = self.m.output(t, x2)

        # Now score them each by how close they are to the measured z
        z_est_score = sum([abs(z_est[key] - z[key]) for key in self.m.outputs])
        z_est2_score = sum([abs(z_est2[key] - z[key]) for key in self.m.outputs])

        # Now choose the closer one
        if z_est2_score < z_est_score: 
            self.state = x2

    @property
    def x(self):
        """
        Measured state
        """
        return ScalarData(self.state)

# Model used in example
class ThrownObject():
    """
    Model that similates an object thrown into the air without air resistance
    """

    inputs = [] # no inputs, no way to control
    states = [
        'x', # Position (m) 
        'v'  # Velocity (m/s)
        ]
    outputs = [ # Anything we can measure
        'x' # Position (m)
    ]
    events = [
        'falling', # Event- object is falling
        'impact' # Event- object has impacted ground
    ]

    # The Default parameters. Overwritten by passing parameters dictionary into constructor
    parameters = {
        'thrower_height': 1.83, # m
        'throwing_speed': 40, # m/s
        'g': -9.81, # Acceleration due to gravity in m/s^2
        'process_noise': 0.0 # amount of noise in each step
    }

    def initialize(self, u = None, z = None):
        self.max_x = 0.0
        return {
            'x': self.parameters['thrower_height'], # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed'] # Velocity at which the ball is thrown - this guy is an professional baseball pitcher
            }
    
    def dx(self, t, x, u = None):
        # apply_process_noise is used to add process noise to each step
        return {
            'x': x['v'],
            'v': self.parameters['g'] # Acceleration of gravity
        }

    def output(self, t, x):
        return {
            'x': x['x']
        }

    def event_state(self, t, x): 
        self.max_x = max(self.max_x, x['x']) # Maximum altitude
        return {
            'falling': max(x['v']/self.parameters['throwing_speed'],0), # Throwing speed is max speed
            'impact': max(x['x']/self.max_x,0) # 1 until falling begins, then it's fraction of height
        }

def run_example():
    # This example creates a new state estimator, instead of using the included algorihtms. 
    # The new state estimator was defined above and can now be used like the UKF or PF
    
    # First we define the model to be used with the state estimator
    m = ThrownObject()

    # Lets pretend we have no idea what the state is, we'll provide an estimate of 0
    x0 = {key : 0 for key in m.states}
    filt = BlindlyStumbleEstimator(m, x0)

    # Now lets simulate it forward and see what it looks like
    dt = 0.1
    x = m.initialize()
    print('t: {}. State: {} (Ground truth: {})'.format(0, filt.x.mean, x))
    for i in range(1, int(8.4/dt)):
        # Update ground truth state
        x = {key : x[key] + m.dx(i*dt, x)[key] * dt for key in m.states}

        # Run estimation step
        filt.estimate(i*dt, None, m.output(i*dt, x))

        # Print result
        print('t: {}. State: {} (Ground truth: {})'.format(i*dt, filt.x.mean, x))

    # The results probably should show that it is estimating the state with a significant delay

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
