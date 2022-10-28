# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example demonstrates use of the Kalman Filter State Estimator with a LinearModel.

First, a linear model is defined. Then the KF State estimator is used with fake data to estimate state.
"""

import numpy as np
from prog_models import LinearModel
from prog_algs.state_estimators import KalmanFilter


# Linear Model for an object thrown into the air
class ThrownObject(LinearModel):
    """
    Model that similates an object thrown into the air without air resistance

    Events (2)
        | falling: The object is falling
        | impact: The object has hit the ground

    Inputs/Loading: (0)

    States: (2)
        | x: Position in space (m)
        | v: Velocity in space (m/s)

    Outputs/Measurements: (1)
        | x: Position in space (m)

    Keyword Args
    ------------
        process_noise : Optional, float or Dict[Srt, float]
          Process noise (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, String
          distribution for process noise (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or Dict[Srt, float]
          Measurement noise (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, String
          distribution for measurement noise (e.g., normal, uniform, triangular)
        g : Optional, float
            Acceleration due to gravity (m/s^2). Default is 9.81 m/s^2 (standard gravity)
        thrower_height : Optional, float
            Height of the thrower (m). Default is 1.83 m
        throwing_speed : Optional, float
            Speed at which the ball is thrown (m/s). Default is 40 m/s
    """

    inputs = []  # no inputs, no way to control
    states = [
        'x',     # Position (m) 
        'v'      # Velocity (m/s)
        ]
    outputs = [
        'x'      # Position (m)
    ]
    events = [
        'impact' # Event- object has impacted ground
    ]

    A = np.array([[0, 1], [0, 0]])
    E = np.array([[0], [-9.81]])
    C = np.array([[1, 0]])
    F = None # Will override method

    # The Default parameters. 
    # Overwritten by passing parameters dictionary into constructor
    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81  # Acceleration due to gravity in m/s^2
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self.parameters['thrower_height'],
            # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed']
            # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            })
    
    # This is actually optional. Leaving thresholds_met empty will use the event state to define thresholds.
    #  Threshold is met when Event State == 0. 
    # However, this implementation is more efficient, so we included it
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

def run_example():
    # Step 1: Instantiate the model
    m = ThrownObject(process_noise = 0, measurement_noise = 0)

    # Step 2: Instantiate the Kalman Filter State Estimator
    # Define the initial state to be slightly off of actual
    x_guess = m.StateContainer({'x': 1.75, 'v': 35}) # Guess of initial state
    # Note: actual is {'x': 1.83, 'v': 40}
    kf = KalmanFilter(m, x_guess)

    # Step 3: Run the Kalman Filter State Estimator
    # Here we're using simulated data from the thrown_object. 
    # In a real application you would be using sensor data from the system
    dt = 0.01  # Time step (s)
    print_freq = 50  # Print every print_freq'th iteration
    x = m.initialize()
    u = m.InputContainer({})  # No input for this model
    
    for i in range(500):
        # Get simulated output (would be measured in a real application)
        z = m.output(x)

        # Estimate New State
        kf.estimate(i*dt, u, z)
        x_est = kf.x.mean

        # Print Results
        if i%print_freq == 0:  # Print every print_freq'th iteration
            print(f"t: {i*dt:.2f}\n\tEstimate: {x_est}\n\tTruth: {x}")
            diff = {key: x_est[key] - x[key] for key in x.keys()}
            print(f"\t Diff: {diff}")

        # Update Real state for next step
        x = m.next_state(x, u, dt)

if __name__ == '__main__':
    run_example()
