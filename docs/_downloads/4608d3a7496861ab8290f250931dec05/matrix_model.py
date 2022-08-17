# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example shows the use of the advanced feature - matrix models. Matrix models represent the state of the system using matricies instead of dictionaries. The provided model.StateContainer, InputContainer, and OutputContainer can be treated as dictionaries but use an underly matrix. This is important for some applications like surrogate and machine-learned models where the state is represented by a tensor, and operations by matrix operations. Simulation functions propogate the state using the matrix form, preventing the inefficiency of having to convert to and from dictionaries.

In this example, a model is designed to simulate a thrown object using matrix notation (instead of dictionary notation as in the standard model). The implementation of the model is comparable to a standard model, except that it uses the x.matrix, u.matrix, and z.matirx to compute matrix operations within each function.
"""

def run_example():
    from prog_models import PrognosticsModel
    import numpy as np

    # Define the model
    class ThrownObject(PrognosticsModel):
        # Define the model properties, this is exactly the same as for a regular model.

        inputs = []  # no inputs, no way to control
        states = [
            'x',    # Position (m) 
            'v'    # Velocity (m/s)
            ]
        outputs = [
            'x'     # Position (m)
        ]
        events = [
            'falling',  # Event- object is falling
            'impact'    # Event- object has impacted ground
        ]

        is_vectorized = True

        # The Default parameters. Overwritten by passing parameters dictionary into constructor
        default_parameters = {
            'thrower_height': 1.83,  # m
            'throwing_speed': 40,  # m/s
            'g': -9.81,  # Acceleration due to gravity in m/s^2
            'process_noise': 0.0  # amount of noise in each step
        }

        # Define the model equations
        def initialize(self, u = None, z = None):
            # Note: states are returned using StateContainer
            return self.StateContainer({
                'x': self.parameters['thrower_height'], 
                'v': self.parameters['throwing_speed']})

        def next_state(self, x, u, dt):
            # Here we will use the matrix version for each variable
            # Note: x.matrix is a column vector
            # Note: u.matrix is a column vector
            #   and u.matrix is in the order of model.inputs, above

            A = np.array([[0, 1], [0, 0]])  # State transition matrix
            B = np.array([[0], [self.parameters['g']]])  # Acceleration due to gravity
            x.matrix += (np.matmul(A, x.matrix) + B) * dt

            return x
            
        def output(self, x):
            # Note- states can still be accessed a dictionary
            return self.OutputContainer({'x': x['x']})

        # This is actually optional. Leaving thresholds_met empty will use the event state to define thresholds.
        #  Threshold = Event State == 0. However, this implementation is more efficient, so we included it
        def threshold_met(self, x):
            return {
                'falling': x['v'] < 0,
                'impact': x['x'] <= 0
            }

        def event_state(self, x): 
            x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2) # Use speed and position to estimate maximum height
            x_max = np.where(x['v'] > 0, x['x'], x_max) # 1 until falling begins
            return {
                'falling': np.maximum(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
                'impact': np.maximum(x['x']/x_max,0)  # then it's fraction of height
            }

    # Now we can use the model
    # Create the model
    thrown_object = ThrownObject()

    # Use the model
    x = thrown_object.initialize()
    print('State at 0.1 seconds: ', thrown_object.next_state(x, {}, 0.1))

    # But you can also initialize state directly, like so:
    x = thrown_object.StateContainer({'x': 1.93, 'v': 40})
    print('State at 0.1 seconds: ', thrown_object.next_state(x, None, 0.1))

    # Now lets use it for simulation.
    def future_loading(t, x=None):
        return thrown_object.InputContainer({})

    thrown_object.simulate_to_threshold(
        future_loading, 
        print = True, 
        threshold_keys = 'impact', 
        dt = 0.1, 
        save_freq = 1)

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
