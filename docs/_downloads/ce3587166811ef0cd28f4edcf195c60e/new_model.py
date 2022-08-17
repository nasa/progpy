# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example defining and testing a new model. 
"""

from prog_models import PrognosticsModel


class ThrownObject(PrognosticsModel):
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
    default_parameters = {
        'thrower_height': 1.83,  # m
        'throwing_speed': 40,  # m/s
        'g': -9.81,  # Acceleration due to gravity in m/s^2
        'process_noise': 0.0  # amount of noise in each step
    }

    def initialize(self, u, z):
        self.max_x = 0.0
        return self.StateContainer({
            'x': self.parameters['thrower_height'],  # Thrown, so initial altitude is height of thrower
            'v': self.parameters['throwing_speed']  # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            })
    
    def dx(self, x, u):
        return self.StateContainer({'x': x['v'],
                'v': self.parameters['g']})  # Acceleration of gravity

    def output(self, x):
        return self.OutputContainer({'x': x['x']})

    # This is actually optional. Leaving thresholds_met empty will use the event state to define thresholds.
    #  Threshold = Event State == 0. However, this implementation is more efficient, so we included it
    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

    def event_state(self, x): 
        self.max_x = max(self.max_x, x['x'])  # Maximum altitude
        return {
            'falling': max(x['v']/self.parameters['throwing_speed'],0),  # Throwing speed is max speed
            'impact': max(x['x']/self.max_x,0)  # 1 until falling begins, then it's fraction of height
        }

def run_example():
    # Demo model
    # Step 1: Create instance of model
    m = ThrownObject()

    # Step 2: Setup for simulation 
    def future_load(t, x=None):
        return m.InputContainer({})  # No inputs, no way to control

    # Step 3: Simulate to impact
    event = 'impact'
    simulated_results = m.simulate_to_threshold(future_load, threshold_keys=[event], dt=0.005, save_freq=1, print = True)
    
    # Print flight time
    print('The object hit the ground in {} seconds'.format(round(simulated_results.times[-1],2)))

    # OK, now lets compare performance on different heavenly bodies. 
    # This requires that we update the cofiguration
    grav_moon = -1.62

    # The first way to change the configuration is to pass in your desired config into construction of the model
    m = ThrownObject(g = grav_moon)
    simulated_moon_results = m.simulate_to_threshold(future_load, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})

    grav_mars = -3.711
    # You can also update the parameters after it's constructed
    m.parameters['g'] = grav_mars
    simulated_mars_results = m.simulate_to_threshold(future_load, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})

    grav_venus = -8.87
    m.parameters['g'] = grav_venus
    simulated_venus_results = m.simulate_to_threshold(future_load, threshold_keys=[event], options={'dt':0.005, 'save_freq':1})

    print('Time to hit the ground: ')
    print('\tvenus: {}s'.format(round(simulated_venus_results.times[-1],2)))
    print('\tearth: {}s'.format(round(simulated_results.times[-1],2)))
    print('\tmars: {}s'.format(round(simulated_mars_results.times[-1],2)))
    print('\tmoon: {}s'.format(round(simulated_moon_results.times[-1],2)))

    # We can also simulate until any event is met by neglecting the threshold_keys argument
    simulated_results = m.simulate_to_threshold(future_load, options={'dt':0.005, 'save_freq':1})
    threshs_met = m.threshold_met(simulated_results.states[-1])
    for (key, met) in threshs_met.items():
        if met:
            event_occured = key
    print('\nThis event that occured first: ', event_occured)
    # It falls before it hits the gorund, obviously

    # Metrics can be analyzed from the simulation results. For example: monotonicity
    print('\nMonotonicity: ', simulated_results.event_states.monotonicity())

# This allows the module to be executed directly 
if __name__=='__main__':
    run_example()
