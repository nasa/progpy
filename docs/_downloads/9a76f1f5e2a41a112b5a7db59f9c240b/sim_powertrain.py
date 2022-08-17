# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a powertrain being simulated for a set amount of time. 
"""

from prog_models.models import Powertrain, ESC, DCMotor

def run_example():
    # Create a model object
    esc = ESC()
    motor = DCMotor()
    powertrain = Powertrain(esc, motor)

    # Define future loading function - 100% duty all the time
    def future_loading(t, x=None):
        return powertrain.InputContainer({
            'duty': 1,
            'v': 23
        })
    
    # Simulate to threshold
    print('\n\n------------------------------------------------')
    print('Simulating to threshold\n\n')
    simulated_results = powertrain.simulate_to(2, future_loading, dt=2e-5, save_freq=0.1, print=True)

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
