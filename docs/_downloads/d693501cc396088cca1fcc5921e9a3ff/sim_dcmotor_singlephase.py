# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a DC motor being simulated for a set amount of time, using the single-phase dcmotor model. 
"""

import math
from prog_models.models import dcmotor_singlephase

def run_example():
    motor = dcmotor_singlephase.DCMotorSP()
    
    def future_loading(t, x=None):
        f = 0.5
        
        # Simple load proportional to rotor speed. 
        # This is a typical, hyper-simplified model of a fixed-pitch propeller directly attached to the motor shaft such that the resistant torque
        # becomes: Cq * omega^2, where Cq is a (assumed to be) constant depending on the propeller profile and omega is the rotor speed.
        # Since there's no transmission, omega is exactly the speed of the motor shaft.
        if x is None:  # First load (before state is initialized)
            t_l = 0.0
        else:
            t_l = 1e-5 * x['v_rot']**2.0
        return motor.InputContainer({
            'v': 10.0 + 2.0 * math.sin(math.tau * f * t),   # voltage input assumed sinusoidal just to show variations in the input. No physical meaning.
            't_l': t_l  # assuming constant load (simple)
            })

    simulated_results = motor.simulate_to(2.0, future_loading, dt=1e-3, save_freq=0.1, print=True)
    simulated_results.states.plot(compact=False)

if __name__ == '__main__':
    print("Simulation of DC single-phase motor")
    run_example()
