# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example further illustrating the concept of 'events' which generalizes EOL. 

.. dropdown:: More details

    :term:`Events<event>` is the term used to describe something to be predicted. Generally in the PHM community these are referred to as End of Life (EOL). However, they can be much more.

    In prog_models, events can be anything that needs to be predicted. Events can represent End of Life (EOL), End of Mission (EOM), warning thresholds, or any Event of Interest (EOI). 

    This example demonstrates how events can be used in your applications. 
"""
import matplotlib.pyplot as plt
from prog_models.models import BatteryElectroChemEOD

def run_example():
    # Example: Warning thresholds
    # In this example we will use the battery model
    # We of course are interested in end of discharge, but for this example we
    # have a requirement that says the battery must not fall below 5% State of Charge (SOC)
    # Note: SOC is the event state for the End of Discharge (EOD) event
    # Event states, like SOC go between 0 and 1, where 1 is healthy and at 0 the event has occurred. 
    # So, 5% SOC corresponds to an 'EOD' event state of 0.05
    # Additionally, we have two warning thresholds (yellow and red)

    YELLOW_THRESH = 0.15
    RED_THRESH = 0.1
    THRESHOLD = 0.05

    # Step 1: Extend the battery model to define the additional events
    class MyBatt(BatteryElectroChemEOD):
        events = BatteryElectroChemEOD.events + ['EOD_warn_yellow', 'EOD_warn_red', 'EOD_requirement_threshold']

        def event_state(self, state):
            # Get event state from parent
            event_state = super().event_state(state)

            # Add yellow, red, and failure states by scaling EOD state
            # Here we scale so the threshold SOC is 0 by their associated events, while SOC of 1 is still 1
            # For example, for yellow we want EOD_warn_yellow to be 1 when SOC is 1, and 0 when SOC is YELLOW_THRESH or lower
            event_state['EOD_warn_yellow'] = (event_state['EOD']-YELLOW_THRESH)/(1-YELLOW_THRESH) 
            event_state['EOD_warn_red'] = (event_state['EOD']-RED_THRESH)/(1-RED_THRESH)
            event_state['EOD_requirement_threshold'] = (event_state['EOD']-THRESHOLD)/(1-THRESHOLD)

            # Return
            return event_state

        def threshold_met(self, x):
            # Get threshold met from parent
            t_met =  super().threshold_met(x)

            # Add yell and red states from event_state
            event_state = self.event_state(x)
            t_met['EOD_warn_yellow'] = event_state['EOD_warn_yellow'] <= 0
            t_met['EOD_warn_red'] = event_state['EOD_warn_red'] <= 0
            t_met['EOD_requirement_threshold'] = event_state['EOD_requirement_threshold'] <= 0

            return t_met

    # Step 2: Use it
    m = MyBatt()

    # 2a: Setup model
    def future_loading(t, x=None):
        # Variable (piece-wise) future loading scheme 
        # For a battery, future loading is in term of current 'i' in amps. 
        if (t < 600):
            i = 2
        elif (t < 900):
            i = 1
        elif (t < 1800):
            i = 4
        elif (t < 3000):
            i = 2     
        else:
            i = 3
        return m.InputContainer({'i': i})
    
    # 2b: Simulate to threshold
    simulated_results = m.simulate_to_threshold(future_loading, threshold_keys=['EOD'], print = True)

    # 2c: Plot results
    simulated_results.event_states.plot()
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
