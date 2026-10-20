# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example demonstrates the Direct Models functionality. Direct models are models that directly estimate time of event from system state, rather than using state transition. This is useful for data-driven models that map from sensor data to time of event, and for physics-based models where state transition differential equations can be solved.

Here a DirectThrownObject model is defined, extending the ThrownObject state transition model from the new_model example. This model is then initialized and used to estimate the time of event from the initial state.
"""

import time
import numpy as np
from progpy.models import ThrownObject


def run_example():
    # Here is how estimating time of event works for a timeseries model
    m = ThrownObject()
    x = m.initialize()
    print(
        m.__class__.__name__, "(Direct Model)" if m.is_direct else "(Timeseries Model)"
    )
    tic = time.perf_counter()
    print("Time of event: ", m.time_of_event(x, dt=0.05))
    toc = time.perf_counter()
    print(f"execution: {(toc - tic) * 1000:0.4f} milliseconds")

    # Step 1: Define DirectModel
    # In this case we're extending the ThrownObject model to include the  time_to_event method, defined in DirectModel
    # In the case of thrown objects, we can solve the differential equation
    # to estimate the time at which the events occur.
    class DirectThrownObject(ThrownObject):
        def time_of_event(self, x, *args, **kwargs):
            # calculate time when object hits ground given x['x'] and x['v']
            # 0 = x0 + v0*t - 0.5*g*t^2
            g = self.parameters["g"]
            t_impact = -(x["v"] + np.sqrt(x["v"] * x["v"] - 2 * g * x["x"])) / g

            # 0 = v0 - g*t
            t_falling = -x["v"] / g

            return {"falling": t_falling, "impact": t_impact}

    # Note that adding *args and **kwargs is optional.
    # Having these arguments makes the function interchangeable with other models
    # which might have arguments or keyword arguments

    # Step 2: Now estimate time of event for a ThrownObject
    m = DirectThrownObject()
    x = m.initialize()  # Using Initial state
    # Now instead of simulating to threshold, we can estimate it directly from the state, like so
    print(
        "\n",
        m.__class__.__name__,
        "(Direct Model)" if m.is_direct else "(Timeseries Model)",
    )
    tic = time.perf_counter()
    print("Time of event: ", m.time_of_event(x))
    toc = time.perf_counter()
    print(f"execution: {(toc - tic) * 1000:0.4f} milliseconds")

    # Notice that execution is MUCH faster for the direct model.
    # This is even more pronounced for events that occur later in the simulation.

    # In this case, the DirectThrownObject has a defined next_state and output equation,
    # allowing it to be used with a state estimator (e..g, Particle Filter)


# This allows the module to be executed directly
if __name__ == "__main__":
    run_example()
