# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of a pneumatic valve being simulated until threshold is met.
"""

from progpy.models.pneumatic_valve import PneumaticValve


def run_example():
    # Create a model object
    valv = PneumaticValve(process_noise=0)

    # Define future loading function
    cycle_time = 20

    def future_loading(t, x=None):
        t = t % cycle_time
        if t < cycle_time / 2:
            return valv.InputContainer(
                {
                    "pL": 3.5e5,
                    "pR": 2.0e5,
                    # Open Valve
                    "uTop": False,
                    "uBot": True,
                }
            )
        return valv.InputContainer(
            {
                "pL": 3.5e5,
                "pR": 2.0e5,
                # Close Valve
                "uTop": True,
                "uBot": False,
            }
        )

    # Simulate to threshold
    print("\n\n------------------------------------------------")
    print("Simulating to threshold\n\n")
    # Configure options
    config = {
        "dt": 0.01,
        "horizon": 800,
        "save_freq": 60,
        "print": True,
        "progress": True,
    }
    # Set wear parameter for spring to 1
    valv.parameters["x0"]["wk"] = 1

    # Define first measured output. This is needed by the simulate_to_threshold method to initialize state
    first_output = valv.output(valv.initialize(future_loading(0)))
    # Simulate
    simulated_results = valv.simulate_to_threshold(
        future_loading, first_output, **config
    )

    # Simulate to threshold again but with a different wear mode
    print("\n\n------------------------------------------------")
    print("Simulating to threshold\n\n")
    # Configure options
    config = {
        "dt": 0.01,
        "horizon": 800,
        "save_freq": 60,
        "print": True,
        "progress": True,
    }
    # Reset wear parameter for spring to 0, set wear parameter for friction to 1
    valv.parameters["x0"]["wk"] = 0
    valv.parameters["x0"]["wr"] = 1

    # Define first measured output. This is needed by the simulate_to_threshold method to initialize state
    first_output = valv.output(valv.initialize(future_loading(0)))
    # Simulate
    simulated_results = valv.simulate_to_threshold(
        future_loading, first_output, **config
    )


# This allows the module to be executed directly
if __name__ == "__main__":
    run_example()
