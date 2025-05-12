# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example illustrating how to use the CompositeModel class to create a composite model from multiple models.

This example creates a composite model of a DC motor with an Electronic Speed Controller and a propeller load. The three composite models are interrelated. The created composite model describes the nature of these interconnections. The resulting powertrain model is then simulated forward with time and the results are plotted.
"""

from progpy.models import DCMotor, ESC, PropellerLoad
from progpy import CompositeModel


def run_example():
    # First, lets define the composite models
    m_motor = DCMotor()
    m_esc = ESC()
    m_load = PropellerLoad()

    # Now let's combine them into a single composite model describing the behavior of a powertrain
    # This model will then behave as a single model
    m_powertrain = CompositeModel(
        (m_esc, m_load, m_motor),
        connections=[
            ("DCMotor.theta", "ESC.theta"),
            ("ESC.v_a", "DCMotor.v_a"),
            ("ESC.v_b", "DCMotor.v_b"),
            ("ESC.v_c", "DCMotor.v_c"),
            ("PropellerLoad.t_l", "DCMotor.t_l"),
            ("DCMotor.v_rot", "PropellerLoad.v_rot"),
        ],
        outputs={"DCMotor.v_rot", "DCMotor.theta"},
    )

    # Print out the inputs, states, and outputs of the composite model
    print("Composite model of DCMotor, ESC, and Propeller load")
    print("inputs: ", m_powertrain.inputs)
    print("states: ", m_powertrain.states)
    print("outputs: ", m_powertrain.outputs)

    # Define future loading function - 100% duty all the time
    def future_loading(t, x=None):
        return m_powertrain.InputContainer({"ESC.duty": 1, "ESC.v": 23})

    # Simulate to threshold
    print("\n\n------------------------------------------------")
    print("Simulating to threshold\n\n")
    simulated_results = m_powertrain.simulate_to(
        2, future_loading, dt=2e-5, save_freq=0.1, print=True
    )

    simulated_results.outputs.plot()


if __name__ == "__main__":
    run_example()
