# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import math
from progpy import PrognosticsModel, create_discrete_state

ValveState = create_discrete_state(2, ["open", "closed"])


class Tank(PrognosticsModel):
    """
    A simple model of liquid in a tank draining through a drain with a controllable valve. This is used as an example for discrete states. Default parameters represent a tank that is a cube with 1m along each edge

    :term:`Events<event>`: (2)
        full: The tank is full
        empty: The tank is empty

    :term:`Inputs/Loading<input>`: (2)
        | q_in: Flowrate into the tank (m^3/s)
        | valve_command (DiscreteState): Discrete state to command the valve

    :term:`States<state>`: (2)
        | state (DiscreteState): state of the valve
        | h: Height of the fluid in the tank (m)

    :term:`Outputs<output>`: (1)
        h: Height of the fluid in the tank (m)

    Keyword Args
    ------------
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state).
          Can be number (e.g., .2) applied to every state, a dictionary of values for each
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)
        crosssection_area : Optional, float
            Crosssectional area of the tank in m^2. Default is 1
        height : Optional, float
            Height of the tank in m. Default is 1
        rho : Optional, float
            Fluid density in kg/m^3. Default is for water
        g : Optional, float
            Acceleration due to gravity in m/s^2. Default is earth gravity at surface
        valve_r : Optional, float
            Radius of valve opening in m
        valve_l : Optional, float
            Length of valve in m
        viscosity : Optional, float
            Viscosity of the fluid in Pa*s. Default is for water
        x0 : Optional, dict
            Initial State
    """

    inputs = ["q_in", "valve_command"]
    states = ["valve", "h"]
    outputs = ["h"]
    events = ["full", "empty"]

    default_parameters = {
        "crosssection_area": 1,
        "height": 1,
        "rho": 1000,
        "g": -9.81,
        "valve_r": 3e-3,
        "valve_l": 0.001,
        "viscosity": 1e-3,
        "x0": {
            "valve": ValveState.closed,
            "h": 0,
        },
    }

    state_limits = {"h": (0, float("inf"))}

    def next_state(self, x, u, dt):
        x["valve"] = ValveState(u["valve_command"])

        # Relative pressure of fluid
        p = self["rho"] * self["g"] * x["h"]
        if x["valve"] == ValveState.open:
            # flow rate out through valve m^3/s
            q_out = (
                p
                * math.pi
                * self["valve_r"] ** 4
                / (8 * self["viscosity"] * self["valve_l"])
            )
        else:
            # Valve is closed, no flow
            q_out = 0
        x["h"] += (u["q_in"] + q_out) * dt / self["crosssection_area"]

        # Limit to height of tank
        x["h"] = min(x["h"], self["height"])
        return x

    def output(self, x):
        return self.OutputContainer({"h": x["h"]})

    def event_state(self, x):
        return {
            "full": (self["height"] - x["h"]) / self["height"],
            "empty": x["h"] / self["height"],
        }
