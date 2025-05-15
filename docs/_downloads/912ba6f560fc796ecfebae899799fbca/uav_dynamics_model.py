# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example of generating a trajectory for a small rotorcraft through a set of coarse waypoints, and simulate the rotorcraft flight using a 6-dof model.
"""

import matplotlib.pyplot as plt
import numpy as np

from progpy.utils.traj_gen import Trajectory
from progpy.models.aircraft_model import SmallRotorcraft
from progpy.loading.controllers import LQR_I, LQR


def run_example():
    # Initialize vehicle
    vehicle = SmallRotorcraft(
        dt=0.05, vehicle_model="tarot18", process_noise=0, measurement_noise=0
    )

    # EXAMPLE 1:
    # Define coarse waypoints: latitudes, longitudes, and altitudes are
    # required, ETAs are optional
    # Latitudes and longitudes must be defined as numpy arrays of size n x 1
    # and with unit radians
    # Altitudes must be defined as numpy arrays of size n x 1 with unit meters
    # ETAs (if included) must be defined as a list of datetime objects
    # If ETAs are not included, speeds must be defined (see Example 2)

    # Here, we specify waypoints in a dictionary and then pass
    # lat/lon/alt/ETAs into the trajectory class
    lat_deg = np.array(
        [
            37.09776,
            37.09776,
            37.09776,
            37.09798,
            37.09748,
            37.09665,
            37.09703,
            37.09719,
            37.09719,
            37.09719,
            37.09719,
            37.09748,
            37.09798,
            37.09776,
            37.09776,
        ]
    )
    lon_deg = np.array(
        [
            -76.38631,
            -76.38629,
            -76.38629,
            -76.38589,
            -76.3848,
            -76.38569,
            -76.38658,
            -76.38628,
            -76.38628,
            -76.38628,
            -76.38628,
            -76.3848,
            -76.38589,
            -76.38629,
            -76.38629,
        ]
    )
    alt_ft = np.array(
        [
            -1.9682394,
            164.01995,
            164.01995,
            164.01995,
            164.01995,
            164.01995,
            164.01995,
            164.01995,
            0.0,
            0.0,
            164.01995,
            164.01995,
            164.01995,
            164.01995,
            0.0,
        ]
    )
    time_unix = [
        1544188336,
        1544188358,
        1544188360,
        1544188377,
        1544188394,
        1544188411,
        1544188428,
        1544188496,
        1544188539,
        1544188584,
        1544188601,
        1544188635,
        1544188652,
        1544188672,
        1544188692,
    ]

    # Generate trajectory
    # =====================
    # Generate trajectory object and pass the route (waypoints, ETA) to it
    traj = Trajectory(
        lat=lat_deg * np.pi / 180.0,
        lon=lon_deg * np.pi / 180.0,
        alt=alt_ft * 0.3048,
        etas=time_unix,
    )

    ref_traj = traj.generate(dt=vehicle.parameters["dt"])

    # Define controller and build scheduled control. The controller acts as a
    # future_loading function when simulating
    # We use a linear quadratic regulator (LQR), which tries to minimize the
    # cost function defined by:
    # J = \int{ x^T Q x + u^T R u \mathrm{d}t }
    # Where x is the state vector, u is the input vector, t is time, Q is the
    # state error penalty matrix, and R is the input generation penalty matrix.
    # The LQR uses a linearized version of the dynamic system
    # (i.e., dxdt = A x + Bu) to find the gain matrix K that minimizes the cost J.
    ctrl = LQR(ref_traj, vehicle)

    # Simulate vehicle to fly trajectory
    traj_results = vehicle.simulate_to_threshold(
        ctrl, dt=vehicle.parameters["dt"], save_freq=vehicle.parameters["dt"]
    )

    # Visualize Results
    vehicle.visualize_traj(pred=traj_results, ref=ref_traj)

    # EXAMPLE 2:
    # In this example, we define another trajectory through the same
    # waypoints but with speeds defined instead of ETAs

    # Generate trajectory object and pass the route (lat/lon/alt, no ETAs)
    # and speed information to it
    traj_speed = Trajectory(
        lat=lat_deg * np.pi / 180.0,
        lon=lon_deg * np.pi / 180.0,
        alt=alt_ft * 0.3048,
        cruise_speed=8.0,
        ascent_speed=2.0,
        descent_speed=3.0,
        landing_speed=2.0,
    )
    ref_traj_speeds = traj_speed.generate(dt=vehicle.parameters["dt"])

    # Define controller and build scheduled control. This time we'll use LQR_I,
    # which is a linear quadratic regulator with integral action.
    # The integral action has the same purpose of "I" in PI or PID controllers,
    # which is to minimize offset errors in the variable of interest.
    # This version of LQR_I compensates for integral errors in the position of
    # the vehicle, i.e., x, y, z variables of the state vector.
    ctrl_speeds = LQR_I(ref_traj_speeds, vehicle)

    # Set simulation options
    options = {"dt": vehicle.parameters["dt"], "save_freq": vehicle.parameters["dt"]}

    # Simulate vehicle to fly trajectory
    traj_results_speeds = vehicle.simulate_to_threshold(ctrl_speeds, **options)

    # Visualize results - notice these results are slightly different, since
    # the speeds through the waypoints (and therefore the resulting trajectory)
    # are different than Example 1
    vehicle.visualize_traj(pred=traj_results_speeds, ref=ref_traj_speeds)

    # EXAMPLE 3:
    # In this example, we just want to simulate a specific portion of the
    # reference trajectory
    # We will simulate the second cruise interval in Example 1,
    # i.e. waypoints 10 - 13 (where the first waypoint is index 0).
    # We will use the reference trajectory (ref_traj) and controller (ctrl)
    # already generated in Example 1

    # First, we'll re-define the ETAs in the waypoints dictionary
    # (since we deleted them from the waypoints in Example 2)
    time_unix = np.array(
        [
            1544188336,
            1544188358,
            1544188360,
            1544188377,
            1544188394,
            1544188411,
            1544188428,
            1544188496,
            1544188539,
            1544188584,
            1544188601,
            1544188635,
            1544188652,
            1544188672,
            1544188692,
        ]
    )

    # Extract time information for desired interval, starting at waypoint 10
    # and ending at waypoint 13
    start_time = time_unix[10] - time_unix[0]
    end_time = time_unix[13] - time_unix[0]
    sim_time = end_time - start_time

    # Define initial state, x0, based on reference trajectory at start_time
    ind = np.where(ref_traj["t"] == start_time)
    x0 = {key: ref_traj[key][ind][0] for key in ref_traj.keys()}
    vehicle.parameters["x0"] = x0

    # Define simulation parameters - note that we must define t0 as start_time
    # since we are not starting at the default of t0 = 0
    options = {
        "dt": vehicle.parameters["dt"],
        "save_freq": vehicle.parameters["dt"],
        "t0": start_time,
    }

    # Simulate starting from this initial state from start_time to end_time
    traj_results_interval = vehicle.simulate_to(sim_time, ctrl, **options)

    # Plot results with Example 1 results to show equivalence on this interval
    z_1 = [output["z"] for output in traj_results.outputs]
    z_4 = [output["z"] for output in traj_results_interval.outputs]

    fig, ax = plt.subplots()
    ax.plot(traj_results.times, z_1, "-b", label="Example 1")
    ax.plot(traj_results_interval.times, z_4, "--r", label="Example 3")
    ax.set_xlabel("time, s", fontsize=14)
    ax.set_ylabel("altitude, m", fontsize=14)
    ax.legend()


# This allows the module to be executed directly
if __name__ == "__main__":
    run_example()
