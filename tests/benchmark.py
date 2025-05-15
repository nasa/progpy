# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import time
from io import StringIO
import sys

FORMAT_STR = "{:40s}"
CLOCK = time.process_time

if __name__ == "__main__":
    print(FORMAT_STR.format("import main"), end="")
    t = CLOCK()
    from progpy import state_estimators, predictors

    t2 = CLOCK()
    print(t2 - t)

    from progpy.models import BatteryElectroChemEOD as Battery

    def future_loading(t, x=None):
        # Variable (piece-wise) future loading scheme
        if t < 600:
            i = 2
        elif t < 900:
            i = 1
        elif t < 1800:
            i = 4
        elif t < 3000:
            i = 2
        else:
            i = 3
        return {"i": i}

    R_vars = {"t": 2, "v": 0.02}
    batt = Battery(measurement_noise=R_vars)
    initial_state = batt.parameters["x0"]

    print("PF")
    print(FORMAT_STR.format("   Initialize"), end="")
    temp_out = StringIO()
    sys.stdout = temp_out
    sys.stderr = temp_out
    t = CLOCK()
    filt = state_estimators.ParticleFilter(batt, initial_state)
    t2 = CLOCK()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(t2 - t)

    print(FORMAT_STR.format("   Step"), end="")
    example_measurements = {"t": 32.2, "v": 3.915}
    t = 0.1
    t = CLOCK()
    filt.estimate(t, future_loading(t), example_measurements)
    t2 = CLOCK()
    print(t2 - t)

    print("UKF")
    print(FORMAT_STR.format("   Initialize "), end="")
    temp_out = StringIO()
    sys.stdout = temp_out
    sys.stderr = temp_out
    t = CLOCK()
    filt = state_estimators.UnscentedKalmanFilter(batt, initial_state)
    t2 = CLOCK()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(t2 - t)

    # print(FORMAT_STR.format('   Step'), end='')
    # example_measurements = {'t': 32.2, 'v': 3.915}
    # t = 0.1
    # temp_out = StringIO()
    # sys.stdout = temp_out
    # sys.stderr = temp_out
    # t = CLOCK()
    # filt.estimate(t, future_loading(t), example_measurements)
    # t2 = CLOCK()
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    # print(t2-t)

    print(FORMAT_STR.format("Plot Results"), end="")
    t = CLOCK()
    filt.x.plot_scatter(label="prior")
    t2 = CLOCK()
    print(t2 - t)

    print("MC")
    print(FORMAT_STR.format("   Initialize "), end="")
    t = CLOCK()
    mc = predictors.MonteCarlo(batt)
    t2 = CLOCK()
    print(t2 - t)

    print(FORMAT_STR.format("   Prediction"), end="")
    NUM_SAMPLES = 5
    STEP_SIZE = 0.1
    t = CLOCK()
    mc_results = mc.predict(
        batt.initialize(), future_loading, n_samples=NUM_SAMPLES, dt=STEP_SIZE
    )
    t2 = CLOCK()
    print(t2 - t)

    print(FORMAT_STR.format("Metrics"), end="")
    t = CLOCK()
    mc_results.time_of_event.percentage_in_bounds([3005.2, 3005.6])
    mc_results.time_of_event.metrics(ground_truth=3005.25)
    t2 = CLOCK()
    print(t2 - t)

    print(FORMAT_STR.format("Plot Scatter"), end="")
    t = CLOCK()
    fig = mc_results.states.snapshot(0).plot_scatter(
        label="t={} s".format(int(mc_results.times[0]))
    )  # 0
    quarter_index = int(len(mc_results.times) / 4)
    mc_results.states.snapshot(quarter_index).plot_scatter(
        fig=fig, label="t={} s".format(int(mc_results.times[quarter_index]))
    )  # 25%
    mc_results.states.snapshot(quarter_index * 2).plot_scatter(
        fig=fig, label="t={} s".format(int(mc_results.times[quarter_index * 2]))
    )  # 50%
    mc_results.states.snapshot(quarter_index * 3).plot_scatter(
        fig=fig, label="t={} s".format(int(mc_results.times[quarter_index * 3]))
    )  # 75%
    mc_results.states.snapshot(-1).plot_scatter(
        fig=fig, label="t={} s".format(int(mc_results.times[-1]))
    )  # 100%
    t2 = CLOCK()
    print(t2 - t)

    print(FORMAT_STR.format("Plot Hist"), end="")
    t = CLOCK()
    mc_results.time_of_event.plot_hist()
    t2 = CLOCK()
    print(t2 - t)
