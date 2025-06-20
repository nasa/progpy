# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example downloading and using a NASA prognostics dataset.

.. dropdown:: More details

    In this example, a battery dataset is downloaded from the NASA PCoE data repository. This dataset is then accessed and plotted.
"""

import matplotlib.pyplot as plt
import pickle
from progpy.datasets import nasa_battery

DATASET_ID = 1


def run_example():
    # Step 1: Download and import the dataset for a single battery
    # Note: This may take some time
    print("Downloading... ", end="")
    (desc, data) = nasa_battery.load_data(DATASET_ID)
    print("done")

    # We recommend saving the dataset to disk for future use
    # This way you don't have to download it each time
    pickle.dump((desc, data), open(f"dataset_{DATASET_ID}.pkl", "wb"))

    # Step 2: Access the dataset description
    print(f"\nDataset {DATASET_ID}")
    print(desc["description"])
    print(f"Procedure: {desc['procedure']}")

    # Step 3: Access the dataset data
    # Data is in format [run_id][time][variable]
    # For the battery the variables are
    #    0: relativeTime (since beginning of run)
    #    1: current (amps)
    #    2: voltage
    #    3: temperature (°C)
    # so that data[a][b, 3] is the temperature at time index b (relative to the start of the run) for run a
    print(f"\nNumber of runs: {len(data)}")
    print("\nAnalyzing run 4")
    print(f"number of time indices: {len(data[4])}")
    print(f"Details of run 4: {desc['runs'][4]}")

    # Plot the run
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data[4]["relativeTime"], data[4]["current"])
    plt.ylabel("Current (A)")

    plt.subplot(2, 1, 2)
    plt.plot(data[4]["relativeTime"], data[4]["voltage"])
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")
    plt.title("Run 4")

    # Graph all reference discharge profiles
    indices = [
        i
        for i, x in enumerate(desc["runs"])
        if "reference discharge" in x["desc"] and "rest" not in x["desc"]
    ]
    plt.figure()
    for i in indices:
        plt.plot(data[i]["relativeTime"], data[i]["voltage"], label=f"Run {i}")
    plt.title("Reference discharge profiles")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.show()


# This allows the module to be executed directly
if __name__ == "__main__":
    run_example()
