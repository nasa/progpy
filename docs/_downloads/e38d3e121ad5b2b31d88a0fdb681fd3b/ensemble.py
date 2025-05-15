# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example using the Ensemble Model feature

.. dropdown:: More details

    Ensemble model is an approach to modeling where one or more different models are simulated together and then aggregated into a single prediction. This is generally done to improve the accuracy of prediction when you have multiple models that each represent part of the behavior or represent a distribution of different behaviors.

    In this example, 4 different equivalent circuit models are setup with different configuration parameters. They are each simulated individually. Then an ensemble model is created for the 4 models, and that is simulated individually. The results are plotted.

    The results are partially skewed by a poorly configured model, so we change the aggregation method to account for that. and resimulate, showing the results

    Finally, an ensemble model is created for two different models with different states. That model is simulated with time and the results are plotted.
"""

from matplotlib import pyplot as plt
import numpy as np
from progpy import EnsembleModel
from progpy.datasets import nasa_battery
from progpy.models import BatteryElectroChemEOD, BatteryCircuit


def run_example():
    # Example 1: Different model configurations
    # Download data
    print("downloading data (this may take a while)...")
    data = nasa_battery.load_data(8)[1]

    # Prepare data
    RUN_ID = 0
    test_input = [{"i": i} for i in data[RUN_ID]["current"]]
    test_time = data[RUN_ID]["relativeTime"]

    # Setup models
    # In this case, we have some uncertainty on the parameters of the model,
    # so we're setting up a few versions of the circuit model with different parameters.
    print("Setting up models...")
    m_circuit = BatteryCircuit(process_noise=0, measurement_noise=0)
    m_circuit_2 = BatteryCircuit(process_noise=0, measurement_noise=0, qMax=7860)
    m_circuit_3 = BatteryCircuit(
        process_noise=0, measurement_noise=0, qMax=6700, Rs=0.055
    )
    m_ensemble = EnsembleModel(
        (m_circuit, m_circuit_2, m_circuit_3), process_noise=0, measurement_noise=0
    )

    # Evaluate models
    print("Evaluating models...")

    def future_loading(t, x=None):
        for i, mission_time in enumerate(test_time):
            if mission_time > t:
                return m_circuit.InputContainer(test_input[i])

    results_ensemble = m_ensemble.simulate_to(test_time.iloc[-1], future_loading)

    # Plot results
    print("Producing figures...")
    plt.plot(test_time, data[RUN_ID]["voltage"], color="green", label="ground truth")
    plt.plot(
        results_ensemble.times,
        [z["v"] for z in results_ensemble.outputs],
        color="red",
        label="ensemble",
    )
    plt.legend()

    # Note: This is a very poor performing model
    # there was an outlier model (m_circuit_3), which effected the quality of the model prediction
    # This can be resolved by using a different aggregation_method. For example, median
    # In a real scenario, you would likely remove this model, this is just to illustrate outlier elimination
    print("Updating with Median ")
    m_ensemble.parameters["aggregation_method"] = np.median
    results_ensemble = m_ensemble.simulate_to(test_time.iloc[-1], future_loading)
    plt.plot(
        results_ensemble.times,
        [z["v"] for z in results_ensemble.outputs],
        color="orange",
        label="ensemble - median",
    )
    plt.legend()

    # Example 2: Different Models
    # The same ensemble approach can be used with different models that have different states
    # In this case, we're using the equivalent circuit and electro chem models
    # These two models share one state, but besides that they have different states

    # Setup Model
    print("Setting up models...")
    m_electro = BatteryElectroChemEOD(process_noise=0, measurement_noise=0)
    m_ensemble = EnsembleModel(
        (m_circuit, m_electro), process_noise=0, measurement_noise=0
    )

    # Evaluate models
    print("Evaluating models...")
    print("\tEnsemble")
    results_ensemble = m_ensemble.simulate_to(test_time.iloc[-1], future_loading)
    print("\tCircuit 1")
    results_circuit1 = m_circuit.simulate_to(test_time.iloc[-1], future_loading)
    print("\tElectroChem")
    results_electro = m_electro.simulate_to(test_time.iloc[-1], future_loading)

    # Plot results
    print("Producing figures...")
    plt.figure()
    plt.plot(test_time, data[RUN_ID]["voltage"], color="green", label="ground truth")
    plt.plot(
        results_circuit1.times,
        [z["v"] for z in results_circuit1.outputs],
        color="blue",
        label="circuit",
    )
    plt.plot(
        results_electro.times,
        [z["v"] for z in results_electro.outputs],
        color="red",
        label="electro chemistry",
    )
    plt.plot(
        results_ensemble.times,
        [z["v"] for z in results_ensemble.outputs],
        color="yellow",
        label="ensemble",
    )
    plt.legend()

    # Note that the result may not be exactly between the other two models.
    # This is because of aggregation is done in 2 steps: at state transition and then at output calculation


# This allows the module to be executed directly
if __name__ == "__main__":
    run_example()
