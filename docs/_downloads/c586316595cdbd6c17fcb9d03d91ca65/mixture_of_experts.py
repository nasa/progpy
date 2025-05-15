# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example using the MixtureOfExpertsModel
"""

from matplotlib import pyplot as plt
from progpy import MixtureOfExpertsModel
from progpy.models import BatteryCircuit


def run_example():
    # Mixture of Experts (MoE) models combine multiple
    # models of the same system, similar to Ensemble
    # models. Unlike Ensemble Models, the aggregation
    # is done by selecting the "best" model. That is
    # the model that has performed the best over the
    # past. Each model will have a 'score' that is
    # tracked in the state, and this determines which
    # model is best.

    # To demonstrate this feature we will repeat the
    # example from the ensemble model section, this
    # time with a mixture of experts model. For this
    # example to work you will have had to have run
    # the ensemble model section example.

    # First, let's combine three battery circuit
    # models into a single mixture of experts model.
    print("setting up....")
    m_circuit = BatteryCircuit(qMax=6700, Rs=0.055)
    m_circuit_2 = BatteryCircuit(qMax=7860)
    m_circuit_3 = BatteryCircuit()

    m = MixtureOfExpertsModel(models=(m_circuit, m_circuit_2, m_circuit_3))

    # Note: The combined model has the same outputs and
    # events as the circuit model.
    print("outputs: ", m.outputs)
    print("events: ", m.events)

    # Its states contain all of the states of each model,
    # kept separate. Each individual model comprising
    # the MoE model will be simulated separately, so the
    # model keeps track of the states propogated through
    # each model separately. The states also include
    # scores for each model.
    print("states: ", m.states)

    # The MoE model inputs include both the comprised
    # model input, i (current) and outputs: v (voltage)
    # and t(temperature). The comprised model outputs
    # are provided to update the scores of each model
    # when performing state transition. If they are
    # not provided when calling next_state, then scores
    # would not be updated.
    print("inputs: ", m.inputs)

    # Now let's evaluate the performance of the combined
    # model using real battery data from NASA's prognostic
    # data repository (note: this may take a while)
    from progpy.datasets import nasa_battery

    print("downloading data... (this may take a while)")
    data = nasa_battery.load_data(batt_id=8)[1]
    RUN_ID = 0
    test_input = [{"i": i} for i in data[RUN_ID]["current"]]
    test_time = data[RUN_ID]["relativeTime"]
    t_end = test_time.iloc[-1]

    # To evaluate the model we first create a future
    # loading function that uses the loading from the data.
    def future_loading(t, x=None):
        for i, mission_time in enumerate(test_time):
            if mission_time > t:
                return m_circuit.InputContainer(test_input[i])
        return m_circuit.InputContainer(test_input[-1])  # Default - last load

    print("\n------------------\nSimulating... (this may also take a while)")
    results_moe = m.simulate_to(t_end, future_loading)
    plt.figure()
    fig = plt.plot(
        test_time, data[RUN_ID]["voltage"], color="green", label="ground truth"
    )
    fig = plt.plot(
        results_moe.times,
        [z["v"] for z in results_moe.outputs],
        color="red",
        label="moe",
    )
    plt.title("MixtureOfExperts, before provided data")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage")
    plt.legend()

    # Here the model performs pretty poorly. If you were to
    # look at the state, we see that the three scores are
    # equal. This is because we haven't provided any output information. The future load function doesn't include
    # the output, just the input (i). When the three scores
    # are equal like this, the first model is used.
    print("Model 1 Score: ", results_moe.states[-1]["BatteryCircuit._score"])
    print("Model 2 Score: ", results_moe.states[-1]["BatteryCircuit_2._score"])
    print("Model 3 Score: ", results_moe.states[-1]["BatteryCircuit_3._score"])

    # Now let's provide the output for a few steps.
    print("\n------------------\nProviding data...\n")
    x0 = m.initialize()
    x = m.next_state(
        x=x0,
        u=m.InputContainer(
            {
                "i": test_input[0]["i"],
                "v": data[RUN_ID]["voltage"][0],
                "t": data[RUN_ID]["temperature"][0],
            }
        ),
        dt=test_time[1] - test_time[0],
    )
    x = m.next_state(
        x=x,
        u=m.InputContainer(
            {
                "i": test_input[1]["i"],
                "v": data[RUN_ID]["voltage"][1],
                "t": data[RUN_ID]["temperature"][1],
            }
        ),
        dt=test_time[1] - test_time[0],
    )

    # Let's take a look at the model scores again
    print("Model 1 Score: ", x["BatteryCircuit._score"])
    print("Model 2 Score: ", x["BatteryCircuit_2._score"])
    print("Model 3 Score: ", x["BatteryCircuit_3._score"])

    # Here we see after a few steps the algorithm has determined that model 3 is the better fitting of the models. Now if we were to repeat the simulation, it would use the best model, resulting in a better fit.
    print("\n------------------\nRe-simulating... (this may also take a while)\n")
    results_moe = m.simulate_to(
        t_end, future_loading, t0=test_time[1] - test_time[0], x=x
    )
    plt.figure()
    fig = plt.plot(
        test_time[2:], data[RUN_ID]["voltage"][2:], color="green", label="ground truth"
    )
    fig = plt.plot(
        results_moe.times[2:],
        [z["v"] for z in results_moe.outputs][2:],
        color="red",
        label="moe",
    )
    plt.title("MixtureOfExperts, after provided data")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage")
    plt.legend()
    plt.show()

    # The fit here is much better. The MoE model learned
    # which of the three models best fit the observed behavior.

    # In a prognostic application, the scores will be
    # updated each time you use a state estimator
    # (so long as you provide the output as part of the input).
    # Then when performing a prediction the scores aren't
    # updated, since outputs are not known.

    # An example of when this would be useful is for cases
    # where there are three common degradation paths or
    # "modes" rather than a single model with uncertainty
    # to represent every mode, the three modes can be
    # represented by three different models. Once enough
    # of the degradation path has been observed the observed
    # mode will be the one reported.

    # If the model fit is expected to be stable
    # (that is, the best model is not expected to change anymore).
    # The best model can be extracted and used directly,
    # like demonstrated below.
    name, m_best = m.best_model(x)
    print(name, " was the best fit")


# This allows the module to be executed directly
if __name__ == "__main__":
    run_example()
