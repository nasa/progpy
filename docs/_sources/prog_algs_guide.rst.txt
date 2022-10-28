prog_algs Guide
===================================================

.. role:: pythoncode(code)
   :language: python

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=prog_algs&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb

The Prognostic Algorithms Package is a python framework for prognostics (computation of remaining useful life or future states) of engineering systems. The package provides an extendable set of algorithms for state estimation and prediction, including uncertainty propagation. The package also include metrics, visualization, and analysis tools needed to measure the prognostic performance. The algorithms use prognostic models (from :ref:`prog_models<prog_models Guide>`) to perform estimation and prediction functions. The package enables the rapid development of prognostics solutions for given models of components and systems. Different algorithms can be easily swapped to do comparative studies and evaluations of different algorithms to select the best for the application at hand.

Installing prog_algs
-----------------------

.. tabs::

    .. tab:: Stable Version (Recommended)

        The latest stable release of prog_algs is hosted on PyPi. For most users (unless you want to contribute to the development of `prog_algs`), this version will be adequate. To install from the command line, use the following command:

        .. code-block:: console

            $ pip install prog_algs

    .. tab:: Pre-Release

        Users who would like to contribute to prog_algs or would like to use pre-release features can do so using the `prog_algs GitHub repo <https://github.com/nasa/prog_algs>`__. This isn't recommended for most users as this version may be unstable. To do this, use the following commands:

        .. code-block:: console

            $ git clone https://github.com/nasa/prog_algs
            $ cd prog_algs
            $ git checkout dev 
            $ pip install -e .

Summary
---------

The structure of the packages is illustrated below:

.. image:: images/package_structure.png

Prognostics is performed using `State Estimators <state_estimators.html>`__ and `Predictors <predictors.html>`__. State Estimators are resposible for estimating the current state of the modeled system using sensor data and a prognostics model (see: `prog_models package <https://github.com/nasa/prog_models>`__). The state estimator then produces an estimate of the system state with uncertainty in the form of an `uncertain data object <uncertain_data.html>`__. This state estimate is used by the predictor to predict when events will occur (Time of Event, ToE - returned as an `uncertain data object <uncertain_data.html>`__), and future system states (returned as a `Prediction object <prediction.html#id1>`__).

Data Structures
***************

A few custom data structures are available for storing and manipulating prognostics data of various forms. These structures are listed below and desribed on their respective pages:
 * :py:class:`prog_models.sim_result.SimResult` : The result of a single simulation (without uncertainty). Can be used to store inputs, outputs, states, event_states, observables, etc. Is returned by the model.simulate_to* methods.
 * :py:class:`prog_algs.uncertain_data.UncertainData`: Used throughout the package to represent data with uncertainty. There are a variety of subclasses of UncertainData to represent data with uncertainty in different forms (e.g., :py:class:`prog_algs.uncertain_data.ScalarData`, :py:class:`prog_algs.uncertain_data.MultivariateNormalDist`, :py:class:`prog_algs.uncertain_data.UnweightedSamples`). Notably, this is used to represent the output of a StateEstimator's `estimate` method, individual snapshots of a prediction, and the time of event estimate from a predictor's `predict` method.
 * :py:class:`prog_algs.predictors.Prediction`: Prediction of future values (with uncertainty) of some variable (e.g., :term:`input`, :term:`state`, :term:`output`, :term:`event state`, etc.). The `predict` method of predictors return this. 
 * :py:class:`prog_algs.predictors.ToEPredictionProfile` : The time of prediction estimates from multiple predictions. This data structure can be treated as a dictionary of time of prediction to toe prediction. 

State Estimation
-----------------

:term:`State estimation<state estimation>` is the process of estimating the internal model :term:`state` (x) using :term:`input` (i.e., loading), :term:`output` (i.e., sensor data), and system :term:`parameters`. State estimation is necessary for cases where model state isn't directly measurable (i.e., `hidden state`) or where there is sensor noise. Most state estimators estimate the state with some representation of uncertainty. 

The foundation of state estimators is the estimate method. The estimate method is called with a time, inputs, and outputs. Each time estimate is called, the internal state estimate is updated. 

.. code-block:: python

    >>> estimator.update(time, inputs, outputs)

The internal state is stored in the estimators x property as a UncertainData subclass (see `UncertainData <https://nasa.github.io/progpy/api_ref/prog_algs/UncertainData.html>`__). State is accessed like so :pythoncode:`x_est = estimator.x`.

.. dropdown:: Included State Estimators

    ProgPy includes a number of state estimators in the *prog_algs.state_estimators* package. The most commonly used of these are highlighted below. See `State Estimators <https://nasa.github.io/progpy/api_ref/prog_algs/StateEstimator.html>`__ for a full list of supported state estimators.

    * **Unscented Kalman Filter (UKF)**: A type of kalman filter for non-linear models where the state distribution is represented by a set of sigma points, calculated by an unscented tranform. Sigma points are propogated forward and then compared with the measurement to update the distribution. The resulting state is represented by a :py:class:`prog_algs.uncertain_data.MultivariateNormalDist`. By it's nature, UKFs are much faster than Particle Filters, but they fit the data to a normal distribution, resulting in some loss of information.
    * **Particle Filter (PF)**: A sample-based state estimation algorithm, where the distribution of likely states is represented by a set of unweighted samples. These samples are propagated forward and then weighted according to the likelihood of the measurement (given those samples) to update the distribution. The resulting state is represented by a :py:class:`prog_algs.uncertain_data.UnweightedSamples`. By its nature, PF is more accurate than a UKF, but much slower. Full accuracy of PF can be adjusted by increasing or decreasing the number of samples
    * **Kalman Filter (KF)**: A Simple efficient Kalman Filter for linear systems where state is represented by a mean and covariance matrix. The resulting state is represented by a :py:class:`prog_algs.uncertain_data.MultivariateNormalDist`. Only works with Prognostic Models inheriting from :py:class:`prog_models.LinearModel`. 

    .. dropdown:: UKF Details

        .. autoclass:: prog_algs.state_estimators.UnscentedKalmanFilter
    
    .. dropdown:: PF Details

        .. autoclass:: prog_algs.state_estimators.ParticleFilter

    .. dropdown:: KF Details

        .. autoclass:: prog_algs.state_estimators.ParticleFilter

.. dropdown:: Example

    Here's an example of its use. In this example we use the unscented kalman filter state estimator and the ThrownObject model. 

    .. code-block:: python

        >>> from prog_models.models import ThrownObject
        >>> from prog_algs.state_estimators import UnscentedKalmanFilter
        >>>
        >>> m = ThrownObject()
        >>> initial_state = m.initialize()
        >>> filt = UnscentedKalmanFilter(m, initial_state)
        >>>
        >>> load = {}  # No load for ThrownObject
        >>> new_data = {'x': 1.8}  # Observed state
        >>> print('Prior: ', filt.x.mean)
        >>> filt.estimate(0.1, load, new_data)
        >>> print('Posterior: ', filt.x.mean)

Extending
************

New :term:`state estimator` are created by extending the :class:`prog_algs.state_estimators.StateEstimator` class. 

See :download:`examples.new_state_estimator_example <../../prog_algs/examples/new_state_estimator_example.py>` for an example of this approach.

Example
^^^^^^^^^^^

* :download:`examples.new_state_estimator_example <../../prog_algs/examples/new_state_estimator_example.py>`
    .. automodule:: new_state_estimator_example

Prediction
-----------

Prediction is the process by which future states are estimated, given the initial state (e.g., from State Estimation), a model, and an estimate of :term:`future load`. An algorithm used to do this is called a :term:`predictor`. Prediction is often computationally expensive, especially for sample-based approaches with strict precision requirements (which therefore require large number of samples).

With this framework, there are a number of results that can be predicted. The exact prediction results are selected based on the needs of the end-user. The most common prediction results are Time of Event (ToE) and Time to Event (TtE). Time of Event at a specific prediction time (:math:`t_P`) is defined as the time when the event is expected to occur (with uncertainty), or equivalently, the time where the event state for that event is zero. Time to Event is defined as the time to ToE (:math:`TtE = ToE - t_P`). In prognostics, ToE and TtE are often referred to as End of Life (EOL) and Remaining Useful Life (RUL), respectively.

Beyond these, results of prediction can also include event state, outputs, performance metrics, and system states at different future times, including at ToE. For approaches that predict ToE with uncertainty, some users consider Probability of Success (PoS) or Probability of Failure (PoF). PoF is defined as the percentage of predictions that result in failure before the prognostics horizon (:math:`PoS \triangleq 1 - PoF`).

A predictors ``predict`` method is used to perform prediction, generally defined below:

.. code-block:: python

    result = predictor.predict(x0, future_loading, **config)

Where x0 is the initial state as an UncertainData object (often the output of state estimation), future_loading is a function defining future loading as a function of state and time, and config is a dictionary of any additional configuration parameters, specific to the predictor being used. See `Predictors <https://nasa.github.io/progpy/api_ref/prog_algs/Predictors.html>`__ for options available for each predictor

The result of the predict method is a named tuple with the following members:

* **times**: array of times for each savepoint such that times[i] corresponds to inputs.snapshot(i)
* **inputs**: :py:class:`prog_algs.predictors.Prediction` object containing inputs used to perform prediction such that inputs.snapshot(i) corresponds to times[i]
* **outputs**: :py:class:`prog_algs.predictors.Prediction` object containing  predicted outputs at each savepoint such that outputs.snapshot(i) corresponds to times[i]
* **event_states**: :py:class:`prog_algs.predictors.Prediction` object containing predicted event states at each savepoint such that event_states.snapshot(i) corresponds to times[i]
* **time_of_event**: :py:class:`prog_algs.uncertain_data.UncertainData` object containing the predicted Time of Event (ToE) for each event. Additionally, final state at time of event is saved at time_of_event.final_state -> :py:class:`prog_algs.uncertain_data.UncertainData` for each event

The stepsize and times at which results are saved can be defined like in a simulation. See `Simulation <https://nasa.github.io/progpy/docs/prog_models_guide.html#simulation>`__.

.. dropdown:: Included Predictors

    ProgPy includes a number of predictors in the *prog_algs.predictors* package. The most commonly used of these are highlighted below. See `Predictors <https://nasa.github.io/progpy/api_ref/prog_algs/Predictors.html>`__ for a full list of supported predictors.

    * **Unscented Transform (UT)**: A type of predictor for non-linear models where the state distribution is represented by a set of sigma points, calculated by an unscented tranform. Sigma points are propogated forward with time until the pass the threshold. The times at which each sigma point passes the threshold are converted to a distribution of time of event. The predicted future states and time of event are represented by a :py:class:`prog_algs.uncertain_data.MultivariateNormalDist`. By it's nature, UTs are much faster than MCs, but they fit the data to a normal distribution, resulting in some loss of information.
    * **Monte Carlo (MC)**: A sample-based prediction algorithm, where the distribution of likely states is represented by a set of unweighted samples. These samples are propagated forward with time. By its nature, MC is more accurate than a PF, but much slower. The predicted future states and time of event are represented by a :py:class:`prog_algs.uncertain_data.UnweightedSamples`. Full accuracy of MC can be adjusted by increasing or decreasing the number of samples

    .. dropdown:: UT Details

        .. autoclass:: prog_algs.predictors.UnscentedTransformPredictor

    .. dropdown:: MC Details

        .. autoclass:: prog_algs.predictors.MonteCarlo

        .. autoclass:: prog_algs.predictors.MonteCarloPredictor

Extending
*************

New :term:`predictor` are created by extending the :class:`prog_algs.predictors.Predictor` class. 


Analyzing Results
--------------------

State Estimation
*******************

The results of the state estimation are stored in an object of type :class:`prog_algs.uncertain_data.UncertainData`. This class contains a number of methods for analyzing a state estimate. This includes methods for obtaining statistics about the distribution, including the following:

* **mean**: The mean value of the state estimate distribution.
* **median**: The median value of the state estimate distribution.
* **cov**: Covariance matrix (in same order as keys in mean)
* **metrics**: A collection of various metrics about the distribution, inlcuding the ones above and percentiles of the state estimate
* **describe**: Similar to metrics, but in human readable format
* **percentage_in_bounds**: The percentage of the state estimate that is within defined bounds.
* **relative_accuracy**: Relative accuracy is how close the mean of the distribution is to the ground truth, on relative terms

There are also a number of figures available to describe a state estimate, described below

.. dropdown:: Scatter Plot

    A scatter plot is one of the best ways to visualize a distribution. The scatter plot will combine all of the states into a single plot, so you can see the correlation between different states as well as the distribution. This figure is made using the :pythoncode:`state.plot_scatter()` method. An example is illustrated below. 
    
    .. raw:: html

        <div style="text-align: center;">

    .. image:: images/single_scatter.png

    .. raw:: html

        </div>

    Multiple states can be overlayed on the same plot. This is typically done to show how a state evolves with time. The following example shows the distribution of states at different future times:

    .. code-block:: python

        >>> results = predictor.predict(...)
        >>> fig = results.states.snapshot(0).plot_scatter(label = "t={} s".format(int(results.times[0])))  # 0
        quarter_index = int(len(results.times)/4)
        >>> results.states.snapshot(quarter_index).plot_scatter(fig = fig, label = "t={} s".format(int(results.times[quarter_index])))  # 25%
        >>> results.states.snapshot(quarter_index*2).plot_scatter(fig = fig, label = "t={} s".format(int(results.times[quarter_index*2])))  # 50%
        >>> results.states.snapshot(quarter_index*3).plot_scatter(fig = fig, label = "t={} s".format(int(results.times[quarter_index*3])))  # 75%
        >>> results.states.snapshot(-1).plot_scatter(fig = fig, label = "t={} s".format(int(results.times[-1])))  # 100%

    .. raw:: html

        <div style="text-align: center;">

    .. image:: images/scatter.png

    .. raw:: html

        </div>

.. dropdown:: Histogram

    The simplest representation of a state estimate is a histogram. A histogram plot is genearted using the built in :pythoncode:`state.plot_hist()` method. The result is one histogram for each value in the state estimate, describing the distribution, as illustrated below:

    .. raw:: html

        <div style="text-align: center;">

    .. image:: images/histogram.png

    .. raw:: html

        </div>

See :class:`prog_algs.uncertain_data.UncertainData` documentation for more details.

Predicted Future States
**************************

Predicted future states, inputs, outputs, and event states come in the form of a :class:`prog_algs.predictors.Prediction` object. Predictions store distributions of predicted future values at multiple future times. Predictions contain a number of tools for analyzing the results, some of which are described below:

* **mean**: Estimate the mean value at each time. The result is a list of dictionaries such that prediction.mean[i] corresponds to times[i]
* **monotonicity**: Given a single prediction, for each event: go through all predicted states and compare those to the next one.
        Calculates monotonicity for each event key using its associated mean value in UncertainData [#Baptista2022]_ [#Coble2021]_


Time of Event (ToE)
**************************

Time of Event is also stored as an object of type :class:`prog_algs.uncertain_data.UncertainData`, so the analysis functions described in :ref:`State Estimation` are also available for a ToE estimate. See :ref:`State Estimation` or :class:`prog_algs.uncertain_data.UncertainData` documentation for details.

In addition to these standard UncertainData metrics, Probability of Success (PoS) is an important metric for prognostics. Probability of Success is the probability that a event will not occur before a defined time. For example, in aeronautics, PoS might be the probability that no failure will occur before end of mission.

Below is an example calculating probability of success:

.. code-block:: python

    >>> from prog_algs.metrics import prob_success
    >>> ps = prob_success(some_distribution, end_of_mission)

ToE Prediction Profile
**************************

A :class:`prog_algs.predictors.ToEPredictionProfile` contains Time of Event (ToE) predictions performed at multiple points. ToEPredictionProfile is frequently used to evaluate the prognostic quality for a given prognostic solution. It contains a number of methods to help with this, including:

* **alpha_lambda**: Whether the prediction falls within specified limits at particular times with respect to a performance measure [#Goebel2017]_ [#Saxena2010]_
* **cumulate relitive accuracy**: The sum of the relative accuracies of each prediction, given a ground truth
* **monotonicity**: The monotonicity of the prediction series [#Baptista2022]_ [#Coble2021]_
* **prognostic_horizon**: The difference between a time :math:`t_i`, when the predictions meet specified performance criteria, and the time corresponding to the true Time of Event (ToE), for each event [#Goebel2017]_ [#Saxena2010]_

A ToEPredictionProfile also contains a plot method (:pythoncode:`profile.plot(...)`), which looks like this:

.. image:: images/alpha_chart.png

This chart shows the distribution of estimated RUL (y-axis) at different prediction times (x-axis) in red. The ground truth and an alpha bound around the ground truth is displayed in green. 

Examples 
----------

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb

The best way to learn how to use `prog_algs` is through the `tutorial <https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb>`__. There are also a number of examples which show different aspects of the package, summarized and linked below:

* :download:`examples.basic_example <../../prog_algs/examples/basic_example.py>`
    .. automodule:: basic_example

* :download:`examples.basic_example_battery <../../prog_algs/examples/basic_example_battery.py>`
    .. automodule:: basic_example_battery

.. * :download:`examples.benchmarking_example <../../prog_algs/examples/benchmarking_example.py>`
..     .. automodule:: benchmarking_example

* :download:`examples.eol_event <../../prog_algs/examples/eol_event.py>`
    .. automodule:: eol_event

* :download:`examples.new_state_estimator_example <../../prog_algs/examples/new_state_estimator_example.py>`
    .. automodule:: new_state_estimator_example

* :download:`examples.horizon <../../prog_algs/examples/horizon.py>`
    .. automodule:: horizon

* :download:`examples.kalman_filter <../../prog_algs/examples/kalman_filter.py>`
    .. automodule:: kalman_filter

* :download:`examples.measurement_eqn_example <../../prog_algs/examples/measurement_eqn_example.py>`
    .. automodule:: measurement_eqn_example

* :download:`examples.playback <../../prog_algs/examples/playback.py>`
    .. automodule:: playback

* :download:`examples.predict_specific_event <../../prog_algs/examples/predict_specific_event.py>`
    .. automodule:: predict_specific_event

* :download:`examples.particle_filter_battery_example <../../prog_algs/examples/particle_filter_battery_example.py>`
    .. automodule:: particle_filter_battery_example

References
-------------

.. [#Goebel2017] Kai Goebel, Matthew John Daigle, Abhinav Saxena, Indranil Roychoudhury, Shankar Sankararaman, and José R Celaya. Prognostics: The science of making predictions. 2017

.. [#Saxena2010] Abhinav Saxena, José Celaya, Sankalita Saha, Bhaskar Saha, and Kai Goebel. Saxena, A., Celaya, J. Metrics for Offline Evaluation of Prognostic Performance. International Journal of Prognostics and Health Management, 1(1), 20. 2010.

.. [#Coble2021] Jamie Coble et al. Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society. http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404, 2021
        
.. [#Baptista2022] Marcia Baptista, et. al.. Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Aritifical Intelligence, Volume 306. https://www.sciencedirect.com/science/article/pii/S0004370222000078, 2022
