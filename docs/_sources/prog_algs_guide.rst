prog_algs Guide
===================================================

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=prog_algs&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb

The Prognostic Algorithms Package is a python framework for prognostics (computation of remaining useful life or future states) of engineering systems. The package provides an extendable set of algorithms for state estimation and prediction, including uncertainty propagation. The package also include metrics, visualization, and analysis tools needed to measure the prognostic performance. The algorithms use prognostic models (from :py:mod:`prog_models`) to perform estimation and prediction functions. The package enables the rapid development of prognostics solutions for given models of components and systems. Different algorithms can be easily swapped to do comparative studies and evaluations of different algorithms to select the best for the application at hand.

Installing
-----------------------

Installing from pip (recommended)
********************************************
The latest stable release of `prog_algs` is hosted on PyPi. For most users (unless you want to contribute to the development of `prog_algs`), this version will be adequate. To install from the command line, use the following command:

.. code-block:: console

    $ pip install prog_algs

Installing Pre-Release Versions with GitHub
********************************************
For users who would like to contribute to `prog_algs` or would like to use pre-release features can do so using the 'dev' branch (or a feature branch) on the `prog_algs GitHub repo <https://github.com/nasa/prog_algs>`__. This isn't recommended for most users as this version may be unstable. To use this version, use the following commands:

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
 * `SimResult (from prog_models) <https://nasa.github.io/prog_models/simresult.html>`__ : The result of a single simulation (without uncertainty). Can be used to store inputs, outputs, states, event_states, observables, etc. Is returned by the model.simulate_to* methods.
 * `UncertainData <uncertain_data.html>`__ : Used throughout the package to represent data with uncertainty. There are a variety of subclasses of UncertainData to represent data with uncertainty in different forms (e.g., ScalarData, MultivariateNormalDist, UnweightedSamples). Notibly, this is used to represent the output of a StateEstimator's `estimate` method, individual snapshots of a prediction, and the time of event estimate from a predictor's `predict` method.
 * `Prediction <prediction.html#id1>`__ : Prediction of future values (with uncertainty) of some variable (e.g., input, state, output, event_states, etc.). The `predict` method of predictors return this. 
 * `ToEPredictionProfile <prediction.html#toe-prediction-profile>`__ : The result of multiple predictions, including time of prediction. This data structure can be treated as a dictionary of time of prediction to toe prediction. 

Use 
----
The best way to learn how to use `prog_algs` is through the `tutorial <https://mybinder.org/v2/gh/nasa/prog_algs/master?labpath=tutorial.ipynb>`__. There are also a number of examples which show different aspects of the package, summarized and linked below:

* :download:`examples.basic_example <../../prog_algs/examples/basic_example.py>`
    .. automodule:: examples.basic_example
    |
* :download:`examples.thrown_object_example <../../prog_algs/examples/thrown_object_example.py>`
    .. automodule:: examples.thrown_object_example
    |
* :download:`examples.utpredictor <../../prog_algs/examples/utpredictor.py>`
    .. automodule:: examples.utpredictor
    |
* :download:`examples.benchmarking_example <../../prog_algs/examples/benchmarking_example.py>`
    .. automodule:: examples.benchmarking_example
    |
* :download:`examples.eol_event <../../prog_algs/examples/eol_event.py>`
    .. automodule:: examples.eol_event
    |
* :download:`examples.horizon <../../prog_algs/examples/horizon.py>`
    .. automodule:: examples.horizon
    |
* :download:`examples.kalman_filter <../../prog_algs/examples/kalman_filter.py>`
    .. automodule:: examples.kalman_filter
    |
* :download:`examples.measurement_eqn_example <../../prog_algs/examples/measurement_eqn_example.py>`
    .. automodule:: examples.measurement_eqn_example
    |
* :download:`examples.new_state_estimator_example <../../prog_algs/examples/new_state_estimator_example.py>`
    .. automodule:: examples.new_state_estimator_example
    |
* :download:`examples.playback <../../prog_algs/examples/playback.py>`
    .. automodule:: examples.playback
    |
* :download:`examples.predict_specific_event <../../prog_algs/examples/predict_specific_event.py>`
    .. automodule:: examples.predict_specific_event
    |
* :download:`examples.particle_filter_battery_example <../../prog_algs/examples/particle_filter_battery_example.py>`
    .. automodule:: examples.particle_filter_battery_example
    |
* :download:`tutorial <../../prog_algs/tutorial.ipynb>`
    |

Extending
---------
New State Estimators and Predictors are created by extending the :class:`prog_algs.state_estimators.StateEstimator` and :class:`prog_algs.predictors.Predictor` class, respectively. 

See :download:`examples.new_state_estimator_example <../../prog_algs/examples/new_state_estimator_example.py>` for an example of this approach.

