Glossary
==============

.. glossary::
    :sorted:

    event
      Something that can be predicted (e.g., system failure). An event has either occurred or not. See also: :term:`threshold`

    event state
      Progress towards :term:`event` occurring. Defined as a number where an event state of 0 indicates the :term:`event` has occurred and 1 indicates no progress towards the :term:`event` (i.e., fully healthy operation for a failure event). For a gradually occurring :term:`event` (e.g., discharge) the number will progress from 1 to 0 as the :term:`event` nears. In prognostics, event state is frequently called "State of Health".

    input
      Control or loading applied to the system being modeled (e.g., current drawn from a battery). Input is frequently denoted by u.

    output
      Measured sensor values from a system (e.g., voltage and temperature of a battery). Output is frequently denoted by z.

    future load
      :term:`input` (i.e., loading) expected to be applied to a system at future times

    performance metric
      Performance characteristics of a system that are a function of system state, but are not directly measured.

    state
      Internal parameters (typically hidden states) used to represent the state of the system- can be same as inputs/outputs but do not have to be.  State is frequently denoted as x

    state estimator
      An algorithm that is used to estimate the :term:`state` of the system, given measurements and a model, defined in the :py:mod:`prog_algs.state_estimators` subpackage (e.g., :py:class:`prog_algs.state_estimators.UnscentedKalmanFilter`).

    process noise
      Represents uncertainty in the model transition (e.g., model uncertainty), applied to :term:`state` during state transition.

    measurement noise
      Represents uncertainty in the measurement process (e.g., sensor sensitivity, sensor misalignements, environmental effects), applied to :term:`output`.

    predictor
      An algorithm that is used to predict future states, given the initial state, a model, and an estimate of :term:`future load`. E.g., :py:class:`prog_algs.predictors.MonteCarlo`.

    prediction
      A prediction of something (e.g., :term:`input`, :term:`state`, :term:`output`, :term:`event state`, etc.), with uncertainty, at one or more future times, a result of a :term:`predictor` prediction step (:py:func:`prog_algs.predictors.Predictor.predict`). For example- a prediction of the future :term:`state` of a system at certain specified savepoints, returned from prediction using a :py:class:`prog_algs.predictors.MonteCarlo` predictor. 

    surrogate
      A model that approximates the behavior of another model. Often used to generate a faster version of a model (e.g., for resource-constrained applications or to be used in optimization) or to test a data model. Generated using :py:func:`prog_models.PrognosticsModel.generate_surrogate` method.

    model
      A subclass of :py:class:`prog_models.PrognosticsModel` the describes the behavior of a system. Models are typically physics-based, data-driven (i.e., subclasses of :py:class:`prog_models.data_models.DataModel`), or some hybrid approach (e.g., physics informed machine learning).

    threshold
      The conditions under which an :term:`event` is considered to have occurred.
      