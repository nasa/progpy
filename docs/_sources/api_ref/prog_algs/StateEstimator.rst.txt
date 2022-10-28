State Estimators
===========================
The State Estimator uses sensor information and a Prognostics Model (see: `prog_models package <https://github.com/nasa/prog_models>`__) to produce an estimate of system state (which can be used to estimate outputs, event_states, and performance metrics). This state estimate can either be used by itself or as input to a `Predictor <predictors.html>`__. A state estimator is typically run each time new information is available.

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

See tutorial and examples for more information and additional features.

Included State Estimators
-------------------------
The following state estimators are included with this package. A new state estimator can be created by subclassing `prog_algs.state_estimators.StateEstimator`. See also: `state_estimator_template.py`

.. tabs:: prog_algs

   .. tab:: Particle Filter

      .. autoclass:: prog_algs.state_estimators.ParticleFilter

   .. tab:: Unscented Kalman Filter

      .. autoclass:: prog_algs.state_estimators.UnscentedKalmanFilter
   
   .. tab:: Kalman Filter

      .. autoclass:: prog_algs.state_estimators.KalmanFilter

State Estimator Interface
-------------------------
.. autoclass:: prog_algs.state_estimators.StateEstimator
   :members:
   :inherited-members:
