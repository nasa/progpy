State Estimators
===========================
The State Estimator uses sensor information and a Prognostics Model to produce an estimate of system state (which can be used to estimate outputs, event_states, and performance metrics). This state estimate can either be used by itself or as input to a `Predictor <https://nasa.github.io/progpy/api_ref/progpy/Predictor.html>`__. A state estimator is typically run each time new information is available.

Here's an example of its use. In this example we use the unscented kalman filter state estimator and the ThrownObject model. 

.. code-block:: python

   >>> from progpy.models import ThrownObject
   >>> from progpy.state_estimators import UnscentedKalmanFilter
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
The following state estimators are included with this package. A new state estimator can be created by subclassing `progpy.state_estimators.StateEstimator`. See also: `state_estimator_template.py`

.. tabs:: progpy

   .. tab:: Particle Filter

      .. autoclass:: progpy.state_estimators.ParticleFilter

   .. tab:: Unscented Kalman Filter

      .. autoclass:: progpy.state_estimators.UnscentedKalmanFilter
   
   .. tab:: Kalman Filter

      .. autoclass:: progpy.state_estimators.KalmanFilter

State Estimator Interface
-------------------------
.. autoclass:: progpy.state_estimators.StateEstimator
   :members:
   :inherited-members:
