Predictors
===========================

The Predictor uses a state estimate (type UncertainData subclass, output of a state estimator), information about expected future loading, and a Prognostics Model (see: `prog_models package <https://github.com/nasa/prog_models>`__) to predict both future states (also outputs, performance metrics, event_states) at predefined points and the time that an event will occur (Time of Event, ToE) with uncertainty.

Here's an example of its use. In this example we use the ThrownObject model and the MonteCarlo predictor, and we save the state every 1s. We also use a scalar first state (i.e., no uncertainty).

.. code-block:: python

   >>> from prog_models.models import ThrownObject
   >>> from prog_algs.predictors import MonteCarlo
   >>> from prog_algs.uncertain_data import ScalarData
   >>>
   >>> m = ThrownObject()
   >>> pred = MonteCarlo(m)
   >>> first_state = ScalarData({'x': 1.7, 'v': 20})  # Initial state for prediction
   >>> def future_loading(t, x): 
   >>>    return {}  # ThrownObject doesn't have a way of loading it
   >>>
   >>> pred_results = pred.predict(first_state, future_loading, save_freq=1)
   >>> pred_results.time_of_event.plot_hist(events='impact')  # Plot a histogram of when the impact event occurred

See tutorial and examples for more information and additional features.

Included Predictors
-----------------------
The following predictors are included with this package. A new predictor can be created by subclassing `prog_algs.predictors.Predictor`. See also: `predictor_template.py`

Monte Carlo Predictor
**********************
.. autoclass:: prog_algs.predictors.MonteCarlo

Unscented Transform Predictor
*****************************
.. autoclass:: prog_algs.predictors.UnscentedTransformPredictor

Predictor Interface
-----------------------
.. autoclass:: prog_algs.predictors.Predictor
   :members:
   :inherited-members:
