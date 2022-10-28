Release Notes
=================

.. ..  contents:: 
..     :backlinks: top

Updates in V1.4
-----------------------

prog_models
**************
* **Data-Driven Models**

  * Created new :py:class:`prog_models.data_models.DataModel` class as interface/superclass for all data-driven models. Data-driven models are interchangeable in use (e.g., simulation, use with prog_algs) with physics-based models. DataModels can be trained using data (:py:meth:`prog_models.data_models.DataModel.from_data`), or an existing model (:py:meth:`prog_models.data_models.DataModel.from_model`)
  * Introduced new LSTM State Transition DataModel (:py:class:`prog_models.data_models.LSTMStateTransitionModel`). See :download:`examples.lstm_model <../../prog_models/examples/lstm_model.py>`, :download:`examples.full_lstm_model <../../prog_models/examples/full_lstm_model.py>`, and :download:`examples.custom_model <../../prog_models/examples/custom_model.py>` for examples of use
  * DMD model (:py:class:`prog_models.data_models.DMDModel`) updated to new data-driven model interface. Can now be created from data as well as an existing model
  * Added ability to integrate training noise to data for DMD Model (:py:class:`prog_models.data_models.DMDModel`)

* **New Model**: Single-Phase DC Motor (:py:class:`prog_models.models.DCMotorSP`)
* Added the ability to select integration method when simulating (see ``integration_method`` keywork argument for :py:func:`prog_models.PrognosticsModel.simulate_to_threshold`). Current options are Euler and RK4
* New feature allowing serialization of model parameters as JSON. See :py:meth:`prog_models.PrognosticsModel.to_json`, :py:meth:`prog_models.PrognosticsModel.from_json`, and serialization example (:download:`examples.serialization <../../prog_models/examples/serialization.py>`)
* Added automatic step size feature in simulation. When enabled, step size will adapt to meet the exact save_pts and save_freq. Step size range can also be bounded
* New Example Model: Simple Paris' Law (:py:class:`prog_models.models.ParisLawCrackGrowth`)
* Added ability to set bounds when estimating parameters (See :py:meth:`prog_models.PrognosticsModel.estimate_params`)
* Initialize method is now optional
* Various bug fixes and performance improvements

prog_algs
**********
* Added new :py:class:`prog_algs.predictors.ToEPredictionProfile` Metric: Monotonicity. See :py:func:`prog_algs.predictors.ToEPredictionProfile.monotonicity`
* Updated to support prog_models v1.4
* Various bug fixes and performance improvements

prog_server and prog_client
****************************
* Added new endpoint (GET /api/v1/session/{id}/model) and client function (:py:meth:`prog_client.Session.get_model`) to get the model from the server.
* Updated to support prog_models and prog_algs v1.4
* Various bug fixes and performance improvements

Updates in V1.3
-----------------------

prog_models
**************
* **Surrogate Models** Added initial draft of new feature to generate surrogate models automatically from :class:`prog_models.PrognosticsModel`. (See :download:`examples.generate_surrogate <../../prog_models/examples/generate_surrogate.py>` example). Initial implementation uses Dynamic Mode Decomposition. Additional Surrogate Model Generation approaches will be explored for future releases. [Developed by NASA's DRF Project]
* **New Example Models** Added new :class:`prog_models.models.DCMotor`, :class:`prog_models.models.ESC`, and :class:`prog_models.models.Powertrain` models (See :download:`examples.sim_powertrain <../../prog_models/examples/sim_powertrain.py>` example) [Developed by NASA's SWS Project]
* **Datasets** Added new feature that allows users to access prognostic datasets programmatically (See :download:`examples.dataset <../../prog_models/examples/dataset.py>`)
* Added new :class:`prog_models.LinearModel` class - Linear Prognostics Models can be represented by a Linear Model. Similar to PrognosticsModels, LinearModels are created by subclassing the LinearModel class. Some algorithms will only work with Linear Models. See :download:`examples.linear_model <../../prog_models/examples/linear_model.py>` example for detail
* Added new StateContainer/InputContainer/OutputContainer objects for classes which allow for data access in matrix form and enforce expected keys. 
* Added new metric for SimResult: :py:func:`prog_models.sim_result.SimResult.monotonicity`.
* :py:func:`prog_models.sim_result.SimResult.plot` now automatically shows legends
* Added drag to :class:`prog_models.models.ThrownObject` model, making the model non-linear. Degree of nonlinearity can be effected using the model parameters (e.g., coefficient of drag cd).
* `observables` from previous releases are now called `performance_metrics`
* model.simulate_to* now returns named tuple, allowing for access by property name (e.g., result.states)
* Updates to :class:`prog_models.sim_result.SimResult` and :class:`prog_models.sim_result.LazySimResult` for robustness
* Various performance improvements and bug fixes

.. :note::

    Now input, states, and output should be represented by model.InputContainer, StateContainer, and OutputContainer, respectively

.. :note::

    Python 3.6 is no longer supported.

prog_algs
**********
* **New State Estimator Added** :class:`prog_algs.state_estimators.KalmanFilter`. Works with models derived from :class:`prog_models.LinearModel`. See :download:`examples.kalman_filter <../../prog_algs/examples/kalman_filter.py>`
* **New Predictor Added** :class:`prog_algs.predictors.UnscentedTransformPredictor`.
* Initial state estimate (x0) can now be passed as `UncertainData` to represent initial state uncertainty. See :download:`examples.playback <../../prog_algs/examples/playback.py>`
* Added new metrics for :class:`prog_algs.predictors.ToEPredictionProfile`: Prognostics horizon, Cumulative Relative Accuracy (CRA). See :download:`examples.playback <../../prog_algs/examples/playback.py>`
* Added ability to plot :class:`prog_algs.predictors.ToEPredictionProfile`: profile.plot(). See :download:`examples.playback <../../prog_algs/examples/playback.py>`
* Added new metric for :class:`prog_algs.predictors.Prediction`: Monotonicity, Relative Accuracy (RA)
* Added new metric for :class:`prog_algs.uncertain_data.UncertainData` (and subclasses): Root Mean Square Error (RMSE)
* Added new describe method for :class:`prog_algs.uncertain_data.UncertainData` (and subclasses)
* Add support for python 3.10
* Various performance improvements and bugfixes

prog_server
************
* Added ability to set state using pickled prog_algs.uncertain_data.UncertainData type

prog_client
************
* Added new set_state method

Updates in V1.2
------------------------

prog_models
**************
* New Feature: Vectorized Models
    * Distributed models were vectorized to support vectorized sample-based prognostics approaches
* New Feature: Dynamic Step Sizes
    * Now step size can be a function of time or state
    * See `examples.dynamic_step_size` for more information
* New Feature: New method model.apply_bounds
    * This method allows for other classes to use applied bound limits
* Simulate_to* methods can now specify initial time. Also, outputs are now optional
* Various bug fixes

prog_algs
**************

.. :note::

    This release includes changes to the return format of the MonteCarlo Predictor's `predict` method. These changes were necessary to support non-sample based predictors. The non backwards-compatible changes are listed below:

    * times: 
        * previous ```List[List[float]]``` where times[n][m] corresponds to timepoint m of sample n. 
        * new ```List[float]``` where times[m] corresponds to timepoint m for all samples.
    * End of Life (EOL)/ Time of Event (ToE) estimates:
        * previous ```List[float]``` where the times correspond to the time that the first event occurs.
        * new ```UnweightedSamples``` where keys correspond to the inidividualevents predicted.
    * State at time of event (ToE).
    * previous: element in states.
    * new: member of ToE structure (e.g., ToE.final_state['event1']).

* New Feature: Histogram and Scatter Plot of UncertainData.
* New Feature: Vectorized particle filter.
    * Particle Filter State Estimator is now vectorized for vectorized models - this significantly improves performance.
* New Feature: Unscented Transform Predictor.
    * New predictor that propogates sigma points forward to estimate time of event and future states.
* New Feature: `Prediction` class to represent predicted future values.
* New Feature: `ToEPredictionProfile` class to represent and operate on the result of multiple predictions generated at different prediction times.
* Added metrics `percentage_in_bounds` and `metrics` and plots to UncertainData .
* Add support for Python3.9.
* General Bugfixes.

Updates in V1.1
------------------------

prog_models
**************
* New Feature: Derived Parameters
    * Users can specify callbacks for parameters that are defined from others. These callbacks will be called when the dependency parameter is updated.
    * See `examples.derived_params` for more information.
* New Feature: Parameter Estimation
    * Users can use the estimate_parameters method to estimate all or select parameters. 
    * see `examples.param_est`
* New Feature: Automatic Noise Generation
    * Now noise is automatically generated when next_state/dx (process_noise) and output (measurement_noise). This removed the need to explicitly call apply_*_noise functions in these methods. 
    * See `examples.noise` for more details in setting noise
    * For any classes users created using V1.0.*, you should remove any call to apply_*_noise functions to prevent double noise application. 
* New Feature: Configurable State Bounds
    * Users can specify the range of valid values for each state (e.g., a temperature in celcius would have to be greater than -273.15 - absolute zero)
* New Feature: Simulation Result Class
    * Simulations now return a simulation result object for each value (e.g., output, input, state, etc) 
    * These simulation result objects can be used just like the previous lists. 
    * Output and Event State are now "Lazily Evaluated". This speeds up simulation when intermediate states are not printed and these properties are not used
    * A plot method has been added directly to the class (e.g., `event_states.plot()`)
* New Feature: Intermediate Result Printing
    * Use the print parameter to enable printing intermediate results during a simulation 
    * e.g., `model.simulate_to_threshold(..., print=True)`
    * Note: This slows down simulation performance
* Added support for python 3.9
* Various bug fixes

ElectroChemistry Model Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New Feature: Added thermal effects. Now the model include how the temperature is effected by use. Previous implementation only included effects of temperature on performance.
* New Feature: Added `degraded_capacity` (i.e., EOL) event to model. There are now three different models: BatteryElectroChemEOL (degraded_capacity only), BatteryElectroChemEOD (discharge only), and BatteryElectroChemEODEOL (combined). BatteryElectroChem is an alias for BatteryElectroChemEODEOL. 
* New Feature: Updated SOC (EOD Event State) calculation to include voltage when near V_EOD. This prevents a situation where the voltage is below lower bound but SOC > 0. 

CentrifugalPump Model Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New Feature: Added CentrifugalPumpBase class where wear rates are parameters instead of part of the state vector. 
    * Some users may use this class for prognostics, then use the parameter estimation tool occasionally to update the wear rates, which change very slowly.
* Bugfix: Fixed bug where some event states were returned as negative
* Bugfix: Fixed bug where some states were saved as parameters instead of part of the state. 
* Added example on use of CentrifugalPump Model (see `examples.sim_pump`)
* Performance improvements

PneumaticValve Model Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New Feature: Added PneumaticValveBase class where wear rates are parameters instead of part of the state vector. 
    * Some users may use this class for prognostics, then use the parameter estimation tool occasionally to update the wear rates, which change very slowly.
* Added example on use of PneumaticValve Model (see `examples.sim_valve`)

