DataModel
=============

The :py:class:`DataModel` class is the base class for all data-based models. It is a subclass of :py:class:`PrognosticsModel`, allowing it to be used interchangeably with physics-based models.

.. ..  contents:: 
..     :backlinks: top

Examples:

* :download:`examples.lstm_model <../../../../prog_models/examples/lstm_model.py>`
* :download:`examples.full_lstm_model <../../../../prog_models/examples/full_lstm_model.py>`
* :download:`examples.custom_model <../../../../prog_models/examples/custom_model.py>`

Training DataModels
-----------------------
There are a few ways to construct a :py:class:`DataModel` object, described below.

From Data
*************************************************
This is the most common way to construct a :py:class:`DataModel` object, using the :py:func:`DataModel.from_data` method. It involves using one or more runs of data to train the model. Each DataModel class expects different data from the following set: times, inputs, states, outputs, and event_states. See documentation for the specific algorithm to see what it expects. Below is an example if it's use with the LSTMStateTransitionModel, which expects inputs and outputs.

.. dropdown:: example

   .. code-block:: python

      >>> from prog_models.models import LSTMStateTransitionModel
      >>> input_data = [run1.inputs, run2.inputs, run3.inputs]
      >>> output_data = [run1.outputs, run2.outputs, run3.outputs]
      >>> m = LSTMStateTransitionModel.from_data(input_data, output_data)

From Another PrognosticsModel (i.e., Surrogate)
*************************************************
Surrogate models are constructed using the :py:func:`DataModel.from_model` Class Method. These models are trained using data from the original model, i.e., as a surrogate for the original model. The original model is not modified. Below is an example if it's use. In this example a surrogate (m2) of the original ThrownObject Model (m) is created, and can then be used interchangeably with the original model.

.. dropdown:: example

   .. code-block:: python

      >>> from prog_models.models import ThrownObject
      >>> from prog_models.models import LSTMStateTransitionModel
      >>> m = ThrownObject()
      >>> def future_loading(t, x=None):
      >>>    return m.InputContainer({})  # No input for thrown object 
      >>> m2 = LSTMStateTransitionModel.from_model(m, future_loading)

.. note::

   Surrogate models are generally less accurate than the original model. This method is used either to create a quicker version of the original model (see :py:class:`DMDModel`) or to test the performance of a :py:class:`DataModel` approach.

.. seealso::

   :py:func:`PrognosticsModel.generate_surrogate`

Using Constructor
**********************
This method is the least frequently used, and it is very specific to the :py:class:`DataModel` class being constructed. For example: :py:class:`DMDModel` classes are constructed using the DMD Matrix, and :py:class:`LSTMStateTransitionModel` classes are constructed using a trained Keras Model.

See example :download:`examples.custom_model <../../../../prog_models/examples/custom_model.py>`

Included DataModels
-------------------------
The following DataModels are included in the package. A new DataModel can be created by subclassing :py:class:`DataModel`, implementing the abstract methods of both :py:class:`DataModel` and :py:class:`PrognosticsModel`.

DMDModel
**************************
.. autoclass:: prog_models.data_models.DMDModel
   :members: from_data, from_model

LSTMStateTransitionModel
**************************
.. autoclass:: prog_models.data_models.LSTMStateTransitionModel
   :members: from_data, from_model

DataModel Interface
---------------------------
.. autoclass:: prog_models.data_models.DataModel
   :members:
   :inherited-members:
   :exclude-members: SimulationResults, generate_model, observables, dx, next_state, initialize, output, event_state, threshold_met, apply_process_noise, apply_measurement_noise, apply_limits, performance_metrics
