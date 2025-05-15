Modeling and Sim Guide
===================================================

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=progpy&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

The Prognostics Python Package (ProgPy) includes tools for defining, building, using, and testing models for :term:`prognostics` of engineering systems. It also provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components and can be used in conjunction with the state estimation and prediction features (see :ref:`State Estimation and Prediction Guide<State Estimation and Prediction Guide>`) to perform research in prognostics methods. 

.. include:: installing.rst

Getting Started 
------------------

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/progpy/master?labpath=tutorial.ipynb

The best way to learn how to use progpy is through the `tutorial <https://mybinder.org/v2/gh/nasa/progpy/master?labpath=tutorial.ipynb>`__. There are also a number of examples that show different aspects of the package, summarized and linked in the below sections

ProgPy Prognostic Model Format
----------------------------------

Prognostics models are the foundation of the prognostics process. They describe how a system or system-of-systems is expected to behave based on how it is used/loaded (i.e., :term:`input`). Prognostic models typically come in one of 4 categories: knowledge-based, :term:`physics-based<physics-based model>`, :term:`data-driven<data-driven model>`, or some combination of those three (i.e., hybrid).

Inputs
^^^^^^^^^^^^^^^^^^^^^^^^

Prognostic model :term:`inputs<input>` are how a system is loaded. These are things that can be controlled, and affect how the system state evolves. The expected inputs for a model are defined by its *inputs* property. For example, a battery is loaded by applying a current, so the only input is *i*, the applied current. Inputs are also sometimes environmental conditions, such as ambient temperature or pressure. 

Inputs are one of the inputs to the state transition model, described in :ref:`States` .

States
^^^^^^^^^^^^^^^^^^^^

ProgPy prognostic models are state-transition models. The internal :term:`state` of the system at any time is represented by one or more (frequently :term:`hidden<hidden state>`) state variables, represented by the custom type StateContainer. Each model has a discrete set of states, the keys of which are defined by the *states* property.

For example, the example ThrownObject model has two states, position (x) and velocity (v).

States are transitioned forward in time using the state transition equation. 

.. raw:: html

    <div style="text-align: center;">

:math:`x(t+dt) = f(t, x(t), u(t), dt, \Theta)`

.. raw:: html

    </div>

where :math:`x(t)` is :term:`state` at time :math:`t`, :math:`u(t)` is :term:`input` at time :math:`t` , :math:`dt` is the stepsize, and :math:`\Theta` are the model :term:`parameters` .

In a ProgPy model, this state transition can be represented one of two ways, either discrete or continuous, depending on the nature of state transition. In the case of :term:`continuous models<continuous model>`, state transition behavior is defined by defining the first derivative, using the :py:func:`progpy.PrognosticsModel.dx` method. For :term:`discrete models <discrete model>`, state transition behavior is defined using the :py:func:`progpy.PrognosticsModel.next_state` method. The continuous state transition behavior is recommended, because defining the first derivative enables some approaches that rely on that information.

.. image:: images/next_state.png
    :width: 70 %
    :align: center

.. image:: images/dx.png
    :width: 70 %
    :align: center

States can also be discrete or continuous. :term:`Discrete states<discrete state>` are those which can only exist in a finite set of values. Continuous states are initialized with a number and discrete states are initialized using the function :py:func:`progpy.create_discrete_state`, like the examples below. Each discrete state represents a unique condition or mode, and transitions between states are governed by defined rules or events, providing clarity and predictability in state management.

.. code-block:: python

    >>> from progpy import create_discrete_state
    >>> ValveState = create_discrete_state(2, ["open", "closed"])
    >>> x["valve"] = ValveState.open

.. code-block:: python

    >>> from progpy import create_discrete_state
    >>> GearState = create_discrete_state(5, transition="sequential")
    >>> x["gear"] = GearState(1)

.. note::
    :term:`Discrete states <discrete state>` are different from :term:`discrete models <discrete model>`. Discrete models are models where state transition is discrete, where discrete states are where the state itself is discrete. Discrete models may have continuous states.

.. dropdown::  State Transition Equation Example

    An example of a state transition equation for a thrown object is included below. In this example, a model is created to describe an object thrown directly into the air. It has two states: position (x) and velocity (v), and no inputs.

    .. code-block:: python

        >>> def dx(self, x, u):
        >>>    # Continuous form
        >>>    dxdt = x['v']
        >>>    dvdt = -9.81  # Acceleration due to gravity
        >>>    return self.StateContainer({'x': dxdt, 'v': dvdt})

    or, alternatively

    .. code-block:: python

        >>> def next_state(self, x, u, dt):
        >>>    # Discrete form
        >>>    new_x = x['x'] + x['v']*dt
        >>>    new_v = x['v'] -9.81*dt  # Acceleration due to gravity
        >>>    return self.StateContainer({'x': new_x, 'v': new_v})


Output (Measurements)
^^^^^^^^^^^^^^^^^^^^^^^^^

The next important part of a prognostic model is the outputs. Outputs are measurable quantities of a system that are a function of system state. When applied in prognostics, generally the outputs are what is being measured or observed in some way. State estimators use the different between predicted and measured values of these outputs to estimate the system state. 

Outputs are a function of only the system state (x) and :term:`parameters` (:math:`\Theta`), as described below. The expected outputs for a model are defined by its *outputs* property. The logic of calculating outputs from system state is provided by the user in the model :py:func:`progpy.PrognosticsModel.output` method.

.. image:: images/output.png
    :width: 70 %
    :align: center

.. raw:: html

    <div style="text-align: center;">

:math:`z(t) = f(x(t), \Theta)`

.. raw:: html
    
    </div>

.. dropdown::  Output Equation Example

    An example of a output equation for a thrown object is included below. In this example, a model is created to describe an object thrown directly into the air. It has two states: position (x) and velocity (v). In this case we're saying that the position of the object is directly measurable. 

    .. code-block:: python

        >>> def output(self, x):
        >>>     # Position is directly measurable
        >>>     position = x['x']
        >>>     return self.OutputContainer({'x': position})

Events 
^^^^^^^^^^^^^^^^^^^^^^^^^^

Traditionally users may have heard the prognostic problem as estimating the Remaining Useful Life (RUL) of a system. ProgPy generalizes this concept with the concept of :term:`events<event>`. ProgPy Prognostic Models contain one or more events which can be predicted. Systems frequently have more than one failure mode, each of these modes can be represented by a separate event. For example, a valve model might have separate events for an internal leak and a leak at the input. Or a battery model might have events for insufficient capacity, thermal runaway, and low-voltage. 

Additionally, events can be used to predict other events of interest beyond failure, such as special system states or warning thresholds. For example, the above battery model might also have an warning event for when battery capacity reaches 50% of the original capacity because of battery aging with use.

The expected events for a model are defined by its *events* property. The logic of events can be defined in two methods: :py:func:`progpy.PrognosticsModel.threshold_met` and :py:func:`progpy.PrognosticsModel.event_state`.

:term:`Thresholds<threshold>` are the conditions under which an event occurs. The logic of the threshold is defined in the :py:func:`progpy.PrognosticsModel.threshold_met` method. This method returns boolean for each event specifying if the event has occured. 

.. image:: images/threshold_met.png
    :width: 70 %
    :align: center

.. raw:: html

    <div style="text-align: center;">

:math:`tm(t) = f(x(t), \Theta)`

.. raw:: html
    
    </div>

:term:`Event states<event state>` are an estimate of the progress towards a threshold. Where thresholds are boolean, event states are a number between 0 and 1, where 0 means the event has occured, 1 means no progress towards the event. Event states are a generalization of State of Health (SOH) for systems with multiple events and non-failure events. The logic of the event states is defined in :py:func:`progpy.PrognosticsModel.event_state`.

.. image:: images/event_state.png
    :width: 70 %
    :align: center

.. raw:: html

    <div style="text-align: center;">

:math:`es(t) = f(x(t), \Theta)`

.. raw:: html
    
    </div>

If threshold_met is not specified, threshold_met is defined as when event_state is 0. Alternately, if event_state is not defined, it will be 0 when threshold_met is True, otherwise 1. If a model has events, at least one of these methods must be defined

.. dropdown:: Event Examples

    An example of a event_state and threshold_met equations for a thrown object is included below. In this example, a model is created to describe an object thrown directly into the air. It has two states: position (x) and velocity (v). The event_state and threshold_met equations for this example are included below

    .. code-block:: python

        >>> def event_state(self, x):
        >>>     # Falling event_state is 0 when velocity hits 0, 1 at maximum speed
        >>>     falling_es = np.maximum(x['v']/self.parameters['throwing_speed'], 0)
        >>>
        >>>     # Impact event_state is 0 when position hits 0, 
        >>>     # 1 when at maximum height or when velocity is positive (going up)
        >>>     if x['v'] > 0:
        >>>         # Event state is 1 until falling starts
        >>>         x_max = 1
        >>>     else:
        >>>         # Use speed and position to estimate maximum height
        >>>         x_max = x['x'] + np.square(x['v'])/(-self.parameters['g']*2) 
        >>>     impact_es = np.maximum(x['x']/x_max,0)
        >>>     return {'falling': falling_es, 'impact': impact_es}
    
    .. code-block:: python

        >>> def threshold_met(self, x):
        >>>     return {
        >>>         'falling': x['v'] < 0,
        >>>         'impact': x['x'] <= 0
        >>>     }


Parameters
^^^^^^^^^^^^^^^

Parameters are used to configure the behavior of a model. For parameterized :term:`physics-based<physics-based model>` models, parameters are used to configure the general system to match the behavior of the specific system. For example, parameters of the general battery model can be used to configure the model to describe the behavior of a specific battery.

Models define a ``default_parameters`` property- the default parameters for that model. After construction, the parameters for a specific model can be accessed using the *parameters* property. For example, for a model `m`

.. code-block:: python

    >>> print(m.parameters)

Parameters can be set in model construction, using the *parameters* property after construction, or using Parameter Estimation feature (See :ref:`Parameter Estimation`). The first two are illustrated below:

.. code-block:: python

    >>> m = SomeModel(some_parameter=10.2, some_other_parameter=2.5)
    >>> m.parameters['some_parameter'] = 11.2  # Overriding parameter

The specific parameters are very specific to the system being modeled. For example, a battery might have parameters for the capacity and internal resistance. When using provided models, see the documentation for that model for details on parameters supported.

.. dropdown:: Derived Parameters

    Sometimes users would like to specify parameters as a function of other parameters. This feature is called "derived parameters". See the derived parameters section in the example below for more details on this feature. 

    * :download:`04 New Models <../../progpy/examples/04_New Models.ipynb>`

Noise
^^^^^^^^^^^^^^^^^^^^^^^

In practice, it is impossible to have absolute knowledge of future states due to uncertainties in the system. There is uncertainty in the estimates of the present state, future inputs, models, and prediction methods [Goebel2017]_. This model-based prognostic approach incorporates this uncertainty in four forms: initial state uncertainty (:math:`x_0`), :term:`process noise`, :term:`measurement noise`, and :term:`future loading noise`.

.. dropdown:: Process Noise

    Process noise is used to represent uncertainty in the state transition process (e.g., uncertainty in the quality of your model or your model configuration :term:`parameters`).

    Process noise is applied in the state transition method (See :ref:`States`). 

.. dropdown:: Measurement Noise

    Measurement noise is used to represent uncertainty in your measurements. This can represent such things as uncertainty in the logic of the model's output method or sensor noise. 

    Measurement noise is applied in the output method (See :ref:`Output (Measurements)`).

.. dropdown:: Future Loading Noise

    Future loading noise is used to represent uncertainty in knowledge of how the system will be loaded in the future (See :ref:`Future Loading`). Future loading noise is applied by the user in their provided future loading method by adding random noise to the estimated future load.

See the noise section in the example below for details on how to configure proccess and measurement noise in ProgPy.

* :download:`01 Simulation <../../progpy/examples/01_Simulation.ipynb>`

Future Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`Future Loading <future load>` is an essential part of prediction and simulation. In order to simulate forward in time, you must have an estimate of how the system will be used (i.e., loaded) during the window of time that the system is simulated. Future load is essentially expected :ref:`Inputs` at future times.

Future loading is provided by the user either using the predifined loading classes in `progpy.loading`, or as a function of time and optional state. For example:

.. code-block:: python

    def future_load(t, x=None):
        # Calculate inputs 
        return m.InputContainer({'input1': ...})

See the future loading section in the example below for details on how to provide future loading information in ProgPy. 

* :download:`01 Simulation <../../progpy/examples/01_Simulation.ipynb>`

General Notes
^^^^^^^^^^^^^^^^

Users of ProgPy will need a model describing the behavior of the system of interest. Users will likely either use one of the models distribued with ProgPy (see `Included Models <https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html>`__), configuring it to their own system using parameter estimation (see :download:`02 Parameter Estimation <../../progpy/examples/02_Parameter Estimation.ipynb>`), use a :term:`data-driven model` class to learn system behavior from data, or build their own model (see `Building New Models`_ section, below). 

Building New Models
----------------------

ProgPy provides a framework for building new models. Generally, models can be divided into three basis categories: :term:`physics-based models<physics-based model>`, :term:`data-driven models<data-driven model>`, and hybrid models. Additionally, models can rely on state-transition for prediction, or they can use what is called direct-prediction. These two categories are described below.

State-Transition Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Physics-Based

        New :term:`physics-based models<physics-based model>` are constructed by subclassing :py:class:`progpy.PrognosticsModel` as illustrated in the first example. To generate a new model, create a new class for your model that inherits from this class. Alternatively, you can copy the template :download:`prog_model_template.ProgModelTemplate <../../progpy/prog_model_template.py>`, replacing the methods with logic defining your specific model. The analysis and simulation tools defined in :class:`progpy.PrognosticsModel` will then work with your new model. 

        For simple linear models, users can choose to subclass the simpler :py:class:`progpy.LinearModel` class, as illustrated in the second example. Some methods and algorithms only function on linear models.

        * :download:`04 New Models <../../progpy/examples/04_New Models.ipynb>`

    .. tab:: Data-Driven

        New :term:`data-driven models<data-driven model>`, such as those using neural networks, are created by subclassing the :py:class:`progpy.data_models.DataModel` class, overriding the ``from_data`` method.
        
        The :py:func:`progpy.data_models.DataModel.from_data` and :py:func:`progpy.data_models.DataModel.from_model` methods are used to construct new models from data or an existing model (i.e., :term:`surrogate`), respectively. The use of these is demonstrated in the following examples.

        .. note:: 
            To use a data-driven model distributed with progpy you need to install the data-driven dependencies.

            .. code-block:: console

                $ pip install progpy[datadriven] 

        * :download:`05_Data Driven <../../progpy/examples/05_Data Driven.ipynb>`

        * :download:`examples.lstm_model <../../progpy/examples/lstm_model.py>`
            .. automodule:: lstm_model
        
        * :download:`examples.full_lstm_model <../../progpy/examples/full_lstm_model.py>`
            .. automodule:: full_lstm_model

        * :download:`examples.pce <../../progpy/examples/pce.py>`
            .. automodule:: pce
        
        * :download:`examples.generate_surrogate <../../progpy/examples/generate_surrogate.py>`
            .. automodule:: generate_surrogate

        .. dropdown:: Advanced features in data models

            * :download:`examples.custom_model <../../progpy/examples/custom_model.py>`
                .. automodule:: custom_model

Direct-Prediction Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:term:`Direct-prediction models<direct-prediction model>` are models that estimate :term:`time of event` directly from the current state and :term:`future load`, instead of being predicted through state transition. When models are pure direct-prediction models, future states cannot be predicted. See the direct models section in the example below for more information.

* :download:`04 New Models <../../progpy/examples/04_New Models.ipynb>`

Using Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whether you're using :term:`data-driven<data-driven model>`, :term:`physics-based<physics-based model>`, expert knowledge, or some hybrid approach, building and validating a model requires data. In the case of data-driven approaches, data is used to train and validate the model. In the case of physics-based, data is used to estimate parameters (see `Parameter Estimation`) and validate the model.

ProgPy includes some example datasets. See `ProgPy Datasets <https://nasa.github.io/progpy/api_ref/progpy/DataSets.html>`_ and the example below for details. 

* :download:`examples.dataset <../../progpy/examples/dataset.py>`
    .. automodule:: dataset

.. note:: To use the dataset feature, you must install the requests package.

Using Provided Models
----------------------------

ProgPy includes a number of predefined models in the :py:mod:`progpy.models` module. These models are parameterized, so they can be configured to represent specific systems (see :ref:`Parameter Estimation`). 

For details on the included models, see `Included Models <https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html>`__. The examples below also illustrate the use of some models provided in the :py:mod:`progpy.models` module.

* :download:`03 Included Models <../../progpy/examples/03_Existing Models.ipynb>`

* :download:`examples.sim_pump <../../progpy/examples/sim_pump.py>`
    .. automodule:: sim_pump

* :download:`examples.sim_valve <../../progpy/examples/sim_valve.py>`
    .. automodule:: sim_valve

* :download:`examples.sim_powertrain <../../progpy/examples/sim_powertrain.py>`
    .. automodule:: sim_powertrain
        
* :download:`examples.sim_dcmotor_singlephase <../../progpy/examples/sim_dcmotor_singlephase.py>`
    .. automodule:: sim_dcmotor_singlephase

* :download:`examples.uav_dynamics_model <../../progpy/examples/uav_dynamics_model.py>`
    .. automodule:: uav_dynamics_model

Simulation
----------------------------

One of the most basic of functions using a model is simulation. Simulation is the process of predicting the evolution of system :term:`state` with time, given a specific :term:`future load` profile. Unlike full prognostics, simulation does not include uncertainty in the state and other product (e.g., :term:`output`) representation. For a prognostics model, simulation is done using the :py:meth:`progpy.PrognosticsModel.simulate_to` and :py:meth:`progpy.PrognosticsModel.simulate_to_threshold` methods.

.. role:: pythoncode(code)
   :language: python

.. dropdown:: Saving Results

    :py:meth:`progpy.PrognosticsModel.simulate_to` and :py:meth:`progpy.PrognosticsModel.simulate_to_threshold` return the inputs, states, outputs, and event states at various points in the simulation. Returning these values for every timestep would require a lot of memory, and is not necessary for most use cases, so ProgPy provides an ability for users to specify what data to save. 

    There are two formats to specify what data to save: the ``save_freq`` and ``save_pts`` arguments, described below

    .. list-table:: 
        :header-rows: 1

        * - Argument
          - Description
          - Example
        * - ``save_freq``
          - The frequency at which data is saved
          - :pythoncode:`m.simulate_to_threshold(..., save_freq=10)`
        * - ``save_pts``
          - Specific times at which data is saved
          - :pythoncode:`m.simulate_to_threshold(..., save_pts=[15, 25, 33])`

    
    .. admonition:: Note
        :class: tip

        Data will always be saved at the next time after the ``save_pt`` or ``save_freq``. As a result, the data may not correspond to the exact time specified. Use automatic step sizes to save at the exact time.

.. dropdown:: Step Size

    Step size is the size of the step taken in integration. It is specified by the ``dt`` argument. It is an important consideration when simulating. Too large of a step size could result in wildly incorrect results, and two small of a step size can be computationally expensive. Step size can be provided in a few different ways, described below:

    * *Static Step Size*: Provide a single number. Simulation will move forward at this rate. Example, :pythoncode:`m.simulate_to_threshold(..., dt=0.1)`
    * *Automatic Dynamic Step Size*: Step size is adjusted automatically to hit each save_pt and save_freq exactly. Example, :pythoncode:`m.simulate_to_threshold(..., dt='auto')`
    * *Bounded Automatic Dynamic Step Size*: Step size is adjusted automatically to hit each save_pt and save_freq exactly, with a maximum step size. Example, :pythoncode:`m.simulate_to_threshold(..., dt=('auto', 0.5))`
    * *Functional Dynamic Step Size*: Step size is provided as a function of time and state. This is the most flexible approach. Example, :pythoncode:`m.simulate_to_threshold(..., dt= lambda t, x : max(0.75 - t*0.01, 0.25))`

    For more details on dynamic step sizes, see the following example:

    * :download:`01 Simulation <../../progpy/examples/01_Simulation.ipynb>`

.. dropdown:: Integration Methods

    Simulation is essentially the process of integrating the model forward with time. By default, simple euler integration is used to propogate the model forward. Advanced users can change the numerical integration method to affect the simulation quality and runtime. This is done using the ``integration_method`` argument in :py:meth:`progpy.PrognosticsModel.simulate_to_threshold` and :py:meth:`progpy.PrognosticsModel.simulate_to`.

    For example, users can use the commonly-used Runge Kutta 4 numerical integration method using the following method call for model m:

    .. code-block:: python

        >>> m.simulate_to_threshold(future_loading, integration_method = 'rk4')

.. dropdown:: Eval Points

    Sometimes users would like to ensure that simulation hits a specific point exactly, regardless of the step size (``dt``). This can be done using the ``eval_pts`` argument in :py:meth:`progpy.PrognosticsModel.simulate_to_threshold` and :py:meth:`progpy.PrognosticsModel.simulate_to`. This argument takes a list of times at which simulation should include. For example, for simulation to evaluate at 10 and 20 seconds, use the following method call for model m:

    .. code-block:: python

        >>> m.simulate_to_threshold(future_loading, eval_pts = [10, 20])

    This feature is especially important for use cases where loading changes dramatically at a specific time. For example, if loading is 10 for the first 5 seconds and 20 afterwards, and you have a  ``dt`` of 4 seconds, here's loading simulation would see:

     * 0-4 seconds: 10
     * 4-8 seconds: 10
     * 8-12 seconds: 20

    That means the load of 10 was applied 3 seconds longer than it was supposed to. Adding a eval point of 5 would apply this load:

     * 0-4 seconds: 10
     * 4-5 seconds: 10
     * 5-9 seconds: 20

    Now loading is applied correctly.

For simulation examples, see the following notebook for details.

* :download:`01 Simulation <../../progpy/examples/01_Simulation.ipynb>`

Parameter Estimation
----------------------------

Parameter estimation is an important step in prognostics. Parameter estimation is used to tune a general model to match the behavior of a specific system. For example, parameters of the general battery model can be used to configure the model to describe the behavior of a specific battery.

Sometimes model parameters are directly measurable (e.g., dimensions of blades on rotor). For these parameters, estimating them is a simple act of direct measurement. For parameters that cannot be directly measured, they're typically estimated using observed data. 

Generally, parameter estimation is done by tuning the parameters of the model so that simulation best matches the behavior observed in some available data. In ProgPy, this is done using the :py:meth:`progpy.PrognosticsModel.estimate_params` method. This method takes :term:`input` and :term:`output` data from one or more runs, and uses scipy.optimize.minimize function to estimate the parameters of the model.

.. code-block:: python
    
    >>> params_to_estimate = ['param1', 'param2']
    >>> m.estimate_params([run1_data, run2_data], params_to_estimate, dt=0.01)

See the example below for more details.

.. admonition:: Note
    :class: tip

    Parameters are changes in-place, so the model on which ``estimate_params`` is called, is now tuned to match the data.

Visualizing Results
----------------------------

Results of a simulation can be visualized using the plot method. For example:

.. code-block:: python

    >>> results = m.simulate_to_threshold(...)
    >>> results.outputs.plot()
    >>> results.states.plot()

See :py:meth:`progpy.sim_result.SimResult.plot` for more details on plotting capabilities

Combination Models
----------------------------

There are two methods in progpy through which multiple models can be combined and used together: composite models and ensemble models, described below. For more details, see the example below.

:download:`06. Combining Models <../../progpy/examples/06_Combining Models.ipynb>`

.. tabs::

    .. tab:: Composite Models

        Composite models are used to represent the behavior of a system of interconnected systems. Each system is represented by its own model. These models are combined into a single composite model which behaves as a single model. When definiting the composite model the user provides a discription of any connections between the state or output of one model and the input of another. For example, 

        .. code-block:: python

            >>> m = CompositeModel(
            >>>     models = [model1, model2],
            >>>     connections = [
            >>>         ('model1.state1', 'model2.input1'),
            >>>         ('model2.state2', 'model1.input2')
            >>>     ]
            >>> )

    .. tab:: Ensemble Models

        Unlike composite models which model a system of systems, ensemble models are used when to combine the logic of multiple models which describe the same system. This is used when there are multiple models representing different system behaviors or conditions. The results of each model are aggregated in a way that can be defined by the user. For example,

        .. code-block:: python

            >>> m = EnsembleModel(
            >>>     models = [model1, model2],
            >>>     aggregator = np.mean
            >>> )

    .. tab:: MixtureOfExperts Models
        
        Mixture of Experts (MoE) models combine multiple models of the same system, similar to Ensemble models. Unlike Ensemble Models, the aggregation is done by selecting the "best" model. That is the model that has performed the best over the past. Each model will have a 'score' that is tracked in the state, and this determines which model is best.

        .. code-block:: python

             >> m = MixtureOfExpertsModel([model1, model2])

Other Examples
----------------------------

* :download:`examples.benchmarking <../../progpy/examples/benchmarking.py>`
    .. automodule:: benchmarking

* :download:`examples.sensitivity <../../progpy/examples/sensitivity.py>`
    .. automodule:: sensitivity

Tips & Best Practices
----------------------
* If you're only doing diagnostics without prognostics- just define a next_state equation with no change of :term:`state` and don't perform prediction. The :term:`state estimator` can still be used to estimate if any of the :term:`events<event>` have occured.
* Sudden :term:`events<event>` use a binary :term:`event state` (1=healthy, 0=failed).
* You can predict as many :term:`events<event>` as you would like, sometimes one :term:`event` must happen before another, in this case the :term:`event` occurance for event 1 can be a part of the equation for event 2 ('event 2': event_1 and [OTHER LOGIC]).
* Minimize the number of state variables whenever possible
* Whenever possible, if calculations dont include state or inputs, include values as parameters or derived parameters instead of calculating within state transition
* Use constant units throughout the model
* Document all assumptions and limitations

References
----------------------------

.. [Goebel2017] Kai Goebel, Matthew John Daigle, Abhinav Saxena, Indranil Roychoudhury, Shankar Sankararaman, and José R Celaya. Prognostics: The science of making predictions. 2017

.. [Celaya2012] J Celaya, A Saxena, and K Goebel. Uncertainty representation and interpretation in model-based prognostics algorithms based on Kalman filter estimation. Annual Conference of the Prognostics and Health Management Society, 2012.

.. [Sankararaman2011] S Sankararaman, Y Ling, C Shantz, and S Mahadevan. Uncertainty quantification in fatigue crack growth prognosis. International Journal of Prognostics and Health Management, vol. 2, no. 1, 2011.
