prog_models Guide
===================================================

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=prog_models&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

The Prognostics Models Package (:py:mod:`prog_models`) is a Python framework for defining, building, using, and testing models for prognostics (computation of remaining useful life) of engineering systems. It also provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components and can be used in conjunction with the Prognostics Algorithms Library to perform research in prognostics methods. 

Installing
-----------------------

The latest stable release of :py:mod:`prog_models` is hosted on PyPi. For most users (unless you want to contribute to the development of :py:mod:`prog_models`), the version on PyPi will be adequate. To install from the command line, use the following command:

.. code-block:: console

    $ pip install prog_models

.. collapse:: Installing Pre-Release Versions

    Users who would like to contribute to :py:mod:`prog_models` or would like to use pre-release features can do so using the `prog_models GitHub repo <https://github.com/nasa/prog_models>`__. This isn't recommended for most users as this version may be unstable. To do this, use the following commands:

        .. code-block:: console

            $ git clone https://github.com/nasa/prog_models
            $ cd prog_models
            $ git checkout dev 
            $ pip install -e .

Getting Started 
------------------

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nasa/prog_models/master?labpath=tutorial.ipynb

The best way to learn how to use :py:mod:`prog_models` is through the `tutorial <https://mybinder.org/v2/gh/nasa/prog_models/master?labpath=tutorial.ipynb>`__. There are also a number of examples that show different aspects of the package, summarized and linked in the below sections

.. |br| raw:: html

     <br>

Building New Models
******************************

New models are constructed by subclassing :py:class:`PrognosticsModel` as illustrated in the first example. To generate a new model, create a new class for your model that inherits from this class. Alternatively, you can copy the template :download:`prog_model_template.ProgModelTemplate <../../prog_models/prog_model_template.py>`, replacing the methods with logic defining your specific model. The analysis and simulation tools defined in :class:`prog_models.PrognosticsModel` will then work with your new model. 

For simple linear models, users can choose to subclass the simpler :py:class:`LinearModel` class, as illustrated in the second example. Some methods and algorithms only function on linear models.

* :download:`examples.new_model <../../prog_models/examples/new_model.py>`
    .. automodule:: new_model

|br|

* :download:`examples.linear_model <../../prog_models/examples/linear_model.py>`
    .. automodule:: linear_model

|br|

.. collapse:: Advanced features in model building

    * :download:`examples.derived_params <../../prog_models/examples/derived_params.py>`
        .. automodule:: derived_params

    |br|

    * :download:`examples.state_limits <../../prog_models/examples/state_limits.py>`
        .. automodule:: state_limits

    |br|

    * :download:`examples.events <../../prog_models/examples/events.py>`
        .. automodule:: events

    |br|

Data Model Examples
******************************
For data-driven models such as those using neural networks, the :py:func:`DataModel.from_data` and :py:func:`DataModel.from_model` methods are used to construct new models. The use of these is demonstrated in the following examples.

* :download:`examples.lstm_model <../../prog_models/examples/lstm_model.py>`
    .. automodule:: lstm_model

|br|
 
* :download:`examples.full_lstm_model <../../prog_models/examples/full_lstm_model.py>`
    .. automodule:: full_lstm_model

|br|
 
* :download:`examples.generate_surrogate <../../prog_models/examples/generate_surrogate.py>`
    .. automodule:: generate_surrogate

|br|

Simulation
******************************

One of the most basic of functions using a model is simulation. Use of simulation is described in the following examples:

* :download:`examples.sim <../../prog_models/examples/sim.py>`
    .. automodule:: sim

|br|

* :download:`examples.noise <../../prog_models/examples/noise.py>`
    .. automodule:: noise

|br|

* :download:`examples.future_loading <../../prog_models/examples/future_loading.py>`
    .. automodule:: future_loading

|br|

* :download:`examples.dynamic_step_size <../../prog_models/examples/dynamic_step_size.py>`
    .. automodule:: dynamic_step_size

|br|

Model-Specific Examples
******************************
These examples illustrate use of the models provided in the :py:mod:`prog_models.models` module.

* :download:`examples.sim_battery_eol <../../prog_models/examples/sim_battery_eol.py>`
    .. automodule:: sim_battery_eol

|br|

* :download:`examples.sim_pump <../../prog_models/examples/sim_pump.py>`
    .. automodule:: sim_pump

|br|

* :download:`examples.sim_valve <../../prog_models/examples/sim_valve.py>`
    .. automodule:: sim_valve

|br|

* :download:`examples.sim_powertrain <../../prog_models/examples/sim_powertrain.py>`
    .. automodule:: sim_powertrain

|br|

* :download:`examples.visualize <../../prog_models/examples/visualize.py>`
    .. automodule:: visualize

|br|


Other
******************************

* :download:`examples.benchmarking <../../prog_models/examples/benchmarking.py>`
    .. automodule:: benchmarking

|br|

* :download:`examples.sensitivity <../../prog_models/examples/sensitivity.py>`
    .. automodule:: sensitivity

|br|

* :download:`examples.dataset <../../prog_models/examples/dataset.py>`
    .. automodule:: dataset

|br|

* :download:`examples.param_est <../../prog_models/examples/param_est.py>`
    .. automodule:: param_est

|br|

Tips
----
* To predict a certain partial :term:`state` (e.g., 50% SOH), create a new :term:`event` (e.g., 'SOH_50') override the event_state and threshold_met equations to also predict that additional state.
* If you're only doing diagnostics without prognostics- just define a next_state equation with no change of :term:`state` and don't perform prediction. The :term:`state estimator` can still be used to estimate if any of the :term:`event`s have occured.
* Sudden :term:`event`s use a binary :term:`event state` (1=healthy, 0=failed).
* You can predict as many :term:`event`s as you would like, sometimes one :term:`event` must happen before another, in this case the :term:`event` occurance for :term:`event` 1 can be a part of the equation for :term:`event` 2 ('event 2': event_1 and [OTHER LOGIC]).
