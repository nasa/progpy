prog_models Guide
===================================================

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=prog_models&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

The Prognostics Models Package (:py:mod:`prog_models`) is a Python framework for defining, building, using, and testing models for prognostics (computation of remaining useful life) of engineering systems. It also provides a set of prognostics models for select components developed within this framework, suitable for use in prognostics applications for these components and can be used in conjunction with the Prognostics Algorithms Library to perform research in prognostics methods. 

Installing
-----------------------

Installing from pip (recommended)
********************************************
The latest stable release of `prog_models` is hosted on PyPi. For most users (unless you want to contribute to the development of `prog_models`), the version on PyPi will be adequate. To install from the command line, use the following command:

.. code-block:: console

    $ pip install prog_models

Installing Pre-Release Versions with GitHub
********************************************
For users who would like to contribute to `prog_models` or would like to use pre-release features can do so using the 'dev' branch (or a feature branch) on the `prog_models GitHub repo <https://github.com/nasa/prog_models>`__. This isn't recommended for most users as this version may be unstable. To use this version, use the following commands:

.. code-block:: console

    $ git clone https://github.com/nasa/prog_models
    $ cd prog_models
    $ git checkout dev 
    $ pip install -e .

Use 
---
The best way to learn how to use `prog_models` is through the `tutorial <https://mybinder.org/v2/gh/nasa/prog_models/master?labpath=tutorial.ipynb>`__. There are also a number of examples that show different aspects of the package, summarized and linked below:

* :download:`examples.sim <../../prog_models/examples/sim.py>`
    .. automodule:: examples.sim
    |
* :download:`examples.benchmarking <../../prog_models/examples/benchmarking.py>`
    .. automodule:: examples.benchmarking
    |
* :download:`examples.new_model <../../prog_models/examples/new_model.py>`
    .. automodule:: examples.new_model
    |
* :download:`examples.sensitivity <../../prog_models/examples/sensitivity.py>`
    .. automodule:: examples.sensitivity
    |
* :download:`examples.events <../../prog_models/examples/events.py>`
    .. automodule:: examples.events
    |
* :download:`examples.noise <../../prog_models/examples/noise.py>`
    .. automodule:: examples.noise
    |
* :download:`examples.dataset <../../prog_models/examples/dataset.py>`
    .. automodule:: examples.dataset
    |
* :download:`examples.generate_surrogate <../../prog_models/examples/generate_surrogate.py>`
    .. automodule:: examples.generate_surrogate
    |
* :download:`examples.linear_model <../../prog_models/examples/linear_model.py>`
    .. automodule:: examples.linear_model
    |
* :download:`examples.lstm_model <../../prog_models/examples/lstm_model.py>`
    .. automodule:: examples.lstm_model
    | 
* :download:`examples.visualize <../../prog_models/examples/visualize.py>`
    .. automodule:: examples.visualize
    |
* :download:`examples.future_loading <../../prog_models/examples/future_loading.py>`
    .. automodule:: examples.future_loading
    |
* :download:`examples.param_est <../../prog_models/examples/param_est.py>`
    .. automodule:: examples.param_est
    |
* :download:`examples.derived_params <../../prog_models/examples/derived_params.py>`
    .. automodule:: examples.derived_params
    |
* :download:`examples.state_limits <../../prog_models/examples/state_limits.py>`
    .. automodule:: examples.state_limits
    |
* :download:`examples.dynamic_step_size <../../prog_models/examples/dynamic_step_size.py>`
    .. automodule:: examples.dynamic_step_size
    |
* :download:`tutorial <../../prog_models/tutorial.ipynb>`
    |

Model-Specific Examples
------------------------
* :download:`examples.sim_battery_eol <../../prog_models/examples/sim_battery_eol.py>`
    .. automodule:: examples.sim_battery_eol
    |
* :download:`examples.sim_pump <../../prog_models/examples/sim_pump.py>`
    .. automodule:: examples.sim_pump
    |
* :download:`examples.sim_valve <../../prog_models/examples/sim_valve.py>`
    .. automodule:: examples.sim_valve
    |
* :download:`examples.sim_powertrain <../../prog_models/examples/sim_powertrain.py>`
    .. automodule:: examples.sim_powertrain
    |

Extending
----------
You can create new models by creating a new subclass of :class:`prog_models.PrognosticsModel` or :class:`prog_models.LinearModel` (for simple linear models).

To generate a new model, create a new class for your model that inherits from this class. Alternatively, you can copy the template :download:`prog_model_template.ProgModelTemplate <../../prog_models/prog_model_template.py>`, replacing the methods with logic defining your specific model.

The analysis and simulation tools defined in :class:`prog_models.PrognosticsModel` will then work with your new model. 

See :download:`examples.new_model <../../prog_models/examples/new_model.py>` for an example of this approach.

Tips
----
* To predict a certain partial state (e.g., 50% SOH), create a new event (e.g., 'SOH_50') override the event_state and threshold_met equations to also predict that additional state.
* If you're only doing diagnostics without prognostics- just define a next_state equation with no change of state and don't perform prediction. The state estimator can still be used to estimate if any of the events have occured.
* Sudden events use a binary event_state (1=healthy, 0=failed).
* You can predict as many events as you would like, sometimes one event must happen before another, in this case the event occurance for event 1 can be a part of the equation for event 2 ('event 2': event_1 and [OTHER LOGIC]).

