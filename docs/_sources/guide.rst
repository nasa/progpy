ProgPy Guide 
=============================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:

   prog_models_guide
   prog_algs_guide
   prog_server_guide

This page is a general guide for ProgPy. ProgPy consists of three packages: prog_models, prog_algs, prog_server. To access a guide specific to the package you're using, select it in the menu below.

.. panels::
    :img-top-cls: pt-2, pb-2
    :header: text-center
    :body: text-center
    :column: col-lg col-lg col-lg

    ---
    :img-top: images/cube.png

    .. link-button:: prog_models Guide
        :type: ref
        :text: prog_models
        :classes: stretched-link btn-outline-primary btn-block

    ---
    :img-top: images/Gear-icon.png

    .. link-button:: prog_algs Guide
        :type: ref
        :text: prog_algs
        :classes: stretched-link btn-outline-primary btn-block

    ---
    :img-top: images/Server_icon_CC0.svg.png

    .. link-button:: prog_server Guide
        :type: ref
        :text: prog_server
        :classes: stretched-link btn-outline-primary btn-block


What is Prognostics
------------------------------
ProgPy uses the following definition for :term:`prognostics`:

.. topic:: Prognostics

   Prediction of (a) future performance and/or (b) the time at which one or more events of interest occur, for a system or a system of systems

This is similar to those described in [#Goebel2017]_. This approach is intended to be generic, capable of describing system behavior based on physical principles (i.e., physics-based), learning from data (i.e., data-based), or hybrid approaches (e.g., Physics-Informed Machine Learning). 

In general, the ProgPy prognostic approach is illustrated below. 

.. image:: images/package_structure.png

The foundation of prognostics is a :term:`model`. Models describe the behavior of a system or system of systems. A prognostics model specifically describes how the state of the system evolves with time. Prognostic models typically come in one of 4 categories: knowledge-based, :term:`physics-based<physics-based model>`, :term:`data-driven<data-driven model>`, or some combination of those three (i.e., hybrid).

Functionality for creation, simulation, and analysis of models can be found in the :ref:`prog_models<prog_models Guide>` package. That package also includes some example models and tools to access relevant data for model creation. For more information see the :ref:`prog_models Guide`.

ProgPy divides the prognostic process into two steps: :term:`state estimation<state estimator>` and :term:`prediction<predictor>`. State estimation is the process of determining the current system state (x), with some uncertainty, given the system parameters (:math:`\Theta`), system loading (u) and measurements (z). There are various methods used for this, such as Kalman Filters and Particle Filters. These methods utilize a prognostics model, comparing measurements (z) with those predicted from the system output equation.

In the prediction step, the state estimate at the prediction time and system model are used together to estimate system degradation with time. This is most commonly done using a variant of the Monte Carlo method with the model state transition equation. Prediction is often computationally expensive, especially for sample-based approaches with strict precision requirements (which therefore require large number of samples). ProgPy provides some potential solutions to combat this, such as :term:`surrogate` models, vectorization, and model configuration options.

Algorithms for :term:`state estimation<state estimator>` and :term:`prediction<predictor>` along with tools analyzing and visualizing results of state estimation and prediction, managing uncertainty, and creating new state estimators or predictors can be found in the :ref:`prog_algs<prog_algs Guide>` package. For more information see the :ref:`prog_algs Guide`.

More information
------------------------------

For more information, see the inidividual pages for each of the three ProgPy Packages

.. panels::
    :img-top-cls: pt-2, pb-2
    :header: text-center
    :body: text-center
    :column: col-lg col-lg col-lg

    ---
    :img-top: images/cube.png

    .. link-button:: prog_models Guide
        :type: ref
        :text: prog_models
        :classes: stretched-link btn-outline-primary btn-block

    ---
    :img-top: images/Gear-icon.png

    .. link-button:: prog_algs Guide
        :type: ref
        :text: prog_algs
        :classes: stretched-link btn-outline-primary btn-block

    ---
    :img-top: images/Server_icon_CC0.svg.png

    .. link-button:: prog_server Guide
        :type: ref
        :text: prog_server
        :classes: stretched-link btn-outline-primary btn-block


References
----------------------------

.. [#Goebel2017] Kai Goebel, Matthew John Daigle, Abhinav Saxena, Indranil Roychoudhury, Shankar Sankararaman, and Jos ́e R Celaya. Prognostics: The science of making predictions. 2017
