Prediction
=======================

Predictions store the result of a prediction (i.e., returned by the predict method of a predictor). They store values (with uncertainty) at different future times. These are used to store states, inputs, outputs, perfomance metrics, and event states with uncertainty at savepoints.

Two types of predictions are distributed with this package: `Prediction` and `UnweightedSamplesPrediction`, described below. `UnweightedSamplesPrediction` extends `Prediction` to allow some operations specific to cases where each prediction is represented by an UnweightedSamples object (e.g., accessing SimResult for a single sample).

.. tabs::
   .. tab:: Prediction

      .. autoclass:: prog_algs.predictors.Prediction
         :members:
         :inherited-members: 

   .. tab:: UnweightedSamplesPrediction

      .. autoclass:: prog_algs.predictors.UnweightedSamplesPrediction
         :members:
         :inherited-members:
         :exclude-members: append, extend, clear, pop, remove, reverse, insert
