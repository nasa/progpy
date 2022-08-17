Uncertain Data
=======================

The `prog_algs.uncertain_data` package includes classes for representing data with uncertainty. All types of UncertainData can be operated on using `the interface <#interface>`__. Inidividual classes for representing uncertain data of different kinds are described below, in `Implemented UncertainData Types <#implemented-uncertaindata-types>`__.

Interface
------------------------
.. autoclass:: prog_algs.uncertain_data.UncertainData
   :members:
   :inherited-members:

Implemented UncertainData Types
--------------------------------

Unweighted Samples
******************
.. autoclass:: prog_algs.uncertain_data.UnweightedSamples
   :members: key

Multivariate Normal Distribution
********************************
.. autoclass:: prog_algs.uncertain_data.MultivariateNormalDist

Scalar Data (i.e., no uncertainty)
**********************************
.. autoclass:: prog_algs.uncertain_data.ScalarData
