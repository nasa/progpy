Uncertain Data
=======================

The `progpy.uncertain_data` package includes classes for representing data with uncertainty. All types of UncertainData can be operated on using `the interface <#interface>`__. Inidividual classes for representing uncertain data of different kinds are described below, in `Implemented UncertainData Types <#implemented-uncertaindata-types>`__.

Interface
------------------------
.. autoclass:: progpy.uncertain_data.UncertainData
   :members:
   :inherited-members:

Implemented UncertainData Types
--------------------------------

.. tabs::

   .. tab:: Unweighted Samples

      .. autoclass:: progpy.uncertain_data.UnweightedSamples
         :members: key

   .. tab:: Multivariate Normal Distribution

      .. autoclass:: progpy.uncertain_data.MultivariateNormalDist

   .. tab:: Scalar

      .. autoclass:: progpy.uncertain_data.ScalarData
