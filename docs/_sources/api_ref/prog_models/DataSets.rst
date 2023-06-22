Datasets
================================================================

The `prog_models` dataset subpackage is used to download labeled prognostics data for use in model building, analysis, or validation. Every dataset comes equipped with a  `load_data` function which loads the specified data. Some datasets require a dataset number or id. This indicates the specific data to load from the larger dataset. The format of the data is specific to the dataset downloaded. Details of the specific datasets are summarized below:

.. note:: To use the dataset feature, you must install the requests package.

Variable Load Battery Data (nasa_battery)
----------------------------------------------------
.. autofunction:: prog_models.datasets.nasa_battery.load_data

CMAPSS Jet Engine Data (nasa_cmapss)
----------------------------------------------------
.. autofunction:: prog_models.datasets.nasa_cmapss.load_data
 