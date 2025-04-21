.. _troubleshooting:

Troubleshooting Guide
==============
This document includes common ProgPy issues and ways to troubleshoot.

Tensorflow Version Compatibility
------------------------
The current datadriven dependencies require ``tensorflow>=2.18.0`` due to compatibility with newer versions of ``numpy`` (from 2.0) and ``numpoly`` (from 1.3.6). Dependency specifications can be found in ``pyproject.toml``.

Older versions of `tensorflow` may work with older ``numpy`` and ``numpoly`` versions. One known version that works is ``tensorflow==2.16.2`` with ``numpy==1.26.4`` and ``numpoly==1.2.12``.

Data-Driven Tools
------------------------
If you are using data-driven tools (e.g., LSTM model), make sure the datadriven dependencies are installed using the following command:

.. tabs::

    .. tab:: Stable Version (Recommended)

        .. code-block:: console

            $ pip install 'progpy[datadriven]'

    .. tab:: Pre-Release

        .. code-block:: console

            $ pip install -e '.[datadriven]'

Installing ProgPy Data-Driven Tools with Python 3.13
------------------------
Tensorflow does not support python3.13 as of the writing of this. Until this is fixed, ProgPy data-driven features may not work correctly. If you are having trouble running data-driven features with Python3.13, try with an earlier version of Python.
