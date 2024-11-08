Installing ProgPy
-----------------------

.. tabs::

    .. tab:: Stable Version (Recommended)

        The latest stable release of ProgPy is hosted on PyPi. For most users, this version will be adequate. To install via the command line, use the following command:

        .. code-block:: console

            $ pip install progpy

        If you will be using the datadriven tools (e.g., LSTM model), install the datadriven dependencies as well using the following command:

        .. code-block:: console

            $ pip install progpy[datadriven]

    .. tab:: Pre-Release

        Users who would like to contribute to ProgPy or would like to use pre-release features can do so using the `ProgPy GitHub repo <https://github.com/nasa/progpy>`__. This isn't recommended for most users as this version may be unstable. To do this, use the following commands:

        .. code-block:: console

            $ git clone https://github.com/nasa/progpy
            $ cd progpy
            $ git checkout dev 
            $ pip install -e .

        If you will be using the datadriven tools (e.g., LSTM model), install the datadriven dependencies as well using the following command:

        .. code-block:: console

            $ pip install -e '.[datadriven]'
