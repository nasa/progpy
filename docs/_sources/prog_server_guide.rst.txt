prog_server Guide
===================================================

.. role:: pythoncode(code)
   :language: python

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=prog_server&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

The Prognostics As-A-Service (PaaS) Sandbox (a.k.a., prog_server) is a simplified implementation of a Service-Oriented Architecture (SOA) for performing prognostics (estimation of time until events and future system states) of engineering systems. The PaaS Sandbox is a wrapper around the :ref:`Prognostics Algorithms Package <prog_algs Guide>`__ and :ref:`Prognostics Models Package <prog_models Guide>`, allowing one or more users to access the features of these packages through a REST API. The package is intended to be used as a research tool to prototype and benchmark Prognostics As-A-Service (PaaS) architectures and work on the challenges facing such architectures, including Generality, Communication, Security, Environmental Complexity, Utility, and Trust.

The PaaS Sandbox is actually two packages, prog_server and prog_client. The prog_server package is a prognostics server that provides the REST API. The prog_client package is a python client that provides functions to interact with the server via the REST API.

prog_server uses the :ref:`Prognostics Algorithms Package <prog_algs Guide>` and :ref:`Prognostics Models Package <prog_models Guide>`.

The PaaS Sandbox is a simplified version of the Prognostics As-A-Service Architecture implented as the PaaS/SWS Safety Service software by the NASA System Wide Safety (SWS) project, building upon the original work of the Convergent Aeronautics Solutions (CAS) project. This implementation is a research tool, and is therefore missing important features that should be present in a full implementation of the PaaS architecture such as authentication and persistent state management.

Installing prog_server
-----------------------

.. tabs::

    .. tab:: Stable Version (Recommended)

        The latest stable release of `prog_server` is hosted on PyPi. For most users (unless you want to contribute to the development of `prog_server`), the version on PyPi will be adequate. To install from the command line, use the following command:

        .. code-block:: console

            $ pip install prog_server

    .. tab:: Pre-release

        Users who would like to contribute to `prog_server` or would like to use pre-release features can do so using the `prog_server GitHub repo <https://github.com/nasa/prog_server>`__. This isn't recommended for most users as this version may be unstable. To use this version, use the following commands:

        .. code-block:: console

            $ git clone https://github.com/nasa/prog_server
            $ cd prog_server
            $ git checkout dev 
            $ pip install -e .

About
---------

`prog_server` uses the :ref:`Prognostics Algorithms Package <prog_algs Guide>` and :ref:`Prognostics Models Package <prog_models Guide>`. The best way to learn how to use prog_server is to first learn how to use these packages. See :ref:`Prognostics Algorithms Package Docs <prog_algs Guide>` and :ref:`Prognostics Models Package Docs <prog_models Guide>` for more details.

The PaaS Sandbox is actually two packages, ``prog_server`` and ``prog_client``. The ``prog_server`` package is the server that provides the REST API. The ``prog_client`` package is a python client that uses the REST API (see `prog_client <prog_client.html>`__). The ``prog_server`` package is the PaaS Sandbox Server. Once started the server can accept requests from one or more applications requesting prognostics, using its REST API (described in `prog_server_api`). 

Starting the prog_server 
--------------------------
There are two methods for starting the prog_server, described below:

.. tabs::

    .. tab:: Command line

        Generally, you can start the prog_server by running the module, like this:

        .. code-block:: console

            $ python -m prog_server

        .. admonition:: Note
            :class: tip

            You can force the server to start in debug mode using the ``debug`` flag. For example, :pythoncode:`python -m prog_server --debug`

    .. tab:: Programatically

        There are two methods to start the prog_server in python. The first, below, is non-blocking allowing users to perform other functions while the server is running.

        .. code-block:: python

            >>> import prog_server
            >>> prog_server.start() # Starts the server in a new process (is non-blocking)
            >>> ...
            >>> prog_server.stop() # Stops the server

        The second method, illustrated below, is blocking, meaning that the python shell will be blocked until the server is exited (e.g., by keyboard interrupt) 

        .. code-block:: python

            >>> import prog_server
            >>> prog_server.run() # Starts the server- blocking.

        Both run and start accept the following optional keyword arguments:

        * **host** (str): Server host address. Defaults to '127.0.0.1'
        * **port** (int): Server port address. Defaults to 5000
        * **debug** (bool): If the server is to be started in debug mode

Examples
------------

The best way to learn how to use prog_server is to look at examples. There are a number of examples included with prog_server, listed below:

* :download:`examples.online_prog <../../prog_server/examples/online_prog.py>`
    .. automodule:: online_prog

* :download:`examples.option_scoring <../../prog_server/examples/option_scoring.py>`
    .. automodule:: option_scoring
