ProgPy Prognostics Python Packages 
=============================================================

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=progpy&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
    <br />

NASA's ProgPy is an open-sourced python package supporting research and development of prognostics and health management and predictive maintenance tools. It implements architectures and common functionality of prognostics, supporting researchers and practitioners. The ProgPy package is a combination of the original prog_models and prog_algs packages.

ProgPy documentation is split into three senctions described below.

* :ref:`Modeling and Simulation<Modeling and Sim Guide>` : defining, building, using, and testing models for prognostics
* :ref:`State Estimation and Prediction<State Estimation and Prediction Guide>` : performing and benchmarking prognostics and state estimation
* :ref:`prog_server<prog_server Guide>` and :ref:`prog_client<prog_server Guide>` : A simplified implementation of a Service-Oriented Architecture (SOA) for performing prognostics and associated client

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:

   guide
   api_ref
   releases
   glossary
   dev_guide

Installing progpy
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

Citing This Repository
-----------------------
Use the following to cite this repository:

@misc{2023_nasa_progpy,
  | author    = {Christopher Teubert and Katelyn Jarvis Griffith and Matteo Corbetta and Chetan Kulkarni and Portia Banerjee and Jason Watkins and Matthew Daigle},
  | title     = {{ProgPy Python Prognostics Packages}},
  | month     = May,
  | year      = 2024,
  | version   = {1.7},
  | url       = {https://nasa.github.io/progpy}
  | doi       = {10.5281/ZENODO.8097013}
  | }

The corresponding reference should look like this:

C. Teubert, K. Jarvis Griffith, M. Corbetta, C. Kulkarni, P. Banerjee, J. Watkins, M. Daigle, ProgPy Python Prognostics Packages, v1.7, May 2024. URL https://github.com/nasa/progpy.

Contributing and Partnering
-----------------------------
ProgPy was developed by researchers of the NASA Prognostics Center of Excellence (PCoE) and `Diagnostics & Prognostics Group <https://www.nasa.gov/content/diagnostics-prognostics>`__, with assistance from our partners. We welcome contributions and are actively interested in partnering with other organizations and researchers. If interested in contibuting, please email Chris Teubert at christopher.a.teubert@nasa.gov.

A big thank you to our partners who have contributed to the design, testing, and/or development of ProgPy:

- German Aerospace Center (DLR) Institute of Maintenance, Repair and Overhaul.
- Northrop Grumman Corporation (NGC)
- Research Institutes of Sweden (RISE)
- Vanderbilt University

Indices and Tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Disclaimers
----------------------

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
