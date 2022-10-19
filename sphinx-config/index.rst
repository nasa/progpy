ProgPy Prognostics Python Packages 
=============================================================

.. raw:: html

    <iframe src="https://ghbtns.com/github-btn.html?user=nasa&repo=prog_models&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
    <br />

The NASA Prognostics Python Packages (ProgPy) are a set of open-sourced python packages supporting research and development of prognostics and health management tools. They implement architectures and common functionality of prognostics, supporting researchers and practitioners.

ProgPy was developed by researchers of the NASA Prognostics Center of Excellence (PCoE) and `Diagnostics & Prognostics Group <https://www.nasa.gov/content/diagnostics-prognostics>`__, with assistance from our partners: Northrop Grumman, Vanderbilt University, and the German Aerospace Center (DLR). We welcome contributions and are actively interested in partnering with other organizations and researchers. If interested in contibuting, please email Chris Teubert at christopher.a.teubert@nasa.gov.

ProgPy consists of a set of packages, described below. See the documentation specific to each package for more information.  

* :py:mod:`prog_models` : Tools for defining, building, using, and testing models for prognostics
* :py:mod:`prog_algs` : Tools for performing and benchmarking prognostics and state estimation
* :py:mod:`prog_server` and :py:mod:`prog_client` : A simplified implementation of a Service-Oriented Architecture (SOA) for performing prognostics and associated client

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:

   prog_models_guide
   prog_algs_guide
   prog_server_guide
   api_ref
   releases
   glossary
   dev_guide


Citing This Repository
-----------------------
Use the following to cite this repository:

  @misc{2022_nasa_progpy,
    | author    = {Christopher Teubert and Chetan Kulkarni and Matteo Corbetta and Katelyn Jarvis and Matthew Daigle},
    | title     = {ProgPy Packages},
    | month     = September,
    | year      = 2022,
    | version   = {1.4},
    | url       = {https://nasa.github.io/prog\_models}
    | }

The corresponding reference should look like this:

C. Teubert, C. Kulkarni, M. Corbetta, K. Jarvis, M. Daigle, Prognostics Model Python Package, v1.4, September 2022. URL https://github.com/nasa/prog_models.

Contributing and Partnering
-----------------------------
We welcome contributions and are actively interested in partnering with other organizations and researchers. If interested in contibuting, please email Chris Teubert at christopher.a.teubert@nasa.gov. 

Indices and Tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Disclaimers
----------------------

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
