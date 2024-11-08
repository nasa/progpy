# Prognostics Python Package (ProgPy)
[![CodeFactor](https://www.codefactor.io/repository/github/nasa/progpy/badge)](https://www.codefactor.io/repository/github/nasa/progpy)
[![GitHub License](https://img.shields.io/badge/License-NOSA-green)](https://github.com/nasa/progpy/blob/master/license.pdf)
[![GitHub Releases](https://img.shields.io/github/release/nasa/progpy.svg)](https://github.com/nasa/progpy/releases)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nasa/progpy/HEAD?tutorial.ipynb)

The NASA Prognostic Package (ProgPy) is a python prognostics framework focused on building, using, and evaluating models and algorithms for prognostics (computation of remaining useful life) and health management of engineering systems, and provides a set of prognostics models for select components and prognostics algorithms developed within this framework, suitable for use in prognostics for these components. It also includes algorithms for state estimation and prediction, including uncertainty propagation.

**ProgPy was the 2024 NASA Software of the Year!**

ProgPy combines NASAs prog_models and prog_algs packages into a single python package, for simplicity and clarity.

## Installation 
`pip3 install progpy`

or 

`pip3 install progpy[datadriven]`

to include dependencies for data driven models

## [Documentation](https://nasa.github.io/progpy/)
See documentation [here](https://nasa.github.io/progpy/)
 
## Repository Directory Structure 
Here is the directory structure for the github repository 
 
`src/progpy/` - The prognostics python package<br />
`examples/` - Example Python scripts using progpy<br />
`tests/` - Tests for progpy<br />
`README.md` - The readme (this file)<br />
`prog_model_template.py` - Template for Prognostics Model<br />
`state_estimator_template.py` - Template for State Estimators<br />
`predictor_template.py` - Template for Predictor<br />

## Citing this repository
Use the following to cite this repository:

```
@misc{2024_nasa_progpy,
  | author    = {Christopher Teubert and Katelyn Jarvis Griffith and Matteo Corbetta and Chetan Kulkarni and Portia Banerjee and Matthew Daigle},
  | title     = {{ProgPy Python Prognostics Packages}},
  | month     = May,
  | year      = 2024,
  | version   = {1.7},
  | url       = {https://nasa.github.io/progpy}
  | }
```

The corresponding reference should look like this:

C. Teubert, K. Jarvis Griffith, M. Corbetta, C. Kulkarni, P. Banerjee, M. Daigle, ProgPy Python Prognostics Packages, v1.7, May 2024. URL https://github.com/nasa/progpy.

## Contributing Organizations
ProgPy was created by a partnership of multiple organizations, working together to build a set of high-quality prognostic tools for the wider PHM Community. We would like to give a big thank you for the ProgPy community, especially the following contributing organizations:

* [NASA's Diagnostics and Prognostics Group](https://www.nasa.gov/content/diagnostics-prognostics)
* German Aerospace Center (DLR)
* Northrop Grumman Corporation (NGC)
* Vanderbilt University

## Acknowledgements
The structure and algorithms of this package are strongly inspired by the [MATLAB Prognostics Model Library](https://github.com/nasa/PrognosticsModelLibrary), [MATLAB Prognostics Algorithm Library](https://github.com/nasa/PrognosticsAlgorithmLibrary), and the [MATLAB Prognostics Metrics Library](https://github.com/nasa/PrognosticsMetricsLibrary). We would like to recognize Matthew Daigle and the rest of the team that contributed to the Prognostics Models, Algorithms, and Metrics Libraries for the contributions their work on the MATLAB library made to the design of progpy.

## Notices
Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

## Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
