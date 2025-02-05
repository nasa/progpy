Developers Guide & Project Plan
================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:

   npr7150

This document includes some details relevant for developers working on any of the Python Prognostics Packages Tools (progpy, prog_server)

Installing from a Branch 
------------------------
To install the package package from a specific branch. First clone the repository and checkout the branch. Then navigate into the repository directory and use the following command:

.. code-block:: console

   $ pip install -e .

This command installs the package using the checked-out version.

Running Tests
------------------------
The run the progpy tests, first clone the repository and checkout the branch, installing the package using the command above. Then navigate into the repository directory. Next install the tests required dependencies, by using the following commands:

.. code-block:: console

   $ pip install '.[test]'
      
      
Then run the tests using the following command:

.. code-block:: console

   $ python -m tests

.. admonition:: Note      

   Tests on data-driven tools (e.g., LSTM model) will need dependencies from the ``datadriven`` option installed.

Contributing 
---------------
New external (non-NASA or NASA contractor) developers must complete either the `organizational or individual Contributor License Agreement (CLA) <https://github.com/nasa/progpy/tree/master/forms>`__. 

Curious about what needs to be done? Have an idea for a new feature? Find a bug? Check out open issues `progpy <https://github.com/nasa/progpy/issues>`__, `prog_server <https://github.com/nasa/prog_server/issues>`__. 

Project Roles
--------------------
* Software Lead: Christopher Teubert
* Software Assurance Officer: Christopher Teubert
* Deputy Software Lead: Katelyn Jarvis
* Software Management Team: Software Lead, Software Assurance Officer, and Deputy Software Lead
* Developers: See `progpy developers <https://github.com/nasa/progpy/graphs/contributors>`__, `prog_server developers <https://github.com/nasa/prog_server/graphs/contributors>`_

Branching Strategy
------------------
Our project is following the git strategy described `here <https://nvie.com/posts/a-successful-git-branching-model/>`__. Release branches are not required. Details specific to each branch are described below. We recommend that developers from within NASA watch `this video <https://nasa-my.sharepoint.com/:v:/g/personal/rduffy_ndc_nasa_gov/EYCJK6qffBZNtSKZTOV9nPMBc9cPPF6fniRuKtG2GWvoPA>`_ on git strategies and best practices.

`master`: Every merge into the master branch is done using a pull request (never commiting directly), is assigned a release number, and must complete the release checklist. The release checklist is a software assurance tool. 

`dev`: Every commit on the dev branch should be functional. The PR checklist must be completed before merging into dev. Merging into dev should only be done through a PR, unless only documentation has been updated.

`Feature or Bugfix Branches`: These branches include changes specific to a new feature or fixing a bug.

`Hotfix Branches`: These branches are to fix an urgent issue in master. Hotfixes are branched directly off of master. Hotfixes are merged into both master and dev, after being approved by the Software Management Team.

PR Checklist
------------------
* If target is dev or master, ensure PR is reviewed by someone other than the author.
   * Reviewer should look for bugs, efficiency, readability, testing, and coverage in examples (if relevant).
* If adding a new feature, ensure there is a test verifying that feature.
* Ensure errors from static analysis must be resolved.
* Review the test coverage reports (if there is a change)
* Review the software benchmarking results (if there is a change)
* For added dependencies
   * Add to ``pyproject.toml``
   * Add to the bottom of dev_guide.rst (this document)
   * Notify Project Manager
* All warnings from static analysis must be reviewed and resolved - if deemed appropriate.

The following items are checked automatically:

* Ensure tests are passing.
* Check all new files have the following notice on them:
    | "Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved."


Note: when making change to this, update the message in the github actions workflows

Release Checklist
------------------
A release is the merging of a PR where the target is the master branch.

* [Complete - done in PRs to dev] Code review - all software must be checked by someone other than the author
* Check that each new feature has corresponding tests
* [Complete - checked automatically in PRs to dev] Confirm that every page has the copyright notice
* Confirm added dependencies are at the following:
   * setup.py,
   * the bottom of npr7150.rst
* Confirm that all issues associated with the release have been closed (i.e., requirements have been met) or assigned to another release
* Run unit tests `python -m tests` on the following computer types:
   * Apple Silicon Mac
   * Intel Mac
   * Windows
   * Linux 
* If present, run manual tests `python -m tests.test_manual`
* Review the template(s)
* Review static-analysis/linter results
* Review the tutorial
* Run and review the examples
* Check that all examples are tested
* Check new files in PR for any accidentally added
* Check documents
   * Check that all desired examples are in docs
   * General review: see if any updates are required
* Rebuild sphinx documents: `sphinx-build sphinx-config/ docs/`
* Write release notes
* Update version number in src/\*/__init__.py and setup.py
* For releases adding new features- ensure that NASA release process has been followed.
* Confirm that on GitHub Releases page, the next release has been started and that a schedule is present including at least Release Date, Release Review Date, and Release Branch Opening Date.

Updating Documentation 
**************************
Use the following command to update documentation (requires sphinx). Documentation is in the progpy repository.

.. code-block:: console

   $ pip install install '.[docs]'
   $ sphinx-build sphinx-config/ docs/

Uploading new version to PyPI
*******************************
New versions are uploaded upon release (i.e., merging into master branch). The Release Checklist must be complete prior to release

.. code-block: bash

    python -m build --sdist
    python -m build --wheel
    twine upload dist/*

See `here <https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project>`_


Note: when making change to this, update the message in the github actions workflows

Post-Release Checklist
-----------------------
* For prog_server: Update openapi specs on `SwaggerHub <https://app.swaggerhub.com/apis/teubert/prog_server/1.0.0-oas3>`__
* Send notes to Software Release Office (SRO) of updated version number
* Publish to PyPi
* Tag release with DOI
* Setup release in GitHub
* Post release in GitHub Discussions
* Merge doc changes
* Merge back into dev to get post-release changes
* Send Highlights - Division, Known Users, LinkedIn, etc.

Notes for Developers
--------------------
* Configuration-controlled items: source code, tests, test workflow, issues (requirements, non-conformances, bugs, etc.), milestones, examples, tutorial, templates, project plan (this document and 7150 document), and documentation (in progpy repo)
* The package itself is stored in the src directory.
* This is a research tool, so when making a design decision between operational efficiency and usability, generally choose the more usable option
* When supplied by or to the user, values with names (e.g., inputs, states, outputs, event_states, event occurance, etc.) should be supplied as dictionaries (or dict-like objects) where they can be referred to by name. 
* subpackages shall be independent (i.e., not have any dependencies with the wider package or other subpackages) when possible
* Whenever possible Models, UncertainData types, State Estimators, and Predictors should be interchangable with any other class of the same type (e.g., any model should be interchangable with any other model)
* Demonstrate common use cases as an example.
* Use collections.abc instead of typing
* Python code should comply with `PEP 8: Python Style Guide <https://peps.python.org/pep-0008/>`__, where appropriate
  * See also: `Writing Clean and Pythonic Code (JPL) <https://trs.jpl.nasa.gov/bitstream/handle/2014/51618/CL%2319-5039.pdf?sequence=1>`__
* Code should be complient with the recommendations of `LGTM <lgtm.com>`__, whenever appropriate
* Every feature should be demonstrated in an example
  * The most commonly used features should be demonstrated in the tutorial
* Except in the most extreme cases, maintain backwards compatibility for the convenience of existing users
  * If a feature is to be removed, mark it as depreciated (using DeprecationWarning) for at least 1 release before removing unless marked experimental
* Examples are included in the examples/ directory. 
   * Examples should cover the major use cases and features. If a major new feature is added, make sure there's an example demonstrating the feature.
   * For new examples- add to examples __all__ and example tests (tests/test_examples).
   * Also for new examples- add to getting started page in sphinx_config.
   * Examples should include comments explicitly describing each step.
* Tests are included in the tests/ directory.
   * Each new feature should have a test. Check this in each PR review.
   * Check test coverage to improve completeness, automatically reported by bot in each PR.
   * For tests- make sure test are quality. They should cover expected input ranges, error handling. 
   * There are some example models in progpy.models.test_models which are useful for testing
* Documentation 
   * Documentation is autogenerated using sphinx from progpy repository
   * Configuration is in sphinx_config.
   * Documentation is rebuilt on each release.
   * On each release, documentation can be seen at `nasa.github.io/progpy <https://nasa.github.io/progpy/>`__.
* GitHub Actions Test Workflow
   * Automated tests are defined in the .github/ directory.
   * The repository administrator can add tests to the set required to pass for each PR must be done by .
* Template
   * An empty template of a prognostics model is maintained at `progpy/prog_model_template.py`.
   * An empty template of a state estimator and predictor is maintained at `progpy/state_estimator_template.py` and `progpy/predictor_template.py`.
   * Any changes to the basic model setup should be documented there.
* A tutorial is included in tutorial.ipynb. This required Juypter Notebooks. All major features should be illustrated here.
