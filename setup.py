# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

INSTALL_REQS = [
        'scipy',
        'pandas',  # For data downloading features
        'matplotlib',
        'requests',  # For data downloading features
        'chaospy',  # For PCE
        'fastdtw',  # For DTW error calculation
        "tensorflow; platform_system!='Darwin' or platform_machine!='arm64'",
        "tensorflow-macos; platform_system=='Darwin' and platform_machine=='arm64'",
        "filterpy",
    ]

setup(
    name='progpy',
    version='1.6.0',
    description='The NASA Prognostic Package (ProgPy) is a python prognostics framework focused on building, using, and evaluating models and algorithms for prognostics (computation of remaining useful life) and health management of engineering systems, and provides a set of prognostics models for select components and prognostics algorithms developed within this framework, including uncertainty propagation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://nasa.github.io/progpy/',
    author='Christopher Teubert',
    author_email='christopher.a.teubert@nasa.gov',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Manufacturing',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: Other/Proprietary License ',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords=['prognostics', 'diagnostics', 'fault detection', 'fdir', 'physics modeling', 'prognostics and health management', 'PHM', 'health management', 'surrogate modeling', 'model tuning', 'simulation', 'ivhm'],
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    python_requires='>=3.7, <3.12',
    install_requires=INSTALL_REQS,
    license='NOSA',
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/nasa/progpy/issues',
        'Docs': 'https://nasa.github.io/progpy/',
        'Organization': 'https://www.nasa.gov/content/diagnostics-prognostics',
        'Source': 'https://github.com/nasa/progpy',
    },
)
