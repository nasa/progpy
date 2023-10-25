# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import io
import numpy as np
import pandas as pd
import requests
import zipfile

cache = None
URL = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"


def load_data(dataset_id: int) -> tuple:
    """
    .. versionadded:: 1.3.0
    
    Loads data for one CMAPSS trajectory from NASA's PCoE Dataset. See '6. Turbofan Engine Degredation Simulation Data Set' at
    https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

    Data Set: 1
        | Train trajectories: 100
        | Test trajectories: 100
        | Conditions: ONE (Sea Level)
        | Fault Modes: ONE (HPC Degradation)

    Data Set: 2
        | Train trajectories: 260
        | Test trajectories: 259
        | Conditions: SIX
        | Fault Modes: ONE (HPC Degradation)

    Data Set: 3
        | Train trajectories: 100
        | Test trajectories: 100
        | Conditions: ONE (Sea Level)
        | Fault Modes: TWO (HPC Degradation, Fan Degradation)

    Data Set: 4
        | Train trajectories: 248
        | Test trajectories: 249
        | Conditions: SIX 
        | Fault Modes: TWO (HPC Degradation, Fan Degradation)

    Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

    The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data. [0]_

    Args:
        dataset_id (int): Dataset id

    Raises:
        ValueError: Data not in dataset (should be 1-4)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.array]: Tuple of data: training data, testing data, time of end of life)
        
        Each row of the training and testing data is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns in the pandas dataframe correspond to:
            1)	unit number
            2)	time, in cycles
            3)	operational setting 1
            4)	operational setting 2
            5)	operational setting 3
            6)	sensor measurement  1
            7)	sensor measurement  2

            ...

            26)	sensor measurement  21
    
    Raises:
        ValueError: Data not in dataset (should be 1-4)
        ConnectionError: Failed to download data. This may be because of issues with your internet connection or the datasets may have moved. Please check your internet connection and make sure you're using the latest version of progpy.
    
    References:
        .. [0] A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
    """
    global cache

    if dataset_id not in range(1, 5):
        raise ValueError(f"Invalid dataset id {dataset_id}")

    dataset_id = f"FD0{dataset_id:02d}"
    if cache is None:
        # Download data
        try:
            response = requests.get(URL, allow_redirects=True)
        except requests.exceptions.RequestException: 
            # handle chain of errors
            raise ConnectionError("Data download failed. This may be because of issues with your internet connection or the datasets may have moved. Please check your internet connection and make sure you're using the latest version of progpy. If the problem persists, please submit an issue on the progpy issue page (https://github.com/nasa/progpy/issues) for further investigation.")

        # Unzip response
        try:
            cache = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            # In this case the url may have been forwarded to another page
            raise ConnectionRefusedError("Data unzip failed- The site may be down or the datasets may have moved. Please try again later and make sure you're using the latest version of progpy. If the problem persists, please submit an issue on the progpy issue page (https://github.com/nasa/progpy/issues) for further investigation.")

    # Read Files
    with cache.open(f'test_{dataset_id}.txt', mode='r') as f:
        with io.BufferedReader(f) as f2:
            test = np.loadtxt(f2)
            test = pd.DataFrame(
                test,
                columns=['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)])

    with cache.open(f'train_{dataset_id}.txt', mode='r') as f:
        with io.BufferedReader(f) as f2:
            train = np.loadtxt(f2)
            train = pd.DataFrame(
                train,
                columns=['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)])

    with cache.open(f'RUL_{dataset_id}.txt', mode='r') as f:
        with io.BufferedReader(f) as f2:
            rul = np.loadtxt(f2)

    # Return results
    return (test, train, rul)

def clear_cache() -> None:
    """
    Clears the cache of downloaded data
    """
    global cache
    cache = None
