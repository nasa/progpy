# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import io
import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat
import zipfile

# Map of battery to url for data
urls = {
    'RW1': "https://zenodo.org/records/15277374/files/3.%20Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post.zip",
    'RW2': "https://zenodo.org/records/15277374/files/3.%20Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post.zip",
    'RW3': "https://zenodo.org/records/15277374/files/2.%20Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post.zip",
    'RW4': "https://zenodo.org/records/15277374/files/2.%20Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post.zip",
    'RW5': "https://zenodo.org/records/15277374/files/2.%20Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post.zip",
    'RW6': "https://zenodo.org/records/15277374/files/2.%20Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post.zip",
    'RW7': "https://zenodo.org/records/15277374/files/3.%20Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post.zip",
    'RW8': "https://zenodo.org/records/15277374/files/3.%20Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post.zip",
    'RW9': "https://zenodo.org/records/15277374/files/1.%20Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip",
    'RW10': "https://zenodo.org/records/15277374/files/1.%20Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip",
    'RW11': "https://zenodo.org/records/15277374/files/1.%20Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip",
    'RW12': "https://zenodo.org/records/15277374/files/1.%20Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip",
    'RW13': "https://zenodo.org/records/15277374/files/7.%20RW_Skewed_Low_Room_Temp_DataSet_2Post.zip",
    'RW14': "https://zenodo.org/records/15277374/files/7.%20RW_Skewed_Low_Room_Temp_DataSet_2Post.zip",
    'RW15': "https://zenodo.org/records/15277374/files/7.%20RW_Skewed_Low_Room_Temp_DataSet_2Post.zip",
    'RW16': "https://zenodo.org/records/15277374/files/7.%20RW_Skewed_Low_Room_Temp_DataSet_2Post.zip",
    'RW17': "https://zenodo.org/records/15277374/files/5.%20RW_Skewed_High_Room_Temp_DataSet_2Post.zip",
    'RW18': "https://zenodo.org/records/15277374/files/5.%20RW_Skewed_High_Room_Temp_DataSet_2Post.zip",
    'RW19': "https://zenodo.org/records/15277374/files/5.%20RW_Skewed_High_Room_Temp_DataSet_2Post.zip",
    'RW20': "https://zenodo.org/records/15277374/files/5.%20RW_Skewed_High_Room_Temp_DataSet_2Post.zip",
    'RW21': "https://zenodo.org/records/15277374/files/6.%20RW_Skewed_Low_40C_DataSet_2Post.zip?download=1",
    'RW22': "https://zenodo.org/records/15277374/files/6.%20RW_Skewed_Low_40C_DataSet_2Post.zip?download=1",
    'RW23': "https://zenodo.org/records/15277374/files/6.%20RW_Skewed_Low_40C_DataSet_2Post.zip?download=1",
    'RW24': "https://zenodo.org/records/15277374/files/6.%20RW_Skewed_Low_40C_DataSet_2Post.zip?download=1",
    'RW25': "https://zenodo.org/records/15277374/files/4.%20RW_Skewed_High_40C_DataSet_2Post.zip",
    'RW26': "https://zenodo.org/records/15277374/files/4.%20RW_Skewed_High_40C_DataSet_2Post.zip",
    'RW27': "https://zenodo.org/records/15277374/files/4.%20RW_Skewed_High_40C_DataSet_2Post.zip",
    'RW28': "https://zenodo.org/records/15277374/files/4.%20RW_Skewed_High_40C_DataSet_2Post.zip",
}

cache = {}  # Cache for downloaded data
# Cache is used to prevent files from being downloaded twice

def load_data(batt_id: str) -> tuple:
    """
    .. versionadded:: 1.3.0

    Loads data for one or more batteries from NASA's PCoE Dataset, '11. Randomized Battery Usage Data Set'
    https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

    Args:
        batt_id (str): Battery name from dataset (RW1-28)

    Raises:
        ValueError: Battery not in dataset (should be RW1-28)

    Returns:
        tuple[dict, list[pd.DataFrame]]: Data and description as a tuple (description, data), where the data is a list of pandas DataFrames such that data[i] is the data for run i, corresponding with details[i], above. The columns of the dataframe are ('relativeTime', 'current' (amps), 'voltage', 'temperature' (°C)) in that order.

    Raises:
        ValueError: Battery id must be a string or int
        ConnectionError: Failed to download data. This may be because of issues with your internet connection or the datasets may have moved. Please check your internet connection and make sure you're using the latest version of progpy.
    """
    if isinstance(batt_id, int):
        # Convert to string
        batt_id = 'RW' + str(batt_id)
    if not isinstance(batt_id, str):
        raise ValueError('Battery ID must be a string')

    if batt_id not in urls:
        raise ValueError('Unknown battery ID: {}'.format(batt_id))

    url = urls[batt_id]

    if url not in cache:
        # Download data
        try:
            response = requests.get(url, allow_redirects=True)
        except requests.exceptions.RequestException:  # handle chain of errors
            raise ConnectionRefusedError("Data download failed. This may be because of issues with your internet connection or the datasets may have moved. Please check your internet connection and make sure you're using the latest version of progpy. If the problem persists, please submit an issue on the progpy issue page (https://github.com/nasa/progpy/issues) for further investigation.")

        # Unzip response
        try:
            cache[url] = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            # In this case the url may have been forwarded to another page
            raise ConnectionRefusedError("Data unzip failed- The site may be down or the datasets may have moved. Please try again later and make sure you're using the latest version of progpy. If the problem persists, please submit an issue on the progpy issue page (https://github.com/nasa/progpy/issues) for further investigation.")

    f = cache[url].open(f'{cache[url].infolist()[0].filename}Matlab/{batt_id}.mat')

    # Load matlab file
    result = loadmat(f)['data']

    # Reformat
    desc = {
        'procedure': str(result['procedure'][0, 0][0]),
        'description': str(result['description'][0, 0][0]),
        'runs':
        [
            {
                'type': str(run_type[0]),
                'desc': str(desc[0]),
                'date': str(date[0])
            } for (run_type, desc, date) in zip(result['step'][0, 0]['type'][0], result['step'][0, 0]['comment'][0], result['step'][0, 0]['date'][0])
        ]
    }

    result = result['step'][0, 0]
    result = [
        pd.DataFrame(
            np.array([
                result[key][0, i][0] for key in ('relativeTime', 'current', 'voltage', 'temperature')
            ], np.float64).T,
            columns=('relativeTime', 'current', 'voltage', 'temperature')
        ) for i in range(result.shape[1])
    ]
    for r in result:
        r.set_index('relativeTime')

    return desc, result

def clear_cache() -> None:
    """Clears the cache of downloaded data"""
    cache.clear()
