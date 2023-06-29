# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import json
import numpy as np

from progpy.utils.containers import DictLikeMatrixWrapper

__all__ = ['CustomEncoder', 'custom_decoder']


class CustomEncoder(json.JSONEncoder):
    """
    Custom encoder to serialize parameters
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return {'_original_type': 'ndarray', '_data': o.tolist()}
        elif isinstance(o, DictLikeMatrixWrapper):
            dict_temp = {k: v for k, v in o.items()}
            dict_temp['_original_type'] = 'DictLikeMatrixWrapper'
            return dict_temp
        elif isinstance(o, np.bool_):
            return bool(o)
        else: 
            import pickle
            from base64 import b64encode
            pkl_temp = b64encode(pickle.dumps(o))
            save_temp = {}
            save_temp['_data'] = pkl_temp.decode()
            save_temp['_original_type'] = 'pickled'
            return save_temp


def custom_decoder(o):
    """
    Custom decoder to deserialize parameters
    """
    if isinstance(o, dict) and '_original_type' in o.keys():
        if o['_original_type'] == 'ndarray':
            return np.array(o['_data'])
        elif o['_original_type'] == 'DictLikeMatrixWrapper':
            del o['_original_type']
            return DictLikeMatrixWrapper(list(o.keys()), o)
        elif o['_original_type'] == 'pickled':
            import pickle
            from base64 import b64decode
            pkl_temp1 = o['_data'].encode()
            pkl_temp2 = b64decode(pkl_temp1)
            return pickle.loads(pkl_temp2)
        else:
            raise Exception(f"Type {o['_original_type']} not supported by PrognosticsModel json decoder")
    return o
