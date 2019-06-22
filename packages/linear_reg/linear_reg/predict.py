import numpy as np
import pandas as pd

from linear_reg.processing.data_management import load_pipeline
from linear_reg.config import config
from linear_reg.processing.validation import validate_inputs

from linear_reg import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_model_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(input_data):
    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _model_pipe.predict(validated_data[config.FEATURES])
    
    
    response = {'predictions': prediction, 'version':_version}
    
    _logger.info(
            f'Making predictions with model version: {_version}'
            f'Inputs: {validated_data}'
            f'Predictions: {results}')
    
    return response