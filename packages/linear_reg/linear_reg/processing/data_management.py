import pandas as pd
import os
from os import listdir
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from linear_reg.config import config
from linear_reg import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def load_dataset(file_name):
    """
    Load training data
    
    Arguments:
        file_name -- name of file
    
    Returns:
        _data -- panda dataframe
    """
    _data = pd.read_csv(os.path.join(config.DATASET_DIR,file_name))
    return _data


def save_pipeline(pipeline_to_persist):
    """
    Save the pipeline
    
    Arguments:
        pipeline_to_persist -- pipeline to save
    """
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = os.path.join(config.TRAINED_MODEL_DIR, save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    
    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f'saved pipeline: {save_file_name}')
    print('Pipeline saved')
    
    
def load_pipeline(file_name):
    """
    Load a saved pipeline
    
    Arguments:
        file_name -- name of the pipeline to load
    """
    file_path = os.path.join(config.TRAINED_MODEL_DIR, file_name)
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(files_to_keep):
    """
    Remove all old models
    """
    for model_file in os.listdir(config.TRAINED_MODEL_DIR):
        if model_file not in [files_to_keep, '__init__.py']:
            os.remove(os.path.join(config.TRAINED_MODEL_DIR,model_file))
    
    


