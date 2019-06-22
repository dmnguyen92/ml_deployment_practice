import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from linear_reg import pipeline
from linear_reg.processing.data_management import load_dataset, save_pipeline
from linear_reg.config import config
from linear_reg import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def run_training():
    """Train the model"""
    
    # Prepare data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    
    X_train, X_test, y_train, y_test = train_test_split(
            data[config.FEATURES],
            data[config.TARGET],
            test_size=0.1,
            random_state=0)
    
    pipeline.model_pipe.fit(X_train[config.FEATURES], y_train)
    
    # Return error
    y_pred = pipeline.model_pipe.predict(X_test[config.FEATURES])
    rmse = mean_squared_error(y_test,y_pred)**(1/2)
    print('Mean squared error: %.3f' %rmse)
        
    _logger.info(f'Saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.model_pipe)
    
if __name__ == '__main__':
    run_training()