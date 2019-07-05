import os
import pathlib
import pandas as pd
from inspect import getsourcefile

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = os.path.abspath(os.path.join(getsourcefile(lambda:0), '..','..'))
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT,'trained_models')
DATASET_DIR = os.path.join(PACKAGE_ROOT,'datasets')

TRAINING_DATA_FILE = 'data.csv'
TESTING_DATA_FILE = 'data_test.csv'

FEATURES = ['X']
TARGET = 'Y'

PIPELINE_NAME = 'linear_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

ACCEPTABLE_MODEL_DIFFERENCE = 0.5


