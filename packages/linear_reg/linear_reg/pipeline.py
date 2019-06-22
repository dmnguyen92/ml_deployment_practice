from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from linear_reg.processing import preprocessors as pp
from linear_reg.config import config

import logging

_logger = logging.getLogger(__name__)

model_pipe = Pipeline([
        ('remove_nan',
             pp.RemoveNan(variables=config.FEATURES)),
         ('Linear_model',
             Ridge(alpha=0.5, random_state=0))
        ])