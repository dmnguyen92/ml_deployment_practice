import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveNan(BaseEstimator, TransformerMixin):
    """Remove nan from dataframe."""
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for variable in self.variables:
            X[variable] = X_copy[variable].fillna(X_copy[variable].mean())
        return X
        
        
        