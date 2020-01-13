import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit


problem_title = 'Arabic News Category prediction'
_target_column_name = 'Type' 
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_classification()
# An object implementing the workflow

class ANC(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        super(FAN, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = ANC()

# define the score (specific score for the FAN problem)
class FAN_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='fan error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        max_true = np.maximum(5., np.log10(np.maximum(1., y_true)))
        max_pred = np.maximum(5., np.log10(np.maximum(1., y_pred)))
        
        loss = np.mean(np.abs(max_true - max_pred))
        
        return loss

score_types = [
    FAN_error(name='fan error', precision=2),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X['Id'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'data_TRAIN.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'data_TEST.csv'
    return _read_data(path, f_name)