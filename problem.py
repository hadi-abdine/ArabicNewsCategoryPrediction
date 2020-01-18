import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score


problem_title = 'Arabic News Category prediction'
_target_column_name = 'Type'
_prediction_label_names = ['أخبار محليّة', 'أخبار فنية', 'أخبار اقتصادية ومالية', 'أخبار رياضية', 'أخبار إقليمية ودولية']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions= rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
# An object implementing the workflow

class ANC(FeatureExtractorClassifier):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        super(ANC, self).__init__(workflow_element_names[:2])
        self.element_names= workflow_element_names

workflow = ANC()

class F1Score(BaseScoreType):
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1 score'):
        self.name = name

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        return f1_score(y_true, y_pred, average='weighted')


score_types= [
    F1Score(name='f1 score'),
]

def get_cv(X, y):
    cv= GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X, y, groups=X['Id'])

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False, compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_test_data(path="."):
    return _read_data(path, 'test.csv.zip')


def get_train_data(path="."):
    return _read_data(path, 'train.csv.zip')
