import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.metrics import f1_score
import os
from sklearn import preprocessing
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

problem_title = 'Arabic News Category prediction'
_target_column_name = 'Type'
_prediction_label_names = [0, 1, 2, 3, 4]
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

    def __init__(self, name='f1 score', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        y_true = np.argmax(y_true, -1)
        y_pred = np.argmax(y_pred, -1)
        return f1_score(y_true, y_pred, average='weighted')


score_types= [
    F1Score(name='f1 score',precision=3),
]

def get_cv(X, y):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    cv= StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    return cv.split(X, y)

def _read_data(path, f_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False, compression='zip')
    y_array = data[_target_column_name].values
    categories = ['أخبار محليّة', 'أخبار فنية', 'أخبار اقتصادية ومالية', 'أخبار رياضية', 'أخبار إقليمية ودولية']
    class2index = dict(zip(categories, range(len(categories))))
    y_array = np.array([class2index[cat] for cat in y_array])
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_test_data(path="."):
    return _read_data(path, 'test.csv.zip')


def get_train_data(path="."):
    return _read_data(path, 'train.csv.zip')
