import pandas as pd
import numpy as np
import seaborn as sns
import re
import datetime
import dateutil.parser as dparser
import nltk
from nltk.corpus import stopwords
import unicodedata as ud
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
import keras.backend as K
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df

        path = os.path.dirname(__file__)
        
        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])

        def convert_to_date(x):
            dicta =  {"كانون الثاني": "january" ,"شباط": "february", "أيار": "may",  "نيسان": "April",
                "آذار": "march", "حزيران": "june", "تموز": "july", "آب": "august", "أيلول": "september",
                "تشرين الأول": "october", "تشرين الثاني": "november", "كانون الأول": "december",
                "الاثنين": "monday", "الثلاثاء": "tuesday", "الأربعاء": "wednesday", "الخميس": "thursday", "الجمعة": "friday",
                "السبت": "saturday", "الأحد": "sunday", "السبت": "saturday"}
            x_new = x
            for arabic, english in dicta.items():
                x_new = x_new.replace(arabic, english)
            x_new = dparser.parse(x_new, fuzzy=True)
            return x_new
        def process_date(X):
            date = X.Date.apply(convert_to_date)
            return np.c_[date.dt.year, date.dt.month, date.dt.day]
        date_transformer = FunctionTransformer(process_date, validate=False)


        def clean_txt(sent):
            """
                text: a string
                return: modified initial string
            """    
            stps_arabic = set(stopwords.words('arabic'))
            arabic_diacritics = re.compile("""
                                        ّ    | # Tashdid
                                        َ    | # Fatha
                                        ً    | # Tanwin Fath
                                        ُ    | # Damma
                                        ٌ    | # Tanwin Damm
                                        ِ    | # Kasra
                                        ٍ    | # Tanwin Kasr
                                        ْ    | # Sukun
                                        ـ     # Tatwil/Kashida
                                    """, re.VERBOSE)
            sent = str(sent)
            text = sent.strip()
            text = re.sub('[\n\r\t\xa0]', ' ', text)
            text = re.sub(arabic_diacritics, '', text)
            text = ''.join(c for c in text if not ud.category(c).startswith('P') and not c.isdigit())
            res = re.sub(' +', ' ', text)
            return [w for w in text.split() if w not in stps_arabic]

        def process_text(X):
            X['Desc'] = X['Desc'].astype('str')
            X['Title'] = X['Title'].astype('str')
    
            X.loc[X['Desc']=='', 'Desc'] = X['Title']
            X['Desc'] = X['Desc'].apply(clean_txt)
            return np.c_[X['Desc']]

	    # The maximum number of words to be used. (Most frequent)
        vocab_size = 20000
        # The padding and truncating types used.
        trunc_type = 'post'
        padding_type = 'post'
        # The OOV token (Out Of Vocabulary) will be included within the dictionary
        oov_tok = '<OOV>'
        # Max number of words in each complaint.
        MAX_SEQUENCE_LENGTH = 200
        # This is fixed.
        EMBEDDING_DIM = 100

        class FeatureSelector(BaseEstimator, TransformerMixin):
            #Class Constructor 
            def __init__(self, feature_names, vocab_size, oov_tok, MAX_SEQUENCE_LENGTH, padding_type, trunc_type):
                self.feature_names = feature_names
                self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
                self.max_length = MAX_SEQUENCE_LENGTH
                self.padding = padding_type
                self.truncating = trunc_type
        
            def fit(self, X, y=None):
                X_1 = process_text(X.loc[:, self.feature_names])
                self.tokenizer.fit_on_texts(X_1[:, 0])
                self.word_index = self.tokenizer.word_index
                return self
    
            def transform(self, X):
                X_1 = process_text(X.loc[:, self.feature_names])
                sequences = self.tokenizer.texts_to_sequences(X_1[:, 0])
                X = pad_sequences(sequences, maxlen=self.max_length, padding=self.padding, truncating=self.truncating)
                return X
    
            def fit_transform(self, X, y=None):
                self.fit(X, y)
                return self.transform(X)

        num_col = ['Id']
        date_col = ['Date']
        text_cols = ['Desc', 'Title']
        drop_cols = ['Image']

        non_text_preprocessor = ColumnTransformer(
                transformers=[
                    ('date', make_pipeline(date_transformer), date_col),
                    ('num', numeric_transformer, num_col),
                    ('drop cols', 'drop', drop_cols),
                    ])

        text_preprocessor = Pipeline(steps=[
            ('text', FeatureSelector(text_cols, vocab_size, oov_tok, MAX_SEQUENCE_LENGTH, padding_type, trunc_type)),
        ])
        X_train = text_preprocessor.fit_transform(X_df)
        X_extra = non_text_preprocessor.fit_transform(X_df)
        return X_extra, X_train
