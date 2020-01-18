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
from sklearn.impute import SimpleImputer


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
            X.loc[X['Desc']=='', 'Desc'] = X['Title']
            X['Title'] = X['Title'].apply(clean_txt)
            X['Desc'] = X['Desc'].apply(clean_txt)
            return np.c_[ X['Title'], X['Desc']]


        num_col = ['Id']
        date_col = ['Date']
        drop_cols = ['Image', 'Title', 'Desc']

        preprocessor = ColumnTransformer(
            transformers=[
                ('date', make_pipeline(date_transformer), date_col),
                ('num', numeric_transformer, num_col),
                ('drop cols', 'drop', drop_cols),
                ])

        X_array = preprocessor.fit_transform(X_encoded)
        return X_array