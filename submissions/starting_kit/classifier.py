from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
import keras.backend as K
from sklearn import preprocessing
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Classifier(object):
    def __init__(self):
        self.vocab_size = 20000
        self.max_length = 200
        self.EMBEDDING_DIM = 100
        self.model = Sequential()
        self.enc = preprocessing.OneHotEncoder()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        K.clear_session()
        self.model.add(Embedding(self.vocab_size, self.EMBEDDING_DIM, input_length=self.max_length))
        self.model.add(SpatialDropout1D(0.25))
        self.model.add(LSTM(32))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[self.get_f1])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #@staticmethod
    def get_f1(self, y_true, y_pred):
        
        y_true = K.argmax(y_true, -1)
        y_pred = K.argmax(y_pred, -1)
        
        prec_num = 0
        prec_den = 0
        recall_num = 0
        recall_den = 0
        
        equal = K.equal(y_true, y_pred)
        not_equal = K.not_equal(y_true, y_pred)
        
        for i in range(5):
            tmp = K.sum(K.cast(K.all(K.stack([equal, K.equal(y_true, i)], axis=0), axis=0), 'int32'))
            prec_num += tmp
            prec_den += tmp + K.sum(K.cast(K.all(K.stack([not_equal, K.equal(y_pred, i)], axis=0), axis=0), 'int32'))
            recall_num += tmp
            recall_den += tmp + K.sum(K.cast(K.all(K.stack([not_equal, K.equal(y_true, i)], axis=0), axis=0), 'int32'))
            
        prec_num = K.cast(prec_num, 'float')
        prec_den = K.cast(prec_den, 'float')
        recall_num = K.cast(recall_num, 'float')
        recall_den = K.cast(recall_den, 'float')
        
        precision = prec_num / prec_den
        recall = recall_num / recall_den
        f1_val = 2*(precision * recall)/(precision + recall)
        return f1_val

    def fit(self, X, y):
        self.enc.fit(y.reshape(-1, 1))
        Y_train = self.enc.transform(y.reshape(-1, 1)).toarray()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.model.fit(X[1], Y_train, epochs=4, batch_size=32, verbose=0)
    
    def predict_proba(self, X):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        return self.model.predict_proba(X[1])


