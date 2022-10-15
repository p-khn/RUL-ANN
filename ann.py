import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn import metrics

import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError



class Ann():
    def __init__(self, random_state=35) -> None:
       self.random_state = random_state
       tensorflow.get_logger().setLevel(logging.ERROR)
       tensorflow.random.set_seed(self.random_state)
       np.random.seed(self.random_state)
       os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def ann_model(self,X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, y_test:pd.Series, epochs=1000, batch_size=50) -> Tuple:
        optimizer1 = Adam()
        n_inputs = X_train.shape[1]

        model = Sequential()
        model.add(Dense(500, input_dim=n_inputs, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer1 , loss='mse',  metrics=[
            MeanSquaredError(),
        ])
        es = EarlyStopping(monitor='val_loss',
                    mode='min',
                    patience=50,
                    restore_best_weights = True)
                    
        history = model.fit(X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=epochs,
                    batch_size=batch_size, 
                    callbacks=[es],
                    verbose=2)
        
        y_pred = model.predict(X_test)
        score = metrics.mean_absolute_error(y_test, y_pred)

        return history, score
