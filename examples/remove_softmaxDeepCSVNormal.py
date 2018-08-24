import warnings
warnings.simplefilter('ignore')
import numpy as np
import imp
import time
import matplotlib.pyplot as plt
import keras
import keras.backend
from keras.models import load_model, Model
from keras import activations
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from keras.layers import Input, Dense, Dropout



dropoutRate = 0.1
nclasses = 4
a = Input(shape=(66,))
x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(a)
x = Dropout(dropoutRate)(x)
x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropoutRate)(x)
x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropoutRate)(x)
x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropoutRate)(x)
x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
predictions = Dense(nclasses, activation='linear',kernel_initializer='lecun_uniform')(x)
model_nosoftmax = Model(inputs=a, outputs=predictions)


model=load_model('DataStuff/DeepCSV_in_cmssw.h5')
model.save_weights('DataStuff/DeepCSV_in_cmssw_weights.h5')                                                                                                                                              
model_nosoftmax.load_weights('DataStuff/DeepCSV_in_cmssw_weights.h5')
model_nosoftmax.compile(learningrate=0.003,
                             loss='categorical_crossentropy',
                             optimizer='Adam')
model_nosoftmax.save('DataStuff/DeepCSV_in_cmssw_nosoftmax.h5')