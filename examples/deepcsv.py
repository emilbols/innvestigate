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

import innvestigate
import innvestigate.applications
import innvestigate.applications.mnist
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from Losses import *

import os
import tempfile

def softmax(x):
                   """Compute softmax values for each sets of scores in x."""
                   e_x = np.exp((x.transpose()-x.max(axis=1)).transpose())
                   return e_x / np.sum(e_x,axis=1)[:,None]

def apply_modifications(model, custom_objects=None):
                   """Applies modifications to the model layers to create a new Graph. For example, simply changing
                   `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated                                             
                   with modified inbound and outbound tensors because of change in layer building function.                                                                    
                   Args:                                                                                                                                                       
                   model: The `keras.models.Model` instance.                                                                                                               
                   Returns:                                                                                                                                                    
                   The modified model with changes applied. Does not mutate the original `model`.                                                                          
                   """
                   # The strategy is to save the modified model and load it back. This is done because setting the activation
                   # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
                   # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
                   # multiple inbound and outbound nodes are allowed with the Graph API.
                   model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
                   try:
                                      model.save(model_path)
                                      return load_model(model_path, custom_objects=custom_objects)
                   finally:
                                      os.remove(model_path)



def heatmap(X):
                   X = ivis.gamma(X, minamp=0, gamma=0.95)
                   return ivis.heatmap(X)

#dropoutRate = 0.1
#nclasses = 4
#a = Input(shape=(66,))
#x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(a)
#x = Dropout(dropoutRate)(x)
#x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
#x = Dropout(dropoutRate)(x)
#x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
#x = Dropout(dropoutRate)(x)
#x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
#x = Dropout(dropoutRate)(x)
#x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
#predictions = Dense(nclasses, activation='linear',kernel_initializer='lecun_uniform')(x)
#model_nosoftmax = Model(inputs=a, outputs=predictions)


eutils = imp.load_source("utils", "utils.py")
imgnetutils = imp.load_source("utils_imagenet", "utils_imagenet.py")

x = np.load('DataStuff/Numpy_ttBar/ttBarDeepCSV_features_0.npy')
y = np.load('DataStuff/Numpy_ttBar/ttBarDeepCSV_truth_0.npy')
x = x[:400000]
y = y[:400000]

#model=load_model('DataStuff/small_DeepCSV.h5')
#model=load_model('DataStuff/DeepCSV_in_cmssw_nosoftmax.h5')
model=load_model('DataStuff/DeepCSV_model7_nosoftmax.h5')
#model = load_model('DataStuff/YetAnotherDeepCSV_nosoftmax.h5')
#model.save_weights('DataStuff/my_model_weights.h5')
#model_nosoftmax.load_weights('DataStuff/my_model_weights.h5')
model.summary()
#model_nosoftmax = model
#model_nosoftmax.layers[-1].activation = activations.linear 
#apply_modifications(model_nosoftmax)

newx = np.logspace(-3, 0, 100)
pred = model.predict(x)
comparison = softmax(pred)
cs = (y[:,3] == 0)
truth = np.clip(y[:,0]+y[:,1],0,1)[cs]
guess = (comparison[:,0]+comparison[:,1])[cs]
tmp_fpr, tmp_tpr, _ = roc_curve(truth, guess)
coords = pd.DataFrame()
coords['fpr'] = tmp_fpr
coords['tpr'] = tmp_tpr
clean = coords.drop_duplicates(subset=['fpr'])
spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
tprs = spline(newx)
plt.plot(tprs, newx)
plt.grid(which='both')
plt.yscale("log")
plt.show()


analyzer = innvestigate.create_analyzer("lrp.sequential_preset_b", model,epsilon = 1)
analysis = analyzer.analyze(x)
#selection = (( (y[:,0] == 1) | (y[:,1] == 1) ) & ( (comparison[:,0] + comparison[:,1]) > 0.7) )
#selection = (( (y[:,0] == 1) | (y[:,1] == 1) ) & ( (comparison[:,3]) > 0.7) )
#weights = analysis[selection]

weights = analysis
row_sums = np.sum(np.abs(weights),axis=1)
weights = weights / row_sums[:,None]
weights = weights[~np.isnan(weights).any(axis=1)]
mean = np.mean(weights,axis=0)
arguments = np.argsort(mean)
std = np.std(weights,axis=0)
std_args = np.argsort(std)
mean_sorted = mean[arguments[:]]
std_sorted = std[arguments[:]]
mean_std_sorted = mean[std_args[:]]
std_std_sorted = std[std_args[:]]


labels = ['jet_pt', 'jet_eta','TagVarCSV_jetNSecondaryVertices',
          'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR',
          'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm',
          'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm',
          'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks',
          'TagVarCSV_jetNTracksEtaRel',
          'track1_TagVarCSVTrk_trackJetDistVal',
          'track2_TagVarCSVTrk_trackJetDistVal',
          'track3_TagVarCSVTrk_trackJetDistVal',
          'track4_TagVarCSVTrk_trackJetDistVal',
          'track5_TagVarCSVTrk_trackJetDistVal',
          'track6_TagVarCSVTrk_trackJetDistVal',
          'track1_TagVarCSVTrk_trackPtRel',
          'track2_TagVarCSVTrk_trackPtRel',
          'track3_TagVarCSVTrk_trackPtRel',
          'track4_TagVarCSVTrk_trackPtRel',
          'track5_TagVarCSVTrk_trackPtRel',
          'track6_TagVarCSVTrk_trackPtRel',
          'track1_TagVarCSVTrk_trackDeltaR',
          'track2_TagVarCSVTrk_trackDeltaR',
          'track3_TagVarCSVTrk_trackDeltaR',
          'track4_TagVarCSVTrk_trackDeltaR',
          'track5_TagVarCSVTrk_trackDeltaR',
          'track6_TagVarCSVTrk_trackDeltaR',
          'track1_TagVarCSVTrk_trackPtRatio', 
          'track2_TagVarCSVTrk_trackPtRatio', 
          'track3_TagVarCSVTrk_trackPtRatio',
          'track4_TagVarCSVTrk_trackPtRatio',
          'track5_TagVarCSVTrk_trackPtRatio',
          'track6_TagVarCSVTrk_trackPtRatio',
          'track1_TagVarCSVTrk_trackSip3dSig',
          'track2_TagVarCSVTrk_trackSip3dSig',
          'track3_TagVarCSVTrk_trackSip3dSig',
          'track4_TagVarCSVTrk_trackSip3dSig',
          'track5_TagVarCSVTrk_trackSip3dSig',
          'track6_TagVarCSVTrk_trackSip3dSig',
          'track1_TagVarCSVTrk_trackSip2dSig',
          'track2_TagVarCSVTrk_trackSip2dSig',
          'track3_TagVarCSVTrk_trackSip2dSig',
          'track4_TagVarCSVTrk_trackSip2dSig',
          'track5_TagVarCSVTrk_trackSip2dSig',
          'track6_TagVarCSVTrk_trackSip2dSig',
          'track1_TagVarCSVTrk_trackDecayLenVal',
          'track2_TagVarCSVTrk_trackDecayLenVal',
          'track3_TagVarCSVTrk_trackDecayLenVal',
          'track4_TagVarCSVTrk_trackDecayLenVal',
          'track5_TagVarCSVTrk_trackDecayLenVal',
          'track6_TagVarCSVTrk_trackDecayLenVal',
          'track1_TagVarCSV_trackEtaRel',
          'track2_TagVarCSV_trackEtaRel',
          'track3_TagVarCSV_trackEtaRel',
          'track4_TagVarCSV_trackEtaRel',
          'TagVarCSV_vertexMass',
          'TagVarCSV_vertexNTracks',
          'TagVarCSV_vertexEnergyRatio',
          'TagVarCSV_vertexJetDeltaR',
          'TagVarCSV_flightDistance2dVal',
          'TagVarCSV_flightDistance2dSig',
          'TagVarCSV_flightDistance3dVal',
          'TagVarCSV_flightDistance3dSig']


lables_sorted = []
labels_std_sorted = []
for n in range(0, len(arguments)):
                   lables_sorted.append(labels[arguments[n]])
                   labels_std_sorted.append(labels[std_args[n]])
                   
even = np.linspace(1,66,66)
plt.errorbar(even,mean_sorted,std_sorted)
plt.xticks(even, lables_sorted, rotation=90)
plt.show()
plt.errorbar(even,mean_std_sorted,std_std_sorted)
plt.xticks(even, labels_std_sorted, rotation=90)
plt.show()

